import os
import pandas as pd

# ---------- Paths ----------
# Path to the raw CSV that contains odds + results. Relative to project root.
RAW_PATH = "data/raw/nba_2008-2025.csv"

# Directory where we will write processed data (created if missing).
PROCESSED_DIR = "data/processed"

# Full path to the processed CSV output file.
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "nba_ats_clean.csv")


def convert_home_spread(df: pd.DataFrame) -> pd.Series:
    """
    Convert raw spread + whos_favored into a signed spread from the HOME team's perspective.

    Assumptions:
    - 'spread' in the raw data is the magnitude of the line (always >= 0).
    - 'whos_favored' is either 'home' or 'away' (or similar).
    
    We define:
      home_spread < 0  -> home is favored by |home_spread| points
      home_spread > 0  -> home is the underdog by home_spread points
    
    Examples:
      whos_favored = 'home',  spread = 6.5  -> home_spread = -6.5  (home -6.5)
      whos_favored = 'away',  spread = 3.5  -> home_spread =  3.5  (away -3.5, home +3.5)
    """
    # Start with the numeric spread column. The raw 'spread' is assumed to
    # be the magnitude of the line (e.g. 5.5) without a sign indicating which
    # team is favored. We will convert it into a signed number from the
    # HOME team's perspective.
    home_spread = df["spread"].astype(float)
    

    # If 'whos_favored' explicitly says 'home', then the home team is
    # favored and we represent that as a negative number (home -points).
    # Example: spread 6.5, whos_favored == 'home' => home_spread = -6.5
    mask_home_fav = df["whos_favored"].str.lower() == "home"
    home_spread[mask_home_fav] = -home_spread[mask_home_fav]

    # If the away team is favored, we leave the spread positive from the
    # home-perspective (home is the underdog). If the line is a pick'em
    # and whos_favored isn't 'home' or 'away', this will just keep the 0.
    return home_spread


def implied_prob(ml):
    """
    Convert American moneyline to implied probability.
    Returns a float in [0, 1] or None if ml is NaN.
    """
    # If no moneyline is provided, return None so downstream code can
    # handle missingness explicitly.
    if pd.isna(ml):
        return None

    ml = float(ml)

    # American moneyline -> implied probability conversion.
    # Positive ML (e.g. +150): implied prob = 100/(ml + 100)
    # Negative ML (e.g. -120): implied prob = -ml/(-ml + 100)
    # The returned value is in range (0,1).
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return -ml / (-ml + 100.0)


def build_dataset() -> pd.DataFrame:
    """
    Load raw NBA odds/results data, clean it, create HOME-ATS targets,
    and add some basic engineered features, all from the HOME team's perspective.
    """
    # ---------- 1. Load ----------
    # Read the raw CSV into a pandas DataFrame. This file should contain
    # one row per game and columns for date, teams, scores, spread, etc.
    df = pd.read_csv(RAW_PATH)

    # ---------- 2. Basic cleaning ----------
    # Convert date column to pandas datetime for ordering and potential
    # time-based feature engineering later.
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["playoffs"] == False]
    df = df[df["regular"] == True]

    # Sort rows by date to maintain chronological order. This is important
    # if you later compute rolling/lag features or split train/test by time.
    df = df.sort_values("date").reset_index(drop=True)

    # Drop any games that are missing essential fields needed to compute
    # ATS (against-the-spread) outcomes: spread, which side was favored,
    # and the final scores for each team.
    df = df.dropna(subset=["spread", "whos_favored", "score_home", "score_away"])

    # ---------- 3. Signed spread from HOME perspective ----------
    # Convert the raw magnitude spread + who was favored into a signed
    # spread from the home team's point of view using the helper above.
    df["spread_home"] = convert_home_spread(df)

    # Overwrite the original 'spread' column with the home-perspective
    # signed spread so later code can consistently use 'spread'. This is
    # a convenience; we keep the 'spread_home' column as well for clarity.
    df["spread"] = df["spread_home"]

    # ---------- 4. Game outcome + ATS targets (HOME perspective) ----------

    # margin_home: positive if home scored more points than away
    df["margin_home"] = df["score_home"] - df["score_away"]

    # Straight-up win indicators (binary 0/1)
    df["home_win"] = (df["margin_home"] > 0).astype(int)
    df["away_win"] = (df["margin_home"] < 0).astype(int)

    # ATS (against-the-spread) result is margin + signed spread. From the
    # home-perspective: if margin_home + spread_home > 0 then the HOME team
    # covered the spread. If it's < 0 the AWAY team covered. If == 0 it's a
    # PUSH (exact tie with the spread).
    df["ats_result_raw"] = df["margin_home"] + df["spread_home"]

    # Binary indicator: 1 when home covers, otherwise 0. Push is treated
    # as not covering (0) here; adjust if you prefer to handle pushes
    # explicitly in modeling.
    df["ats_home_cover"] = (df["ats_result_raw"] > 0).astype(int)

    # Human-readable label for ATS outcome: 'home_cover', 'away_cover', or 'push'
    def label_ats(x: float) -> str:
        if x > 0:
            return "home_cover"
        elif x < 0:
            return "away_cover"
        else:
            return "push"

    df["ats_label"] = df["ats_result_raw"].apply(label_ats)

    # ---------- 5. Basic features (HOME perspective) ----------

    # Binary flag indicating if the home team was favored on the line.
    df["is_home_favored"] = (df["whos_favored"].str.lower() == "home").astype(int)

    # Magnitude of the spread (how many points) regardless of sign.
    df["abs_spread"] = df["spread"].abs()

    # Total points in the game (useful for over/under related features).
    df["total_points_scored"] = df["score_home"] + df["score_away"]

    # Absolute final margin (how close the game was irrespective of winner).
    df["abs_margin_home"] = df["margin_home"].abs()

    # If moneyline columns exist, convert to implied probabilities and
    # compute a couple of simple derived features comparing ML to the
    # actual outcome.
    if "moneyline_home" in df.columns and "moneyline_away" in df.columns:
        df["p_home_ml"] = df["moneyline_home"].apply(implied_prob)
        df["p_away_ml"] = df["moneyline_away"].apply(implied_prob)

        # Simple calibration/edge feature: did the home team win (0/1) minus
        # the moneyline-implied probability. Positive means home outperformed
        # the moneyline expectation on average for that sample row.
        df["p_home_edge_vs_ml"] = df["home_win"] - df["p_home_ml"]

        # Difference between home and away implied win probabilities.
        df["p_home_minus_away_ml"] = df["p_home_ml"] - df["p_away_ml"]

    # You can add more engineered features here (rolling averages, team
    # strength ratings, rest days, head-to-head, injuries, ELO, betting edges,
    # etc.) depending on the modeling approach.

    return df


def main():
    # Build the cleaned + feature-engineered dataset
    df = build_dataset()

    # ---------- 6. Ensure processed directory exists ----------
    # Create target directory if it doesn't exist. exist_ok=True avoids
    # raising an error when the directory already exists.
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ---------- 7. Save to CSV ----------
    # Write the processed DataFrame to CSV for use by modeling/training code.
    df.to_csv(PROCESSED_PATH, index=False)

    # Print a short summary so the user knows where the file was written and
    # how large it is.
    print(f"Saved processed dataset to: {PROCESSED_PATH}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    # CLI entrypoint: run this from the project root as:
    #   python3 src/build_dataset.py
    main()