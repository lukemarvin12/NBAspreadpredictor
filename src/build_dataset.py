import os
import pandas as pd

# ---------- Paths ----------
RAW_PATH = "data/raw/nba_2008-2025.csv"
PROCESSED_DIR = "data/processed"
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
    home_spread = df["spread"].astype(float)

    # Home favored -> negative spread from home POV
    mask_home_fav = df["whos_favored"].str.lower() == "home"
    home_spread[mask_home_fav] = -home_spread[mask_home_fav]

    # Away favored -> positive spread from home POV
    # (if not home, we assume away; pick'em will just be 0)
    # If you have a special value for pick'em, you can handle it here.

    return home_spread


def implied_prob(ml):
    """
    Convert American moneyline to implied probability.
    Returns a float in [0, 1] or None if ml is NaN.
    """
    if pd.isna(ml):
        return None
    ml = float(ml)
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
    df = pd.read_csv(RAW_PATH)

    # ---------- 2. Basic cleaning ----------
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Drop rows with no spread (can't do ATS without it)
    df = df.dropna(subset=["spread", "whos_favored", "score_home", "score_away"])

    # ---------- 3. Signed spread from HOME perspective ----------
    df["spread_home"] = convert_home_spread(df)

    # If you want all downstream code to just use 'spread' as home-signed:
    df["spread"] = df["spread_home"]

    # ---------- 4. Game outcome + ATS targets (HOME perspective) ----------

    # Raw margin from HOME team's perspective
    df["margin_home"] = df["score_home"] - df["score_away"]

    # Home straight-up win / loss
    df["home_win"] = (df["margin_home"] > 0).astype(int)
    df["away_win"] = (df["margin_home"] < 0).astype(int)

    # ATS adjusted margin:
    # margin_home + spread_home > 0 -> HOME covers
    # margin_home + spread_home < 0 -> AWAY covers
    # margin_home + spread_home = 0 -> PUSH
    df["ats_result_raw"] = df["margin_home"] + df["spread_home"]

    df["ats_home_cover"] = (df["ats_result_raw"] > 0).astype(int)   # 1 if home covers, 0 otherwise

    def label_ats(x: float) -> str:
        if x > 0:
            return "home_cover"
        elif x < 0:
            return "away_cover"
        else:
            return "push"

    df["ats_label"] = df["ats_result_raw"].apply(label_ats)

    # ---------- 5. Basic features (HOME perspective) ----------

    # Who is favored (home vs away), as a binary feature
    df["is_home_favored"] = (df["whos_favored"].str.lower() == "home").astype(int)

    # Absolute size of the (home POV) spread
    df["abs_spread"] = df["spread"].abs()

    # Total points scored in the game
    df["total_points_scored"] = df["score_home"] + df["score_away"]

    # Absolute final score margin
    df["abs_margin_home"] = df["margin_home"].abs()

    # Moneyline implied probabilities if you have them
    if "moneyline_home" in df.columns and "moneyline_away" in df.columns:
        df["p_home_ml"] = df["moneyline_home"].apply(implied_prob)
        df["p_away_ml"] = df["moneyline_away"].apply(implied_prob)
        df["p_home_edge_vs_ml"] = df["home_win"] - df["p_home_ml"]
        # Difference in implied win prob home vs away
        df["p_home_minus_away_ml"] = df["p_home_ml"] - df["p_away_ml"]

    # You can add more features later (rolling stats, ELO, rest days, etc.)

    return df


def main():
    # Build the cleaned + feature-engineered dataset
    df = build_dataset()

    # ---------- 6. Ensure processed directory exists ----------
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ---------- 7. Save to CSV ----------
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed dataset to: {PROCESSED_PATH}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    # CLI entrypoint: run this from the project root as:
    #   python3 src/build_dataset.py
    main()