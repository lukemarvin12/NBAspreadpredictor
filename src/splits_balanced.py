"""
src/splits_balanced.py

Goal:
    Create train / val / test splits that are:
      - Based only on recent seasons (to avoid old-era basketball)
      - Balanced across:
            * seasons
            * days of the week
            * parts of the season (early / mid / late)
      - So that each split has a similar mix of game types:
            * not all 2020 games in test
            * not all Friday games in train
            * not all late-season games in val

Output:
    data/processed/nba_ats_features_balanced_splits.csv
    with a new column: 'split' ∈ {train, val, test}
"""

import numpy as np
import pandas as pd

# ---- File paths ----
INPUT_PATH = "data/processed/nba_ats_features_basic.csv"
OUTPUT_PATH = "data/processed/nba_ats_features_balanced_splits.csv"

# ---- Era / data settings ----
MIN_SEASON = 2018   # try to only use 2018+ (modern NBA)
MIN_GAMES  = 5000   # if we get fewer than this after filtering, extend backwards

# ---- Split proportions ----
TRAIN_FRACTION = 0.7
VAL_FRACTION   = 0.15
# TEST_FRACTION = 1 - TRAIN_FRACTION - VAL_FRACTION


def choose_recent_era(df: pd.DataFrame,
                      min_season: int = MIN_SEASON,
                      min_games: int = MIN_GAMES) -> pd.DataFrame:
    """
    Filter to a "recent era" of seasons.

    Strategy:
      1) Try to keep only seasons >= min_season (e.g. 2018+).
      2) If that leaves too few games (< min_games), extend backwards
         season-by-season until we have at least min_games.

    This follows your idea:
      "Use data from a few recent years (assuming there is enough data,
       if not use further back)."
    """
    if "season" not in df.columns:
        raise ValueError("Dataframe must have a 'season' column for era selection.")

    # First, try the simple filter
    df_recent = df[df["season"] >= min_season]
    if len(df_recent) >= min_games:
        print(f"Using seasons >= {min_season} only (rows: {len(df_recent)}).")
        return df_recent

    # If not enough rows, extend backwards until we hit min_games
    print(
        f"Not enough rows with season >= {min_season} "
        f"({len(df_recent)} < {min_games}). Extending backwards..."
    )

    # Sort unique seasons from most recent to oldest
    seasons = sorted(df["season"].unique(), reverse=True)

    selected_seasons = []
    running_total = 0
    for s in seasons:
        selected_seasons.append(s)
        running_total += (df["season"] == s).sum()
        if running_total >= min_games:
            break

    print(f"Selected seasons: {sorted(selected_seasons)} (rows: {running_total}).")
    return df[df["season"].isin(selected_seasons)]


def add_strata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the columns we want to stratify on:

      - day of week (dow)
      - part of season (early / mid / late)
      - season number
      - a combined 'strata' key that groups games with the same:
            (season, dow, season_part)

    Every strata group will be entirely assigned to train OR val OR test.
    This way, each split has a similar mix of game "types".
    """
    df = df.copy()

    # ---- Day of week (0=Mon, 6=Sun) ----
    df["dow"] = df["date"].dt.dayofweek

    # ---- Season part: early / mid / late within each season ----
    # Idea:
    #   For each season, cut the dates into 3 roughly equal time bins using
    #   the 33% and 66% date quantiles.
    season_part = pd.Series(index=df.index, dtype="object")

    for season, grp in df.groupby("season"):
        # Get the 33% and 66% quantiles of the dates for this season
        q1, q2 = grp["date"].quantile([0.33, 0.66])

        is_early = grp["date"] <= q1
        is_mid   = (grp["date"] > q1) & (grp["date"] <= q2)
        # Remaining games are "late"
        is_late  = ~(is_early | is_mid)

        season_part.loc[grp.index[is_early]] = "early"
        season_part.loc[grp.index[is_mid]]   = "mid"
        season_part.loc[grp.index[is_late]]  = "late"

    df["season_part"] = season_part

    # ---- Combined strata label ----
    # This uniquely identifies "buckets" like:
    #   2021, Friday, mid-season
    df["strata"] = (
        df["season"].astype(str)
        + "_d" + df["dow"].astype(str)
        + "_p" + df["season_part"].astype(str)
    )

    return df


def make_balanced_splits(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    random_state: int = 42,
) -> pd.Series:
    """
    Assign each game to train / val / test, in a way that:

      - operates at the 'strata' group level
      - ensures each split contains a mix of:
            seasons, days of week, and season parts
      - avoids all of test being from one weird subset (e.g., all Fridays)

    We:
      1) Take the unique strata labels.
      2) Shuffle them.
      3) Assign some strata to train, some to val, some to test.
      4) All games in the same strata go to the same split.
    """
    if "strata" not in df.columns:
        raise ValueError("Dataframe must contain a 'strata' column. Run add_strata_columns first.")

    rng = np.random.default_rng(random_state)

    strata_labels = df["strata"].unique()
    rng.shuffle(strata_labels)

    n = len(strata_labels)
    n_train = int(n * train_fraction)
    n_val   = int(n * val_fraction)
    # Test gets whatever is left
    n_test  = n - n_train - n_val

    train_strata = set(strata_labels[:n_train])
    val_strata   = set(strata_labels[n_train:n_train + n_val])
    test_strata  = set(strata_labels[n_train + n_val:])

    print("Number of strata groups:", n)
    print("  Train strata:", len(train_strata))
    print("  Val strata:  ", len(val_strata))
    print("  Test strata: ", len(test_strata))

    # Assign splits based on which strata each row belongs to
    split = pd.Series(index=df.index, dtype="object")
    split[df["strata"].isin(train_strata)] = "train"
    split[df["strata"].isin(val_strata)]   = "val"
    split[df["strata"].isin(test_strata)]  = "test"

    # Any rows not covered (should be zero, but just in case)
    split.fillna("unassigned", inplace=True)

    return split


def summarize_splits(df: pd.DataFrame) -> None:
    """
    Print some summary tables so you can see that:

      - each split has representation from each season
      - each split has games on each day of the week
      - each split has early / mid / late season games
    """
    print("\n=== Split counts ===")
    print(df["split"].value_counts())

    print("\n=== Split by season ===")
    print(pd.crosstab(df["season"], df["split"]))

    print("\n=== Split by day-of-week (0=Mon..6=Sun) ===")
    print(pd.crosstab(df["dow"], df["split"]))

    print("\n=== Split by season_part ===")
    print(pd.crosstab(df["season_part"], df["split"]))


def main():
    print("Loading:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    # 1) Choose only recent seasons (and extend back if needed)
    df = choose_recent_era(df)

    # 2) Add strata columns (season, dow, season_part, strata)
    df = add_strata_columns(df)

    # 3) Build balanced splits by strata
    print("Creating balanced splits...")
    df["split"] = make_balanced_splits(df)

    # 4) Sanity checks and summaries
    summarize_splits(df)

    # 5) Sort within each split by date (makes downstream usage cleaner)
    df = df.sort_values(["split", "date"]).reset_index(drop=True)

    print("Saving with balanced splits to:", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()