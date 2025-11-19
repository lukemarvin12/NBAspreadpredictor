"""
build_dataset.py

Takes the merged odds + stats file and builds a clean, home-team–centric
ATS dataset with a binary target `covered_home`.

Input:
    data/processed/nba_2018-2025_with_stats_merged.csv

Output:
    data/processed/nba_ats_clean.csv
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_PATH = "data/processed/nba_2018-2025_with_stats_merged.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nba_ats_clean.csv")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_first_column(df: pd.DataFrame, candidates: list[str], logical_name: str) -> str:
    """
    Return the first column name from `candidates` that exists in df.
    If none exist, raise a clear error showing what we tried and what exists.
    """
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find a column for {logical_name}.\n"
        f"Tried any of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def compute_home_signed_spread(row: pd.Series) -> float | pd.NA:
    """
    Convert the raw spread + whos_favored into a HOME-SIGNED spread.

    Assumptions about raw columns:
      - `spread` is a positive magnitude (e.g., 6.5 means the favorite is -6.5).
      - `whos_favored` is 'home' or 'away':
            'home'  => home is favorite  => home spread = -spread
            'away'  => away is favorite  => home spread = +spread
    """
    spread = row["spread"]
    side = str(row["whos_favored"]).strip().lower()

    if pd.isna(spread):
        return pd.NA

    try:
        spread = float(spread)
    except (TypeError, ValueError):
        return pd.NA

    if side == "home":
        return -spread
    elif side == "away":
        return spread
    else:
        # Unknown favorite flag; treat as missing
        return pd.NA


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # Ensure date is datetime
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------------------
    # 1. Check for basic structural columns
    # --------------------------------------------------------------
    basic_required = ["season", "date", "home", "away", "spread", "whos_favored"]
    missing_basic = [c for c in basic_required if c not in df.columns]
    if missing_basic:
        raise ValueError(
            f"Missing required columns in input: {missing_basic}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Regular / playoffs flags (create if missing)
    if "regular" not in df.columns:
        df["regular"] = True
    if "playoffs" not in df.columns:
        df["playoffs"] = False

    # --------------------------------------------------------------
    # 2. Find final scores (home / away) from merged columns
    # --------------------------------------------------------------
    home_score_col = find_first_column(
        df,
        candidates=["score_home", "home_score", "home_pts", "home_pts_x", "home_pts_y"],
        logical_name="home final score",
    )
    away_score_col = find_first_column(
        df,
        candidates=["score_away", "away_score", "away_pts", "away_pts_x", "away_pts_y"],
        logical_name="away final score",
    )

    df["score_home"] = df[home_score_col]
    df["score_away"] = df[away_score_col]

    # Convert scores to numeric (in case they are strings)
    df["score_home"] = pd.to_numeric(df["score_home"], errors="coerce")
    df["score_away"] = pd.to_numeric(df["score_away"], errors="coerce")

    # --------------------------------------------------------------
    # 3. Sort chronologically and drop rows missing essentials
    # --------------------------------------------------------------
    df = df.sort_values("date").reset_index(drop=True)

    before_drop = len(df)
    df = df.dropna(subset=["spread", "whos_favored", "score_home", "score_away"])
    after_drop = len(df)
    print(f"Dropped {before_drop - after_drop} rows due to missing spread/favorite/scores.")

    # --------------------------------------------------------------
    # 4. Compute home-signed spread
    # --------------------------------------------------------------
    df["spread_home"] = df.apply(compute_home_signed_spread, axis=1)

    # Drop rows where we couldn't interpret spread/favorite
    before_drop = len(df)
    df = df.dropna(subset=["spread_home"]).copy()
    after_drop = len(df)
    print(f"Dropped {before_drop - after_drop} rows due to invalid whos_favored/spread.")

    # Overwrite main `spread` column with home-signed spread
    df["spread"] = df["spread_home"]

    # --------------------------------------------------------------
    # 5. Game outcome and ATS labels (home perspective)
    # --------------------------------------------------------------
    # Raw scoring margin from home perspective
    df["margin_home"] = df["score_home"] - df["score_away"]

    # Straight-up result
    df["home_win"] = (df["margin_home"] > 0).astype(int)
    df["away_win"] = (df["margin_home"] < 0).astype(int)

    # ATS margin: by how many points did the home team beat the spread?
    df["ats_margin_home"] = df["margin_home"] + df["spread"]

    # Push, cover, no-cover flags
    df["ats_push"] = (df["ats_margin_home"] == 0).astype(int)
    df["ats_home_cover"] = (df["ats_margin_home"] > 0).astype(int)

    # For modeling, we usually want a clean binary label with no pushes
    before_drop = len(df)
    df = df[df["ats_push"] == 0].copy()
    after_drop = len(df)
    print(f"Dropped {before_drop - after_drop} rows due to ATS pushes (spread exactly matched).")

    # Our main target for models: did the home team cover?
    df["covered_home"] = df["ats_home_cover"]

    # --------------------------------------------------------------
    # 6. Final sanity check & save
    # --------------------------------------------------------------
    if len(df) == 0:
        raise RuntimeError(
            "After dropping invalid rows and pushes, the dataset is empty. "
            "Check that scores and spread/favorite data are populated correctly."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")


if __name__ == "__main__":
    main()