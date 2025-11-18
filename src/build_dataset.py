# src/build_dataset.py
"""
Build a clean ATS dataset from the enriched Excel file.

Assumes input file has at least:
  - season
  - date
  - regular
  - playoffs
  - home, away
  - whos_favored  ('home' or 'away')
  - spread        (absolute value, e.g. 6.5)
  - home_pts
  - away_pts

Outputs:
  data/processed/nba_ats_clean.csv
"""

import os
import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/nba_2018-2025_with_stats.xlsx"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nba_ats_clean.csv")


def main():
    print("Loading:", INPUT_PATH)
    df = pd.read_excel(INPUT_PATH)

    # Basic sanity checks
    required_cols = [
        "season",
        "date",
        "home",
        "away",
        "regular",
        "playoffs",
        "whos_favored",
        "spread",
        "home_pts",
        "away_pts",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    # Normalize date to YYYY-MM-DD string
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    # Make sure spread is numeric
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")

    # Clean whos_favored
    df["whos_favored"] = df["whos_favored"].astype(str).str.lower().str.strip()

    # Compute signed spread from HOME perspective:
    #   spread_home < 0 -> home is favored by |spread_home|
    #   spread_home > 0 -> home is underdog by spread_home
    spread_mag = df["spread"].abs()
    fav = df["whos_favored"]

    spread_home = np.where(fav == "home", -spread_mag, spread_mag)
    df["spread_home"] = spread_home

    # Score / margin
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df["home_margin"] = df["home_pts"] - df["away_pts"]

    # Moneyline-style win flag for home team
    df["home_win"] = (df["home_pts"] > df["away_pts"]).astype(int)

    # ATS margin from the HOME perspective:
    #   home_margin + spread_home
    ats_margin_home = df["home_margin"] + df["spread_home"]
    df["ats_margin_home"] = ats_margin_home

    # Pushes (exactly on the number): we drop them
    df["is_push"] = (ats_margin_home == 0)

    # ATS labels:
    # covered_home = 1 if HOME covers against the spread, 0 otherwise
    df["covered_home"] = (ats_margin_home > 0).astype(int)
    df["covered_away"] = 1 - df["covered_home"]

    # Drop rows where we do not have everything we need
    before = len(df)
    df = df[
        df[["home_pts", "away_pts", "spread_home"]].notna().all(axis=1)
        & (~df["is_push"])
    ].copy()
    after = len(df)
    print(f"Dropped {before - after} rows due to missing data or pushes.")

    # Keep a clean set of columns for modeling
    keep_cols = [
        "season",
        "date",
        "regular",
        "playoffs",
        "home",
        "away",
        "spread_home",
        "home_pts",
        "away_pts",
        "home_margin",
        "home_win",
        "ats_margin_home",
        "covered_home",
        "covered_away",
    ]
    # Keep any intersection of keep_cols & existing columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_clean = df[keep_cols].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(df_clean))


if __name__ == "__main__":
    main()