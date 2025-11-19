# src/merge_nbaapi_sidecar.py

"""
Merge existing processed odds+stats file with nba_api sidecar stats.

Inputs:
  base_path = data/processed/nba_2018-2025_with_stats.xlsx
  api_path  = data/processed/nba_2018-2025_nbaapi_stats.xlsx

Output:
  data/processed/nba_2018-2025_with_stats_merged.csv   <-- CSV, not Excel
"""

import pandas as pd

BASE_PATH = "data/processed/nba_2018-2025_with_stats.xlsx"
API_PATH = "data/processed/nba_2018-2025_nbaapi_stats.xlsx"
OUTPUT_PATH = "data/processed/nba_2018-2025_with_stats_merged.csv"


def main():
    print(f"Loading base file: {BASE_PATH}")
    base = pd.read_excel(BASE_PATH)

    print(f"Loading nba_api stats sidecar: {API_PATH}")
    api = pd.read_excel(API_PATH)

    # Ensure key columns exist
    key_cols = ["season", "date", "home", "away"]
    for col in key_cols:
        if col not in base.columns:
            raise ValueError(f"Missing key column '{col}' in base file.")
        if col not in api.columns:
            raise ValueError(f"Missing key column '{col}' in nba_api sidecar file.")

    # Normalize date types for merge
    base["date"] = pd.to_datetime(base["date"])
    api["date"] = pd.to_datetime(api["date"])

    # Only bring in columns that are NOT already in base
    extra_cols = [c for c in api.columns if c not in key_cols]
    print("Extra nba_api columns to merge:", extra_cols)

    api_sub = api[key_cols + extra_cols]

    print("Merging base + nba_api stats on [season, date, home, away]...")
    merged = base.merge(api_sub, on=key_cols, how="left")

    # ✅ Save as CSV (fast, no Excel engine issues)
    print(f"Saving merged file to: {OUTPUT_PATH}")
    merged.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()