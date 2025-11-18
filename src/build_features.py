# src/build_features.py

import os
import pandas as pd
from features import add_basic_features

INPUT_PATH = "data/processed/nba_ats_clean.csv"
OUTPUT_DIR = "data/processed"
# keep same name so splits + old train script still work
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nba_ats_features_winpct.csv")


def main():
    print("Loading:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    # Normalize date
    df["date"] = df["date"].dt.date.astype(str)

    print("Adding basic features (win%, recent form, rest, abs_spread)...")
    df_fe = add_basic_features(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_fe.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(df_fe), "Columns:", df_fe.shape[1])


if __name__ == "__main__":
    main()