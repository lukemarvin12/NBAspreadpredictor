# src/build_features.py

import pandas as pd
from features import add_basic_features

INPUT_PATH = "data/processed/nba_ats_clean.csv"
OUTPUT_PATH = "data/processed/nba_ats_features_basic.csv"

def main():
    print("Loading:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    print("Applying feature engineering...")
    df_fe = add_basic_features(df)

    df_fe.to_csv(OUTPUT_PATH, index=False)
    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(df_fe), "Columns:", df_fe.shape[1])

if __name__ == "__main__":
    main()