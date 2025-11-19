import pandas as pd

INPUT_PATH = "data/processed/nba_ats_features_winpct.csv"
OUTPUT_PATH = "data/processed/nba_ats_features_time_splits.csv"

def main():
    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Extract year
    df["year"] = df["date"].dt.year

    # ----- TIME-BASED SPLITS -----
    # Train on early years
    train_years = [2018, 2019, 2020, 2021, 2022]

    # Validate on next year
    val_years = [2023]

    # Test on newest years
    test_years = [2024, 2025]

    df["split"] = "train"
    df.loc[df["year"].isin(val_years), "split"] = "val"
    df.loc[df["year"].isin(test_years), "split"] = "test"

    print("Split counts:")
    print(df["split"].value_counts())

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()