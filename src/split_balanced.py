# src/splits_balanced.py
"""
Create balanced train/val/test splits for the feature dataset.

We aim for:
  - Each season appears in all splits
  - Each day-of-week appears across splits

Input:
  data/processed/nba_ats_features_winpct.csv

Output:
  data/processed/nba_ats_features_balanced_splits.csv
  with a new 'split' column in {'train', 'val', 'test'}
"""

import os
import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/nba_ats_features_winpct.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nba_ats_features_balanced_splits.csv")

RNG_SEED = 42
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2


def assign_splits_balanced(df: pd.DataFrame) -> pd.Series:
    """
    For each (season, day-of-week) group, randomly split rows into
    train/val/test with approximately the global fractions.

    Returns a 'split' Series aligned with df.index.
    """
    rng = np.random.default_rng(RNG_SEED)
    split = pd.Series(index=df.index, dtype="object")

    # Ensure date is datetime so we can get day-of-week
    df_dt = df.copy()
    df_dt["date_dt"] = pd.to_datetime(df_dt["date"])

    df_dt["dow"] = df_dt["date_dt"].dt.dayofweek  # 0=Mon, 6=Sun

    # Iterate by season & dow group
    group_cols = ["season", "dow"]
    for (season, dow), idx in df_dt.groupby(group_cols).groups.items():
        idx = list(idx)
        if len(idx) == 0:
            continue

        # Shuffle indices within group
        idx_shuffled = list(idx)
        rng.shuffle(idx_shuffled)

        n = len(idx_shuffled)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        # Remaining goes to test
        n_test = n - n_train - n_val

        train_idx = idx_shuffled[:n_train]
        val_idx = idx_shuffled[n_train : n_train + n_val]
        test_idx = idx_shuffled[n_train + n_val :]

        split.loc[train_idx] = "train"
        split.loc[val_idx] = "val"
        split.loc[test_idx] = "test"

    # Some tiny groups might end up with NaN split (e.g., rare combinations)
    # Assign them randomly, but with global fractions.
    mask_unassigned = split.isna()
    if mask_unassigned.any():
        unassigned_idx = split[mask_unassigned].index
        n_unassigned = len(unassigned_idx)
        print(f"Unassigned rows due to tiny groups: {n_unassigned}")

        choices = ["train", "val", "test"]
        probs = [TRAIN_FRAC, VAL_FRAC, TEST_FRAC]
        rand_splits = rng.choice(choices, size=n_unassigned, p=probs)
        split.loc[unassigned_idx] = rand_splits

    return split


def main():
    print("Loading:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    if "season" not in df.columns or "date" not in df.columns:
        raise ValueError("Expected at least 'season' and 'date' columns in features file.")

    print("Assigning balanced splits...")
    df["split"] = assign_splits_balanced(df)

    print("Split counts:")
    print(df["split"].value_counts())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
    