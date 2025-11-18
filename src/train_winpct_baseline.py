# src/train_winpct_baseline.py
"""
Train a simple logistic regression baseline using ONLY win% features:

  X = [team_win_pct_before_game, opp_win_pct_before_game]
  y = covered_home

Input:
  data/processed/nba_ats_features_balanced_splits.csv

Prints metrics for train/val/test.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/processed/nba_ats_features_balanced_splits.csv"


def evaluate_split(name: str, y_true, y_proba):
    """
    Compute and print accuracy, log loss, and AUC for a split.
    """
    y_pred = (y_proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    try:
        ll = log_loss(y_true, y_proba, labels=[0, 1])
    except ValueError:
        ll = float("nan")

    print(f"\n=== {name.upper()} METRICS ===")
    print(f"Rows:     {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log loss: {ll:.4f}")
    print(f"AUC:      {auc:.4f}")


def main():
    print("Loading data from:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    required_cols = {
        "team_win_pct_before_game",
        "opp_win_pct_before_game",
        "covered_home",
        "split",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Features: ONLY win% features
    feature_cols = [
        "team_win_pct_before_game",
        "opp_win_pct_before_game",
    ]

    # Drop any rows with NaNs in features or target
    mask_good = df[feature_cols + ["covered_home"]].notna().all(axis=1)
    df = df[mask_good].copy()

    # Split
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print("Split sizes:")
    print("  Train:", len(train_df))
    print("  Val:  ", len(val_df))
    print("  Test: ", len(test_df))

    X_train = train_df[feature_cols].values
    y_train = train_df["covered_home"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["covered_home"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["covered_home"].values

    # Build pipeline: standardize -> logistic regression
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                ),
            ),
        ]
    )

    print("\nTraining logistic regression on win% features only...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    evaluate_split("train", y_train, y_train_proba)
    evaluate_split("val", y_val, y_val_proba)
    evaluate_split("test", y_test, y_test_proba)


if __name__ == "__main__":
    main()