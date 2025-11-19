# src/train_winpct_baseline.py
"""
Baseline logistic regression using only win% features.

Runs twice:
  1) On balanced splits   -> nba_ats_features_balanced_splits.csv
  2) On time-based splits -> nba_ats_features_time_splits.csv

Saves models to:
  models/logreg_winpct_balanced.joblib
  models/logreg_winpct_time.joblib
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

BALANCED_PATH = "data/processed/nba_ats_features_balanced_splits.csv"
TIME_PATH = "data/processed/nba_ats_features_time_splits.csv"
MODEL_DIR = "models"


def evaluate_split(name: str, y_true, y_proba):
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


def run_baseline(input_path: str, model_path: str, label: str):
    print("\n" + "=" * 70)
    print(f"Running win% baseline on {label} splits")
    print("=" * 70)
    print("Loading data from:", input_path)

    df = pd.read_csv(input_path)

    required_cols = {
        "split",
        "covered_home",
        "team_win_pct_before_game",
        "opp_win_pct_before_game",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    feature_cols = [
        "team_win_pct_before_game",
        "opp_win_pct_before_game",
    ]

    # Drop rows with NaNs in features or target
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

    # Pipeline: standardize -> logistic regression
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

    print("\nTraining logistic regression (win% only)...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    evaluate_split("train", y_train, y_train_proba)
    evaluate_split("val", y_val, y_val_proba)
    evaluate_split("test", y_test, y_test_proba)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(clf, model_path)
    print(f"\nSaved baseline model to: {model_path}")


def main():
    run_baseline(BALANCED_PATH, os.path.join(MODEL_DIR, "logreg_winpct_balanced.joblib"), "BALANCED")
    run_baseline(TIME_PATH, os.path.join(MODEL_DIR, "logreg_winpct_time.joblib"), "TIME-BASED")


if __name__ == "__main__":
    main()