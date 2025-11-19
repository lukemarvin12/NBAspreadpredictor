# src/train_lgbm_full_features.py

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

INPUT_PATH = "data/processed/nba_ats_features_balanced_splits.csv"
OUTPUT_DIR = "data/processed"
FI_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "lgbm_feature_importances.csv")


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


def main():
    print("Loading data from:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    # All possible features we might use; we'll keep only those that actually exist in df
    possible_feature_cols = [
        "team_win_pct_before_game",
        "opp_win_pct_before_game",
        "team_recent_win_pct_3",
        "opp_recent_win_pct_3",
        "abs_spread",
        "team_rest",
        "opp_rest",
        "team_pts_for_5",
        "team_pts_against_5",
        "team_margin_5",
        "opp_pts_for_5",
        "opp_pts_against_5",
        "opp_margin_5",
        "delta_win_pct",
        "delta_recent_win_pct_3",
        "delta_rest",
        "delta_pts_for_5",
        "delta_margin_5",
        # Advanced OffRtg/DefRtg/Pace + deltas (if present)
        "team_offrtg_5",
        "team_defrtg_5",
        "team_pace_5",
        "opp_offrtg_5",
        "opp_defrtg_5",
        "opp_pace_5",
        "delta_offrtg_5",
        "delta_defrtg_5",
        "delta_pace_5",
        # Rest/travel flags
        "team_is_b2b",
        "opp_is_b2b",
        "team_is_3in4",
        "opp_is_3in4",
        # Altitude
        "is_altitude_home",
        "is_altitude_away",
    ]

    feature_cols = [c for c in possible_feature_cols if c in df.columns]

    if not feature_cols:
        raise ValueError("No feature columns found in dataframe. Check feature engineering step.")

    print("Using feature columns:")
    for c in feature_cols:
        print("  -", c)

    required_cols = {"covered_home", "split"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns in input file: {missing_required}")

    # Drop rows with NaNs in features or target
    mask_good = df[feature_cols + ["covered_home"]].notna().all(axis=1)
    df = df[mask_good].copy()

    # Split into train / val / test
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

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        n_jobs=-1,
    )

    print("\nTraining LightGBM on full feature set with early stopping...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = model.best_iteration_
    print(f"\nBest iteration (n_estimators used): {best_iter}")

    y_train_proba = model.predict_proba(X_train, num_iteration=best_iter)[:, 1]
    y_val_proba = model.predict_proba(X_val, num_iteration=best_iter)[:, 1]
    y_test_proba = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]

    evaluate_split("train", y_train, y_train_proba)
    evaluate_split("val", y_val, y_val_proba)
    evaluate_split("test", y_test, y_test_proba)

    # Save feature importances
    print("\nSaving feature importances to:", FI_OUTPUT_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    )
    fi.sort_values("importance_gain", ascending=False, inplace=True)
    fi.to_csv(FI_OUTPUT_PATH, index=False)

    print("Done.")


if __name__ == "__main__":
    main()