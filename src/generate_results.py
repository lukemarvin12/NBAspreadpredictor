"""
Generate results summary for HTML dashboard
Collects metrics from all trained models and dataset statistics
"""
import pandas as pd
import json
import os
from pathlib import Path

def generate_results():
    results = {
        "pipeline_status": {},
        "dataset_stats": {},
        "baseline_metrics": {},
        "full_model_metrics": {},
        "feature_importance": {}
    }
    
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "processed"
    
    # Check pipeline completion
    files_to_check = {
        "enriched_data": "nba_2018-2025_with_stats.xlsx",
        "clean_dataset": "nba_ats_clean.csv",
        "features": "nba_ats_features_winpct.csv",
        "splits": "nba_ats_features_balanced_splits.csv"
    }
    
    for name, filename in files_to_check.items():
        filepath = data_path / filename
        results["pipeline_status"][name] = {
            "exists": filepath.exists(),
            "path": str(filepath)
        }
    
    # Load dataset statistics
    try:
        df_clean = pd.read_csv(data_path / "nba_ats_clean.csv")
        results["dataset_stats"]["total_games"] = len(df_clean)
        results["dataset_stats"]["home_covered"] = int(df_clean["covered_home"].sum())
        results["dataset_stats"]["home_covered_pct"] = float(df_clean["covered_home"].mean() * 100)
        results["dataset_stats"]["seasons"] = df_clean["SEASON_ID"].nunique() if "SEASON_ID" in df_clean.columns else "N/A"
    except Exception as e:
        results["dataset_stats"]["error"] = str(e)
    
    # Load split statistics
    try:
        df_splits = pd.read_csv(data_path / "nba_ats_features_balanced_splits.csv")
        for split in ["train", "val", "test"]:
            split_df = df_splits[df_splits["split"] == split]
            results["dataset_stats"][f"{split}_count"] = len(split_df)
            if len(split_df) > 0:
                results["dataset_stats"][f"{split}_covered_pct"] = float(split_df["covered_home"].mean() * 100)
    except Exception as e:
        results["dataset_stats"]["split_error"] = str(e)
    
    # Feature statistics
    try:
        df_features = pd.read_csv(data_path / "nba_ats_features_winpct.csv")
        feature_cols = [col for col in df_features.columns if col not in 
                       ["GAME_ID", "GAME_DATE", "covered_home", "split", "SEASON_ID", "home_margin", "spread_home"]]
        results["feature_importance"]["total_features"] = len(feature_cols)
        results["feature_importance"]["feature_list"] = feature_cols
    except Exception as e:
        results["feature_importance"]["error"] = str(e)
    
    # Save results
    output_path = base_path / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    return results

if __name__ == "__main__":
    generate_results()
