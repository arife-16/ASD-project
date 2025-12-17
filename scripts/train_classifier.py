import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
import pickle

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from asd_pipeline.model import evaluate_models, evaluate_models_tuned

def main():
    parser = argparse.ArgumentParser(description="Train Classifiers on Normative Model Features")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing z_scores.npy and subjects_summary.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save classification results")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (slower)")
    parser.add_argument("--cv_strategy", type=str, default="stratified", choices=["stratified", "site_stratified", "loso"], help="Cross-validation strategy")
    parser.add_argument("--use_harmonized", action="store_true", help="Use z_scores_harmonized.npy if available")
    parser.add_argument("--add_summary_features", action="store_true", help="Append summary metrics (mean_abs_z, outlier counts) to feature vector")
    parser.add_argument("--advanced_features_file", type=str, help="Path to advanced features CSV (from extract_features.py) to use instead of/in addition to raw Z-scores")
    parser.add_argument("--features_mode", type=str, default="raw", choices=["raw", "advanced", "combined"], help="Which features to use: 'raw' (Z-scores), 'advanced' (extracted), or 'combined' (both)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...", flush=True)
    
    # Load Summary (Phenotypes)
    summary_path = os.path.join(args.results_dir, "subjects_summary.csv")
    if not os.path.exists(summary_path):
         raise FileNotFoundError("Could not find subjects_summary.csv in results_dir")
    df = pd.read_csv(summary_path)
    
    # Determine Features X
    X = None
    
    # Option A: Advanced Features Only or Combined
    if args.features_mode in ["advanced", "combined"]:
        if not args.advanced_features_file:
            # Try default location
            default_adv = os.path.join(args.results_dir, "advanced_features.csv") # Assume in results_dir by default if not passed?
            # Or assume output of extract_features.py might be elsewhere.
            # Let's check if the file exists at args.advanced_features_file or in results_dir
            if os.path.exists(default_adv):
                args.advanced_features_file = default_adv
            else:
                raise ValueError("features_mode is 'advanced' or 'combined' but --advanced_features_file not provided and not found in results_dir")
        
        print(f"Loading advanced features from {args.advanced_features_file}...", flush=True)
        adv_df = pd.read_csv(args.advanced_features_file)
        
        # Ensure alignment
        # Assuming rows are aligned or use FILE_ID
        # Let's check columns to exclude (ID, Group, Site)
        drop_cols = [c for c in adv_df.columns if c in ["FILE_ID", "SUB_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX"]]
        X_adv = adv_df.drop(columns=drop_cols).values
        
        # Check alignment length
        if len(adv_df) != len(df):
            print("Warning: Advanced features length mismatch with summary. Assuming aligned by index or subset.", flush=True)
            # If IDs present, align?
            if "FILE_ID" in adv_df.columns and "FILE_ID" in df.columns:
                 adv_df = adv_df.set_index("FILE_ID")
                 df_temp = df.set_index("FILE_ID")
                 # Intersect
                 common = adv_df.index.intersection(df_temp.index)
                 adv_df = adv_df.loc[common]
                 df = df_temp.loc[common].reset_index()
                 X_adv = adv_df.drop(columns=[c for c in drop_cols if c in adv_df.columns]).values
            else:
                 raise ValueError("Length mismatch and no FILE_ID to align.")
    
    # Option B: Raw Z-scores
    if args.features_mode in ["raw", "combined"]:
        z_scores_path = os.path.join(args.results_dir, "z_scores.npy")
        if args.use_harmonized:
            harm_path = os.path.join(args.results_dir, "z_scores_harmonized.npy")
            if os.path.exists(harm_path):
                z_scores_path = harm_path
                print("Using harmonized z-scores.", flush=True)
        
        if not os.path.exists(z_scores_path):
            if args.features_mode == "combined":
                 print("Warning: Raw Z-scores not found, proceeding with Advanced only.", flush=True)
                 X = X_adv
            else:
                 raise FileNotFoundError(f"Z-scores not found at {z_scores_path}")
        else:
            X_raw = np.load(z_scores_path)
            # Align if needed (if df was filtered above)
            if len(df) != X_raw.shape[0]:
                 # This implies X_raw corresponds to original summary, but df might have been subsetted?
                 # If we did subsetting above, we have a problem unless we know indices.
                 # Re-read original summary to find indices
                 orig_df = pd.read_csv(summary_path)
                 if "FILE_ID" in orig_df.columns and "FILE_ID" in df.columns:
                     # Find indices of current df in orig_df
                     indices = orig_df[orig_df["FILE_ID"].isin(df["FILE_ID"])].index
                     X_raw = X_raw[indices]
                 else:
                     raise ValueError("Cannot align Raw Z-scores: Length mismatch and no IDs.")

            X = X_raw
            
    # Combine if needed
    if args.features_mode == "combined":
        if X is None: X = X_adv # Fallback
        else:
            print(f"Combining Raw ({X.shape[1]}) and Advanced ({X_adv.shape[1]}) features...", flush=True)
            X = np.hstack([X, X_adv])
    elif args.features_mode == "advanced":
        X = X_adv
        
    # Check shape
    print(f"Final Feature Matrix Shape: {X.shape}", flush=True)
    
    if len(df) != X.shape[0]:
        raise ValueError(f"Mismatch: Phenotype has {len(df)} subjects, Features have {X.shape[0]}")
        
    # 2. Prepare Labels and Groups
    # Assuming DX_GROUP: 1=ASD, 2=TD
    if "DX_GROUP" not in df.columns:
        raise ValueError("DX_GROUP column missing in summary csv")
        
    y = df["DX_GROUP"].values
    groups = df["SITE_ID"].values if "SITE_ID" in df.columns else None
    
    # Filter for valid classes just in case
    # Convert to 0/1 if needed, but sklearn handles multi-class or arbitrary labels usually
    # But usually 1=ASD, 2=TD. Let's make sure we are classifying ASD (1) vs TD (2)
    # Or map to 0 (TD) and 1 (ASD) for consistency
    
    # Map: TD (2) -> 0, ASD (1) -> 1
    # Check what values exist
    classes = np.unique(y)
    print(f"Classes found: {classes}", flush=True)
    
    # Simple mapping
    y_bin = np.zeros(len(y), dtype=int)
    
    # Check if we have standard 1/2 labels
    if 1 in classes and 2 in classes:
        print("Mapping labels: 1 (ASD) -> 1, 2 (TD) -> 0", flush=True)
        y_bin[y == 1] = 1 # ASD
        y_bin[y == 2] = 0 # TD
    elif 0 in classes and 1 in classes:
        print("Labels are already 0/1. Assuming 1=ASD, 0=TD.", flush=True)
        y_bin = y.astype(int)
    else:
        print(f"Warning: Unexpected labels {classes}. Treating {classes[0]} as 0 and {classes[-1]} as 1.", flush=True)
        y_bin[y == classes[-1]] = 1
        y_bin[y == classes[0]] = 0
        
    print(f"Class distribution after mapping: {np.bincount(y_bin)} (0=TD, 1=ASD)", flush=True)
    
    # Optionally augment features with summary metrics
    if args.add_summary_features:
        cols = []
        for c in ["mean_abs_z", "outlier_count_total", "outlier_count_pos", "outlier_count_neg"]:
            if c in df.columns:
                cols.append(c)
        if len(cols) > 0:
            print(f"Appending summary features: {cols}", flush=True)
            X = np.hstack([X, df[cols].values])
        else:
            print("No summary feature columns found to append.", flush=True)

    # 3. Train and Evaluate
    print(f"Starting training with strategy: {args.cv_strategy}...", flush=True)
    
    if args.tune:
        results = evaluate_models_tuned(X, y_bin, cv_strategy=args.cv_strategy, groups=groups)
    else:
        results = evaluate_models(X, y_bin, cv_strategy=args.cv_strategy, groups=groups)
        
    # 4. Save Results
    print("Training complete. Saving results...", flush=True)
    
    # Print summary to console
    print("\nResults Summary (ROC AUC):")
    for model, metrics in results.items():
        print(f"  {model}: {metrics.get('roc_auc', 0.0):.4f}")
        
    results_file = os.path.join(args.output_dir, "classification_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    # Also save as CSV for easier reading
    rows = []
    for model, metrics in results.items():
        row = {"Model": model}
        row.update(metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "classification_results.csv"), index=False)
    
    print(f"Results saved to {args.output_dir}", flush=True)

if __name__ == "__main__":
    main()
