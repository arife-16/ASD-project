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
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...", flush=True)
    z_scores_path = os.path.join(args.results_dir, "z_scores.npy")
    summary_path = os.path.join(args.results_dir, "subjects_summary.csv")
    
    if not os.path.exists(z_scores_path) or not os.path.exists(summary_path):
        raise FileNotFoundError("Could not find z_scores.npy or subjects_summary.csv in results_dir")
        
    X = np.load(z_scores_path)
    df = pd.read_csv(summary_path)
    
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
    y_bin = np.zeros_like(y)
    y_bin[y == 1] = 1 # ASD
    y_bin[y == 2] = 0 # TD
    
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
