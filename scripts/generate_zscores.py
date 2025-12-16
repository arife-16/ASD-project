import argparse
import os
import sys
import numpy as np
import pandas as pd
import pickle

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from asd_pipeline.normative import predict_normative_model, load_normative_model

# Try to import ComBat
try:
    from asd_pipeline.harmonize import combat_harmonize
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Generate Z-scores using trained Normative Model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--phenotype", type=str, required=True, help="Path to phenotype CSV")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory containing precomputed feature .npy files or stack")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save Z-scores")
    parser.add_argument("--covariates", type=str, default="AGE_AT_SCAN,SEX", help="Covariate columns for model")
    parser.add_argument("--harmonize", action="store_true", help="Apply ComBat harmonization to Z-scores (requires SITE_ID)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for prediction")
    
    args = parser.parse_args()
    
    # 1. Load Model
    print(f"Loading model from {args.model}...", flush=True)
    model_data = load_normative_model(args.model)
    
    # 2. Load Phenotype
    print("Loading phenotype...", flush=True)
    pheno = pd.read_csv(args.phenotype)
    
    if "FILE_ID" not in pheno.columns:
        # Try to infer or fail
        if "SUB_ID" in pheno.columns:
             pheno["FILE_ID"] = pheno["SUB_ID"]
        else:
             raise ValueError("Phenotype file must contain 'FILE_ID' column")
             
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str)
    
    # Remove subjects with no filename
    pheno = pheno[pheno["FILE_ID"] != "no_filename"].reset_index(drop=True)
    print(f"Processing {len(pheno)} subjects from phenotype file.", flush=True)
    
    # 3. Load Features
    print("Loading features...", flush=True)
    stack_path = os.path.join(args.features_dir, "_stack.npy")
    index_path = os.path.join(args.features_dir, "_index.txt")
    
    X_features = None
    
    if os.path.exists(stack_path) and os.path.exists(index_path):
        # Load full stack metadata
        with open(index_path, "r") as f:
            stack_ids = [line.strip() for line in f]
        
        id_to_idx = {sid: i for i, sid in enumerate(stack_ids)}
        
        # Match phenotype subjects to stack
        valid_indices = []
        valid_pheno_indices = []
        
        for i, row in pheno.iterrows():
            sid = row["FILE_ID"]
            if sid in id_to_idx:
                valid_indices.append(id_to_idx[sid])
                valid_pheno_indices.append(i)
        
        if not valid_indices:
            raise ValueError("No matching subjects found in feature stack.")
            
        print(f"Found {len(valid_indices)} subjects with features available.", flush=True)
        
        # Filter phenotype to matched subjects
        pheno = pheno.iloc[valid_pheno_indices].reset_index(drop=True)
        
        # Load features
        X_all = np.load(stack_path, mmap_mode='r')
        
        # Load subset into RAM
        # Note: fancy indexing on mmap returns a copy in RAM
        print(f"Loading feature subset into RAM ({len(valid_indices)} subjects)...", flush=True)
        X_features = X_all[valid_indices]
        
    else:
        raise FileNotFoundError(f"Feature stack not found in {args.features_dir}. Please run feature extraction first.")

    # 4. Prepare Covariates
    cov_cols = args.covariates.split(",")
    covs = []
    for _, row in pheno.iterrows():
        vals = []
        for col in cov_cols:
            if col not in row:
                 raise ValueError(f"Covariate {col} not found in phenotype")
            val = row[col]
            if col.upper() in ["SEX", "GENDER"] and isinstance(val, str):
                val = 0 if val.lower() in ["m", "male", "1"] else 1
            vals.append(float(val))
        covs.append(vals)
    covs = np.array(covs)
    
    # 5. Predict and Generate Z-scores
    print("Generating Z-scores...", flush=True)
    results = predict_normative_model(model_data, covs, X_features, use_gpu=args.use_gpu)
    
    # 6. Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    
    z_scores = results["feature_z"]
    pred_mean = results["pred_mean"]
    pred_std = results["pred_std"]
    
    # 6a. Harmonize if requested
    if args.harmonize:
        if not COMBAT_AVAILABLE:
            print("Warning: ComBat requested but neuroHarmonize not installed or failed to import. Skipping harmonization.", flush=True)
        elif "SITE_ID" not in pheno.columns:
             print("Warning: ComBat requested but SITE_ID not in phenotype. Skipping.", flush=True)
        else:
             print("Applying ComBat harmonization to Z-scores...", flush=True)
             # Prepare covariates for ComBat (Diagnosis, Age, Sex should be preserved if possible, but usually we just want to remove Site effects)
             # Standard ComBat: remove Site effect, preserving biological covariates
             # We should preserve 'DX_GROUP' if we want to keep group differences?
             # Actually, if we are harmonizing Z-scores (deviations), we expect TD to be around 0 and ASD to deviate.
             # If we include DX as covariate to preserve, we are good.
             # If we don't, ComBat might remove the ASD signal if it correlates with site.
             # Safest is to use DX_GROUP, AGE, SEX as covariates to preserve.
             
             combat_covs = pheno.copy()
             # Ensure numeric
             if "DX_GROUP" in combat_covs.columns:
                 combat_covs["DX_GROUP"] = combat_covs["DX_GROUP"].astype(int)
             
             # Need to pass categorical/continuous lists
             # Assuming AGE is continuous, SEX/DX categorical
             cat_cols = []
             cont_cols = []
             
             if "AGE_AT_SCAN" in combat_covs.columns:
                 cont_cols.append("AGE_AT_SCAN")
             if "SEX" in combat_covs.columns:
                 cat_cols.append("SEX")
             if "DX_GROUP" in combat_covs.columns:
                 cat_cols.append("DX_GROUP")
                 
             try:
                 z_scores_harm, _ = combat_harmonize(
                     z_scores, 
                     combat_covs, 
                     "SITE_ID", 
                     continuous_covars=cont_cols, 
                     categorical_covars=cat_cols
                 )
                 print("ComBat complete.", flush=True)
                 # Save harmonized version
                 np.save(os.path.join(args.output_dir, "z_scores_harmonized.npy"), z_scores_harm)
                 # Update z_scores variable for metrics calculation?
                 # User might want to compare both. Let's calculate metrics on the *harmonized* one if available?
                 # Or better, calculate metrics on BOTH.
                 # Let's stick to calculating on the 'main' z_scores. 
                 # If harmonized, we should probably treat harmonized as the 'main' result for downstream analysis.
                 z_scores = z_scores_harm
             except Exception as e:
                 print(f"ComBat failed: {e}", flush=True)

    print(f"Saving results to {args.output_dir}...", flush=True)
    np.save(os.path.join(args.output_dir, "z_scores.npy"), z_scores)
    np.save(os.path.join(args.output_dir, "pred_mean.npy"), pred_mean)
    np.save(os.path.join(args.output_dir, "pred_std.npy"), pred_std)
    
    # Save Index
    with open(os.path.join(args.output_dir, "subjects.txt"), "w") as f:
        for sid in pheno["FILE_ID"]:
            f.write(f"{sid}\n")
            
    # Also save as DataFrame with IDs if small enough? No, 300k features.
    # But we can save a small summary
    
    # Calculate Mean Absolute Z-score per subject
    mean_abs_z = np.mean(np.abs(z_scores), axis=1)
    
    # Calculate Extreme Value Metrics (|Z| > 1.96)
    threshold = 1.96
    outlier_count_pos = np.sum(z_scores > threshold, axis=1)
    outlier_count_neg = np.sum(z_scores < -threshold, axis=1)
    outlier_count_total = np.sum(np.abs(z_scores) > threshold, axis=1)
    
    pheno["mean_abs_z"] = mean_abs_z
    pheno["outlier_count_pos"] = outlier_count_pos
    pheno["outlier_count_neg"] = outlier_count_neg
    pheno["outlier_count_total"] = outlier_count_total
    
    pheno.to_csv(os.path.join(args.output_dir, "subjects_summary.csv"), index=False)
            
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
