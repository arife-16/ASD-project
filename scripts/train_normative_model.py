import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import glob
import json
from asd_pipeline.normative import fit_and_save_normative_model
from scripts.run_pipeline import find_ts_file
from asd_pipeline.precision_mapping import precision_mapping_workflow
from asd_pipeline.atlas_labels import load_cc_labels
from asd_pipeline.connectome import build_connectome_feature_vector
from asd_pipeline.features import alff, dynamic_state_features, build_feature_vector
from asd_pipeline.preprocess import regress_confounds

def main():
    parser = argparse.ArgumentParser(description="Train Normative Model on TD Subjects")
    parser.add_argument("--phenotype", type=str, required=True, help="Path to phenotypic CSV")
    parser.add_argument("--ts_dir", type=str, required=False, help="Directory containing NIfTI/timeseries files (optional if using precomputed features)")
    parser.add_argument("--features_dir", type=str, required=False, help="Directory containing precomputed feature .npy files")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save trained model (e.g. model.pkl or model.npz)")
    
    # Feature Extraction Options
    parser.add_argument("--atlas", type=str, default="", help="Path to Atlas NIfTI (if using fixed atlas)")
    parser.add_argument("--labels_tsv", type=str, default="", help="Path to labels TSV (for CC400/CC200)")
    parser.add_argument("--precision_mapping", action="store_true", help="Use precision mapping features")
    parser.add_argument("--template_labels_path", type=str, default="", help="Path to template labels for precision mapping")
    
    # Normative Options
    parser.add_argument("--normative_model", type=str, default="linear", choices=["linear", "gpr", "lowess"])
    parser.add_argument("--covariates", type=str, default="AGE_AT_SCAN,SEX", help="Comma-separated list of covariate columns")
    parser.add_argument("--site_col", type=str, default="SITE_ID")
    parser.add_argument("--dx_col", type=str, default="DX_GROUP", help="Column for diagnosis")
    parser.add_argument("--td_label", type=int, default=2, help="Value for TD subjects in dx_col")
    
    # Preprocessing
    parser.add_argument("--tr", type=float, default=2.0)
    parser.add_argument("--confounds_dir", type=str, default="")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--adj_thr", type=float, default=0.3)
    
    args = parser.parse_args()
    
    # 1. Load Phenotype and Filter TD
    print("Loading phenotype...", flush=True)
    pheno = pd.read_csv(args.phenotype)
    
    # Ensure FILE_ID is string
    if "FILE_ID" not in pheno.columns:
        raise ValueError("Phenotype file must contain 'FILE_ID' column")
        
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str)
    
    # Filter TD
    td_mask = (pheno[args.dx_col] == args.td_label) & (pheno["FILE_ID"] != "no_filename")
    td_pheno = pheno[td_mask].reset_index(drop=True)
    print(f"Found {len(td_pheno)} valid TD subjects out of {len(pheno)} total (removed 'no_filename').", flush=True)
    
    if len(td_pheno) == 0:
        raise ValueError("No TD subjects found with current filter settings.")
        
    # 2. Extract Features for TD
    print("Extracting features...", flush=True)
    
    features_list = []
    covariates_list = []
    
    # Prepare covariate columns
    cov_cols = args.covariates.split(",")
    for col in cov_cols:
        if col not in td_pheno.columns:
            raise ValueError(f"Covariate {col} not found in phenotype file")
            
    # Helper to safe extract covariates
    def extract_covariates(row, cols):
        vals = []
        for col in cols:
            val = row[col]
            # specific handling for SEX/Gender if string
            if col.upper() in ["SEX", "GENDER"] and isinstance(val, str):
                val = 0 if val.lower() in ["m", "male", "1"] else 1
            vals.append(float(val))
        return vals

    # Load Precomputed Features if provided
    if args.features_dir:
        print(f"Loading precomputed features from {args.features_dir}...", flush=True)
        # Check for stack file first
        stack_path = os.path.join(args.features_dir, "_stack.npy")
        index_path = os.path.join(args.features_dir, "_index.txt")
        
        if os.path.exists(stack_path) and os.path.exists(index_path):
            # Load stack
            X_all = np.load(stack_path)
            with open(index_path, "r") as f:
                stack_ids = [line.strip() for line in f]
            
            # Create map
            id_to_idx = {sid: i for i, sid in enumerate(stack_ids)}
            
            # Filter and match TD subjects
            valid_indices = []
            for idx, row in td_pheno.iterrows():
                sid = row["FILE_ID"]
                if sid in id_to_idx:
                    features_list.append(X_all[id_to_idx[sid]])
                    covariates_list.append(extract_covariates(row, cov_cols))
                else:
                    print(f"Warning: Subject {sid} not found in feature stack.", flush=True)
        else:
            # Load individual files
            for idx, row in td_pheno.iterrows():
                sid = row["FILE_ID"]
                fpath = os.path.join(args.features_dir, f"{sid}.npy")
                if os.path.exists(fpath):
                    features_list.append(np.load(fpath))
                    covariates_list.append(extract_covariates(row, cov_cols))
                else:
                    print(f"Warning: Subject {sid} feature file not found.", flush=True)
                    
    else:
        # Calculate from Time Series (Legacy Mode)
        if not args.ts_dir:
             raise ValueError("Either --features_dir or --ts_dir must be provided.")
             
        # Handle template loading for precision mapping once
        template_labels = None
    if not args.features_dir:
        # Labels for Atlas
        networks = None
        if args.labels_tsv:
            _, networks = load_cc_labels(args.labels_tsv)
        
        sites_for_id = {str(row["FILE_ID"]): str(row[args.site_col]) if args.site_col in pheno.columns else "" for _, row in pheno.iterrows()}
        
        for idx, row in td_pheno.iterrows():
            sid = row["FILE_ID"]
            site = sites_for_id.get(sid, "")
            
            # Find file
            ts_path = find_ts_file(sid, site, args.ts_dir)
            if not ts_path:
                print(f"Skipping {sid}: Time series file not found", flush=True)
                continue

            try:
                # 1. Regress Confounds (if provided)
                confounds_path = None
                if args.confounds_dir:
                    confounds_path = os.path.join(args.confounds_dir, f"{sid}_confounds.tsv")
                    if not os.path.exists(confounds_path):
                         # Try recursive search if flat structure fails
                         c_files = glob.glob(os.path.join(args.confounds_dir, "**", f"*{sid}*confounds*.tsv"), recursive=True)
                         if c_files:
                             confounds_path = c_files[0]
                
                # 2. Extract Feature
                if args.precision_mapping:
                    # Precision Mapping Workflow
                    feat = precision_mapping_workflow(
                        ts_path, 
                        template_labels if template_labels is not None else args.template_labels_path,
                        confounds=confounds_path
                    )
                else:
                    # Standard Atlas-based Connectome
                    feat = build_connectome_feature_vector(
                        ts_path, 
                        args.atlas, 
                        networks, 
                        kind='correlation', 
                        confounds=confounds_path
                    )
                
                features_list.append(feat)
                covariates_list.append(extract_covariates(row, cov_cols))
                
            except Exception as e:
                print(f"Error processing {sid}: {e}", flush=True)

    # 3. Train Model
    if len(features_list) == 0:
        raise ValueError("No valid features extracted or loaded.")
        
    X = np.array(features_list)
    covariates = np.array(covariates_list)
    
    print(f"Training model on {len(X)} subjects with {X.shape[1]} features...", flush=True)
    
    # Explicitly delete unused variables to free RAM before training
    del pheno, td_pheno, covariates_list
    import gc
    gc.collect()
    
    model_data = fit_and_save_normative_model(
        X, 
        covariates, 
        args.output_model, 
        model_type=args.normative_model
    )
    
    from asd_pipeline.normative import save_normative_model
    save_normative_model(model_data, args.output_model)
    
    print(f"Model saved to {args.output_model}", flush=True)

if __name__ == "__main__":
    main()
