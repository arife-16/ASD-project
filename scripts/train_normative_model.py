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
    parser.add_argument("--ts_dir", type=str, required=True, help="Directory containing NIfTI/timeseries files")
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
    td_mask = pheno[args.dx_col] == args.td_label
    td_pheno = pheno[td_mask].reset_index(drop=True)
    print(f"Found {len(td_pheno)} TD subjects out of {len(pheno)} total.", flush=True)
    
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
            
    # Handle template loading for precision mapping once
    template_labels = None
    if args.precision_mapping:
        if not args.template_labels_path:
            raise ValueError("Precision mapping requires --template_labels_path")
        if args.template_labels_path.endswith(".npy"):
            template_labels = np.load(args.template_labels_path)
        else:
            import nibabel as nib
            template_labels = nib.load(args.template_labels_path).get_fdata().ravel()
            
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
            ts = np.load(ts_path)
            
            # Feature Extraction Logic
            vec = None
            
            if args.precision_mapping:
                # Precision Mapping
                if ts.shape[0] < ts.shape[1]: # (n_rois, n_time) -> (n_time, n_vertices)
                    ts = ts.T
                vec = precision_mapping_workflow(ts, template_labels)
                
            elif args.labels_tsv:
                # Atlas-based Connectome
                # Regress confounds first?
                # Assume ts is already cleaned or raw? Pipeline usually cleans.
                # Let's do basic cleaning if confounds provided
                confounds = None
                if args.confounds_dir:
                     cpath = os.path.join(args.confounds_dir, f"{sid}.npy")
                     if os.path.exists(cpath):
                         confounds = np.load(cpath)
                
                ts_clean = regress_confounds(ts, confounds=confounds, tr=args.tr)
                
                # Connectome features
                # Reuse run_pipeline logic: FC + ALFF
                v_conn, _ = build_connectome_feature_vector(
                    ts_clean, window_size=args.window_size, step=args.step, 
                    thr=args.adj_thr, networks=networks, compute_partial=True
                )
                v_alff = alff(ts_clean, args.tr)
                vec = np.concatenate([v_conn, v_alff])
                
            else:
                # Default / Fallback (e.g. simple ROI correlations)
                # Just use build_feature_vector
                ts_clean = regress_confounds(ts, tr=args.tr)
                vec, _ = build_feature_vector(ts_clean, tr=args.tr, window_size=args.window_size, step=args.step)
            
            if vec is not None:
                features_list.append(vec)
                # Get covariates
                vals = []
                for col in cov_cols:
                    val = row[col]
                    # Encode sex if string
                    if col == "SEX" and isinstance(val, str):
                        val = 0 if val.lower() in ["m", "male", "1"] else 1
                    vals.append(float(val))
                covariates_list.append(vals)
                
        except Exception as e:
            print(f"Error processing {sid}: {e}", flush=True)
            continue
            
    if not features_list:
        raise RuntimeError("No features extracted successfully.")
        
    X_td = np.stack(features_list)
    cov_td = np.array(covariates_list)
    
    print(f"Training Normative Model ({args.normative_model}) on {len(X_td)} subjects...", flush=True)
    print(f"Features shape: {X_td.shape}, Covariates shape: {cov_td.shape}", flush=True)
    
    # 3. Fit and Save Model
    model_data = fit_and_save_normative_model(
        X_td, cov_td, args.output_model, model_type=args.normative_model
    )
    
    print(f"Model saved to {args.output_model}", flush=True)

if __name__ == "__main__":
    main()
