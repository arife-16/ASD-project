import os
import sys
import json
import numpy as np
import pandas as pd
import requests
import argparse
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asd_pipeline.atlas import extract_roi_timeseries
from asd_pipeline.preprocess import regress_confounds
from asd_pipeline.connectome import build_connectome_feature_vector
from asd_pipeline.normative import personalized_deviation_maps
from asd_pipeline.model import evaluate_models


def main():
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotype_path", type=str, default="")
    ap.add_argument("--nifti_dir", type=str, default="")
    ap.add_argument("--nifti_pattern", type=str, default="")
    ap.add_argument("--timeseries_dir", type=str, default="")
    ap.add_argument("--atlas_path", type=str, default="")
    ap.add_argument("--labels_path", type=str, default="")
    ap.add_argument("--strict_bna", action="store_true")
    ap.add_argument("--out_dir", type=str, default=os.path.join(proj, "experiment_results"))
    ap.add_argument("--max_subjects", type=int, default=int(os.environ.get("MAX_SUBJECTS", "0")))
    ap.add_argument("--debug_paths", action="store_true")
    args = ap.parse_args()
    args.phenotype_path = args.phenotype_path or os.environ.get("PHENO", "")
    args.nifti_dir = args.nifti_dir or os.environ.get("NIFTI_DIR", "")
    args.nifti_pattern = args.nifti_pattern or os.environ.get("NIFTI_PATTERN", "")
    args.atlas_path = args.atlas_path or os.environ.get("ATLAS_PATH", "")
    args.labels_path = args.labels_path or os.environ.get("LABELS_PATH", "")
    args.out_dir = args.out_dir or os.environ.get("OUT_BNA", os.path.join(proj, "experiment_results"))
    out_dir = args.out_dir or os.path.join(proj, "experiment_results")
    args.out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    pheno_path = args.phenotype_path
    if not pheno_path:
        raise FileNotFoundError("Phenotype CSV path not provided. Set --phenotype_path or PHENO environment variable to your Drive CSV.")
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"Phenotype CSV not found at '{pheno_path}'. Check the path or set PHENO env to your Drive CSV.")
    pheno = pd.read_csv(pheno_path)
    ids = pheno["FILE_ID"].astype(str).tolist() if "FILE_ID" in pheno.columns else pheno["SUB_ID"].astype(str).tolist()
    key_col = "FILE_ID" if "FILE_ID" in pheno.columns else "SUB_ID"
    site_by_id = {}
    if "SITE" in pheno.columns:
        for _, row in pheno.iterrows():
            site_by_id[str(row[key_col])] = str(row["SITE"]) if not pd.isna(row["SITE"]) else ""
    if not args.nifti_dir or not os.path.isdir(args.nifti_dir):
        raise FileNotFoundError(f"NIFTI directory not found at '{args.nifti_dir}'. Mount Drive and set --nifti_dir to the folder containing .nii.gz files.")
    if args.max_subjects and args.max_subjects > 0:
        ids = ids[: args.max_subjects]
    third_party_dir = os.path.join(proj, "third_party", "autism_connectome")
    atlas_path = args.atlas_path or os.path.join(third_party_dir, "fullbrain_atlas_thr0-2mm.nii.gz")
    labels_path = args.labels_path or os.path.join(third_party_dir, "BNA_subregions.xlsx")
    os.makedirs(third_party_dir, exist_ok=True)
    def download_if_url(name, url, dest_path):
        if not url:
            return False
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            return False
    use_bna = os.path.exists(atlas_path) and os.path.exists(labels_path)
    if not use_bna:
        atlas_url = os.environ.get("BNA_ATLAS_URL", "")
        labels_url = os.environ.get("BNA_LABELS_URL", "")
        if not os.path.exists(atlas_path) and atlas_url:
            if download_if_url("atlas", atlas_url, atlas_path):
                pass
        if not os.path.exists(labels_path) and labels_url:
            if download_if_url("labels", labels_url, labels_path):
                pass
        use_bna = os.path.exists(atlas_path) and os.path.exists(labels_path)
    if args.strict_bna and not use_bna:
        raise RuntimeError("BNA atlas/labels not found under third_party/autism_connectome and strict mode is enabled")
    networks = None
    if use_bna:
        from asd_pipeline.atlas_labels import load_bna_labels
        names, networks = load_bna_labels(labels_path)
    # Build features directly from NIFTI per subject (no ROI .npy files)
    if not use_bna:
        from nilearn import datasets
        ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_path = ho["maps"]
    feats = []
    found_ids = []
    debug_missing = []
    for sid in ids:
        if not args.nifti_dir:
            continue
        candidates = []
        site = site_by_id.get(sid, "") if 'site_by_id' in locals() else ""
        site_variants = [site]
        if site:
            site_variants += [site.lower(), site.upper(), site.title(), site.replace(" ", "_")]
        if args.nifti_pattern:
            for sv in site_variants:
                try:
                    candidates.append(os.path.join(args.nifti_dir, args.nifti_pattern.format(SUB_ID=sid, FILE_ID=sid, SITE=sv)))
                except Exception:
                    pass
        candidates.extend([
            os.path.join(args.nifti_dir, f"{sid}.nii.gz"),
            os.path.join(args.nifti_dir, f"{sid}_func_preproc.nii.gz"),
            os.path.join(args.nifti_dir, f"{sid}_func_preprop.nii.gz"),
            os.path.join(args.nifti_dir, f"{sid}_bold.nii.gz"),
            os.path.join(args.nifti_dir, f"{sid}_bold_preproc.nii.gz"),
        ])
        if site:
            candidates.extend([
                os.path.join(args.nifti_dir, f"{site}_{sid}_func_preproc.nii.gz"),
                os.path.join(args.nifti_dir, f"{site}-{sid}_func_preproc.nii.gz"),
                os.path.join(args.nifti_dir, f"{site}_{sid}_func_preprop.nii.gz"),
                os.path.join(args.nifti_dir, f"{site}_{sid}_bold.nii.gz"),
            ])
        candidates = [p for p in candidates if p]
        nii = next((p for p in candidates if os.path.isfile(p)), "")
        if not nii:
            if args.debug_paths:
                debug_missing.append({"sid": sid, "site": site, "candidates": candidates})
            continue
        ts = extract_roi_timeseries(nii, atlas_path, atlas_type="labels", tr=2.0, high_pass=0.01, low_pass=0.1)
        ts = regress_confounds(ts, confounds=None, tr=2.0, high_pass=0.01, low_pass=0.1)
        vec, _ = build_connectome_feature_vector(ts, window_size=30, step=15, thr=0.4, networks=networks, compute_partial=False)
        feats.append(vec)
        found_ids.append(sid)
    if not feats:
        if args.debug_paths and debug_missing:
            sample = debug_missing[:3]
            print(json.dumps({"debug_no_matches": sample}, indent=2))
        raise FileNotFoundError("No NIFTI files matched. Verify --nifti_dir exists and --nifti_pattern. Pattern supports {SUB_ID},{FILE_ID},{SITE}. Consider --debug_paths to print attempted paths.")
    X = np.stack(feats, axis=0)
    key_col = "FILE_ID" if "FILE_ID" in pheno.columns else "SUB_ID"
    ph_sub = pheno[pheno[key_col].astype(str).isin(found_ids)].copy()
    y = ph_sub["DX_GROUP"].values.astype(int)
    td_mask = y == 2
    covars = ph_sub[["AGE_AT_SCAN", "SEX"]].values.astype(float)
    dev = personalized_deviation_maps(X[td_mask], X, covars=covars)
    X_dev = dev["feature_z"]
    groups = ph_sub["SITE"].values
    metrics = evaluate_models(X_dev, y, cv_strategy="site_stratified", groups=groups, cv_splits=5)
    with open(os.path.join(out_dir, "bna_results.json"), "w") as f:
        json.dump({"models": metrics}, f)
    print(json.dumps({"models": metrics}))


if __name__ == "__main__":
    main()
