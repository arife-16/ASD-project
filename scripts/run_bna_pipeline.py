import os
import sys
import json
import numpy as np
import pandas as pd
import requests
import argparse

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
    ap.add_argument("--timeseries_dir", type=str, default="")
    ap.add_argument("--atlas_path", type=str, default="")
    ap.add_argument("--labels_path", type=str, default="")
    ap.add_argument("--strict_bna", action="store_true")
    ap.add_argument("--out_dir", type=str, default=os.path.join(proj, "experiment_results"))
    ap.add_argument("--max_subjects", type=int, default=int(os.environ.get("MAX_SUBJECTS", "0")))
    args = ap.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    roi_dir = os.path.join(out_dir, "_roi_ts")
    os.makedirs(roi_dir, exist_ok=True)
    pheno_path = args.phenotype_path or os.path.join(out_dir, "phenotype.csv")
    pheno = pd.read_csv(pheno_path)
    sids = pheno["SUB_ID"].astype(str).tolist()
    if args.max_subjects and args.max_subjects > 0:
        sids = sids[: args.max_subjects]
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
    need_extraction = any(not os.path.exists(os.path.join(roi_dir, f"{sid}.npy")) for sid in sids)
    if need_extraction:
        if not use_bna:
            from nilearn import datasets
            ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
            atlas_path = ho["maps"]
        for sid in sids:
            if args.timeseries_dir:
                ts_path_npy = os.path.join(args.timeseries_dir, f"{sid}.npy")
                ts_path_1d = os.path.join(args.timeseries_dir, f"{sid}.1D")
                if os.path.exists(ts_path_npy):
                    ts = np.load(ts_path_npy)
                    np.save(os.path.join(roi_dir, f"{sid}.npy"), ts)
                    continue
                if os.path.exists(ts_path_1d):
                    arr = np.loadtxt(ts_path_1d)
                    ts = arr.T if arr.shape[0] > arr.shape[1] else arr
                    np.save(os.path.join(roi_dir, f"{sid}.npy"), ts)
                    continue
            if args.nifti_dir:
                nii = os.path.join(args.nifti_dir, f"{SITE}_{sid}_func_preproc.nii.gz")
                if not os.path.exists(nii):
                    continue
                ts = extract_roi_timeseries(nii, atlas_path, atlas_type="labels", tr=2.0, high_pass=0.01, low_pass=0.1)
                ts = regress_confounds(ts, confounds=None, tr=2.0, high_pass=0.01, low_pass=0.1)
                np.save(os.path.join(roi_dir, f"{sid}.npy"), ts)
    feats = []
    for sid in sids:
        ts = np.load(os.path.join(roi_dir, f"{sid}.npy"))
        vec, _ = build_connectome_feature_vector(ts, window_size=30, step=15, thr=0.4, networks=networks, compute_partial=False)
        feats.append(vec)
    X = np.stack(feats, axis=0)
    ph_sub = pheno[pheno["SUB_ID"].astype(str).isin(sids)].copy()
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
