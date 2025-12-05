import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asd_pipeline.atlas import extract_roi_timeseries
from asd_pipeline.preprocess import regress_confounds
from asd_pipeline.connectome import build_connectome_feature_vector
from asd_pipeline.normative import personalized_deviation_maps
from asd_pipeline.model import evaluate_models


def main():
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(proj, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    roi_dir = os.path.join(out_dir, "_roi_ts")
    os.makedirs(roi_dir, exist_ok=True)
    pheno_path = os.path.join(out_dir, "phenotype.csv")
    pheno = pd.read_csv(pheno_path)
    sids = pheno["SUB_ID"].astype(str).tolist()
    max_n = int(os.environ.get("MAX_SUBJECTS", "0"))
    if max_n > 0:
        sids = sids[:max_n]
    third_party_dir = os.path.join(proj, "third_party", "autism_connectome")
    atlas_path = os.path.join(third_party_dir, "fullbrain_atlas_thr0-2mm.nii.gz")
    labels_path = os.path.join(third_party_dir, "BNA_subregions.xlsx")
    use_bna = os.path.exists(atlas_path) and os.path.exists(labels_path)
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--strict_bna", action="store_true")
    try:
        args, _ = ap.parse_known_args()
    except SystemExit:
        class _A:
            strict_bna = False
        args = _A()
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
            nii = os.path.join(proj, "Olin_0050115_func_preproc.nii.gz")
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
