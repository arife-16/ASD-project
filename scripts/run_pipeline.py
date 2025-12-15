import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
import glob

from asd_pipeline.preprocess import regress_confounds
from asd_pipeline.features import build_feature_vector, build_feature_vector_with_states, dynamic_state_features, alff
from asd_pipeline.harmonize import combat_harmonize
from asd_pipeline.normative import personalized_deviation_maps
from asd_pipeline.model import evaluate_classifier, evaluate_models
from asd_pipeline.atlas import extract_roi_timeseries
from asd_pipeline.confounds import build_confounds
from asd_pipeline.connectome import build_connectome_feature_vector
from asd_pipeline.atlas_labels import load_cc_labels


def load_timeseries(subject_ids: List[str], ts_dir: str) -> List[np.ndarray]:
    series = []
    for sid in subject_ids:
        path = os.path.join(ts_dir, f"{sid}.npy")
        arr = np.load(path)
        series.append(arr)
    return series


def build_features(subject_ids: List[str], ts_dir: str, tr: float, window_size: int, step: int, confounds_dir: str = "") -> np.ndarray:
    feats = []
    for sid in subject_ids:
        ts = np.load(os.path.join(ts_dir, f"{sid}.npy"))
        confounds = None
        if confounds_dir:
            cpath = os.path.join(confounds_dir, f"{sid}.npy")
            if os.path.exists(cpath):
                confounds = np.load(cpath)
        ts_clean = regress_confounds(ts, confounds=confounds, tr=tr, high_pass=0.01, low_pass=0.1)
        vec, _ = build_feature_vector(ts_clean, tr=tr, window_size=window_size, step=step)
        feats.append(vec)
    return np.stack(feats, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotype", type=str, required=True)
    parser.add_argument("--ts_dir", type=str, default="")
    parser.add_argument("--nifti_dir", type=str, default="")
    parser.add_argument("--atlas", type=str, default="")
    parser.add_argument("--atlas_type", type=str, default="labels")
    parser.add_argument("--labels_tsv", type=str, default="")
    parser.add_argument("--confounds_dir", type=str, default="")
    parser.add_argument("--confounds_tsv_dir", type=str, default="")
    parser.add_argument("--wm_mask", type=str, default="")
    parser.add_argument("--csf_mask", type=str, default="")
    parser.add_argument("--site_col", type=str, default="SITE")
    parser.add_argument("--label_col", type=str, default="DX_GROUP")
    parser.add_argument("--age_col", type=str, default="AGE_AT_SCAN")
    parser.add_argument("--sex_col", type=str, default="SEX")
    parser.add_argument("--tr", type=float, default=2.0)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--n_states", type=int, default=5)
    parser.add_argument("--cv_strategy", type=str, default="stratified")
    parser.add_argument("--loso", action="store_true")
    parser.add_argument("--tune_models", action="store_true")
    parser.add_argument("--output", type=str, default="pipeline_results.json")
    args = parser.parse_args()

    pheno = pd.read_csv(args.phenotype)
    ids = pheno["FILE_ID"].astype(str)
    valid_mask = (~ids.isna()) & (ids.str.len() > 0) & (~ids.str.lower().str.contains("no_filename"))
    pheno = pheno.loc[valid_mask].reset_index(drop=True)
    subject_ids = pheno["FILE_ID"].astype(str).tolist()
    sites_for_id = {str(row["FILE_ID"]): str(row[args.site_col]) if args.site_col in pheno.columns else "" for _, row in pheno.iterrows()}
    if args.nifti_dir and args.atlas:
        ts_dir_tmp = os.path.join(args.nifti_dir, "_roi_ts")
        os.makedirs(ts_dir_tmp, exist_ok=True)
        for sid in subject_ids:
            nifti_path = os.path.join(args.nifti_dir, f"{sid}.nii.gz")
            if not os.path.exists(nifti_path):
                continue
            ts = extract_roi_timeseries(nifti_path, args.atlas, atlas_type=args.atlas_type, tr=args.tr, high_pass=0.01, low_pass=0.1)
            confounds = None
            if args.confounds_tsv_dir or args.wm_mask or args.csf_mask:
                tsv_path = os.path.join(args.confounds_tsv_dir, f"{sid}_confounds.tsv") if args.confounds_tsv_dir else None
                confounds = build_confounds(nifti_path, tsv_path, args.wm_mask or None, args.csf_mask or None, args.tr)
            if confounds is not None:
                ts = regress_confounds(ts, confounds=confounds, tr=args.tr, high_pass=0.01, low_pass=0.1)
            np.save(os.path.join(ts_dir_tmp, f"{sid}.npy"), ts)
        def build(sids, dirpath):
            feats = []
            networks = None
            if args.labels_tsv:
                _, networks = load_cc_labels(args.labels_tsv)
            for sid in sids:
                ts = np.load(os.path.join(dirpath, f"{sid}.npy"))
                use_connectome = bool(args.labels_tsv) or ("CC400" in os.path.basename(args.atlas))
                if use_connectome:
                    v_conn, _ = build_connectome_feature_vector(ts, window_size=args.window_size, step=args.step, thr=0.3, networks=networks, compute_partial=True)
                    v_alff = alff(ts, args.tr)
                    parts = [v_conn, v_alff]
                    if args.n_states and args.n_states > 0:
                        state = dynamic_state_features(ts, args.window_size, args.step, n_states=args.n_states)
                        parts.extend([state["state_occ"], state["transitions"], state["dwell_mean"], state["dwell_std"], state["entropy"], state["asymmetry"]])
                    vec = np.concatenate(parts, axis=0)
                    feats.append(vec)
                else:
                    if args.n_states and args.n_states > 0:
                        vec, _ = build_feature_vector_with_states(ts, tr=args.tr, window_size=args.window_size, step=args.step, n_states=args.n_states)
                    else:
                        vec, _ = build_feature_vector(ts, tr=args.tr, window_size=args.window_size, step=args.step)
                    feats.append(vec)
            return np.stack(feats, axis=0)
        X = build(subject_ids, ts_dir_tmp)
    else:
        feats = []
        networks = None
        if args.labels_tsv:
            _, networks = load_cc_labels(args.labels_tsv)
        for sid in subject_ids:
            site = sites_for_id.get(sid, "")
            direct = os.path.join(args.ts_dir, f"{sid}.npy")
            alt1 = os.path.join(args.ts_dir, f"{site}_{sid}.npy") if site else ""
            alt2 = os.path.join(args.ts_dir, f"{site}-{sid}.npy") if site else ""
            pattern = glob.glob(os.path.join(args.ts_dir, f"*{sid}*.npy"))
            chosen = None
            for p in [direct, alt1, alt2] + pattern:
                if p and os.path.exists(p):
                    chosen = p
                    break
            if not chosen:
                continue
            ts = np.load(chosen)
            confounds = None
            if args.confounds_dir:
                cpath = os.path.join(args.confounds_dir, f"{sid}.npy")
                if os.path.exists(cpath):
                    confounds = np.load(cpath)
            ts_clean = regress_confounds(ts, confounds=confounds, tr=args.tr, high_pass=0.01, low_pass=0.1)
            use_connectome = bool(args.labels_tsv)
            if use_connectome:
                v_conn, _ = build_connectome_feature_vector(ts_clean, window_size=args.window_size, step=args.step, thr=0.3, networks=networks, compute_partial=True)
                v_alff = alff(ts_clean, args.tr)
                parts = [v_conn, v_alff]
                if args.n_states and args.n_states > 0:
                    state = dynamic_state_features(ts_clean, args.window_size, args.step, n_states=args.n_states)
                    parts.extend([state["state_occ"], state["transitions"], state["dwell_mean"], state["dwell_std"], state["entropy"], state["asymmetry"]])
                vec = np.concatenate(parts, axis=0)
            else:
                if args.n_states and args.n_states > 0:
                    vec, _ = build_feature_vector_with_states(ts_clean, tr=args.tr, window_size=args.window_size, step=args.step, n_states=args.n_states)
                else:
                    vec, _ = build_feature_vector(ts_clean, tr=args.tr, window_size=args.window_size, step=args.step)
            feats.append(vec)
        X = np.stack(feats, axis=0)
    covars = pheno[[args.site_col, args.age_col, args.sex_col]].copy()
    continuous = [args.age_col]
    categorical = [args.sex_col]
    X_h, _ = combat_harmonize(X, covars[[args.site_col, args.age_col, args.sex_col]], site_col=args.site_col, continuous_covars=continuous, categorical_covars=categorical)
    y = pheno[args.label_col].values.astype(int)
    td_mask = y == 2
    covars_arr = covars[[args.age_col, args.sex_col]].values.astype(float)
    dev = personalized_deviation_maps(X_h[td_mask], X_h, covars=covars_arr)
    X_dev = dev["feature_z"]
    groups = pheno[args.site_col].values
    if args.tune_models:
        cv_name = "loso" if args.loso else ("group" if args.cv_strategy == "group" else ("site_stratified" if args.cv_strategy == "site_stratified" else "stratified"))
        model_metrics = evaluate_models(X_dev, y, cv_strategy=cv_name, groups=groups, cv_splits=5)
        out = {"models": model_metrics}
    else:
        metrics = evaluate_classifier(X_dev, y)
        out = {"metrics": metrics}
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
