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
from asd_pipeline.precision_mapping import precision_mapping_workflow


def find_ts_file(sid: str, site: str, ts_dir: str) -> str:
    direct = os.path.join(ts_dir, f"{sid}.npy")
    alt1 = os.path.join(ts_dir, f"{site}_{sid}.npy") if site else ""
    alt2 = os.path.join(ts_dir, f"{site}-{sid}.npy") if site else ""
    pattern = glob.glob(os.path.join(ts_dir, f"*{sid}*.npy"))
    chosen = None
    for p in [direct, alt1, alt2] + pattern:
        if p and os.path.exists(p):
            chosen = p
            break
    return chosen

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
    parser.add_argument("--features_out_dir", type=str, default="")
    parser.add_argument("--norm_features_out_dir", type=str, default="")
    parser.add_argument("--skip_models", action="store_true")
    parser.add_argument("--components", type=str, default="")
    parser.add_argument("--components_out_dir", type=str, default="")
    parser.add_argument("--combine_only", action="store_true")
    parser.add_argument("--adj_thr", type=float, default=0.3)
    parser.add_argument("--external_norm_params", type=str, default="")
    parser.add_argument("--normative_model", type=str, default="linear", choices=["linear", "lowess", "gpr"], help="Type of normative model: linear, lowess, or gpr")
    parser.add_argument("--precision_mapping", action="store_true", help="Use precision functional mapping (dense connectivity -> Infomap)")
    parser.add_argument("--template_labels_path", type=str, default="", help="Path to template labels NIfTI or CIFTI for precision mapping")
    parser.add_argument("--gm_mask", type=str, default="", help="Path to Gray Matter mask for precision mapping (reduces voxel count)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for acceleration where possible")
    args = parser.parse_args()

    pheno = pd.read_csv(args.phenotype)
    ids = pheno["FILE_ID"].astype(str)
    valid_mask = (~ids.isna()) & (ids.str.len() > 0) & (~ids.str.lower().str.contains("no_filename"))
    pheno = pheno.loc[valid_mask].reset_index(drop=True)
    subject_ids = pheno["FILE_ID"].astype(str).tolist()
    sites_for_id = {str(row["FILE_ID"]): str(row[args.site_col]) if args.site_col in pheno.columns else "" for _, row in pheno.iterrows()}
    if args.combine_only and args.components_out_dir and args.features_out_dir:
        os.makedirs(args.features_out_dir, exist_ok=True)
        comps = [c.strip() for c in (args.components.split(",") if args.components else []) if c.strip()]
        if comps:
            order = comps
        elif args.precision_mapping:
            order = ["network_area"]
        else:
            order = ["fc_mean","fc_std","pc_mean","pc_std","strength_mean","strength_std","cluster_mean","cluster_std","alff","state_occ","transitions","dwell_mean","dwell_std","entropy","asymmetry"]
        feats = []
        n_done = 0
        print(json.dumps({"combine_start": {"n_subjects": len(subject_ids), "components_out_dir": args.components_out_dir, "features_out_dir": args.features_out_dir, "components": order}}), flush=True)
        for sid in subject_ids:
            parts = []
            ok = True
            for name in order:
                p = os.path.join(args.components_out_dir, name, f"{sid}.npy")
                if not os.path.exists(p):
                    print(json.dumps({"missing_component": {"sid": sid, "component": name, "path": p}}), flush=True)
                    ok = False
                    break
                parts.append(np.load(p))
            if not ok:
                continue
            vec = np.concatenate(parts, axis=0)
            np.save(os.path.join(args.features_out_dir, f"{sid}.npy"), vec)
            feats.append(vec)
            n_done += 1
            if n_done % 10 == 0:
                print(json.dumps({"combine_progress": n_done}), flush=True)
        if len(feats) > 0:
            X = np.stack(feats, axis=0)
            np.save(os.path.join(args.features_out_dir, "_stack.npy"), X)
            with open(os.path.join(args.features_out_dir, "_index.txt"), "w") as f:
                for sid in subject_ids:
                    p = os.path.join(args.features_out_dir, f"{sid}.npy")
                    if os.path.exists(p):
                        f.write(f"{sid}\n")
        else:
            X = np.empty((0,))
        print(json.dumps({"combine_done": n_done}), flush=True)
        if args.skip_models:
            out = {"combine_done": n_done, "saved_features": True, "n_subjects": len(subject_ids)}
            with open(args.output, "w") as f:
                json.dump(out, f)
            print(json.dumps(out), flush=True)
            return
    if args.precision_mapping:
        # Fallback: if template_labels_path is missing but labels_tsv is a NIfTI, use it
        if not args.template_labels_path and args.labels_tsv and (args.labels_tsv.endswith(".nii") or args.labels_tsv.endswith(".nii.gz")):
            args.template_labels_path = args.labels_tsv
            
        if not args.template_labels_path:
            raise ValueError("--template_labels_path (or --labels_tsv with a NIfTI file) is required for precision mapping")
        
        # Load template
        # Assume template is in same space as timeseries (or we resample)
        # For simplicity, we assume .npy labels or .nii
        template_labels = args.template_labels_path # pass path to workflow if needed, or load here
        if args.template_labels_path.endswith(".npy"):
            template_labels = np.load(args.template_labels_path)
        # If NIfTI, we let workflow handle it alongside masking
            
        feats = []
        for sid in subject_ids:
            # Check if already processed
            outd_chk = os.path.join(args.components_out_dir, "network_area") if args.components_out_dir else None
            if outd_chk and os.path.exists(os.path.join(outd_chk, f"{sid}.npy")):
                print(f"Skipping {sid}, already processed.", flush=True)
                feats.append(np.load(os.path.join(outd_chk, f"{sid}.npy")))
                continue

            # Find timeseries file
            # Priority: NIfTI file in ts_dir or nifti_dir
            # Fallback: .npy file
            
            chosen = None
            # Check for NIfTI first if precision mapping (raw data preferred)
            if args.nifti_dir:
                nii = os.path.join(args.nifti_dir, f"{sid}.nii.gz")
                if os.path.exists(nii):
                    chosen = nii
            
            if not chosen:
                site = sites_for_id.get(sid, "")
                # Try finding NIfTI in ts_dir
                pat = glob.glob(os.path.join(args.ts_dir, f"*{sid}*.nii.gz"))
                if pat:
                    chosen = pat[0]
                else:
                    # Fallback to .npy (assuming it's voxelwise if user knows what they're doing)
                    chosen = find_ts_file(sid, site, args.ts_dir)
            
            if not chosen:
                print(f"Warning: Could not find timeseries (NIfTI or .npy) for {sid}", flush=True)
                continue
                
            # Run workflow
            try:
                if chosen.endswith(".npy"):
                    ts = np.load(chosen)
                    if ts.shape[0] < ts.shape[1]:
                        ts = ts.T
                    # If .npy, we assume it's already masked or user accepts high dim
                    # We can't apply NIfTI mask to .npy easily without knowing shape
                    areas = precision_mapping_workflow(ts, template_labels, use_gpu=args.use_gpu)
                else:
                    # Pass path to workflow to handle loading & masking
                    # If gm_mask is not provided, precision_loader will default to MNI GM mask
                    areas = precision_mapping_workflow(chosen, template_labels, mask_img=args.gm_mask or None, use_gpu=args.use_gpu)
            
                # Save if components out
                if args.components_out_dir:
                    outd = os.path.join(args.components_out_dir, "network_area")
                    os.makedirs(outd, exist_ok=True)
                    np.save(os.path.join(outd, f"{sid}.npy"), areas)
                    
                feats.append(areas)
            except Exception as e:
                print(f"Error processing {sid} for precision mapping: {e}", flush=True)
                continue
            
        if len(feats) > 0:
            X = np.stack(feats, axis=0)
        else:
            X = np.empty((0,))
            
    elif args.nifti_dir and args.atlas:
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
                    if args.components_out_dir:
                        comps = [c.strip() for c in (args.components.split(",") if args.components else []) if c.strip()]
                        start = 0
                        n_time = ts.shape[1]
                        n_rois = ts.shape[0]
                        fc_sum = np.zeros((n_rois, n_rois), dtype=float)
                        fc_sumsq = np.zeros((n_rois, n_rois), dtype=float)
                        pc_sum = np.zeros((n_rois, n_rois), dtype=float)
                        pc_sumsq = np.zeros((n_rois, n_rois), dtype=float)
                        str_sum = np.zeros(n_rois, dtype=float)
                        str_sumsq = np.zeros(n_rois, dtype=float)
                        clu_sum = np.zeros(n_rois, dtype=float)
                        clu_sumsq = np.zeros(n_rois, dtype=float)
                        n_w = 0
                        while start + args.window_size <= n_time:
                            w = ts[:, start : start + args.window_size]
                            mat = None
                            if any(k in comps for k in ["fc_mean", "fc_std", "strength_mean", "strength_std", "cluster_mean", "cluster_std"]) or len(comps) == 0:
                                mat = np.corrcoef((w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8))
                                fc_sum += mat
                                fc_sumsq += mat * mat
                                s = np.sum(np.abs(mat), axis=1)
                                str_sum += s
                                str_sumsq += s * s
                                a = (np.abs(mat) >= args.adj_thr).astype(np.int8)
                                n = a.shape[0]
                                cc = np.zeros(n, dtype=float)
                                for i in range(n):
                                    nbrs = np.where(a[i] > 0)[0]
                                    k = len(nbrs)
                                    if k < 2:
                                        cc[i] = 0.0
                                        continue
                                    sub = a[np.ix_(nbrs, nbrs)]
                                    t = (np.sum(sub) - np.trace(sub)) / 2.0
                                    cc[i] = (t) / (k * (k - 1) / 2.0)
                                clu_sum += cc
                                clu_sumsq += cc * cc
                            if any(k in comps for k in ["pc_mean", "pc_std"]) or len(comps) == 0:
                                from sklearn.covariance import LedoitWolf
                                lw = LedoitWolf().fit(w.T)
                                P = lw.precision_
                                d = np.sqrt(np.diag(P))
                                denom = np.outer(d, d)
                                pc = -P / denom
                                np.fill_diagonal(pc, 1.0)
                                pc_sum += pc
                                pc_sumsq += pc * pc
                            n_w += 1
                            start += args.step
                        inv_n = 1.0 / max(n_w, 1)
                        def vec_ut(m):
                            idx = np.triu_indices(m.shape[0], k=1)
                            return m[idx]
                        os.makedirs(args.components_out_dir, exist_ok=True)
                        if (not comps) or ("fc_mean" in comps):
                            v = vec_ut(fc_sum * inv_n)
                            outd = os.path.join(args.components_out_dir, "fc_mean")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or ("fc_std" in comps):
                            var = (fc_sumsq - (fc_sum * fc_sum) * inv_n) / max(n_w - 1, 1)
                            var = np.where(var < 0, 0, var)
                            v = vec_ut(np.sqrt(var))
                            outd = os.path.join(args.components_out_dir, "fc_std")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or ("pc_mean" in comps):
                            v = vec_ut(pc_sum * inv_n)
                            outd = os.path.join(args.components_out_dir, "pc_mean")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or ("pc_std" in comps):
                            var = (pc_sumsq - (pc_sum * pc_sum) * inv_n) / max(n_w - 1, 1)
                            var = np.where(var < 0, 0, var)
                            v = vec_ut(np.sqrt(var))
                            outd = os.path.join(args.components_out_dir, "pc_std")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or ("strength_mean" in comps):
                            v = str_sum * inv_n
                            outd = os.path.join(args.components_out_dir, "strength_mean")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or ("strength_std" in comps):
                            var = (str_sumsq - (str_sum * str_sum) * inv_n) / max(n_w - 1, 1)
                            var = np.where(var < 0, 0, var)
                            v = np.sqrt(var)
                            outd = os.path.join(args.components_out_dir, "strength_std")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or ("cluster_mean" in comps):
                            v = clu_sum * inv_n
                            outd = os.path.join(args.components_out_dir, "cluster_mean")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or ("cluster_std" in comps):
                            var = (clu_sumsq - (clu_sum * clu_sum) * inv_n) / max(n_w - 1, 1)
                            var = np.where(var < 0, 0, var)
                            v = np.sqrt(var)
                            outd = os.path.join(args.components_out_dir, "cluster_std")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or ("alff" in comps):
                            v = alff(ts, args.tr)
                            outd = os.path.join(args.components_out_dir, "alff")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), v)
                        if (not comps) or any(k in comps for k in ["state_occ", "transitions", "dwell_mean", "dwell_std", "entropy", "asymmetry"]):
                            st = dynamic_state_features(ts, args.window_size, args.step, n_states=args.n_states)
                            if (not comps) or ("state_occ" in comps):
                                outd = os.path.join(args.components_out_dir, "state_occ")
                                os.makedirs(outd, exist_ok=True)
                                np.save(os.path.join(outd, f"{sid}.npy"), st["state_occ"])
                            if (not comps) or ("transitions" in comps):
                                outd = os.path.join(args.components_out_dir, "transitions")
                                os.makedirs(outd, exist_ok=True)
                                np.save(os.path.join(outd, f"{sid}.npy"), st["transitions"])
                            if (not comps) or ("dwell_mean" in comps):
                                outd = os.path.join(args.components_out_dir, "dwell_mean")
                                os.makedirs(outd, exist_ok=True)
                                np.save(os.path.join(outd, f"{sid}.npy"), st["dwell_mean"])
                            if (not comps) or ("dwell_std" in comps):
                                outd = os.path.join(args.components_out_dir, "dwell_std")
                                os.makedirs(outd, exist_ok=True)
                                np.save(os.path.join(outd, f"{sid}.npy"), st["dwell_std"])
                            if (not comps) or ("entropy" in comps):
                                outd = os.path.join(args.components_out_dir, "entropy")
                                os.makedirs(outd, exist_ok=True)
                                np.save(os.path.join(outd, f"{sid}.npy"), st["entropy"])
                            if (not comps) or ("asymmetry" in comps):
                                outd = os.path.join(args.components_out_dir, "asymmetry")
                                os.makedirs(outd, exist_ok=True)
                                np.save(os.path.join(outd, f"{sid}.npy"), st["asymmetry"])
                        if args.combine_only:
                            continue
                        v_conn, _ = build_connectome_feature_vector(ts, window_size=args.window_size, step=args.step, thr=args.adj_thr, networks=networks, compute_partial=True)
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
            chosen = find_ts_file(sid, site, args.ts_dir)
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
                if args.components_out_dir:
                    comps = [c.strip() for c in (args.components.split(",") if args.components else []) if c.strip()]
                    start = 0
                    n_time = ts_clean.shape[1]
                    n_rois = ts_clean.shape[0]
                    fc_sum = np.zeros((n_rois, n_rois), dtype=float)
                    fc_sumsq = np.zeros((n_rois, n_rois), dtype=float)
                    pc_sum = np.zeros((n_rois, n_rois), dtype=float)
                    pc_sumsq = np.zeros((n_rois, n_rois), dtype=float)
                    str_sum = np.zeros(n_rois, dtype=float)
                    str_sumsq = np.zeros(n_rois, dtype=float)
                    clu_sum = np.zeros(n_rois, dtype=float)
                    clu_sumsq = np.zeros(n_rois, dtype=float)
                    n_w = 0
                    while start + args.window_size <= n_time:
                        w = ts_clean[:, start : start + args.window_size]
                        mat = None
                        if any(k in comps for k in ["fc_mean", "fc_std", "strength_mean", "strength_std", "cluster_mean", "cluster_std"]) or len(comps) == 0:
                            mat = np.corrcoef((w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8))
                            fc_sum += mat
                            fc_sumsq += mat * mat
                            s = np.sum(np.abs(mat), axis=1)
                            str_sum += s
                            str_sumsq += s * s
                            a = (np.abs(mat) >= args.adj_thr).astype(np.int8)
                            n = a.shape[0]
                            cc = np.zeros(n, dtype=float)
                            for i in range(n):
                                nbrs = np.where(a[i] > 0)[0]
                                k = len(nbrs)
                                if k < 2:
                                    cc[i] = 0.0
                                    continue
                                sub = a[np.ix_(nbrs, nbrs)]
                                t = (np.sum(sub) - np.trace(sub)) / 2.0
                                cc[i] = (t) / (k * (k - 1) / 2.0)
                            clu_sum += cc
                            clu_sumsq += cc * cc
                        if any(k in comps for k in ["pc_mean", "pc_std"]) or len(comps) == 0:
                            from sklearn.covariance import LedoitWolf
                            lw = LedoitWolf().fit(w.T)
                            P = lw.precision_
                            d = np.sqrt(np.diag(P))
                            denom = np.outer(d, d)
                            pc = -P / denom
                            np.fill_diagonal(pc, 1.0)
                            pc_sum += pc
                            pc_sumsq += pc * pc
                        n_w += 1
                        start += args.step
                    inv_n = 1.0 / max(n_w, 1)
                    def vec_ut(m):
                        idx = np.triu_indices(m.shape[0], k=1)
                        return m[idx]
                    os.makedirs(args.components_out_dir, exist_ok=True)
                    if (not comps) or ("fc_mean" in comps):
                        v = vec_ut(fc_sum * inv_n)
                        outd = os.path.join(args.components_out_dir, "fc_mean")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or ("fc_std" in comps):
                        var = (fc_sumsq - (fc_sum * fc_sum) * inv_n) / max(n_w - 1, 1)
                        var = np.where(var < 0, 0, var)
                        v = vec_ut(np.sqrt(var))
                        outd = os.path.join(args.components_out_dir, "fc_std")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or ("pc_mean" in comps):
                        v = vec_ut(pc_sum * inv_n)
                        outd = os.path.join(args.components_out_dir, "pc_mean")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or ("pc_std" in comps):
                        var = (pc_sumsq - (pc_sum * pc_sum) * inv_n) / max(n_w - 1, 1)
                        var = np.where(var < 0, 0, var)
                        v = vec_ut(np.sqrt(var))
                        outd = os.path.join(args.components_out_dir, "pc_std")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or ("strength_mean" in comps):
                        v = str_sum * inv_n
                        outd = os.path.join(args.components_out_dir, "strength_mean")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or ("strength_std" in comps):
                        var = (str_sumsq - (str_sum * str_sum) * inv_n) / max(n_w - 1, 1)
                        var = np.where(var < 0, 0, var)
                        v = np.sqrt(var)
                        outd = os.path.join(args.components_out_dir, "strength_std")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or ("cluster_mean" in comps):
                        v = clu_sum * inv_n
                        outd = os.path.join(args.components_out_dir, "cluster_mean")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or ("cluster_std" in comps):
                        var = (clu_sumsq - (clu_sum * clu_sum) * inv_n) / max(n_w - 1, 1)
                        var = np.where(var < 0, 0, var)
                        v = np.sqrt(var)
                        outd = os.path.join(args.components_out_dir, "cluster_std")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or ("alff" in comps):
                        v = alff(ts_clean, args.tr)
                        outd = os.path.join(args.components_out_dir, "alff")
                        os.makedirs(outd, exist_ok=True)
                        np.save(os.path.join(outd, f"{sid}.npy"), v)
                    if (not comps) or any(k in comps for k in ["state_occ", "transitions", "dwell_mean", "dwell_std", "entropy", "asymmetry"]):
                        st = dynamic_state_features(ts_clean, args.window_size, args.step, n_states=args.n_states)
                        if (not comps) or ("state_occ" in comps):
                            outd = os.path.join(args.components_out_dir, "state_occ")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), st["state_occ"])
                        if (not comps) or ("transitions" in comps):
                            outd = os.path.join(args.components_out_dir, "transitions")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), st["transitions"])
                        if (not comps) or ("dwell_mean" in comps):
                            outd = os.path.join(args.components_out_dir, "dwell_mean")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), st["dwell_mean"])
                        if (not comps) or ("dwell_std" in comps):
                            outd = os.path.join(args.components_out_dir, "dwell_std")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), st["dwell_std"])
                        if (not comps) or ("entropy" in comps):
                            outd = os.path.join(args.components_out_dir, "entropy")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), st["entropy"])
                        if (not comps) or ("asymmetry" in comps):
                            outd = os.path.join(args.components_out_dir, "asymmetry")
                            os.makedirs(outd, exist_ok=True)
                            np.save(os.path.join(outd, f"{sid}.npy"), st["asymmetry"])
                    if args.combine_only:
                        continue
                v_conn, _ = build_connectome_feature_vector(ts_clean, window_size=args.window_size, step=args.step, thr=args.adj_thr, networks=networks, compute_partial=True)
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
        if len(feats) > 0:
            X = np.stack(feats, axis=0)
        else:
            X = np.empty((0,))
    if args.combine_only and args.components_out_dir and args.features_out_dir:
        os.makedirs(args.features_out_dir, exist_ok=True)
        comps = [c.strip() for c in (args.components.split(",") if args.components else []) if c.strip()]
        order = comps if comps else ["fc_mean","fc_std","pc_mean","pc_std","strength_mean","strength_std","cluster_mean","cluster_std","alff","state_occ","transitions","dwell_mean","dwell_std","entropy","asymmetry"]
        feats = []
        for sid in subject_ids:
            parts = []
            ok = True
            for name in order:
                p = os.path.join(args.components_out_dir, name, f"{sid}.npy")
                if not os.path.exists(p):
                    ok = False
                    break
                parts.append(np.load(p))
            if not ok:
                continue
            vec = np.concatenate(parts, axis=0)
            np.save(os.path.join(args.features_out_dir, f"{sid}.npy"), vec)
            feats.append(vec)
        if len(feats) > 0:
            X = np.stack(feats, axis=0)
        else:
            X = np.empty((0,))
    if args.features_out_dir:
        os.makedirs(args.features_out_dir, exist_ok=True)
        for sid, row in zip(subject_ids, X):
            np.save(os.path.join(args.features_out_dir, f"{sid}.npy"), row)
        np.save(os.path.join(args.features_out_dir, "_stack.npy"), X)
        with open(os.path.join(args.features_out_dir, "_index.txt"), "w") as f:
            for sid in subject_ids:
                f.write(f"{sid}\n")
    if args.site_col not in pheno.columns:
        # Fallback for common variations
        if "SITE_ID" in pheno.columns:
            args.site_col = "SITE_ID"
        elif "site_id" in pheno.columns:
            args.site_col = "site_id"
        else:
            print(f"Warning: Site column '{args.site_col}' not found in phenotype file. Assuming single site or skipping harmonization if used.", flush=True)
            # Create a dummy column if needed to prevent crash, but harmonization requires site
            pheno[args.site_col] = "SITE_1"
    
    covars = pheno[[args.site_col, args.age_col, args.sex_col]].copy()
    continuous = [args.age_col]
    categorical = [args.sex_col]
    if args.combine_only and args.components_out_dir and args.features_out_dir:
        os.makedirs(args.features_out_dir, exist_ok=True)
        comps = [c.strip() for c in (args.components.split(",") if args.components else []) if c.strip()]
        order = comps if comps else ["fc_mean","fc_std","pc_mean","pc_std","strength_mean","strength_std","cluster_mean","cluster_std","alff","state_occ","transitions","dwell_mean","dwell_std","entropy","asymmetry"]
        feats = []
        n_done = 0
        print(json.dumps({"combine_start": {"n_subjects": len(subject_ids), "components_out_dir": args.components_out_dir, "features_out_dir": args.features_out_dir, "components": order}}), flush=True)
        for sid in subject_ids:
            parts = []
            ok = True
            for name in order:
                p = os.path.join(args.components_out_dir, name, f"{sid}.npy")
                if not os.path.exists(p):
                    print(json.dumps({"missing_component": {"sid": sid, "component": name, "path": p}}), flush=True)
                    ok = False
                    break
                parts.append(np.load(p))
            if not ok:
                continue
            vec = np.concatenate(parts, axis=0)
            np.save(os.path.join(args.features_out_dir, f"{sid}.npy"), vec)
            feats.append(vec)
            n_done += 1
            if n_done % 10 == 0:
                print(json.dumps({"combine_progress": n_done}), flush=True)
        if len(feats) > 0:
            X = np.stack(feats, axis=0)
            np.save(os.path.join(args.features_out_dir, "_stack.npy"), X)
            with open(os.path.join(args.features_out_dir, "_index.txt"), "w") as f:
                for sid in subject_ids:
                    p = os.path.join(args.features_out_dir, f"{sid}.npy")
                    if os.path.exists(p):
                        f.write(f"{sid}\n")
        else:
            X = np.empty((0,))
        print(json.dumps({"combine_done": n_done}), flush=True)
        if args.skip_models:
            out = {"combine_done": n_done, "saved_features": True, "n_subjects": len(subject_ids)}
            with open(args.output, "w") as f:
                json.dump(out, f)
            print(json.dumps(out), flush=True)
            return
    X_h, _ = combat_harmonize(X, covars[[args.site_col, args.age_col, args.sex_col]], site_col=args.site_col, continuous_covars=continuous, categorical_covars=categorical)
    y = pheno[args.label_col].values.astype(int)
    td_mask = y == 2
    covars_arr = covars[[args.age_col, args.sex_col]].values.astype(float)
    if args.external_norm_params and os.path.exists(args.external_norm_params):
        params = None
        if args.external_norm_params.lower().endswith(".npz"):
            d = np.load(args.external_norm_params)
            params = {k: d[k] for k in d.files}
        else:
            with open(args.external_norm_params, "r") as f:
                import json as _json
                jd = _json.load(f)
                params = {k: np.asarray(v) for k, v in jd.items()}
        mean = params.get("mean", None)
        std = params.get("std", None)
        if mean is None or std is None:
            X_dev = personalized_deviation_maps(X_h[td_mask], X_h, covars=covars_arr)["feature_z"]
        else:
            s = np.where(std < 1e-8, 1e-8, std)
            X_dev = (X_h - mean) / s
    else:
        dev = personalized_deviation_maps(X_h[td_mask], X_h, covars=covars_arr, model_type=args.normative_model)
        X_dev = dev["feature_z"]
    if args.norm_features_out_dir:
        os.makedirs(args.norm_features_out_dir, exist_ok=True)
        for sid, row in zip(subject_ids, X_dev):
            np.save(os.path.join(args.norm_features_out_dir, f"{sid}.npy"), row)
        np.save(os.path.join(args.norm_features_out_dir, "_stack.npy"), X_dev)
        with open(os.path.join(args.norm_features_out_dir, "_index.txt"), "w") as f:
            for sid in subject_ids:
                f.write(f"{sid}\n")
    if args.skip_models:
        out = {"saved_features": bool(args.features_out_dir), "saved_norm_features": bool(args.norm_features_out_dir), "n_subjects": len(subject_ids)}
        with open(args.output, "w") as f:
            json.dump(out, f)
        print(json.dumps(out))
        return
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
