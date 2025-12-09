import os
import sys
import json
import numpy as np
import pandas as pd
import requests
import argparse
import glob
import re
import shutil
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asd_pipeline.atlas import extract_roi_timeseries
from asd_pipeline.preprocess import regress_confounds
from asd_pipeline.connectome import build_connectome_feature_vector
from asd_pipeline.normative import personalized_deviation_maps
from asd_pipeline.model import evaluate_models
from asd_pipeline.features import dynamic_fc_sequence


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
    ap.add_argument("--pipeline_stage", type=str, default=os.environ.get("BNA_STAGE", "all"))
    ap.add_argument("--roi_dir", type=str, default="")
    ap.add_argument("--features_dir", type=str, default="")
    ap.add_argument("--sequence_dir", type=str, default="")
    ap.add_argument("--index_path", type=str, default="")
    ap.add_argument("--copy_to_local", action="store_true")
    ap.add_argument("--norm_mode", type=str, default=os.environ.get("NORM_MODE", "mvn"))
    ap.add_argument("--z_dir", type=str, default="")
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
    roi_dir = args.roi_dir or os.path.join(out_dir, "_roi_ts")
    features_dir = args.features_dir or os.path.join(out_dir, "_features")
    sequence_dir = args.sequence_dir or os.path.join(out_dir, "_seq_fc")
    index_path = args.index_path or os.path.join(out_dir, "match_index.json")
    z_dir = args.z_dir or os.path.join(out_dir, "_z")
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(sequence_dir, exist_ok=True)
    os.makedirs(z_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "_cache_nii")
    os.makedirs(cache_dir, exist_ok=True)

    def copy_with_retry(src, dst, tries=3, wait=1.0):
        for i in range(tries):
            try:
                shutil.copy2(src, dst)
                return True
            except Exception:
                time.sleep(wait * (i + 1))
        return False

    def materialize_nii(src_path):
        bn = os.path.basename(src_path)
        dst_path = os.path.join(cache_dir, bn)
        if os.path.exists(dst_path):
            return dst_path
        ok = copy_with_retry(src_path, dst_path, tries=3)
        return dst_path if ok else src_path
    
    pheno_path = args.phenotype_path
    if not pheno_path:
        raise FileNotFoundError("Phenotype CSV path not provided. Set --phenotype_path or PHENO environment variable to your Drive CSV.")
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"Phenotype CSV not found at '{pheno_path}'. Check the path or set PHENO env to your Drive CSV.")
    pheno = pd.read_csv(pheno_path)
    def pick_col(df, choices):
        for c in choices:
            if c in df.columns:
                return c
        return ""
    key_col = "FILE_ID" if "FILE_ID" in pheno.columns else ("SUB_ID" if "SUB_ID" in pheno.columns else "")
    if not key_col:
        kc = pick_col(pheno, ["FILE_ID", "SUB_ID", "SUBID", "subject_id", "ID"])
        key_col = kc
    ids = pheno[key_col].astype(str).tolist()
    site_by_id = {}
    site_col = pick_col(pheno, ["SITE", "site", "Site", "SITE_NAME"]) 
    if site_col:
        for _, row in pheno.iterrows():
            site_by_id[str(row[key_col])] = str(row[site_col]) if not pd.isna(row[site_col]) else ""
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
    all_niftis = glob.glob(os.path.join(args.nifti_dir, "**", "*.nii.gz"), recursive=True) if args.nifti_dir else []
    patts = [
        re.compile(r"^(?P<site>[A-Za-z0-9]+)[_-](?P<fileid>\d+)_func_preproc\.nii\.gz$"),
        re.compile(r"^(?P<site>[A-Za-z0-9]+)[_-](?P<fileid>\d+)_func_preprop\.nii\.gz$"),
        re.compile(r"^(?P<site>[A-Za-z0-9]+)[_-](?P<fileid>\d+)_bold(?:_preproc)?\.nii\.gz$"),
        re.compile(r"^(?P<fileid>\d+)_func_preproc\.nii\.gz$"),
        re.compile(r"^(?P<subid>\d+)\.nii\.gz$"),
        re.compile(r"^sub-(?P<subid>\d+)_task-rest_bold\.nii\.gz$"),
        re.compile(r"^(?P<fileid>\d+)_bold\.nii\.gz$"),
        re.compile(r"^(?P<site>[A-Za-z0-9]+)[_-](?P<subid>\d+)\.nii\.gz$"),
    ]
    by_file_site = {}
    by_sub_site = {}
    by_file = {}
    by_sub = {}
    for fp in all_niftis:
        bn = os.path.basename(fp)
        m = None
        for r in patts:
            mm = r.match(bn)
            if mm:
                m = mm
                break
        if not m:
            continue
        gd = m.groupdict()
        fileid = gd.get("fileid", "")
        subid = gd.get("subid", "")
        sitev = gd.get("site", "")
        if fileid:
            by_file[fileid] = by_file.get(fileid, fp)
            if sitev:
                for sv in [sitev, sitev.lower(), sitev.upper(), sitev.title(), sitev.replace(" ", "_")]:
                    by_file_site[(fileid, sv)] = fp
        if subid:
            by_sub[subid] = by_sub.get(subid, fp)
            if sitev:
                for sv in [sitev, sitev.lower(), sitev.upper(), sitev.title(), sitev.replace(" ", "_")]:
                    by_sub_site[(subid, sv)] = fp
    def build_index():
        out = []
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
            else:
                for sv in site_variants:
                    if key_col == "FILE_ID":
                        p = by_file_site.get((sid, sv), "")
                        if p:
                            candidates.append(p)
                    elif key_col == "SUB_ID":
                        p = by_sub_site.get((sid, sv), "")
                        if p:
                            candidates.append(p)
                if key_col == "FILE_ID" and sid in by_file:
                    candidates.append(by_file[sid])
                if key_col == "SUB_ID" and sid in by_sub:
                    candidates.append(by_sub[sid])
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
            out.append({"sid": sid, "path": nii})
        return out, debug_missing

    stage = (args.pipeline_stage or "all").lower()
    idx = None
    if stage in ["discover", "timeseries", "features", "sequence", "normative", "models", "all"]:
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    d = json.load(f)
                    if isinstance(d, dict) and "found" in d:
                        idx = d.get("found", [])
                    elif isinstance(d, list):
                        idx = d
            except Exception:
                idx = None
        if idx is None:
            idx, debug_missing = build_index()
            payload = {"found": idx}
            if args.debug_paths and debug_missing:
                payload["debug_no_matches"] = debug_missing[:3]
            with open(index_path, "w") as f:
                json.dump(payload, f)
        if stage == "discover":
            print(json.dumps({"indexed": len(idx)}))
            return
        if stage in ["timeseries", "all"]:
            n_saved = 0
            for it in idx:
                sid = str(it["sid"])
                nii = it["path"]
                outp = os.path.join(roi_dir, f"{sid}.npy")
                if os.path.exists(outp):
                    continue
                use_nii = materialize_nii(nii) if args.copy_to_local else nii
                use_atlas = materialize_nii(atlas_path) if args.copy_to_local else atlas_path
                ts = extract_roi_timeseries(use_nii, use_atlas, atlas_type="labels", tr=2.0, high_pass=0.01, low_pass=0.1)
                ts = regress_confounds(ts, confounds=None, tr=2.0, high_pass=0.01, low_pass=0.1)
                np.save(outp, ts)
                n_saved += 1
            if stage == "timeseries":
                print(json.dumps({"timeseries_saved": n_saved}))
                return
        if stage in ["features", "all"]:
            n_saved = 0
            for it in idx:
                sid = str(it["sid"])
                tsp = os.path.join(roi_dir, f"{sid}.npy")
                outp = os.path.join(features_dir, f"{sid}.npy")
                if os.path.exists(outp):
                    continue
                if not os.path.exists(tsp):
                    continue
                ts = np.load(tsp)
                vec, _ = build_connectome_feature_vector(ts, window_size=30, step=15, thr=0.4, networks=networks, compute_partial=False)
                np.save(outp, vec)
                n_saved += 1
            if stage == "features":
                print(json.dumps({"features_saved": n_saved}))
                return
        if stage in ["sequence", "all"]:
            n_saved = 0
            for it in idx:
                sid = str(it["sid"])
                tsp = os.path.join(roi_dir, f"{sid}.npy")
                outp = os.path.join(sequence_dir, f"{sid}.npz")
                if os.path.exists(outp):
                    continue
                if not os.path.exists(tsp):
                    continue
                ts = np.load(tsp)
                seq = dynamic_fc_sequence(ts, window_size=30, step=15)
                np.savez(outp, fc_vectors=seq["fc_vectors"], window_indices=seq["window_indices"]) 
                n_saved += 1
            if stage == "sequence":
                print(json.dumps({"sequence_saved": n_saved}))
                return
        if stage in ["normative", "models", "all"]:
            feats = []
            found_ids = []
            for it in idx:
                sid = str(it["sid"])
                fp = os.path.join(features_dir, f"{sid}.npy")
                if os.path.exists(fp):
                    vec = np.load(fp)
                    feats.append(vec)
                    found_ids.append(sid)
            if not feats:
                raise FileNotFoundError("No features found. Run with stage 'timeseries' and 'features' first.")
            X = np.stack(feats, axis=0)
            key_col2 = "FILE_ID" if "FILE_ID" in pheno.columns else ("SUB_ID" if "SUB_ID" in pheno.columns else key_col)
            ph_sub = pheno[pheno[key_col2].astype(str).isin(found_ids)].copy()
            y_col = pick_col(ph_sub, ["DX_GROUP", "DX", "diagnosis", "DX_GROUP_BIN", "Label"])
            y_raw = ph_sub[y_col].values
            if np.issubdtype(y_raw.dtype, np.number):
                y = y_raw.astype(int)
            else:
                ya = pd.Series(y_raw).astype(str).str.upper().tolist()
                y = np.array([1 if v == "ASD" else (2 if v == "TD" else 0) for v in ya], dtype=int)
            td_mask = y == 2
            age_col = pick_col(ph_sub, ["AGE_AT_SCAN", "AGE", "AgeAtScan"]) 
            sex_col = pick_col(ph_sub, ["SEX", "sex", "Gender"]) 
            covars = ph_sub[[age_col, sex_col]].values.astype(float)
            if (args.norm_mode or "mvn").lower() == "z_only":
                td_X = X[td_mask]
                m = td_X.mean(axis=0)
                v = td_X.var(axis=0)
                s = np.sqrt(np.maximum(v, 1e-8))
                n_subj = X.shape[0]
                n_feat = X.shape[1]
                mm_path = os.path.join(out_dir, "X_dev.dat")
                mm = np.memmap(mm_path, dtype="float32", mode="w+", shape=(n_subj, n_feat))
                for i in range(n_subj):
                    z = (X[i] - m) / s
                    mm[i, :] = z.astype(np.float32)
                    np.save(os.path.join(z_dir, f"{found_ids[i]}.npy"), z)
                del mm
                meta = {"n_subjects": int(n_subj), "n_features": int(n_feat)}
                with open(os.path.join(out_dir, "X_dev_meta.json"), "w") as f:
                    json.dump(meta, f)
                groups = ph_sub[site_col].values if site_col else np.array([""]*ph_sub.shape[0])
                np.save(os.path.join(out_dir, "y.npy"), y)
                np.save(os.path.join(out_dir, "groups.npy"), groups)
                with open(os.path.join(out_dir, "ids.json"), "w") as f:
                    json.dump(found_ids, f)
                if stage == "normative":
                    print(json.dumps({"normative_saved": True, "n_subjects": int(n_subj), "n_features": int(n_feat), "mode": "z_only"}))
                    return
                try:
                    X_dev = np.load(os.path.join(out_dir, "X_dev.npy"))
                except Exception:
                    meta = json.load(open(os.path.join(out_dir, "X_dev_meta.json")))
                    n_subj = int(meta.get("n_subjects", n_subj))
                    n_feat = int(meta.get("n_features", n_feat))
                    X_dev = np.memmap(os.path.join(out_dir, "X_dev.dat"), dtype="float32", mode="r", shape=(n_subj, n_feat))
                metrics = evaluate_models(X_dev, y, cv_strategy="site_stratified", groups=groups, cv_splits=5)
            else:
                dev = personalized_deviation_maps(X[td_mask], X, covars=covars)
                X_dev = dev["feature_z"]
                groups = ph_sub[site_col].values if site_col else np.array([""]*ph_sub.shape[0])
                np.save(os.path.join(out_dir, "X_dev.npy"), X_dev)
                np.save(os.path.join(out_dir, "y.npy"), y)
                np.save(os.path.join(out_dir, "groups.npy"), groups)
                with open(os.path.join(out_dir, "ids.json"), "w") as f:
                    json.dump(found_ids, f)
                if stage == "normative":
                    print(json.dumps({"normative_saved": True, "n_subjects": int(X_dev.shape[0]), "n_features": int(X_dev.shape[1]), "mode": "mvn"}))
                    return
                metrics = evaluate_models(X_dev, y, cv_strategy="site_stratified", groups=groups, cv_splits=5)
            with open(os.path.join(out_dir, "bna_results.json"), "w") as f:
                json.dump({"models": metrics}, f)
            print(json.dumps({"models": metrics}))
            return
    raise RuntimeError("Invalid stage")


if __name__ == "__main__":
    main()
