import os
import sys
import os
import json
import glob
import numpy as np
import pandas as pd
from nilearn import datasets
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asd_pipeline.atlas import extract_roi_timeseries
from asd_pipeline.features import build_feature_vector_with_states
from asd_pipeline.harmonize import combat_harmonize
from asd_pipeline.normative import personalized_deviation_maps
from asd_pipeline.model import evaluate_models


def find_example_nifti(root: str) -> str:
    pats = ["*.nii", "*.nii.gz"]
    for p in pats:
        files = glob.glob(os.path.join(root, p))
        if files:
            return files[0]
    raise FileNotFoundError("No NIfTI found")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(root)
    nifti_path = find_example_nifti(proj)
    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_path = ho["maps"]
    out_dir = os.path.join(proj, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    roi_dir = os.path.join(out_dir, "_roi_ts")
    os.makedirs(roi_dir, exist_ok=True)
    n_subjects = 40
    subject_ids = [f"EX_{i:03d}" for i in range(n_subjects)]
    sites = ["Olin", "Caltech", "NYU"]
    labels = []
    ages = []
    sexes = []
    for i, sid in enumerate(subject_ids):
        ts = extract_roi_timeseries(nifti_path, atlas_path, atlas_type="labels", tr=2.0, high_pass=0.01, low_pass=0.1)
        noise = 0.05 * np.random.standard_normal(ts.shape)
        ts = ts + noise
        np.save(os.path.join(roi_dir, f"{sid}.npy"), ts)
        labels.append(2 if i % 2 == 0 else 1)
        ages.append(float(np.random.uniform(8, 30)))
        sexes.append(int(np.random.randint(0, 2)))
    pheno = pd.DataFrame({
        "SUB_ID": subject_ids,
        "SITE": [sites[i % len(sites)] for i in range(n_subjects)],
        "DX_GROUP": labels,
        "AGE_AT_SCAN": ages,
        "SEX": sexes,
    })
    pheno.to_csv(os.path.join(out_dir, "phenotype.csv"), index=False)
    feats = []
    for sid in subject_ids:
        ts = np.load(os.path.join(roi_dir, f"{sid}.npy"))
        vec, _ = build_feature_vector_with_states(ts, tr=2.0, window_size=50, step=10, n_states=5)
        feats.append(vec)
    X = np.stack(feats, axis=0)
    covars = pheno[["SITE", "AGE_AT_SCAN", "SEX"]].copy()
    try:
        X_h, _ = combat_harmonize(X, covars[["SITE", "AGE_AT_SCAN", "SEX"]], site_col="SITE", continuous_covars=["AGE_AT_SCAN"], categorical_covars=["SEX"])
    except Exception:
        X_h = X
    y = pheno["DX_GROUP"].values.astype(int)
    td_mask = y == 2
    covars_arr = covars[["AGE_AT_SCAN", "SEX"]].values.astype(float)
    dev = personalized_deviation_maps(X_h[td_mask], X_h, covars=covars_arr)
    X_dev = dev["feature_z"]
    groups = pheno["SITE"].values
    metrics = evaluate_models(X_dev, y, cv_strategy="site_stratified", groups=groups, cv_splits=5)
    out = {"models": metrics}
    with open(os.path.join(out_dir, "example_results.json"), "w") as f:
        json.dump(out, f)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
