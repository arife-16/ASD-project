import os
import sys
import json
import numpy as np
import pandas as pd
from nilearn import datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asd_pipeline.pcn_adapter import setup_pcn, fit_normative_pcn, predict_normative_pcn
from asd_pipeline.features import build_feature_vector_with_states
from asd_pipeline.plots import save_histogram, save_bar, save_waterfall


def triu_pairs(n):
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def roi_labels_harvard_oxford():
    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    labels = ho["labels"]
    return [lbl for lbl in labels if lbl and lbl.lower() != "background"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcn_path", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="experiment_results/pcn_normative")
    ap.add_argument("--subject_id", type=str, default="")
    args = ap.parse_args()

    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(proj, "experiment_results")
    pheno_path = os.path.join(exp_dir, "phenotype.csv")
    roi_dir = os.path.join(exp_dir, "_roi_ts")
    pheno = pd.read_csv(pheno_path)
    sids = pheno["SUB_ID"].astype(str).tolist()
    feats = []
    idxs_list = []
    n_rois = None
    for sid in sids:
        ts = np.load(os.path.join(roi_dir, f"{sid}.npy"))
        n_rois = ts.shape[0]
        vec, idxs = build_feature_vector_with_states(ts, tr=2.0, window_size=50, step=10, n_states=5)
        feats.append(vec)
        idxs_list.append(idxs)
    X = np.stack(feats, axis=0)
    covars = pheno[["AGE_AT_SCAN", "SEX"]].values.astype(float)
    y = pheno["DX_GROUP"].values.astype(int)
    td_mask = y == 2
    os.environ["PCN_PATH"] = args.pcn_path
    setup_pcn(args.pcn_path)
    model = fit_normative_pcn(X[td_mask], covars[td_mask], out_dir=args.out_dir)
    mu, sigma, z = predict_normative_pcn(model, X, covars)
    if z is None:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "z_all.npy"), z)
    sid = args.subject_id or sids[0]
    idx_map = {sid: i for i, sid in enumerate(sids)}
    si = idx_map[sid]
    z_subj = z[si]
    idxs = idxs_list[si]
    save_histogram(z_subj, f"Subject {sid} z-score histogram", os.path.join(args.out_dir, f"{sid}_z_hist.png"))
    a_fc, b_fc = idxs["fc"]
    z_fc = z_subj[a_fc:b_fc]
    labels = roi_labels_harvard_oxford()
    pairs = triu_pairs(n_rois)
    names = [f"{labels[i] if i < len(labels) else i}-{labels[j] if j < len(labels) else j}" for i, j in pairs]
    save_bar(names, z_fc, f"Subject {sid} top FC deviations", os.path.join(args.out_dir, f"{sid}_top_fc.png"), top_k=20)
    save_waterfall(z_subj, f"Subject {sid} z-score waterfall", os.path.join(args.out_dir, f"{sid}_waterfall.png"))
    out_json = {"subject_id": sid, "mean_abs_z": float(np.mean(np.abs(z_subj)))}
    with open(os.path.join(args.out_dir, f"{sid}_summary.json"), "w") as f:
        json.dump(out_json, f)
    print(json.dumps(out_json))


if __name__ == "__main__":
    main()
