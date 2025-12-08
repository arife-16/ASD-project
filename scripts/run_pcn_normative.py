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
try:
    from pcntoolkit import BLR, NormativeModel, NormData
    try:
        from pcntoolkit import GPR
    except Exception:
        GPR = None
except Exception:
    BLR = None
    NormativeModel = None
    NormData = None
    GPR = None


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
    ap.add_argument("--compare_blr_gpr", action="store_true")
    ap.add_argument("--feature_index", type=int, default=0)
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
    if not args.compare_blr_gpr:
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
        return
    os.makedirs(args.out_dir, exist_ok=True)
    fi = max(0, min(args.feature_index, X.shape[1] - 1))
    Y_all = X[:, fi:fi+1]
    try:
        train_data = NormData.from_ndarrays("train", X=covars[td_mask], Y=Y_all[td_mask])
        test_data = NormData.from_ndarrays("test", X=covars, Y=Y_all)
    except Exception:
        return
    blr_model = None
    gpr_model = None
    blr_test = None
    gpr_test = None
    if BLR is not None and NormativeModel is not None:
        try:
            blr_model = NormativeModel(BLR(), inscaler="standardize", outscaler="standardize", save_dir=os.path.join(args.out_dir, "blr"))
            blr_test = blr_model.fit_predict(train_data, test_data)
        except Exception:
            blr_model = None
            blr_test = None
    if GPR is not None and NormativeModel is not None:
        try:
            gpr_model = NormativeModel(GPR(), inscaler="standardize", outscaler="standardize", save_dir=os.path.join(args.out_dir, "gpr"))
            gpr_test = gpr_model.fit_predict(train_data, test_data)
        except Exception:
            gpr_model = None
            gpr_test = None
    x_axis = np.arange(Y_all.shape[0])
    y_true = Y_all[:, 0]
    if blr_test is not None:
        blr_yhat = blr_test["Yhat"].values[:, 0]
        blr_z = blr_test["Z"].values[:, 0]
        save_histogram(blr_z, "BLR z-score histogram", os.path.join(args.out_dir, "blr_z_hist.png"))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,4))
        plt.scatter(x_axis, y_true, s=10, label="true")
        plt.scatter(x_axis, blr_yhat, s=10, label="blr")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "blr_pred_scatter.png"))
        plt.close()
    if gpr_test is not None:
        gpr_yhat = gpr_test["Yhat"].values[:, 0]
        gpr_z = gpr_test["Z"].values[:, 0]
        save_histogram(gpr_z, "GPR z-score histogram", os.path.join(args.out_dir, "gpr_z_hist.png"))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,4))
        plt.scatter(x_axis, y_true, s=10, label="true")
        plt.scatter(x_axis, gpr_yhat, s=10, label="gpr")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "gpr_pred_scatter.png"))
        plt.close()
    out = {}
    if blr_test is not None:
        out["blr_mean_abs_z"] = float(np.mean(np.abs(blr_test["Z"].values[:, 0])))
    if gpr_test is not None:
        out["gpr_mean_abs_z"] = float(np.mean(np.abs(gpr_test["Z"].values[:, 0])))
    out["feature_index"] = int(fi)
    with open(os.path.join(args.out_dir, "blr_gpr_summary.json"), "w") as f:
        json.dump(out, f)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
