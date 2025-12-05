import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asd_pipeline.preprocess import regress_confounds
from asd_pipeline.features import build_subject_feature_bundle


def save_array(path: str, arr: np.ndarray, fmt: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "npy":
        np.save(path, arr)
    elif fmt == "csv":
        np.savetxt(path, arr, delimiter=",")
    elif fmt == "json":
        with open(path, "w") as f:
            json.dump(arr.tolist(), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotype", required=True)
    ap.add_argument("--ts_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--window_size", type=int, default=50)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--n_states", type=int, default=5)
    ap.add_argument("--format", type=str, default="npy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pheno = pd.read_csv(args.phenotype)
    sids = pheno["SUB_ID"].astype(str).tolist()
    for sid in sids:
        sub_dir = os.path.join(args.out_dir, "subjects", sid)
        os.makedirs(sub_dir, exist_ok=True)
        marker = os.path.join(sub_dir, "_SUCCESS")
        if os.path.exists(marker) and not args.overwrite:
            continue
        ts_path = os.path.join(args.ts_dir, f"{sid}.npy")
        if not os.path.exists(ts_path):
            continue
        ts = np.load(ts_path)
        bundle = build_subject_feature_bundle(ts, tr=args.tr, window_size=args.window_size, step=args.step, n_states=args.n_states)
        save_array(os.path.join(sub_dir, "static_fc_matrix." + args.format), bundle["static_fc_matrix"], args.format)
        save_array(os.path.join(sub_dir, "static_fc_vector." + args.format), bundle["static_fc_vector"], args.format)
        save_array(os.path.join(sub_dir, "dynamic_fc_matrices." + args.format), bundle["dynamic_fc_matrices"], args.format)
        save_array(os.path.join(sub_dir, "dynamic_fc_vectors." + args.format), bundle["dynamic_fc_vectors"], args.format)
        save_array(os.path.join(sub_dir, "window_indices." + args.format), bundle["window_indices"], args.format)
        save_array(os.path.join(sub_dir, "dynamic_pc_matrices." + args.format), bundle["dynamic_pc_matrices"], args.format)
        save_array(os.path.join(sub_dir, "dynamic_pc_vectors." + args.format), bundle["dynamic_pc_vectors"], args.format)
        save_array(os.path.join(sub_dir, "dynamic_node_strength." + args.format), bundle["dynamic_node_strength"], args.format)
        save_array(os.path.join(sub_dir, "dynamic_clustering." + args.format), bundle["dynamic_clustering"], args.format)
        save_array(os.path.join(sub_dir, "dynamic_intra_network." + args.format), bundle["dynamic_intra_network"], args.format)
        save_array(os.path.join(sub_dir, "dynamic_inter_network." + args.format), bundle["dynamic_inter_network"], args.format)
        save_array(os.path.join(sub_dir, "alff." + args.format), bundle["alff"], args.format)
        save_array(os.path.join(sub_dir, "state_labels." + args.format), np.asarray(bundle["state_labels"]) if isinstance(bundle["state_labels"], list) else bundle["state_labels"], args.format)
        save_array(os.path.join(sub_dir, "state_occ." + args.format), bundle["state_occ"], args.format)
        save_array(os.path.join(sub_dir, "transitions." + args.format), bundle["transitions"], args.format)
        save_array(os.path.join(sub_dir, "dwell_mean." + args.format), bundle["dwell_mean"], args.format)
        save_array(os.path.join(sub_dir, "dwell_std." + args.format), bundle["dwell_std"], args.format)
        save_array(os.path.join(sub_dir, "entropy." + args.format), bundle["entropy"], args.format)
        save_array(os.path.join(sub_dir, "asymmetry." + args.format), bundle["asymmetry"], args.format)
        meta = {
            "subject_id": sid,
            "n_rois": int(ts.shape[0]),
            "n_time": int(ts.shape[1]),
            "n_windows": int(bundle["dynamic_fc_vectors"].shape[0]),
            "window_size": args.window_size,
            "step": args.step,
            "format": args.format,
        }
        with open(os.path.join(sub_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        with open(marker, "w") as f:
            f.write("ok")


if __name__ == "__main__":
    main()
