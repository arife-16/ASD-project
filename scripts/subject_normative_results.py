import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asd_pipeline.features import build_feature_vector_with_states
from asd_pipeline.normative import personalized_deviation_maps


def roi_labels_fallback(n_rois: int):
    return [f"ROI_{i}" for i in range(n_rois)]


def roi_labels_bna():
    import os
    default = os.path.expanduser("~/Downloads/Autism-Connectome-Analysis-master/atlas/BNA_subregions.xlsx")
    xlsx_path = os.environ.get("BNA_XLSX", default)
    try:
        from asd_pipeline.atlas_labels import load_bna_labels
        names, _ = load_bna_labels(xlsx_path)
        return names
    except Exception:
        return []


def triu_pairs(n):
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def main(subject_id: str = None):
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(proj, "experiment_results")
    pheno_path = os.path.join(out_dir, "phenotype.csv")
    roi_dir = os.path.join(out_dir, "_roi_ts")
    pheno = pd.read_csv(pheno_path)
    if subject_id is None:
        subject_id = pheno["SUB_ID"].astype(str).iloc[0]
    sids = pheno["SUB_ID"].astype(str).tolist()
    idx_map = {sid: i for i, sid in enumerate(sids)}
    if subject_id not in idx_map:
        raise RuntimeError("Subject not found in phenotype.csv")
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
    y = pheno["DX_GROUP"].values.astype(int)
    td_mask = y == 2
    covars = pheno[["AGE_AT_SCAN", "SEX"]].values.astype(float)
    dev = personalized_deviation_maps(X[td_mask], X, covars=covars)
    Z = dev["feature_z"]
    M = dev["mahalanobis"]
    si = idx_map[subject_id]
    z_subj = Z[si]
    idxs = idxs_list[si]
    # component summaries
    comps = ["fc", "dyn_mean", "dyn_std", "alff", "state_occ", "transitions", "dwell_mean", "dwell_std", "entropy", "asymmetry"]
    summary = {}
    for c in comps:
        a, b = idxs[c]
        vals = z_subj[a:b]
        summary[c] = {
            "mean_abs_z": float(np.mean(np.abs(vals))) if len(vals) > 0 else 0.0,
            "max_abs_z": float(np.max(np.abs(vals))) if len(vals) > 0 else 0.0,
        }
    # top edges in FC
    a_fc, b_fc = idxs["fc"]
    z_fc = z_subj[a_fc:b_fc]
    pairs = triu_pairs(n_rois)
    labels = roi_labels_bna()
    if not labels:
        labels = roi_labels_fallback(n_rois)
    order = np.argsort(np.abs(z_fc))[::-1][:20]
    top_edges = []
    for k in order:
        i, j = pairs[k]
        top_edges.append({
            "roi_i": int(i),
            "roi_j": int(j),
            "name_i": labels[i] if i < len(labels) else f"ROI_{i}",
            "name_j": labels[j] if j < len(labels) else f"ROI_{j}",
            "z": float(z_fc[k]),
        })
    out = {
        "subject_id": subject_id,
        "mahalanobis": float(M[si]),
        "overall_mean_abs_z": float(np.mean(np.abs(z_subj))),
        "component_summary": summary,
        "top_fc_edges": top_edges,
    }
    out_path = os.path.join(out_dir, f"normative_subject_{subject_id}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(json.dumps(out))


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    main(sid)
