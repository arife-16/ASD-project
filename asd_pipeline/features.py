from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from typing import Any
from asd_pipeline.connectome import precision_partial_corr, threshold_adjacency, node_strength, local_clustering, network_connectivity


def correlation_matrix(ts: np.ndarray) -> np.ndarray:
    ts = ts - ts.mean(axis=1, keepdims=True)
    ts = ts / (ts.std(axis=1, keepdims=True) + 1e-8)
    return np.corrcoef(ts)


def vectorize_upper_triangle(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def sliding_windows(ts: np.ndarray, window_size: int, step: int) -> List[np.ndarray]:
    n_time = ts.shape[1]
    windows = []
    start = 0
    while start + window_size <= n_time:
        windows.append(ts[:, start : start + window_size])
        start += step
    return windows


def dynamic_fc_features(ts: np.ndarray, window_size: int, step: int) -> Dict[str, np.ndarray]:
    wins = sliding_windows(ts, window_size, step)
    if len(wins) == 0:
        mat = correlation_matrix(ts)
        vec = vectorize_upper_triangle(mat)
        return {"static_fc": vec, "dyn_fc_mean": vec, "dyn_fc_std": np.zeros_like(vec)}
    vecs = []
    for w in wins:
        mat = correlation_matrix(w)
        vecs.append(vectorize_upper_triangle(mat))
    vecs = np.stack(vecs, axis=0)
    return {"static_fc": vecs.mean(axis=0), "dyn_fc_mean": vecs.mean(axis=0), "dyn_fc_std": vecs.std(axis=0, ddof=1)}


def alff(ts: np.ndarray, tr: float) -> np.ndarray:
    n_time = ts.shape[1]
    freqs = np.fft.rfftfreq(n_time, d=tr)
    fft = np.fft.rfft(ts, axis=1)
    mask = (freqs >= 0.01) & (freqs <= 0.1)
    power = np.abs(fft) ** 2
    return power[:, mask].mean(axis=1)


def build_feature_vector(ts: np.ndarray, tr: float, window_size: int, step: int) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    dyn = dynamic_fc_features(ts, window_size, step)
    fc = dyn["static_fc"]
    alff_vals = alff(ts, tr)
    parts = [fc, dyn["dyn_fc_mean"], dyn["dyn_fc_std"], alff_vals]
    sizes = [len(fc), len(dyn["dyn_fc_mean"]), len(dyn["dyn_fc_std"]), len(alff_vals)]
    idxs: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    keys = ["fc", "dyn_mean", "dyn_std", "alff"]
    for k, s in zip(keys, sizes):
        idxs[k] = (cursor, cursor + s)
        cursor += s
    feat = np.concatenate(parts, axis=0)
    return feat, idxs


def dynamic_state_features(ts: np.ndarray, window_size: int, step: int, n_states: int = 5, random_state: int = 42) -> Dict[str, np.ndarray]:
    wins = sliding_windows(ts, window_size, step)
    if len(wins) == 0:
        return {"state_occ": np.zeros(n_states), "transitions": np.zeros(n_states * n_states), "dwell_mean": np.zeros(n_states)}
    vecs = []
    for w in wins:
        mat = correlation_matrix(w)
        vecs.append(vectorize_upper_triangle(mat))
    V = np.stack(vecs, axis=0)
    km = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    labels = km.fit_predict(V)
    occ = np.bincount(labels, minlength=n_states).astype(float)
    occ = occ / occ.sum() if occ.sum() > 0 else occ
    trans = np.zeros((n_states, n_states), dtype=float)
    if len(labels) > 1:
        for a, b in zip(labels[:-1], labels[1:]):
            trans[a, b] += 1.0
    trans_row = trans.sum(axis=1, keepdims=True)
    trans_prob = np.divide(trans, trans_row, out=np.zeros_like(trans), where=trans_row > 0)
    dwell = np.zeros(n_states, dtype=float)
    cur = labels[0]
    length = 1
    runs = []
    for l in labels[1:]:
        if l == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = l
            length = 1
    runs.append((cur, length))
    for s in range(n_states):
        lens = [r[1] for r in runs if r[0] == s]
        dwell[s] = float(np.mean(lens)) if len(lens) > 0 else 0.0
    p = occ
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    asym = 0.0
    for i in range(n_states):
        for j in range(n_states):
            asym += abs(trans_prob[i, j] - trans_prob[j, i])
    runs_by_state = {}
    cur = labels[0]
    length = 1
    runs = []
    for l in labels[1:]:
        if l == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = l
            length = 1
    runs.append((cur, length))
    dwell_std = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        lens = [r[1] for r in runs if r[0] == s]
        dwell_std[s] = float(np.std(lens)) if len(lens) > 0 else 0.0
    return {"state_occ": occ, "transitions": trans_prob.flatten(), "dwell_mean": dwell, "dwell_std": dwell_std, "entropy": np.array([entropy]), "asymmetry": np.array([asym])}


def build_feature_vector_with_states(ts: np.ndarray, tr: float, window_size: int, step: int, n_states: int = 5) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    base_feat, base_idxs = build_feature_vector(ts, tr, window_size, step)
    state = dynamic_state_features(ts, window_size, step, n_states=n_states)
    parts = [base_feat, state["state_occ"], state["transitions"], state["dwell_mean"], state["dwell_std"], state["entropy"], state["asymmetry"]]
    sizes = [len(base_feat), len(state["state_occ"]), len(state["transitions"]), len(state["dwell_mean"]), len(state["dwell_std"]), 1, 1]
    idxs: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    keys = ["base", "state_occ", "transitions", "dwell_mean", "dwell_std", "entropy", "asymmetry"]
    for k, s in zip(keys, sizes):
        idxs[k] = (cursor, cursor + s)
        cursor += s
    # include base sub-indices
    idxs["fc"] = base_idxs["fc"]
    idxs["dyn_mean"] = base_idxs["dyn_mean"]
    idxs["dyn_std"] = base_idxs["dyn_std"]
    idxs["alff"] = base_idxs["alff"]
    feat = np.concatenate(parts, axis=0)
    return feat, idxs


def dynamic_fc_sequence(ts: np.ndarray, window_size: int, step: int) -> Dict[str, Any]:
    wins = sliding_windows(ts, window_size, step)
    mats = []
    vecs = []
    idxs = []
    start = 0
    n_time = ts.shape[1]
    while start + window_size <= n_time:
        w = ts[:, start : start + window_size]
        m = correlation_matrix(w)
        mats.append(m)
        vecs.append(vectorize_upper_triangle(m))
        idxs.append((start, start + window_size))
        start += step
    if len(mats) == 0:
        m = correlation_matrix(ts)
        mats = [m]
        vecs = [vectorize_upper_triangle(m)]
        idxs = [(0, ts.shape[1])]
    return {"fc_matrices": np.stack(mats, axis=0), "fc_vectors": np.stack(vecs, axis=0), "window_indices": np.array(idxs, dtype=int)}


def dynamic_state_features_with_labels(ts: np.ndarray, window_size: int, step: int, n_states: int = 5, random_state: int = 42) -> Dict[str, Any]:
    wins = sliding_windows(ts, window_size, step)
    vecs = []
    for w in wins:
        mat = correlation_matrix(w)
        vecs.append(vectorize_upper_triangle(mat))
    if len(vecs) == 0:
        return {"labels": np.zeros(1, dtype=int), "state_occ": np.zeros(n_states), "transitions": np.zeros(n_states * n_states), "dwell_mean": np.zeros(n_states), "dwell_std": np.zeros(n_states), "entropy": np.array([0.0]), "asymmetry": np.array([0.0])}
    V = np.stack(vecs, axis=0)
    km = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    labels = km.fit_predict(V)
    occ = np.bincount(labels, minlength=n_states).astype(float)
    occ = occ / occ.sum() if occ.sum() > 0 else occ
    trans = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        trans[a, b] += 1.0
    trans_row = trans.sum(axis=1, keepdims=True)
    trans_prob = np.divide(trans, trans_row, out=np.zeros_like(trans), where=trans_row > 0)
    dwell = np.zeros(n_states, dtype=float)
    cur = labels[0]
    length = 1
    runs = []
    for l in labels[1:]:
        if l == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = l
            length = 1
    runs.append((cur, length))
    dwell_std = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        lens = [r[1] for r in runs if r[0] == s]
        dwell[s] = float(np.mean(lens)) if len(lens) > 0 else 0.0
        dwell_std[s] = float(np.std(lens)) if len(lens) > 0 else 0.0
    p = occ
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    asym = 0.0
    for i in range(n_states):
        for j in range(n_states):
            asym += abs(trans_prob[i, j] - trans_prob[j, i])
    return {"labels": labels, "state_occ": occ, "transitions": trans_prob.flatten(), "dwell_mean": dwell, "dwell_std": dwell_std, "entropy": np.array([entropy]), "asymmetry": np.array([asym])}


def build_subject_feature_bundle(ts: np.ndarray, tr: float, window_size: int, step: int, n_states: int = 5) -> Dict[str, Any]:
    full_mat = correlation_matrix(ts)
    full_vec = vectorize_upper_triangle(full_mat)
    seq = dynamic_fc_sequence(ts, window_size, step)
    start = 0
    n_time = ts.shape[1]
    pc_mats = []
    pc_vecs = []
    str_seq = []
    clu_seq = []
    intra_seq = []
    inter_seq = []
    while start + window_size <= n_time:
        w = ts[:, start : start + window_size]
        m = correlation_matrix(w)
        p = precision_partial_corr(w)
        a = threshold_adjacency(m, thr=0.3)
        s = node_strength(m)
        c = local_clustering(a)
        pc_mats.append(p)
        pc_vecs.append(vectorize_upper_triangle(p))
        str_seq.append(s)
        clu_seq.append(c)
        intra, inter = network_connectivity(m, networks=None)
        if intra.size > 0:
            intra_seq.append(intra)
        if inter.size > 0:
            inter_seq.append(inter.flatten())
        start += step
    if len(pc_mats) == 0:
        m = correlation_matrix(ts)
        p = precision_partial_corr(ts)
        a = threshold_adjacency(m, thr=0.3)
        s = node_strength(m)
        c = local_clustering(a)
        pc_mats = [p]
        pc_vecs = [vectorize_upper_triangle(p)]
        str_seq = [s]
        clu_seq = [c]
        intra_seq = []
        inter_seq = []
    alff_vals = alff(ts, tr)
    state = dynamic_state_features_with_labels(ts, window_size, step, n_states=n_states)
    return {
        "static_fc_matrix": full_mat,
        "static_fc_vector": full_vec,
        "dynamic_fc_matrices": seq["fc_matrices"],
        "dynamic_fc_vectors": seq["fc_vectors"],
        "window_indices": seq["window_indices"],
        "dynamic_pc_matrices": np.stack(pc_mats, axis=0),
        "dynamic_pc_vectors": np.stack(pc_vecs, axis=0),
        "dynamic_node_strength": np.stack(str_seq, axis=0),
        "dynamic_clustering": np.stack(clu_seq, axis=0),
        "dynamic_intra_network": np.stack(intra_seq, axis=0) if len(intra_seq) > 0 else np.empty((0,)),
        "dynamic_inter_network": np.stack(inter_seq, axis=0) if len(inter_seq) > 0 else np.empty((0,)),
        "alff": alff_vals,
        "state_labels": state["labels"],
        "state_occ": state["state_occ"],
        "transitions": state["transitions"],
        "dwell_mean": state["dwell_mean"],
        "dwell_std": state["dwell_std"],
        "entropy": state["entropy"],
        "asymmetry": state["asymmetry"],
    }
