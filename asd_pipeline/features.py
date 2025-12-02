from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans


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
    feat = np.concatenate(parts, axis=0)
    return feat, idxs
