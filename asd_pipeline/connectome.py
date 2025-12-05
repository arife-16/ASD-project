from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.covariance import LedoitWolf


def fc_matrix(ts: np.ndarray) -> np.ndarray:
    ts = ts - ts.mean(axis=1, keepdims=True)
    ts = ts / (ts.std(axis=1, keepdims=True) + 1e-8)
    return np.corrcoef(ts)


def precision_partial_corr(ts: np.ndarray) -> np.ndarray:
    lw = LedoitWolf().fit(ts.T)
    P = lw.precision_
    d = np.sqrt(np.diag(P))
    denom = np.outer(d, d)
    pc = -P / denom
    np.fill_diagonal(pc, 1.0)
    return pc


def threshold_adjacency(mat: np.ndarray, thr: float = 0.3) -> np.ndarray:
    return (np.abs(mat) >= thr).astype(np.int8)


def node_strength(mat: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(mat), axis=1)


def local_clustering(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    cc = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        k = len(nbrs)
        if k < 2:
            cc[i] = 0.0
            continue
        sub = adj[np.ix_(nbrs, nbrs)]
        t = (np.sum(sub) - np.trace(sub)) / 2.0
        cc[i] = (t) / (k * (k - 1) / 2.0)
    return cc


def network_connectivity(mat: np.ndarray, networks: Optional[list]) -> Tuple[np.ndarray, np.ndarray]:
    if networks is None:
        return np.array([]), np.array([])
    uniq = sorted(list(set(networks)))
    net_idx = {u: [i for i, n in enumerate(networks) if n == u] for u in uniq}
    intra = np.zeros(len(uniq), dtype=float)
    inter = np.zeros((len(uniq), len(uniq)), dtype=float)
    for a, ua in enumerate(uniq):
        idx_a = net_idx[ua]
        if len(idx_a) > 1:
            sub = mat[np.ix_(idx_a, idx_a)]
            intra[a] = float(np.mean(sub[np.triu_indices_from(sub, k=1)]))
        else:
            intra[a] = 0.0
        for b, ub in enumerate(uniq):
            idx_b = net_idx[ub]
            if a == b:
                continue
            inter[a, b] = float(np.mean(mat[np.ix_(idx_a, idx_b)])) if len(idx_a) > 0 and len(idx_b) > 0 else 0.0
    return intra, inter


def build_connectome_feature_vector(ts: np.ndarray, window_size: int, step: int, thr: float = 0.3, networks: Optional[list] = None, compute_partial: bool = True) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    n_time = ts.shape[1]
    n_rois = ts.shape[0]
    start = 0
    n_w = 0
    fc_sum = np.zeros((n_rois, n_rois), dtype=float)
    fc_sumsq = np.zeros((n_rois, n_rois), dtype=float)
    pc_sum = np.zeros((n_rois, n_rois), dtype=float)
    pc_sumsq = np.zeros((n_rois, n_rois), dtype=float)
    str_sum = np.zeros(n_rois, dtype=float)
    str_sumsq = np.zeros(n_rois, dtype=float)
    clu_sum = np.zeros(n_rois, dtype=float)
    clu_sumsq = np.zeros(n_rois, dtype=float)
    intra_sum = None
    intra_sumsq = None
    inter_sum = None
    inter_sumsq = None
    while start + window_size <= n_time:
        w = ts[:, start : start + window_size]
        m = fc_matrix(w)
        p = precision_partial_corr(w) if compute_partial else m
        a = threshold_adjacency(m, thr=thr)
        s = node_strength(m)
        c = local_clustering(a)
        fc_sum += m
        fc_sumsq += m * m
        pc_sum += p
        pc_sumsq += p * p
        str_sum += s
        str_sumsq += s * s
        clu_sum += c
        clu_sumsq += c * c
        if networks is not None:
            intra, inter = network_connectivity(m, networks)
            inter_flat = inter.flatten()
            if intra_sum is None:
                intra_sum = np.zeros_like(intra, dtype=float)
                intra_sumsq = np.zeros_like(intra, dtype=float)
                inter_sum = np.zeros_like(inter_flat, dtype=float)
                inter_sumsq = np.zeros_like(inter_flat, dtype=float)
            intra_sum += intra
            intra_sumsq += intra * intra
            inter_sum += inter_flat
            inter_sumsq += inter_flat * inter_flat
        n_w += 1
        start += step
    if n_w == 0:
        m = fc_matrix(ts)
        p = precision_partial_corr(ts) if compute_partial else m
        a = threshold_adjacency(m, thr=thr)
        s = node_strength(m)
        c = local_clustering(a)
        fc_mean = m
        pc_mean = p
        str_mean = s
        clu_mean = c
        fc_std = np.zeros_like(m)
        pc_std = np.zeros_like(p)
        str_std = np.zeros_like(s)
        clu_std = np.zeros_like(c)
        if networks is not None:
            intra, inter = network_connectivity(m, networks)
            intra_mean = intra
            inter_mean = inter.flatten()
            intra_std = np.zeros_like(intra_mean)
            inter_std = np.zeros_like(inter_mean)
        else:
            intra_mean = np.array([])
            inter_mean = np.array([])
            intra_std = np.array([])
            inter_std = np.array([])
    else:
        inv_n = 1.0 / n_w
        fc_mean = fc_sum * inv_n
        pc_mean = pc_sum * inv_n
        str_mean = str_sum * inv_n
        clu_mean = clu_sum * inv_n
        def std_from_sums(sum_arr, sumsq_arr):
            var = (sumsq_arr - (sum_arr * sum_arr) * inv_n) / max(n_w - 1, 1)
            var = np.where(var < 0, 0, var)
            return np.sqrt(var)
        fc_std = std_from_sums(fc_sum, fc_sumsq)
        pc_std = std_from_sums(pc_sum, pc_sumsq)
        str_std = std_from_sums(str_sum, str_sumsq)
        clu_std = std_from_sums(clu_sum, clu_sumsq)
        if networks is not None and intra_sum is not None:
            intra_mean = intra_sum * inv_n
            inter_mean = inter_sum * inv_n
            intra_std = std_from_sums(intra_sum, intra_sumsq)
            inter_std = std_from_sums(inter_sum, inter_sumsq)
        else:
            intra_mean = np.array([])
            inter_mean = np.array([])
            intra_std = np.array([])
            inter_std = np.array([])
    def vec_ut(m):
        idx = np.triu_indices(m.shape[0], k=1)
        return m[idx]
    parts = [
        vec_ut(fc_mean), vec_ut(fc_std), vec_ut(pc_mean), vec_ut(pc_std),
        str_mean, str_std, clu_mean, clu_std,
        intra_mean, inter_mean, intra_std, inter_std
    ]
    names = ["fc_mean", "fc_std", "pc_mean", "pc_std", "strength_mean", "strength_std", "cluster_mean", "cluster_std", "intra_mean", "inter_mean", "intra_std", "inter_std"]
    idxs: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for k, arr in zip(names, parts):
        s = arr.shape[0]
        idxs[k] = (cursor, cursor + s)
        cursor += s
    feat = np.concatenate(parts, axis=0)
    return feat, idxs
