from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression


def fit_mvn(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = x.mean(axis=0)
    lw = LedoitWolf().fit(x)
    c = lw.covariance_
    return m, c


def mahalanobis(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.pinv(cov)
    d = x - mean
    return np.sqrt(np.einsum("...i,ij,...j->...", d, inv, d))


def zscores(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    s = np.where(std < 1e-8, 1e-8, std)
    return (x - mean) / s


def residualize_with_covariates(td_features: np.ndarray, td_covars: np.ndarray, all_covars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_t = td_covars
    X_a = all_covars
    Y_t = td_features
    coef = np.linalg.pinv(X_t) @ Y_t
    intercept = Y_t.mean(axis=0) - X_t.mean(axis=0) @ coef
    Y_a = all_covars @ coef + intercept
    resid_td = Y_t - (X_t @ coef + intercept)
    resid_all = Y_a
    return resid_td, resid_all


def personalized_deviation_maps(td_features: np.ndarray, all_features: np.ndarray, covars: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    if covars is not None:
        resid_td, resid_all = residualize_with_covariates(td_features, covars[: td_features.shape[0]], covars)
        td_features = resid_td
        all_features = resid_all
    mean = td_features.mean(axis=0)
    std = td_features.std(axis=0, ddof=1)
    mvn_mean, mvn_cov = fit_mvn(td_features)
    z = zscores(all_features, mean, std)
    m = mahalanobis(all_features, mvn_mean, mvn_cov)
    return {"feature_z": z, "mahalanobis": m}
