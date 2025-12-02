from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from nilearn.signal import clean


def regress_confounds(ts: np.ndarray, confounds: Optional[np.ndarray] = None, tr: Optional[float] = None, high_pass: Optional[float] = None, low_pass: Optional[float] = None, detrend: bool = True, standardize: str = "zscore") -> np.ndarray:
    if confounds is not None and confounds.size > 0:
        lr = LinearRegression()
        lr.fit(confounds, ts)
        residuals = ts - lr.predict(confounds)
    else:
        residuals = ts
    return clean(
        residuals,
        detrend=detrend,
        standardize=standardize,
        t_r=tr,
        high_pass=high_pass,
        low_pass=low_pass,
    )


def zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, ddof=1, keepdims=True)
    s = np.where(s < eps, eps, s)
    return (x - m) / s, m.squeeze(), s.squeeze()
