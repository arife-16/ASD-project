import os
import sys
import numpy as np


def setup_pcn(path: str):
    if path and path not in sys.path:
        sys.path.append(path)


def fit_normative_pcn(X: np.ndarray, covars: np.ndarray, out_dir: str):
    setup_pcn(os.environ.get("PCN_PATH", ""))
    try:
        from pcntoolkit.normative_model import NormativeModel
    except Exception:
        return None
    os.makedirs(out_dir, exist_ok=True)
    nm = NormativeModel(output_path=out_dir)
    nm.fit(X, covars)
    return nm


def predict_normative_pcn(model, X: np.ndarray, covars: np.ndarray):
    if model is None:
        return None, None, None
    mu, sigma, z = model.predict(X, covars)
    return mu, sigma, z
