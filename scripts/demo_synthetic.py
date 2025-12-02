import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from asd_pipeline.normative import personalized_deviation_maps
from asd_pipeline.model import evaluate_classifier


def synthetic_data(n_td: int = 150, n_asd: int = 150, n_features: int = 300, shift: float = 1.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_features, n_features))
    cov = A @ A.T + 1e-3 * np.eye(n_features)
    td = rng.multivariate_normal(np.zeros(n_features), cov, size=n_td)
    delta = np.zeros(n_features)
    idx = rng.choice(n_features, n_features // 5, replace=False)
    delta[idx] = shift
    asd = rng.multivariate_normal(delta, cov, size=n_asd)
    X = np.vstack([td, asd])
    y = np.array([0] * n_td + [1] * n_asd)
    return X, y


def run():
    X, y = synthetic_data()
    td_mask = y == 0
    dev = personalized_deviation_maps(X[td_mask], X)
    X_dev = dev["feature_z"]
    metrics = evaluate_classifier(X_dev, y)
    td_dev = float(np.mean(np.abs(X_dev[y == 0])))
    asd_dev = float(np.mean(np.abs(X_dev[y == 1])))
    print(json.dumps({"metrics": metrics, "group_deviation_abs_mean": {"td": td_dev, "asd": asd_dev}}))


if __name__ == "__main__":
    run()
