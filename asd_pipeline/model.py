from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut, cross_validate
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin


def build_classifier(C: float = 1.0, penalty: str = "l2") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver="liblinear", class_weight="balanced")),
    ])


def evaluate_classifier(X: np.ndarray, y: np.ndarray, cv_splits: int = 5, random_state: int = 42) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    clf = build_classifier()
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}
    res = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=None)
    return {k: float(np.mean(v)) for k, v in res.items() if k.startswith("test_")}


def build_models() -> List[Pipeline]:
    models = []
    models.append(Pipeline([("scaler", StandardScaler()), ("selector", NormDiffSelector(k=500)), ("clf", LogisticRegression(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced"))]))
    models.append(Pipeline([("scaler", StandardScaler()), ("selector", NormDiffSelector(k=500)), ("clf", LogisticRegression(C=1.0, penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=2000, class_weight="balanced"))]))
    svc = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC(C=1.0, class_weight="balanced"))])
    models.append(CalibratedClassifierCV(estimator=svc, cv=3, method="sigmoid"))
    return models


def site_stratified_kfold(groups: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    unique_sites = np.unique(groups)
    fold_assign = -np.ones(groups.shape[0], dtype=int)
    for site in unique_sites:
        idx = np.where(groups == site)[0]
        ys = y[idx]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        i = 0
        for _, test in skf.split(np.zeros_like(ys), ys):
            fold_assign[idx[test]] = i
            i += 1
    splits = []
    for f in range(n_splits):
        test_idx = np.where(fold_assign == f)[0]
        train_idx = np.where(fold_assign != f)[0]
        splits.append((train_idx, test_idx))
    return splits


def evaluate_models(X: np.ndarray, y: np.ndarray, cv_strategy: str = "stratified", groups: np.ndarray = None, cv_splits: int = 5, random_state: int = 42) -> Dict[str, Dict[str, float]]:
    if cv_strategy == "loso" and groups is not None:
        cv = LeaveOneGroupOut()
    elif cv_strategy == "group" and groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
    elif cv_strategy == "site_stratified" and groups is not None:
        cv = site_stratified_kfold(groups, y, n_splits=cv_splits, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    models = build_models()
    names = ["logistic_l2", "logistic_elasticnet", "linear_svc_calibrated"]
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}
    out: Dict[str, Dict[str, float]] = {}
    for name, model in zip(names, models):
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None, groups=groups)
        out[name] = {k.replace("test_", ""): float(np.mean(v)) for k, v in res.items() if k.startswith("test_")}
    return out


class NormDiffSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 500):
        self.k = k
        self.idx_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        if 2 in classes and 1 in classes:
            c0, c1 = 2, 1
        elif 0 in classes and 1 in classes:
            c0, c1 = 0, 1
        else:
            c0, c1 = classes[0], classes[-1]
        m0 = X[y_arr == c0].mean(axis=0) if np.any(y_arr == c0) else X.mean(axis=0)
        m1 = X[y_arr == c1].mean(axis=0) if np.any(y_arr == c1) else X.mean(axis=0)
        diff = np.abs(m1 - m0)
        self.idx_ = np.argsort(diff)[::-1][: self.k]
        return self

    def transform(self, X: np.ndarray):
        return X[:, self.idx_]
