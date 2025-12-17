from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut, cross_validate, GridSearchCV
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
    models.append(Pipeline([("scaler", StandardScaler()), ("selector", NormDiffSelector(k=500)), ("svc", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42))]))
    models.append(Pipeline([("selector", NormDiffSelector(k=1000)), ("rf", RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42))]))
    models.append(Pipeline([("scaler", StandardScaler()), ("selector", NormDiffSelector(k=1500)), ("pca", PCA(n_components=0.95, whiten=True, random_state=42)), ("svc", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42))]))
    models.append(Pipeline([("selector", NormDiffSelector(k=1000)), ("gb", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42))]))
    return models


def site_stratified_kfold(groups: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42):
    unique_sites = np.unique(groups)
    classes = np.unique(y)
    min_class_counts = []
    for site in unique_sites:
        idx = np.where(groups == site)[0]
        ys = y[idx]
        counts = [np.sum(ys == c) for c in classes]
        min_class_counts.append(min(counts) if len(counts) > 0 else 0)
    global_min = int(min(min_class_counts)) if len(min_class_counts) > 0 else 0
    n_eff = max(2, min(n_splits, global_min))
    if n_eff < 2:
        gkf = GroupKFold(n_splits=min(n_splits, len(unique_sites)))
        return list(gkf.split(np.zeros_like(y), y, groups=groups))
    fold_assign = -np.ones(groups.shape[0], dtype=int)
    for site in unique_sites:
        idx = np.where(groups == site)[0]
        ys = y[idx]
        skf = StratifiedKFold(n_splits=n_eff, shuffle=True, random_state=random_state)
        i = 0
        for _, test in skf.split(np.zeros_like(ys), ys):
            fold_assign[idx[test]] = i
            i += 1
    splits = []
    for f in range(n_eff):
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
    names = ["logistic_l2", "logistic_elasticnet", "linear_svc_calibrated", "svm_rbf", "random_forest", "svm_rbf_pca", "gradient_boosting"]
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}
    out: Dict[str, Dict[str, float]] = {}
    for name, model in zip(names, models):
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None, groups=groups)
        out[name] = {k.replace("test_", ""): float(np.mean(v)) for k, v in res.items() if k.startswith("test_")}
    return out


def evaluate_models_tuned(X: np.ndarray, y: np.ndarray, cv_strategy: str = "stratified", groups: np.ndarray = None, cv_splits: int = 5, random_state: int = 42) -> Dict[str, Dict[str, float]]:
    if cv_strategy == "loso" and groups is not None:
        cv = LeaveOneGroupOut()
    elif cv_strategy == "group" and groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
    elif cv_strategy == "site_stratified" and groups is not None:
        cv = site_stratified_kfold(groups, y, n_splits=cv_splits, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    models = build_models()
    names = ["logistic_l2", "logistic_elasticnet", "linear_svc_calibrated", "svm_rbf", "random_forest", "svm_rbf_pca", "gradient_boosting"]
    grids = {
        "logistic_l2": {"selector__k": [300, 500, 1000], "clf__C": [0.1, 1.0, 10.0]},
        "logistic_elasticnet": {"selector__k": [300, 500, 1000], "clf__C": [0.1, 1.0, 10.0], "clf__l1_ratio": [0.2, 0.5, 0.8]},
        "linear_svc_calibrated": {"estimator__svc__svc__C": [0.1, 1.0, 10.0]},
        "svm_rbf": {"selector__k": [500, 1000, 2000], "svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", 0.01, 0.001]},
        "random_forest": {"selector__k": [500, 1000], "rf__n_estimators": [200, 500, 800], "rf__max_depth": [None, 10, 20], "rf__min_samples_leaf": [1, 2, 5]},
        "svm_rbf_pca": {"selector__k": [1000, 1500], "pca__n_components": [0.90, 0.95, 0.99], "svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", 0.01]},
        "gradient_boosting": {"gb__n_estimators": [200, 300, 500], "gb__learning_rate": [0.03, 0.05, 0.1], "gb__max_depth": [2, 3, 4]},
    }
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}
    out: Dict[str, Dict[str, float]] = {}
    for name, model in zip(names, models):
        grid = grids.get(name, None)
        if grid:
            gcv = GridSearchCV(model, param_grid=grid, cv=cv, scoring="roc_auc", n_jobs=None, refit=False)
            if groups is not None:
                gcv.fit(X, y, groups=groups)
            else:
                gcv.fit(X, y)
            scores = gcv.cv_results_["mean_test_score"]
            best_idx = int(np.argmax(scores))
            best_params = gcv.cv_results_["params"][best_idx]
            model.set_params(**best_params)
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
