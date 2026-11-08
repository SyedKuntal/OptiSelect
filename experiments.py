"""
experiments.py - the model-evaluation engine used by app.py and generate_results.py.

Protocol (identical for every dataset, model and method):
  * ONE stratified 80/20 train/test split, random_state=42.
  * Imputation -> scaling -> feature selection -> classifier live in ONE sklearn
    Pipeline, so every step is fitted on training data only (no leakage).
  * k is compared with 5-fold stratified cross-validation on the TRAINING split only.
  * The test set is used once per reported configuration, never to pick k or features.
  * Baseline = same pipeline without the selector (all features).
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dataset_loaders import Dataset, load_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = max(1, min(CV_FOLDS, os.cpu_count() or 1))

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
METHOD_NAMES = ["SelectKBest", "RFE"]


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def make_model(model_name: str):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model '{model_name}'. Choose from {MODEL_NAMES}")


def make_pipeline(model_name: str, method: Optional[str] = None, k: Optional[int] = None) -> Pipeline:
    """imputer -> scaler -> [selector] -> model.  method=None gives the all-features baseline."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if method == "SelectKBest":                       # filter approach
        steps.append(("selector", SelectKBest(score_func=f_classif, k=k)))
    elif method == "RFE":                             # wrapper approach (estimator per project guide)
        steps.append(("selector", RFE(estimator=LogisticRegression(max_iter=2000),
                                      n_features_to_select=k)))
    elif method is not None:
        raise ValueError(f"Unknown method '{method}'. Choose from {METHOD_NAMES}")
    steps.append(("model", make_model(model_name)))
    return Pipeline(steps)


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def get_split(dataset_name: str, variant: Optional[str] = None):
    """Returns (dataset, X_train, X_test, y_train, y_test) - the one shared split."""
    ds = load_dataset(dataset_name, variant)
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, stratify=ds.y, random_state=RANDOM_STATE)
    return ds, X_train, X_test, y_train, y_test


def _cv(pipe: Pipeline, X_train, y_train) -> dict:
    res = cross_validate(pipe, X_train, y_train, cv=make_cv(),
                         scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                         n_jobs=N_JOBS)
    return {
        "cv_mean": float(res["test_accuracy"].mean()),
        "cv_std": float(res["test_accuracy"].std()),
        "cv_f1": float(res["test_f1_macro"].mean()),
        "fit_time": float(res["fit_time"].mean()),   # mean seconds per fold (parallel => approximate)
    }


def selected_names(pipe: Pipeline, columns) -> list:
    if "selector" in pipe.named_steps:
        return list(columns[pipe.named_steps["selector"].get_support()])
    return list(columns)


def feature_scores(pipe: Pipeline, columns):
    """(Series of one score per input feature, axis label). Needs a fitted pipeline with a selector."""
    sel = pipe.named_steps["selector"]
    if isinstance(sel, SelectKBest):
        return pd.Series(np.nan_to_num(sel.scores_), index=columns), "ANOVA F-score (SelectKBest)"
    n = len(columns)
    # RFE ranking: 1 = kept to the end. Flip so that a longer survival = a higher bar.
    return pd.Series(n - sel.ranking_ + 1, index=columns), "RFE retention score (higher = eliminated later)"


def _fit_and_test(pipe: Pipeline, X_train, y_train, X_test, y_test, n_classes: int) -> dict:
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)                        # imputer, scaler, selector, model: all train-only
    train_time = time.perf_counter() - t0
    pred = pipe.predict(X_test)
    return {
        "test_acc": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred, average="macro")),
        "train_time": float(train_time),
        "cm": confusion_matrix(y_test, pred, labels=list(range(n_classes))),
    }


# --------------------------------------------------------------------------- #
# k sweep (cross-validation on the training split only)
# --------------------------------------------------------------------------- #
def compute_sweep(X_train, y_train, model_name: str) -> pd.DataFrame:
    """CV accuracy for every k = 1..n_features and both methods."""
    n = X_train.shape[1]
    rows = []
    for method in METHOD_NAMES:
        for k in range(1, n + 1):
            r = _cv(make_pipeline(model_name, method, k), X_train, y_train)
            rows.append({"method": method, "k": k, **r})
    return pd.DataFrame(rows)


def best_k(sweep: pd.DataFrame, method: str) -> int:
    """k with the highest mean CV accuracy; ties go to the SMALLER k (fewer inputs)."""
    sub = sweep[sweep["method"] == method].copy()
    sub["r"] = sub["cv_mean"].round(10)
    return int(sub.sort_values(["r", "k"], ascending=[False, True]).iloc[0]["k"])


# --------------------------------------------------------------------------- #
# One full experiment = what the "Run Experiment" button does
# --------------------------------------------------------------------------- #
def run_experiment(dataset_name: str, model_name: str, method: str, k: int,
                   variant: Optional[str] = None, sweep: Optional[pd.DataFrame] = None) -> dict:
    ds, X_train, X_test, y_train, y_test = get_split(dataset_name, variant)
    n_features = X_train.shape[1]
    n_classes = len(ds.class_names)
    if not 1 <= k <= n_features:
        raise ValueError(f"k must be between 1 and {n_features} for {ds.label}")
    if sweep is None:
        sweep = compute_sweep(X_train, y_train, model_name)

    # Baseline: all features
    base_pipe = make_pipeline(model_name)
    base = {**_cv(base_pipe, X_train, y_train),
            **_fit_and_test(base_pipe, X_train, y_train, X_test, y_test, n_classes)}

    # Optimized: chosen method with k features (CV numbers come from the same sweep)
    pipe = make_pipeline(model_name, method, k)
    opt = _fit_and_test(pipe, X_train, y_train, X_test, y_test, n_classes)
    row = sweep[(sweep["method"] == method) & (sweep["k"] == k)].iloc[0]
    opt.update(cv_mean=float(row["cv_mean"]), cv_std=float(row["cv_std"]),
               cv_f1=float(row["cv_f1"]), fit_time=float(row["fit_time"]))
    scores, score_label = feature_scores(pipe, X_train.columns)
    opt["features"] = selected_names(pipe, X_train.columns)
    opt["scores"] = scores
    opt["score_label"] = score_label

    return {
        "dataset": ds.name, "variant": ds.variant, "label": ds.label,
        "model": model_name, "method": method, "k": int(k), "n_features": n_features,
        "n_train": len(X_train), "n_test": len(X_test),
        "class_names": ds.class_names, "all_features": list(X_train.columns),
        "sweep": sweep,
        "cv_best_k": {m: best_k(sweep, m) for m in METHOD_NAMES},
        "baseline": base, "optimized": opt,
    }


def run_manual(dataset_name: str, model_name: str, features: list,
               variant: Optional[str] = None) -> dict:
    """User-picked features vs all features, on the identical split and CV protocol."""
    ds, X_train, X_test, y_train, y_test = get_split(dataset_name, variant)
    n_classes = len(ds.class_names)
    out = {}
    for tag, cols in (("manual", list(features)), ("all", list(X_train.columns))):
        pipe = make_pipeline(model_name)
        cv = _cv(pipe, X_train[cols], y_train)
        fit = _fit_and_test(pipe, X_train[cols], y_train, X_test[cols], y_test, n_classes)
        out[tag] = {**cv, **fit, "features": cols}
    return out


# --------------------------------------------------------------------------- #
# Plain-language captions built ONLY from measured values
# --------------------------------------------------------------------------- #
def describe_results(res: dict) -> dict:
    sw, base, opt = res["sweep"], res["baseline"], res["optimized"]
    n, k = res["n_features"], res["k"]
    out = {}

    parts = []
    for m in METHOD_NAMES:
        kb = res["cv_best_k"][m]
        acc = sw[(sw.method == m) & (sw.k == kb)].iloc[0]["cv_mean"]
        parts.append(f"{m} peaks at k={kb} (CV accuracy {acc:.3f})")
    out["fig1"] = "; ".join(parts) + f". All-features CV accuracy: {base['cv_mean']:.3f}."

    d_acc, d_f1 = opt["test_acc"] - base["test_acc"], opt["test_f1"] - base["test_f1"]
    verdict = ("preserved or improved" if d_acc >= -1e-9 and d_f1 >= -1e-9
               else "slightly reduced" if d_acc > -0.03 else "reduced")
    out["fig2"] = (f"With {k} of {n} features, test accuracy is {opt['test_acc']:.3f} vs "
                   f"{base['test_acc']:.3f} for all features ({d_acc:+.3f}); macro-F1 is "
                   f"{opt['test_f1']:.3f} vs {base['test_f1']:.3f} ({d_f1:+.3f}). "
                   f"Performance is {verdict} with fewer inputs.")

    top = opt["scores"].sort_values(ascending=False).head(3).index.tolist()
    out["fig3"] = f"Highest-scoring inputs: {', '.join(top)}. Selected: {', '.join(opt['features'])}."

    cm = opt["cm"].copy()
    np.fill_diagonal(cm, 0)
    if cm.sum() == 0:
        out["fig4"] = "No test samples were misclassified."
    else:
        i, j = np.unravel_index(cm.argmax(), cm.shape)
        out["fig4"] = (f"{int(opt['cm'].sum() - np.trace(opt['cm']))} of {res['n_test']} test samples "
                       f"misclassified; most common confusion: true '{res['class_names'][i]}' "
                       f"predicted as '{res['class_names'][j]}' ({int(cm[i, j])}x).")

    out["fig5"] = (f"Training time: {opt['train_time']:.3f}s with {res['method']} (k={k}) vs "
                   f"{base['train_time']:.3f}s for all features. "
                   "Dots in the chart are CV fit time per fold (approximate, since folds run in parallel).")
    return out
