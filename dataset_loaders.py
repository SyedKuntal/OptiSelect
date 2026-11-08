"""
dataset_loaders.py - one small loader per dataset + a registry.

HOW TO ADD / CHANGE A DATASET
-----------------------------
1. Write a function  _load_xxx(variant=None) -> Dataset  (see the examples below).
2. Add one DatasetSpec(...) entry to the DATASETS dictionary at the bottom.
Everything else (engine, figures, Streamlit app, results table) picks it up
automatically.

Note: this file is called dataset_loaders.py (not datasets.py) on purpose, so it
never clashes with the Hugging Face "datasets" package.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

DATA_DIR = Path(__file__).resolve().parent / "data"

# Shown in the app wherever a member's name is credited.
PENDING_NAME = "Group member (name to be added)"


@dataclass
class Dataset:
    """What every loader returns."""
    name: str
    X: pd.DataFrame            # input features only (may contain NaN - handled in the Pipeline)
    y: pd.Series               # integer class labels 0..C-1
    class_names: list          # class_names[i] is the readable name of label i
    target_note: str           # how the target was defined (goes into the report)
    data_note: str             # source / preprocessing notes
    variant: Optional[str] = None

    @property
    def label(self) -> str:
        """Name used in figure titles and in the results table."""
        return f"{self.name} ({self.variant})" if self.variant else self.name


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    member_no: int
    contributor: str
    source: str
    suggested_sizes: Tuple[int, ...]   # example subset sizes from the project guide
    default_k: int                     # slider default in the app
    loader: Callable[..., Dataset]
    variants: Tuple[str, ...] = ()     # optional sub-choices (used by Wine Quality)


# --------------------------------------------------------------------------- #
# Member 1 - Iris
# --------------------------------------------------------------------------- #
def _load_iris(variant=None) -> Dataset:
    b = load_iris(as_frame=True)
    return Dataset(
        name="Iris",
        X=b.data.copy(),
        y=b.target.rename("target").astype(int),
        class_names=[str(c) for c in b.target_names],
        target_note="3 species: setosa, versicolor, virginica (50 samples each).",
        data_note="scikit-learn load_iris(); 150 rows, 4 numeric features, no missing values.",
    )


# --------------------------------------------------------------------------- #
# Member 2 - Wine  (Syed Kuntal)
# --------------------------------------------------------------------------- #
def _load_wine(variant=None) -> Dataset:
    b = load_wine(as_frame=True)
    return Dataset(
        name="Wine",
        X=b.data.copy(),
        y=b.target.rename("target").astype(int),
        class_names=[str(c) for c in b.target_names],
        target_note="3 cultivars of Italian wine (class_0, class_1, class_2).",
        data_note="scikit-learn load_wine(); 178 rows, 13 numeric chemical features, no missing values.",
    )


# --------------------------------------------------------------------------- #
# Member 3 - Breast Cancer Wisconsin
# --------------------------------------------------------------------------- #
def _load_breast_cancer(variant=None) -> Dataset:
    b = load_breast_cancer(as_frame=True)
    return Dataset(
        name="Breast Cancer",
        X=b.data.copy(),
        y=b.target.rename("target").astype(int),
        class_names=[str(c) for c in b.target_names],  # ['malignant', 'benign']
        target_note="Binary: 0 = malignant, 1 = benign (scikit-learn's own encoding).",
        data_note="scikit-learn load_breast_cancer() (Wisconsin Diagnostic); 569 rows, 30 numeric features, no missing values.",
    )


# --------------------------------------------------------------------------- #
# Member 4 - Heart Disease (UCI, Cleveland subset)
# --------------------------------------------------------------------------- #
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def _load_heart() -> Dataset:
    path = DATA_DIR / "heart_disease" / "processed.cleveland.data"
    # File has no header row and marks missing values with "?".
    df = pd.read_csv(path, header=None, names=HEART_COLUMNS, na_values="?")
    # Original target is 0 (no disease) .. 4 (severe). Standard practice: 0 vs 1-4.
    y = (df.pop("target") > 0).astype(int).rename("target")
    n_missing = int(df.isna().sum().sum())
    return Dataset(
        name="Heart Disease",
        X=df.astype(float),
        y=y,
        class_names=["No disease", "Disease"],
        target_note="UCI 'num' column (0-4) converted to binary: 0 = no disease, 1-4 = disease present.",
        data_note=(
            f"UCI Heart Disease, Cleveland file (processed.cleveland.data); {len(df)} rows, "
            f"13 commonly used features. {n_missing} missing values (marked '?', in 'ca' and 'thal') are "
            "left as NaN here and imputed with the training-set median inside the Pipeline, "
            "so nothing from the test split leaks into the imputer. "
            "Categorical codes (cp, restecg, slope, thal...) are treated as numeric, as in most UCI studies."
        ),
    )


def _load_heart_variant(variant=None) -> Dataset:
    return _load_heart()


# --------------------------------------------------------------------------- #
# Member 5 - Wine Quality (UCI)    (Syed Kuntal)
# --------------------------------------------------------------------------- #
WINE_QUALITY_VARIANTS = ("Red", "White", "Combined")


def _load_wine_quality(variant: Optional[str] = None) -> Dataset:
    variant = variant or "Red"
    if variant not in WINE_QUALITY_VARIANTS:
        raise ValueError(f"Wine Quality variant must be one of {WINE_QUALITY_VARIANTS}")
    folder = DATA_DIR / "wine_quality"
    # These UCI files are SEMICOLON separated.
    red = pd.read_csv(folder / "winequality-red.csv", sep=";")
    white = pd.read_csv(folder / "winequality-white.csv", sep=";")
    df = {"Red": red, "White": white,
          "Combined": pd.concat([red, white], ignore_index=True)}[variant]
    # Documented classification target: quality >= 6  ->  "Good" (1), otherwise 0.
    y = (df["quality"] >= 6).astype(int).rename("target")
    X = df.drop(columns=["quality"]).astype(float)
    return Dataset(
        name="Wine Quality",
        X=X,
        y=y,
        class_names=["Not good (<=5)", "Good (>=6)"],
        target_note="Binary target defined as quality >= 6 -> 'Good' (1), quality <= 5 -> 'Not good' (0).",
        data_note=(
            f"UCI Wine Quality, '{variant}' data; {len(X)} rows, 11 physico-chemical features, no missing "
            "values. The raw 'quality' score (0-10) is dropped from the inputs so it cannot leak the target. "
            "'Combined' simply stacks red and white rows (no wine-type feature is added). "
            "Note: the UCI files contain exact duplicate rows, which are kept as-is; a duplicate can fall in "
            "both train and test, which makes test scores slightly optimistic."
        ),
        variant=variant,
    )


# --------------------------------------------------------------------------- #
# Registry - the order here is the order in the app's dropdown.
# --------------------------------------------------------------------------- #
DATASETS = {
    "Iris": DatasetSpec(
        "Iris", 1, PENDING_NAME, "scikit-learn built-in", (2, 3, 4), 3, _load_iris),
    "Wine": DatasetSpec(
        "Wine", 2, "Syed Kuntal", "scikit-learn built-in", (5, 7, 10, 13), 7, _load_wine),
    "Breast Cancer": DatasetSpec(
        "Breast Cancer", 3, PENDING_NAME, "scikit-learn built-in", (5, 10, 15, 30), 10, _load_breast_cancer),
    "Heart Disease": DatasetSpec(
        "Heart Disease", 4, PENDING_NAME, "UCI Machine Learning Repository (Cleveland)", (5, 8, 13), 8,
        _load_heart_variant),
    "Wine Quality": DatasetSpec(
        "Wine Quality", 5, "Syed Kuntal", "UCI Machine Learning Repository", (4, 6, 8, 11), 6,
        _load_wine_quality, variants=WINE_QUALITY_VARIANTS),
}
DATASET_NAMES = list(DATASETS)


@lru_cache(maxsize=None)
def load_dataset(name: str, variant: Optional[str] = None) -> Dataset:
    """Load a dataset by its registry name (cached; never modify the returned frames)."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {DATASET_NAMES}")
    spec = DATASETS[name]
    if spec.variants and variant is None:
        variant = spec.variants[0]
    return spec.loader(variant)
