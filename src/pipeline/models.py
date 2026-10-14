"""Model definitions. Each model is a Pipeline(preprocessor -> estimator)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config

# CPU threads for the added models (-1 = all cores). Set PIPELINE_N_JOBS to limit load and heat.
N_JOBS = int(__import__("os").environ.get("PIPELINE_N_JOBS", "-1"))


def make_preprocessor():
    # Same as notebooks/02_preprocessing.ipynb; remaining columns (Semester_ID) are dropped
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLUMNS),
            ("numerical", StandardScaler(), config.NUMERICAL_COLUMNS),
        ]
    )


def make_midterm_only_preprocessor():
    return ColumnTransformer(transformers=[("numerical", StandardScaler(), ["Midterm_Mark"])])


def make_ft_preprocessor():
    # FT-Transformer tokens: 21 scaled numeric columns first, then 5 categorical codes (-1 = unknown)
    return ColumnTransformer(transformers=[
        ("numerical", StandardScaler(), config.NUMERICAL_COLUMNS),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), config.CATEGORICAL_COLUMNS),
    ])


class EncodedLabels(ClassifierMixin, BaseEstimator):
    """Fits the wrapped classifier on integer labels (needed by XGBoost) and reports string classes_."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.estimator_ = clone(self.estimator).fit(X, y_idx)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class ThresholdedClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a probabilistic classifier with tunable decision thresholds.

    predict: High Risk if P(High) >= t_high, else Low Risk if P(Low) >= t_low, else Medium Risk.
    With no thresholds set it falls back to argmax. predict_proba is unchanged.
    """

    def __init__(self, estimator=None, t_high=None, t_low=None):
        self.estimator = estimator
        self.t_high = t_high
        self.t_low = t_low

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return predict_with_thresholds(self.predict_proba(X), self.classes_, self.t_high, self.t_low)


def predict_with_thresholds(proba, classes, t_high=None, t_low=None):
    classes = np.asarray(classes)
    if t_high is None or t_low is None:
        return classes[proba.argmax(axis=1)]
    high = proba[:, list(classes).index("High Risk")] >= t_high
    low = proba[:, list(classes).index("Low Risk")] >= t_low
    return np.where(high, "High Risk", np.where(low, "Low Risk", "Medium Risk"))


# name -> label, whether to run k-fold CV, preprocessor factory, estimator factory
MODEL_SPECS = {
    "majority": {
        "label": "Majority class (baseline)",
        "cv": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, **_: DummyClassifier(strategy="most_frequent"),
    },
    # C, class_weight and both decision thresholds are tuned on the validation set
    "midterm_lr": {
        "label": "LR, Midterm_Mark only (baseline)",
        "cv": True,
        "preprocessor": make_midterm_only_preprocessor,
        "estimator": lambda seed, C=1.0, class_weight=None, t_high=None, t_low=None, **_: ThresholdedClassifier(
            LogisticRegression(C=C, class_weight=class_weight, max_iter=1000, random_state=seed),
            t_high=t_high,
            t_low=t_low,
        ),
    },
    "logreg": {
        "label": "Logistic Regression",
        "cv": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, **_: LogisticRegression(max_iter=1000, random_state=seed),
    },
    "logreg_balanced": {
        "label": "Logistic Regression (balanced)",
        "cv": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, **_: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    },
    "rf_capped": {
        "label": "RF (capped, tuned)",
        "cv": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, max_depth=20, min_samples_leaf=5, **_: RandomForestClassifier(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    },
    # Original settings; final fit only (no CV) because each fit is very slow and large
    "rf_unconstrained": {
        "label": "RF (unconstrained)",
        "cv": False,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, **_: RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    },
    "hgb": {
        "label": "HistGradientBoosting",
        "cv": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, **_: HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=15,
            random_state=seed,
        ),
    },
    # ---- added models (tuned lightly on the validation set; see config.EXTRA_TUNING_GRIDS)
    "xgb": {
        "label": "XGBoost",
        "cv": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, n_estimators=300, max_depth=6, learning_rate=0.1, device="cpu", **_: EncodedLabels(
            __import__("xgboost").XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                tree_method="hist", device=device, n_jobs=N_JOBS, random_state=seed, verbosity=0,
            )
        ),
    },
    "lgbm": {
        "label": "LightGBM",
        "cv": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, n_estimators=300, num_leaves=31, learning_rate=0.1, device="cpu", **_: __import__("lightgbm").LGBMClassifier(
            n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
            device_type="cpu", n_jobs=N_JOBS, random_state=seed, verbose=-1,
        ),
    },
    "catboost": {
        "label": "CatBoost",
        "cv": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, iterations=500, depth=6, learning_rate=0.1, device="cpu", **_: __import__("catboost").CatBoostClassifier(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            task_type="GPU" if device == "cuda" else "CPU",
            random_seed=seed, thread_count=N_JOBS, verbose=0, allow_writing_files=False,
        ),
    },
    "mlp": {
        "label": "MLP (PyTorch; 100k subsample)",
        "cv": True,
        "deep": True,
        "preprocessor": make_preprocessor,
        "estimator": lambda seed, hidden=256, n_layers=2, lr=1e-3, epochs=20, device="cpu", **_: __import__(
            "pipeline.deep", fromlist=["MLPClassifierTorch"]).MLPClassifierTorch(
            hidden=hidden, n_layers=n_layers, lr=lr, epochs=epochs, random_state=seed, device=device,
        ),
    },
    "ft_transformer": {
        # No CV: one fit costs ~13 min on this CPU even on the subsample (final fit per seed only)
        "label": "FT-Transformer (100k subsample; no CV)",
        "cv": False,
        "deep": True,
        "preprocessor": make_ft_preprocessor,
        "estimator": lambda seed, d_token=64, n_blocks=2, lr=1e-3, epochs=10, device="cpu", **_: __import__(
            "pipeline.deep", fromlist=["FTTransformerClassifier"]).FTTransformerClassifier(
            n_num=len(config.NUMERICAL_COLUMNS), d_token=d_token, n_blocks=n_blocks, lr=lr, epochs=epochs,
            random_state=seed, device=device,
        ),
    },
}


def build_model(name, seed, device="cpu", **params):
    spec = MODEL_SPECS[name]
    return Pipeline([
        ("preprocessor", spec["preprocessor"]()),
        ("model", spec["estimator"](seed, device=device, **params)),
    ])


def label(name):
    return MODEL_SPECS[name]["label"]


def uses_cv(name):
    return MODEL_SPECS[name]["cv"]


def is_deep(name):
    return MODEL_SPECS[name].get("deep", False)
