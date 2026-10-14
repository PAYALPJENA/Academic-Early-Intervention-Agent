"""Evaluation metrics and mean ± std aggregation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import CLASS_ORDER


def compute_metrics(y_true, y_pred, proba=None, proba_classes=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0),
    }

    for i, cls in enumerate(CLASS_ORDER):
        key = cls.lower().replace(" ", "_")
        metrics[f"{key}_precision"] = precision[i]
        metrics[f"{key}_recall"] = recall[i]
        metrics[f"{key}_f1"] = f1[i]

    if proba is not None:
        classes = list(proba_classes)
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=classes
        )
        # PR-AUC (average precision), one-vs-rest per class; threshold-free
        y_arr = np.asarray(y_true)
        pr_aucs = []
        for cls in CLASS_ORDER:
            ap = average_precision_score(y_arr == cls, proba[:, classes.index(cls)])
            metrics[f"{cls.lower().replace(' ', '_')}_pr_auc"] = ap
            pr_aucs.append(ap)
        metrics["macro_pr_auc"] = float(np.mean(pr_aucs))

    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["support"] = {cls: int(s) for cls, s in zip(CLASS_ORDER, support)}
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist()
    return metrics


def scalar_keys(metrics):
    return [k for k, v in metrics.items() if isinstance(v, float)]


def mean_std(metric_dicts):
    """{metric: {"mean", "std", "n"}} over a list of metric dicts (std uses ddof=1)."""
    out = {}
    for k in scalar_keys(metric_dicts[0]):
        values = np.array([m[k] for m in metric_dicts], dtype=float)
        out[k] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
        }
    return out
