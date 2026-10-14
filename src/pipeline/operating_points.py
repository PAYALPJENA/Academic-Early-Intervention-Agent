"""High-Risk operating points for one model: thresholds on P(High) chosen on validation, applied once to test."""

import numpy as np
from sklearn.metrics import precision_recall_curve

from .metrics import mean_std

RECALL_TARGETS = [0.70, 0.80]


def _prf(is_high, flagged):
    tp = np.count_nonzero(flagged & is_high)
    n_flag, n_pos = np.count_nonzero(flagged), np.count_nonzero(is_high)
    precision = tp / n_flag if n_flag else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "flagged_share": n_flag / len(is_high)}


def choose_thresholds(is_high_val, p_high_val):
    """Thresholds chosen on validation only.

    f1_optimal: maximises High-Risk F1.
    recall_ge_X: the threshold with the highest validation precision among those with recall >= X.
    """
    precision, recall, thresholds = precision_recall_curve(is_high_val, p_high_val)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    chosen = {"f1_optimal": float(thresholds[np.argmax(f1)])}
    for target in RECALL_TARGETS:
        ok = np.flatnonzero(recall >= target)
        best = ok[np.argmax(precision[ok])]
        chosen[f"recall_ge_{target:.2f}"] = float(thresholds[best])
    return chosen


def evaluate(seed_data):
    """seed_data: {seed: (is_high_val, p_high_val, is_high_test, p_high_test)}"""
    per_seed = {}
    for seed, (yv, pv, yt, pt) in seed_data.items():
        per_seed[str(seed)] = {
            name: {"threshold": t, "val": _prf(yv, pv >= t), "test": _prf(yt, pt >= t)}
            for name, t in choose_thresholds(yv, pv).items()
        }
    names = list(next(iter(per_seed.values())))
    summary = {}
    for name in names:
        rows = [s[name] for s in per_seed.values()]
        summary[name] = {
            "threshold": mean_std([{"t": r["threshold"]} for r in rows])["t"],
            "val": mean_std([r["val"] for r in rows]),
            "test": mean_std([r["test"] for r in rows]),
        }
    return {"per_seed": per_seed, "summary": summary}
