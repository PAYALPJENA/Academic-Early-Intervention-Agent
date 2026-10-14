"""Reproducible training and evaluation pipeline.

    py -3.13 src/pipeline/run_pipeline.py                 # full run
    py -3.13 src/pipeline/run_pipeline.py --sample 20000 --seeds 0 1 --results-dir tmp/results --figures-dir tmp/figures

Stages
  1. Load data, build labels (checked against the original counts), stratified 70/15/15 split (seed 42).
  2. Tune the capped Random Forest on the validation set.
  3. For every model and seed: 5-fold stratified CV on the training set, then a final fit on the
     full training set, scored on the validation set.
  4. Select the model on validation scores only and write results/selection.json.
  5. Only then score the test set, and write results/test.json.
  6. Write figures/ and the summary table.

Runs are resumable: each (model, seed) result is saved to results/runs/ as soon as it finishes.
"""

import argparse
import json
import os
import platform
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import config, operating_points, plots  # noqa: E402
from pipeline.data import load_dataset, split_dataset, stratified_sample  # noqa: E402
from pipeline.metrics import compute_metrics, mean_std  # noqa: E402
from pipeline.models import build_model, label, uses_cv, is_deep, predict_with_thresholds, MODEL_SPECS  # noqa: E402

# Results of the original notebooks (80/20 split, seed 42, notebook 04), kept for reference only
LEGACY_80_20 = {
    "note": "Original notebook results on the old 80/20 split (100,000 test students, single seed). "
            "Not comparable one-to-one with the new 70/15/15 results.",
    "models": {
        "Logistic Regression": {"accuracy": 0.8608, "high_risk_precision": 0.6901, "high_risk_recall": 0.2048, "high_risk_f1": 0.3158},
        "RF (unconstrained)": {"accuracy": 0.8739, "high_risk_precision": 0.8517, "high_risk_recall": 0.2477, "high_risk_f1": 0.3838,
                               "saved_model_size_mb": 819.2},
        "HistGradientBoosting": {"accuracy": 0.8823, "high_risk_precision": 0.7641, "high_risk_recall": 0.3838, "high_risk_f1": 0.5109,
                                 "saved_model_size_mb": 0.58},
    },
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fit_and_score(name, seed, params, X_fit, y_fit, X_eval, y_eval, device="cpu"):
    model = build_model(name, seed, device=device, **params)
    t0 = time.time()
    model.fit(X_fit, y_fit)
    fit_s = time.time() - t0
    proba = model.predict_proba(X_eval)
    pred = predict_with_thresholds(proba, model.classes_, params.get("t_high"), params.get("t_low"))
    return model, compute_metrics(y_eval, pred, proba, model.classes_), fit_s


def tune_thresholds(proba, classes, y_true):
    """Grid-search t_high and t_low on one set of probabilities (validation only).

    Criterion: High-Risk F1, ties broken by macro F1 (same as model selection).
    """
    classes = list(classes)
    p_high = proba[:, classes.index("High Risk")]
    p_low = proba[:, classes.index("Low Risk")]
    y = np.asarray(y_true)
    true_masks = [y == c for c in config.CLASS_ORDER]  # High, Medium, Low
    true_counts = [m.sum() for m in true_masks]

    def f1(pred_mask, k):
        # Same as sklearn f1_score per class with zero_division=0
        tp = np.count_nonzero(pred_mask & true_masks[k])
        denom = np.count_nonzero(pred_mask) + true_counts[k]
        return 2 * tp / denom if denom else 0.0

    best, best_key = None, None
    for th in config.THRESHOLD_GRID:
        high = p_high >= th
        f1_high = f1(high, 0)
        for tl in config.THRESHOLD_GRID:
            low = ~high & (p_low >= tl)
            medium = ~high & ~low
            key = (f1_high, (f1_high + f1(medium, 1) + f1(low, 2)) / 3)  # (High-Risk F1, macro F1)
            if best_key is None or key > best_key:
                best, best_key = {"t_high": th, "t_low": tl}, key
    return best


def load_predictions(out_dir, name, seed, split):
    return np.load(os.path.join(out_dir, "runs", f"{name}_seed{seed}_{split}_predictions.npz"), allow_pickle=True)


_SUBSAMPLE_CACHE = {}
_DEADLINE = [None]


class TimeBudgetReached(Exception):
    pass


def check_budget(what):
    """Stop cleanly before starting another unit of work once the time budget is used up."""
    if _DEADLINE[0] is not None and time.time() > _DEADLINE[0]:
        raise TimeBudgetReached(what)


def resolve_device(requested):
    """Resolve the optional device without requiring PyTorch for CPU-only runs."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        if requested == "auto":
            return "cpu"
        raise RuntimeError("CUDA was requested, but PyTorch is not installed.") from exc
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return "cuda" if available else "cpu"


def training_data(name, splits):
    """Full training set, or for deep models a fixed stratified subsample of it (seed 42)."""
    X_train, y_train = splits["train"]
    if not is_deep(name) or len(y_train) <= config.DEEP_TRAIN_SUBSAMPLE:
        return X_train, y_train
    key = id(splits)
    if key not in _SUBSAMPLE_CACHE:
        _SUBSAMPLE_CACHE[key] = stratified_sample(X_train, y_train, config.DEEP_TRAIN_SUBSAMPLE, seed=config.SPLIT_SEED)
    return _SUBSAMPLE_CACHE[key]


def tune_extra(name, splits, out_dir, device="cpu"):
    """Light grid search for an added model, scored on the validation set only.

    Criterion: validation High-Risk F1 (argmax), ties by macro F1, as for the Random Forest.
    For deep models the number of epochs is chosen the same way (best epoch up to DEEP_MAX_EPOCHS).
    """
    path = os.path.join(out_dir, f"tuning_{name}.json")
    if os.path.exists(path):
        log(f"{label(name)} tuning: reusing {os.path.basename(path)}")
        return read_json(path)

    X_fit, y_fit = training_data(name, splits)
    X_val, y_val = splits["val"]
    grid = config.EXTRA_TUNING_GRIDS[name]
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in __import__("itertools").product(*[grid[k] for k in keys])]
    key = lambda m: (m[config.SELECTION_METRIC], m[config.SELECTION_TIEBREAK])  # noqa: E731
    partial_path = os.path.join(out_dir, "runs", "tuning_partial", f"tuning_{name}.json")
    rows = read_json(partial_path)["results"] if os.path.exists(partial_path) else []
    done = [r["grid_point"] for r in rows]
    for params in combos:
        if params in done:
            continue
        check_budget(f"{name} tuning")
        t0 = time.time()
        if is_deep(name):
            max_epochs = config.DEEP_MAX_EPOCHS[name]
            pipe = build_model(name, config.RF_TUNING_SEED, device=device, **params, epochs=max_epochs)
            prep = pipe.named_steps["preprocessor"].fit(X_fit)
            Z_fit, Z_val = prep.transform(X_fit), prep.transform(X_val)
            Z_fit = Z_fit.toarray() if hasattr(Z_fit, "toarray") else Z_fit
            Z_val = Z_val.toarray() if hasattr(Z_val, "toarray") else Z_val
            per_epoch = []

            def on_epoch(epoch, est):
                proba = est.predict_proba(Z_val)
                pred = predict_with_thresholds(proba, est.classes_)
                m = compute_metrics(y_val, pred, proba, est.classes_)
                per_epoch.append({"epoch": epoch, "val": m})
                log(f"    epoch {epoch:>2}: val High-Risk F1 {m['high_risk_f1']:.4f}, macro F1 {m['macro_f1']:.4f}")

            pipe.named_steps["model"].fit(Z_fit, y_fit, eval_callback=on_epoch)
            best_epoch = max(per_epoch, key=lambda e: key(e["val"]))
            row = {"params": {**params, "epochs": best_epoch["epoch"]}, "val": best_epoch["val"],
                   "per_epoch": [{"epoch": e["epoch"], "high_risk_f1": e["val"]["high_risk_f1"],
                                  "macro_f1": e["val"]["macro_f1"]} for e in per_epoch]}
        else:
            _, val, _ = fit_and_score(name, config.RF_TUNING_SEED, params, X_fit, y_fit, X_val, y_val,
                                      device=device)
            row = {"params": params, "val": val}
        row["seconds"] = time.time() - t0
        row["grid_point"] = params
        rows.append(row)
        write_json(partial_path, {"results": rows})  # checkpoint after every grid point
        log(f"  {label(name)} {row['params']}: val High-Risk F1 {row['val']['high_risk_f1']:.4f}, macro F1 {row['val']['macro_f1']:.4f} ({row['seconds']:.0f}s)")

    best = max(rows, key=lambda r: key(r["val"]))
    result = {
        "model": name, "label": label(name), "seed": config.RF_TUNING_SEED, "scored_on": "validation",
        "criterion": [config.SELECTION_METRIC, config.SELECTION_TIEBREAK], "grid": grid,
        "training_rows": int(len(y_fit)), "results": rows, "best_params": best["params"],
    }
    if is_deep(name):
        result["max_epochs"] = config.DEEP_MAX_EPOCHS[name]
    write_json(path, result)
    return result


def run_operating_points(splits, seeds, out_dir, fig_dir):
    """Best model (selected on validation under the threshold rule): three High-Risk thresholds chosen on
    validation (F1-optimal, recall >= 0.70, recall >= 0.80), each applied once to the test set."""
    selection = read_json(os.path.join(out_dir, "selection.json"))
    name = selection["tuned_threshold_rule"]["selected_model"]
    y_val = splits["val"][1].to_numpy() == "High Risk"
    y_test = splits["test"][1].to_numpy() == "High Risk"

    def p_high(seed, split):
        d = load_predictions(out_dir, name, seed, split)
        return d["proba"][:, list(d["classes"]).index("High Risk")]

    data = {s: (y_val, p_high(s, "val"), y_test, p_high(s, "test")) for s in seeds}
    result = operating_points.evaluate(data)
    write_json(os.path.join(out_dir, "operating_points.json"), {
        "model": name,
        "label": label(name),
        "chosen_because": "selected on validation under the tuned-threshold rule (selection.json)",
        "task": "High Risk vs rest; predict High Risk if P(High) >= threshold",
        "threshold_rules": {
            "f1_optimal": "maximises High-Risk F1 on validation",
            "recall_ge_0.70": "highest validation precision among thresholds with validation recall >= 0.70",
            "recall_ge_0.80": "highest validation precision among thresholds with validation recall >= 0.80",
        },
        "protocol": "thresholds chosen per seed on the validation set, then applied once to the test set",
        "flagged_share": "fraction of students predicted High Risk",
        **result,
    })
    plots.operating_points_plot((y_val, p_high(seeds[0], "val")), (y_test, p_high(seeds[0], "test")),
                                result["summary"], label(name), seeds[0], fig_dir)

    print(f"\nHigh-Risk operating points for {label(name)} (thresholds chosen on validation, test once; mean ± std over seeds)")
    print(f"{'Rule':16} {'Threshold':>15} {'Precision':>17} {'Recall':>17} {'F1':>17} {'Flagged':>8}")
    for rule, s in result["summary"].items():
        t = s["test"]
        print(f"{rule:16} {s['threshold']['mean']:.3f} ± {s['threshold']['std']:.3f} "
              f"{t['precision']['mean']:.4f} ± {t['precision']['std']:.4f} {t['recall']['mean']:.4f} ± {t['recall']['std']:.4f} "
              f"{t['f1']['mean']:.4f} ± {t['f1']['std']:.4f} {t['flagged_share']['mean']:>7.1%}")
    return result


def tune_midterm_lr(splits, out_dir):
    """Tune C, class_weight and both decision thresholds on the validation set only."""
    path = os.path.join(out_dir, "tuning_midterm_lr.json")
    if os.path.exists(path):
        log("Midterm-only LR tuning: reusing results/tuning_midterm_lr.json")
        return read_json(path)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    key = lambda m: (m[config.SELECTION_METRIC], m[config.SELECTION_TIEBREAK])  # noqa: E731
    rows = []
    for C in config.MIDTERM_LR_GRID["C"]:
        for cw in config.MIDTERM_LR_GRID["class_weight"]:
            model = build_model("midterm_lr", config.RF_TUNING_SEED, C=C, class_weight=cw).fit(X_train, y_train)
            proba = model.predict_proba(X_val)
            t = tune_thresholds(proba, model.classes_, y_val)
            best = (t, compute_metrics(y_val, predict_with_thresholds(proba, model.classes_, t["t_high"], t["t_low"]),
                                       proba, model.classes_))
            argmax_val = compute_metrics(y_val, predict_with_thresholds(proba, model.classes_), proba, model.classes_)
            rows.append({"params": {"C": C, "class_weight": cw, **best[0]}, "val": best[1], "val_argmax": argmax_val})
            log(f"  Midterm LR C={C:<5} class_weight={str(cw):8}: best thresholds {best[0]}, val High-Risk F1 {best[1]['high_risk_f1']:.4f}, macro F1 {best[1]['macro_f1']:.4f}")

    best_row = max(rows, key=lambda r: key(r["val"]))
    result = {
        "seed": config.RF_TUNING_SEED,
        "scored_on": "validation",
        "criterion": [config.SELECTION_METRIC, config.SELECTION_TIEBREAK],
        "grid": {**config.MIDTERM_LR_GRID, "t_high": config.THRESHOLD_GRID, "t_low": config.THRESHOLD_GRID},
        "decision_rule": "High Risk if P(High) >= t_high, else Low Risk if P(Low) >= t_low, else Medium Risk",
        "results": rows,
        "best_params": best_row["params"],
    }
    write_json(path, result)
    return result


def display_path(path):
    try:
        return os.path.relpath(path, config.ROOT)
    except ValueError:  # different drive on Windows
        return path


def saved_size_mb(model, keep_path=None):
    if keep_path:
        joblib.dump(model, keep_path)
        return os.path.getsize(keep_path) / 1e6
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.joblib")
        joblib.dump(model, path)
        return os.path.getsize(path) / 1e6


def split_summary(splits):
    info = {}
    for s, (X, y) in splits.items():
        counts = y.value_counts()
        info[s] = {
            "n": int(len(y)),
            "class_counts": {c: int(counts.get(c, 0)) for c in config.CLASS_ORDER},
            "class_share": {c: float(counts.get(c, 0) / len(y)) for c in config.CLASS_ORDER},
        }
    return info


def tune_rf(splits, out_dir):
    path = os.path.join(out_dir, "rf_tuning.json")
    if os.path.exists(path):
        log("RF tuning: reusing results/rf_tuning.json")
        return read_json(path)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    rows = []
    for depth in config.RF_TUNING_GRID["max_depth"]:
        for leaf in config.RF_TUNING_GRID["min_samples_leaf"]:
            params = {"max_depth": depth, "min_samples_leaf": leaf}
            _, val, fit_s = fit_and_score("rf_capped", config.RF_TUNING_SEED, params, X_train, y_train, X_val, y_val)
            rows.append({"params": params, "val": val, "fit_seconds": fit_s})
            log(f"  RF depth={depth:>2} leaf={leaf:>2}: val High-Risk F1 {val['high_risk_f1']:.4f}, macro F1 {val['macro_f1']:.4f} ({fit_s:.0f}s)")

    best = max(rows, key=lambda r: (r["val"][config.SELECTION_METRIC], r["val"][config.SELECTION_TIEBREAK]))
    result = {
        "seed": config.RF_TUNING_SEED,
        "scored_on": "validation",
        "criterion": [config.SELECTION_METRIC, config.SELECTION_TIEBREAK],
        "fixed": {"n_estimators": 100, "class_weight": "balanced", "n_jobs": -1},
        "grid": config.RF_TUNING_GRID,
        "results": rows,
        "best_params": best["params"],
    }
    write_json(path, result)
    return result


def run_model_seed(name, seed, params, splits, folds, out_dir, models_dir, keep_model, device="cpu"):
    """CV on train + final fit scored on validation. Test predictions are stored, not scored."""
    run_path = os.path.join(out_dir, "runs", f"{name}_seed{seed}.json")
    pred_path = os.path.join(out_dir, "runs", f"{name}_seed{seed}_test_predictions.npz")
    val_pred_path = os.path.join(out_dir, "runs", f"{name}_seed{seed}_val_predictions.npz")
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    if os.path.exists(run_path) and os.path.exists(pred_path) and os.path.exists(val_pred_path):
        log(f"{label(name)} seed {seed}: reusing saved run")
        return read_json(run_path)

    X_train, y_train = training_data(name, splits)
    X_val, y_val = splits["val"]
    X_test, _ = splits["test"]

    record = {"model": name, "label": label(name), "seed": seed, "params": params, "cv": [],
              "training_rows": int(len(y_train))}

    if uses_cv(name):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(X_train, y_train)):
            # Each fold is checkpointed, so an interrupted run loses at most one fit
            fold_root = os.path.join(out_dir, "runs", "folds")
            # Keep existing CPU fold checkpoints intact when a CUDA run resumes.
            if device == "cuda":
                fold_root = os.path.join(fold_root, "cuda")
            fold_path = os.path.join(fold_root, f"{name}_seed{seed}_fold{k}.json")
            if os.path.exists(fold_path):
                m = read_json(fold_path)
            else:
                check_budget(f"{name} seed {seed} fold {k + 1}")
                _, m, fit_s = fit_and_score(name, seed, params, X_train.iloc[tr], y_train.iloc[tr],
                                            X_train.iloc[te], y_train.iloc[te], device=device)
                m["fit_seconds"] = fit_s
                write_json(fold_path, m)
            record["cv"].append(m)
            log(f"{label(name)} seed {seed} fold {k + 1}/{folds}: macro F1 {m['macro_f1']:.4f}, High-Risk F1 {m['high_risk_f1']:.4f} ({m['fit_seconds']:.0f}s)")

    check_budget(f"{name} seed {seed} final fit")
    model, val, fit_s = fit_and_score(name, seed, params, X_train, y_train, X_val, y_val,
                                      device=device)
    np.savez_compressed(val_pred_path, proba=model.predict_proba(X_val).astype(np.float32), classes=model.classes_)
    record["val"] = val
    record["final_fit_seconds"] = fit_s
    log(f"{label(name)} seed {seed} final: val macro F1 {val['macro_f1']:.4f}, High-Risk F1 {val['high_risk_f1']:.4f} ({fit_s:.0f}s)")

    if keep_model:
        os.makedirs(models_dir, exist_ok=True)
        keep_path = os.path.join(models_dir, f"{name}.joblib") if name != "rf_unconstrained" else None
        record["saved_model_size_mb"] = saved_size_mb(model, keep_path)
        record["saved_model_path"] = display_path(keep_path) if keep_path else None
        log(f"{label(name)}: saved model size {record['saved_model_size_mb']:.1f} MB")

    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        record["impurity_importance"] = {
            "features": [n.split("__", 1)[1] for n in model.named_steps["preprocessor"].get_feature_names_out()],
            "values": est.feature_importances_.tolist(),
        }
    if name == "hgb":
        n = min(20000, len(y_val))
        idx = np.random.default_rng(seed).choice(len(y_val), size=n, replace=False)
        pi = permutation_importance(model, X_val.iloc[idx], y_val.iloc[idx], scoring="f1_macro",
                                    n_repeats=3, random_state=seed)
        record["permutation_importance_val"] = {
            "features": list(X_val.columns), "mean": pi.importances_mean.tolist(), "std": pi.importances_std.tolist(),
            "scoring": "f1_macro", "n_samples": n, "n_repeats": 3,
        }

    # Store test predictions now so the model need not be kept in memory; they are scored in stage 5
    proba = model.predict_proba(X_test)
    np.savez_compressed(pred_path, proba=proba.astype(np.float32), classes=model.classes_)

    write_json(run_path, record)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    ap.add_argument("--seeds", nargs="+", type=int, default=config.EXPERIMENT_SEEDS)
    ap.add_argument("--folds", type=int, default=config.CV_FOLDS)
    ap.add_argument("--sample", type=int, default=None, help="stratified subsample size for quick smoke runs")
    ap.add_argument("--results-dir", default=config.RESULTS_DIR)
    ap.add_argument("--figures-dir", default=config.FIGURES_DIR)
    ap.add_argument("--models-dir", default=config.MODELS_DIR)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                    help="compute device for XGBoost, CatBoost and PyTorch models; LightGBM remains on CPU")
    ap.add_argument("--operating-points-only", action="store_true",
                    help="only recompute the best model's High-Risk operating points from saved predictions")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="minutes; stop cleanly (exit code 3) before starting new work after this time. "
                         "Rerun to continue from the checkpoints.")
    args = ap.parse_args()
    if args.time_budget:
        _DEADLINE[0] = time.time() + 60 * args.time_budget
    try:
        run(args)
    except TimeBudgetReached as stop:
        log(f"Time budget reached before: {stop}. Rerun the same command to continue.")
        sys.exit(3)


def run(args):

    device = None if args.operating_points_only else resolve_device(args.device)
    if device is not None:
        log(f"Execution device: {device}")
    out_dir, fig_dir = args.results_dir, args.figures_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---- 1. data
    log("Loading data")
    X, y = load_dataset()
    if args.sample:
        X, y = stratified_sample(X, y, args.sample)
    splits = split_dataset(X, y)
    split_info = split_summary(splits)
    write_json(os.path.join(out_dir, "config.json"), {
        "labels": {
            "source": config.TARGET_SOURCE,
            "rule": f"High Risk if score < {config.HIGH_RISK_BELOW}; Medium Risk if {config.HIGH_RISK_BELOW} <= score < {config.LOW_RISK_FROM}; Low Risk if score >= {config.LOW_RISK_FROM}",
            "checked_against_original_counts": config.EXPECTED_CLASS_COUNTS,
        },
        "split": {"seed": config.SPLIT_SEED, "train": 1 - config.VAL_FRACTION - config.TEST_FRACTION,
                  "val": config.VAL_FRACTION, "test": config.TEST_FRACTION, "stratified": True},
        "sample": args.sample,
        "cv_folds": args.folds,
        "seeds": args.seeds,
        "models": {m: label(m) for m in args.models},
        "selection": {"on": "validation", "metric": config.SELECTION_METRIC, "tiebreak": config.SELECTION_TIEBREAK},
        "dropped_columns": config.DROP_COLUMNS + [config.TARGET_SOURCE, "Semester_ID (dropped by the preprocessor)"],
        "deep_models": {
            "models": [label(m) for m in args.models if is_deep(m)],
            "training_rows": config.DEEP_TRAIN_SUBSAMPLE,
            "note": "Deep models are trained on a stratified subsample of the training set (seed 42) because of CPU "
                    "time; validation and test sets are unchanged. FT-Transformer has no CV (final fit per seed only).",
        },
        "added_models_cpu_threads": os.environ.get("PIPELINE_N_JOBS", "all"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    })
    write_json(os.path.join(out_dir, "data_split.json"), split_info)
    log(f"Split: train {split_info['train']['n']:,} / val {split_info['val']['n']:,} / test {split_info['test']['n']:,}")

    if args.operating_points_only:
        run_operating_points(splits, args.seeds, out_dir, fig_dir)
        return

    # ---- 2. RF tuning (validation only)
    params = {m: {} for m in args.models}
    if "rf_capped" in args.models:
        log("Tuning capped Random Forest on the validation set")
        params["rf_capped"] = tune_rf(splits, out_dir)["best_params"]
        log(f"RF best params: {params['rf_capped']}")
    if "midterm_lr" in args.models:
        log("Tuning midterm-only Logistic Regression on the validation set")
        best = tune_midterm_lr(splits, out_dir)["best_params"]
        # C and class_weight are fixed here; decision thresholds are handled per rule in stage 4
        params["midterm_lr"] = {"C": best["C"], "class_weight": best["class_weight"]}
        log(f"Midterm LR best params: {best}")
    for m in args.models:
        if m in config.EXTRA_TUNING_GRIDS:
            log(f"Tuning {label(m)} on the validation set")
            params[m] = tune_extra(m, splits, out_dir, device=device)["best_params"]
            log(f"{label(m)} best params: {params[m]}")

    # ---- 3. CV + validation
    runs = {m: [] for m in args.models}
    for m in args.models:
        for i, seed in enumerate(args.seeds):
            runs[m].append(run_model_seed(m, seed, params[m], splits, args.folds, out_dir,
                                          args.models_dir, keep_model=(i == 0), device=device))

    cv_summary, val_summary = {}, {}
    for m in args.models:
        if uses_cv(m):
            seed_means = [{k: float(np.mean([f[k] for f in r["cv"]])) for k in r["cv"][0] if isinstance(r["cv"][0][k], float)}
                          for r in runs[m]]
            cv_summary[m] = {
                "label": label(m),
                "across_seeds": mean_std(seed_means),
                "across_all_folds": mean_std([f for r in runs[m] for f in r["cv"]]),
                "per_seed_fold_mean": {str(r["seed"]): s for r, s in zip(runs[m], seed_means)},
            }
        else:
            cv_summary[m] = {"label": label(m), "skipped": "No CV for this model (final fit per seed only) to limit runtime."}
        val_summary[m] = {
            "label": label(m),
            "params": params[m],
            "summary": mean_std([r["val"] for r in runs[m]]),
            "per_seed": {str(r["seed"]): r["val"] for r in runs[m]},
        }
    write_json(os.path.join(out_dir, "cv.json"), cv_summary)
    write_json(os.path.join(out_dir, "validation.json"), val_summary)

    # ---- 4. decision thresholds, tuned per model and seed on the validation set only
    _, y_val = splits["val"]
    thresholds, val_tuned_summary = {}, {}
    for m in args.models:
        thresholds[m] = {}
        for r in runs[m]:
            d = load_predictions(out_dir, m, r["seed"], "val")
            t = tune_thresholds(d["proba"], d["classes"], y_val)
            pred = predict_with_thresholds(d["proba"], d["classes"], t["t_high"], t["t_low"])
            thresholds[m][str(r["seed"])] = {**t, "val": compute_metrics(y_val, pred, d["proba"], d["classes"])}
        val_tuned_summary[m] = mean_std([v["val"] for v in thresholds[m].values()])
    threshold_summary = {
        m: {"t_high": mean_std([{"t": v["t_high"]} for v in thresholds[m].values()])["t"],
            "t_low": mean_std([{"t": v["t_low"]} for v in thresholds[m].values()])["t"]}
        for m in args.models
    }
    write_json(os.path.join(out_dir, "thresholds_validation.json"), {
        "decision_rule": "High Risk if P(High) >= t_high, else Low Risk if P(Low) >= t_low, else Medium Risk",
        "grid": config.THRESHOLD_GRID,
        "criterion": [config.SELECTION_METRIC, config.SELECTION_TIEBREAK],
        "tuned_on": "validation set, separately for every model and seed",
        "models": {m: {"label": label(m), **threshold_summary[m], "per_seed": thresholds[m],
                       "val_summary": val_tuned_summary[m]} for m in args.models},
    })

    # ---- 5. selection on validation only (both decision rules)
    def rank(summaries):
        return sorted(args.models, key=lambda m: (summaries[m][config.SELECTION_METRIC]["mean"],
                                                  summaries[m][config.SELECTION_TIEBREAK]["mean"]), reverse=True)

    def ranking(order, summaries):
        return [{"model": m, "label": label(m),
                 config.SELECTION_METRIC: summaries[m][config.SELECTION_METRIC],
                 config.SELECTION_TIEBREAK: summaries[m][config.SELECTION_TIEBREAK]} for m in order]

    val_argmax = {m: val_summary[m]["summary"] for m in args.models}
    rank_a, rank_b = rank(val_argmax), rank(val_tuned_summary)
    selection = {
        "criterion": f"highest mean validation {config.SELECTION_METRIC}, ties by {config.SELECTION_TIEBREAK}",
        "argmax_rule": {"selected_model": rank_a[0], "label": label(rank_a[0]), "ranking": ranking(rank_a, val_argmax)},
        "tuned_threshold_rule": {"selected_model": rank_b[0], "label": label(rank_b[0]),
                                 "ranking": ranking(rank_b, val_tuned_summary)},
        "decided_before_test_evaluation": True,
    }
    write_json(os.path.join(out_dir, "selection.json"), selection)
    log(f"Selected on validation: {label(rank_a[0])} (argmax rule), {label(rank_b[0])} (tuned thresholds)")

    # ---- 6. final test evaluation, once per decision rule
    _, y_test = splits["test"]
    rules = {
        "argmax": lambda m, s: (None, None),
        "tuned_thresholds": lambda m, s: (thresholds[m][str(s)]["t_high"], thresholds[m][str(s)]["t_low"]),
    }
    test, test_cms = {}, {}
    for rule, get_t in rules.items():
        test[rule], test_cms[rule] = {}, {}
        for m in args.models:
            per_seed = {}
            for r in runs[m]:
                d = load_predictions(out_dir, m, r["seed"], "test")
                pred = predict_with_thresholds(d["proba"], d["classes"], *get_t(m, r["seed"]))
                per_seed[str(r["seed"])] = compute_metrics(y_test, pred, d["proba"], d["classes"])
            test[rule][m] = {"label": label(m), "params": params[m], "summary": mean_std(list(per_seed.values())),
                             "per_seed": per_seed}
            test_cms[rule][m] = np.sum([s["confusion_matrix"] for s in per_seed.values()], axis=0).tolist()
        write_json(os.path.join(out_dir, f"test_{rule}.json"), {"decision_rule": rule, "models": test[rule]})
    write_json(os.path.join(out_dir, "legacy_80_20.json"), LEGACY_80_20)

    sizes = {m: runs[m][0].get("saved_model_size_mb") for m in args.models}
    table_keys = ["accuracy", "macro_f1", "high_risk_precision", "high_risk_recall", "high_risk_f1",
                  "high_risk_pr_auc", "macro_pr_auc", "roc_auc_ovr_macro"]

    def build_table(rule, val_summaries):
        rows = []
        for m in args.models:
            row = {"model": label(m), "params": params[m], "saved_model_size_mb": sizes[m],
                   "training_rows": runs[m][0].get("training_rows", split_info["train"]["n"]),
                   "cross_validated": uses_cv(m)}
            if rule == "tuned_thresholds":
                row.update(threshold_summary[m])
            for k in table_keys:
                if rule == "argmax":  # CV uses the argmax rule
                    row[f"cv_{k}"] = cv_summary[m]["across_seeds"][k] if uses_cv(m) else None
                row[f"val_{k}"] = val_summaries[m][k]
                row[f"test_{k}"] = test[rule][m]["summary"][k]
            if "midterm_lr" in args.models and m != "midterm_lr":
                # What the other features add over the midterm mark alone, under the same decision rule (test means)
                row["test_gain_over_midterm_only"] = {
                    k: test[rule][m]["summary"][k]["mean"] - test[rule]["midterm_lr"]["summary"][k]["mean"]
                    for k in table_keys
                }
            rows.append(row)
        return rows

    write_json(os.path.join(out_dir, "summary_table.json"), {
        "notes": [
            "All values are mean ± std over seeds. Test = final evaluation, run once after selection on validation.",
            "Table A: every model predicts by argmax of its class probabilities.",
            "Table B: every model uses the same rule with t_high and t_low tuned on the validation set, "
            "separately for each model and seed, then applied once to the test set.",
            "PR-AUC and ROC-AUC do not depend on the decision rule, so they are identical in both tables.",
            "RF (unconstrained) has no CV (final fit per seed only).",
        ],
        "table_A_argmax": {"selected_on_validation": selection["argmax_rule"]["label"],
                           "rows": build_table("argmax", val_argmax)},
        "table_B_validation_tuned_thresholds": {"selected_on_validation": selection["tuned_threshold_rule"]["label"],
                                                "rows": build_table("tuned_thresholds", val_tuned_summary)},
        "high_risk_at_tuned_threshold": [
            {"model": label(m), "t_high": threshold_summary[m]["t_high"],
             **{f"test_{k}": test["tuned_thresholds"][m]["summary"][k]
                for k in ["high_risk_precision", "high_risk_recall", "high_risk_f1"]}}
            for m in args.models
        ],
    })

    # ---- 7. figures
    labels = {m: label(m) for m in args.models}
    plots.class_distribution(split_info, fig_dir)
    keys = ["accuracy", "macro_f1", "high_risk_precision", "high_risk_recall", "high_risk_f1", "high_risk_pr_auc"]
    first_seed_proba = {}
    for m in args.models:
        d = load_predictions(out_dir, m, args.seeds[0], "test")
        first_seed_proba[m] = d["proba"][:, list(d["classes"]).index("High Risk")]
    plots.pr_curves_high_risk(y_test.to_numpy() == "High Risk", first_seed_proba, labels,
                              {m: test["argmax"][m]["summary"]["high_risk_pr_auc"]["mean"] for m in args.models},
                              args.seeds[0], fig_dir)
    cv_models = [m for m in args.models if uses_cv(m)]
    if cv_models:
        plots.metric_bars({m: cv_summary[m]["across_seeds"] for m in cv_models}, cv_models, labels, keys,
                          f"{args.folds}-fold CV on the training set, argmax rule ({len(args.seeds)} seeds)",
                          fig_dir, "cv_metrics.png")
    plots.metric_bars(val_argmax, args.models, labels, keys,
                      f"Validation set, argmax rule ({len(args.seeds)} seeds)", fig_dir, "validation_metrics_argmax.png")
    plots.metric_bars({m: test["argmax"][m]["summary"] for m in args.models}, args.models, labels, keys,
                      f"Table A: test set, argmax rule ({len(args.seeds)} seeds)", fig_dir, "test_metrics_argmax.png")
    plots.metric_bars({m: test["tuned_thresholds"][m]["summary"] for m in args.models}, args.models, labels, keys,
                      f"Table B: test set, validation-tuned thresholds ({len(args.seeds)} seeds)",
                      fig_dir, "test_metrics_tuned_thresholds.png")
    plots.confusion_matrices(test_cms["argmax"], labels, fig_dir, "confusion_matrices_test_argmax.png")
    plots.confusion_matrices(test_cms["tuned_thresholds"], labels, fig_dir,
                             "confusion_matrices_test_tuned_thresholds.png")
    if "rf_capped" in args.models:
        plots.rf_tuning_heatmap(read_json(os.path.join(out_dir, "rf_tuning.json"))["results"], fig_dir)
        imp = np.array([r["impurity_importance"]["values"] for r in runs["rf_capped"]])
        plots.importance_bars(runs["rf_capped"][0]["impurity_importance"]["features"], imp.mean(0),
                              imp.std(0, ddof=1) if len(imp) > 1 else imp.std(0),
                              "RF (capped): impurity importance, mean ± std over seeds", "Mean decrease in impurity",
                              fig_dir, "feature_importance_rf_capped.png")
    if "hgb" in args.models:
        pim = np.array([r["permutation_importance_val"]["mean"] for r in runs["hgb"]])
        plots.importance_bars(runs["hgb"][0]["permutation_importance_val"]["features"], pim.mean(0),
                              pim.std(0, ddof=1) if len(pim) > 1 else pim.std(0),
                              "HistGradientBoosting: permutation importance on validation", "Drop in macro F1",
                              fig_dir, "feature_importance_hgb_permutation.png")

    # ---- 8. operating points of the best model
    run_operating_points(splits, args.seeds, out_dir, fig_dir)

    # ---- console summary
    def fmt(s):
        return f"{s['mean']:.4f} ± {s['std']:.4f}" if s else "—"
    cols = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("high_risk_precision", "HR prec"),
            ("high_risk_recall", "HR recall"), ("high_risk_f1", "HR F1"), ("high_risk_pr_auc", "HR PR-AUC")]
    titles = {"argmax": "TABLE A: test set, argmax rule",
              "tuned_thresholds": "TABLE B: test set, thresholds tuned on validation (per model and seed)"}
    for rule in rules:
        print(f"\n{titles[rule]}, mean ± std over seeds")
        print(f"{'Model':34} " + " ".join(f"{h:>17}" for _, h in cols) + f" {'Size MB':>8}")
        for m in args.models:
            s = test[rule][m]["summary"]
            size = f"{sizes[m]:.1f}" if sizes[m] is not None else "—"
            print(f"{label(m):34} " + " ".join(f"{fmt(s[k]):>17}" for k, _ in cols) + f" {size:>8}")
    log(f"Done. Selected on validation: {label(rank_a[0])} (argmax), {label(rank_b[0])} (tuned thresholds)")


if __name__ == "__main__":
    main()
