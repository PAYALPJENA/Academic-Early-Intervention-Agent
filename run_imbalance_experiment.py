"""Train an isolated class-imbalance comparison for CatBoost and HistGradientBoosting.

This script reads the same raw CSV, label logic, stratified 70/15/15 split,
preprocessors, model builders, metrics, and validation threshold tuner as the main
pipeline. All experiment artifacts are written under new imbalance-specific paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import config  # noqa: E402
from pipeline.data import load_dataset, split_dataset, stratified_sample  # noqa: E402
from pipeline.metrics import compute_metrics, mean_std  # noqa: E402
from pipeline.models import build_model, label, predict_with_thresholds  # noqa: E402
from pipeline.run_pipeline import resolve_device, tune_thresholds  # noqa: E402


MODELS = ("catboost", "hgb")
SETTINGS = ("current", "class_weight", "smote", "medium_undersample")
SETTING_LABELS = {
    "current": "Current (no resampling)",
    "class_weight": "Inverse-frequency sample weights",
    "smote": "SMOTE after preprocessing",
    "medium_undersample": "Medium Risk undersampled",
}
METRIC_KEYS = (
    "high_risk_precision",
    "high_risk_recall",
    "high_risk_f1",
    "high_risk_pr_auc",
    "macro_f1",
)
_DEADLINE: float | None = None


class TimeBudgetReached(Exception):
    """Raised between fit units when the requested time budget is exhausted."""


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def write_json(path: Path, payload: dict) -> None:
    """Atomically create/update a checkpoint inside the new imbalance namespace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp_path, path)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_budget(unit: str) -> None:
    if _DEADLINE is not None and time.time() >= _DEADLINE:
        raise TimeBudgetReached(unit)


def class_counts(y) -> dict[str, int]:
    values = np.asarray(y)
    return {cls: int(np.count_nonzero(values == cls)) for cls in config.CLASS_ORDER}


def count_rows(values) -> int:
    return int(values.shape[0])


def prepare_training_rows(X_fit, y_fit, setting: str, random_state: int):
    """Resample/weight only a preprocessed training matrix and its training labels."""
    y_fit = np.asarray(y_fit)
    before = class_counts(y_fit)
    sample_weight = None
    result_X, result_y = X_fit, y_fit

    if setting == "class_weight":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    elif setting == "smote":
        sampler = SMOTE(random_state=random_state, k_neighbors=5)
        result_X, result_y = sampler.fit_resample(X_fit, y_fit)
    elif setting == "medium_undersample":
        # Keep Medium Risk at the larger of the two other class counts.
        # This reduces the majority class without oversampling either minority class.
        target_medium = max(before["High Risk"], before["Low Risk"])
        if before["Medium Risk"] > target_medium:
            sampler = RandomUnderSampler(
                sampling_strategy={"Medium Risk": target_medium},
                random_state=random_state,
            )
            result_X, result_y = sampler.fit_resample(X_fit, y_fit)

    audit = {
        "input_rows": int(len(y_fit)),
        "output_rows": int(len(result_y)),
        "class_counts_before": before,
        "class_counts_after": class_counts(result_y),
        "sample_weighted": sample_weight is not None,
        "resampler_received_validation_or_test": False,
    }
    if sample_weight is not None:
        weight_by_class = {}
        for cls in config.CLASS_ORDER:
            cls_weights = sample_weight[y_fit == cls]
            weight_by_class[cls] = float(cls_weights[0]) if len(cls_weights) else None
        audit["per_class_sample_weight"] = weight_by_class
    return result_X, np.asarray(result_y), sample_weight, audit


def new_model(model_name: str, seed: int, device: str, params: dict):
    model_pipeline = build_model(model_name, seed, device=device, **params)
    return model_pipeline.named_steps["preprocessor"], model_pipeline.named_steps["model"]


def fit_fold(model_name, setting, seed, device, params, X_fit, y_fit, X_eval, y_eval, sampler_seed):
    """Fit preprocessing and model on a CV training fold; score its held-out fold."""
    prep, estimator = new_model(model_name, seed, device, params)
    Z_fit = prep.fit_transform(X_fit)
    Z_eval = prep.transform(X_eval)
    Z_resampled, y_resampled, sample_weight, audit = prepare_training_rows(
        Z_fit, y_fit, setting, sampler_seed
    )
    started = time.time()
    if sample_weight is None:
        estimator.fit(Z_resampled, y_resampled)
    else:
        estimator.fit(Z_resampled, y_resampled, sample_weight=sample_weight)
    elapsed = time.time() - started
    proba = estimator.predict_proba(Z_eval)
    pred = predict_with_thresholds(proba, estimator.classes_)
    metrics = compute_metrics(y_eval, pred, proba, estimator.classes_)
    return {
        "metrics_argmax": metrics,
        "fit_seconds": float(elapsed),
        "fit_rows": audit,
        "evaluation_rows": int(len(y_eval)),
        "preprocessor_fit_rows": int(len(y_fit)),
    }


def load_model_params() -> dict[str, dict]:
    tuning_path = config.RESULTS_DIR
    with (Path(tuning_path) / "tuning_catboost.json").open(encoding="utf-8") as handle:
        catboost_tuning = json.load(handle)
    return {"catboost": dict(catboost_tuning["best_params"]), "hgb": {}}


def signature_for(device, seeds, folds, sample_size, params, smoke):
    data_path = Path(config.RAW_DATA_PATH)
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "dataset_sha256": data_hash,
        "split_seed": config.SPLIT_SEED,
        "seeds": list(seeds),
        "folds": int(folds),
        "sample_size": sample_size,
        "device": device,
        "params": params,
        "smoke": bool(smoke),
        "medium_undersampling": "Medium Risk reduced to max(High Risk, Low Risk) training count",
        "smote": "after fold-train preprocessing; one-hot features can become fractional",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], payload


def run_seed(model_name, setting, seed, device, params, splits, folds, checkpoint_root):
    seed_path = checkpoint_root / "seeds" / model_name / setting / f"seed_{seed}.json"
    if seed_path.exists():
        existing = read_json(seed_path)
        log(f"Reusing completed {model_name}/{setting} seed {seed}")
        return existing

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    cv_records = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (fit_pos, eval_pos) in enumerate(skf.split(X_train, y_train)):
        fold_path = (
            checkpoint_root / "folds" / model_name / setting
            / f"seed_{seed}_fold_{fold + 1}.json"
        )
        if fold_path.exists():
            fold_record = read_json(fold_path)
            log(f"Reusing {model_name}/{setting} seed {seed} fold {fold + 1}/{folds}")
        else:
            check_budget(f"{model_name}/{setting} seed {seed} CV fold {fold + 1}")
            fold_seed = int(seed * 100 + fold + 1)
            fold_record = fit_fold(
                model_name, setting, seed, device, params,
                X_train.iloc[fit_pos], y_train.iloc[fit_pos],
                X_train.iloc[eval_pos], y_train.iloc[eval_pos], fold_seed,
            )
            fold_record.update({
                "model": model_name,
                "setting": setting,
                "seed": int(seed),
                "fold": int(fold + 1),
                "sampler_seed": fold_seed,
                "params": params,
            })
            write_json(fold_path, fold_record)
            log(
                f"{model_name}/{setting} seed {seed} fold {fold + 1}/{folds}: "
                f"High-Risk F1 {fold_record['metrics_argmax']['high_risk_f1']:.4f} "
                f"({fold_record['fit_seconds']:.1f}s; rows "
                f"{fold_record['fit_rows']['input_rows']}->{fold_record['fit_rows']['output_rows']})"
            )
        cv_records.append(fold_record)

    check_budget(f"{model_name}/{setting} seed {seed} final fit")

    # The final preprocessor is fit on training rows only. The selected resampler
    # receives only the transformed training rows and labels.
    prep, estimator = new_model(model_name, seed, device, params)
    Z_train = prep.fit_transform(X_train)
    Z_val = prep.transform(X_val)
    Z_resampled, y_resampled, sample_weight, final_audit = prepare_training_rows(
        Z_train, y_train, setting, int(seed * 100 + 99)
    )
    final_audit["preprocessor_fit_rows"] = int(len(y_train))
    final_audit["validation_rows"] = int(len(y_val))
    final_audit["validation_resampled"] = False
    final_audit["test_rows_before_threshold_selection"] = 0

    fit_started = time.time()
    if sample_weight is None:
        estimator.fit(Z_resampled, y_resampled)
    else:
        estimator.fit(Z_resampled, y_resampled, sample_weight=sample_weight)
    fit_seconds = time.time() - fit_started

    # Threshold selection reads only final-fit validation probabilities/labels.
    val_proba = estimator.predict_proba(Z_val)
    val_argmax_pred = predict_with_thresholds(val_proba, estimator.classes_)
    thresholds = tune_thresholds(val_proba, estimator.classes_, y_val)
    val_tuned_pred = predict_with_thresholds(
        val_proba, estimator.classes_, thresholds["t_high"], thresholds["t_low"]
    )
    validation = {
        "argmax": compute_metrics(y_val, val_argmax_pred, val_proba, estimator.classes_),
        "tuned_thresholds": compute_metrics(y_val, val_tuned_pred, val_proba, estimator.classes_),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
    }

    # Test is not transformed or scored until validation thresholds are fixed.
    check_budget(f"{model_name}/{setting} seed {seed} test score")
    X_test, y_test = splits["test"]
    Z_test = prep.transform(X_test)
    test_proba = estimator.predict_proba(Z_test)  # one test probability pass per seed/setting
    test_argmax = predict_with_thresholds(test_proba, estimator.classes_)
    test_tuned = predict_with_thresholds(
        test_proba, estimator.classes_, thresholds["t_high"], thresholds["t_low"]
    )
    test = {
        "argmax": compute_metrics(y_test, test_argmax, test_proba, estimator.classes_),
        "tuned_thresholds": compute_metrics(y_test, test_tuned, test_proba, estimator.classes_),
    }

    fold_metrics = [r["metrics_argmax"] for r in cv_records]
    cv_fold_summary = mean_std(fold_metrics)
    cv_seed_mean = {metric: values["mean"] for metric, values in cv_fold_summary.items()}
    record = {
        "model": model_name,
        "label": label(model_name),
        "setting": setting,
        "setting_label": SETTING_LABELS[setting],
        "seed": int(seed),
        "params": params,
        "device": device,
        "cv_folds": cv_records,
        "cv_mean_over_folds_argmax": cv_seed_mean,
        "cv_fold_summary_argmax": cv_fold_summary,
        "validation": validation,
        "test": test,
        "thresholds_tuned_on": "validation only",
        "test_probability_passes": 1,
        "final_fit_seconds": float(fit_seconds),
        "final_training_audit": final_audit,
        "test_scored_after_validation_thresholds": True,
    }
    write_json(seed_path, record)
    log(
        f"{model_name}/{setting} seed {seed}: tuned thresholds "
        f"{thresholds['t_high']:.2f}/{thresholds['t_low']:.2f}; test tuned High-Risk F1 "
        f"{test['tuned_thresholds']['high_risk_f1']:.4f}"
    )
    return record


def aggregate(seed_records, models, seeds, device, params, folds, sample_size, smoke,
              run_signature, split_counts):
    output = {
        "schema_version": 1,
        "experiment": "class-imbalance supplementary experiment",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "split": "same stratified 70/15/15 split as the main pipeline",
            "split_seed": config.SPLIT_SEED,
            "split_counts": split_counts,
            "run_signature": run_signature,
            "seeds": list(seeds),
            "cv_folds": int(folds),
            "device": device,
            "test_set_scored_once_per_model_setting_seed": True,
            "thresholds": "t_high and t_low tuned on validation only; applied to the test probabilities",
            "preprocessing": "fit on each CV training fold; final preprocessor fit on the full training split",
            "resampling": "training rows only, after preprocessing, within each CV fold and for the final fit",
            "smote_limitation": (
                "SMOTE is applied after one-hot encoding; synthetic rows can have fractional one-hot values. "
                "This keeps the same numeric preprocessing interface across both models."
            ),
            "medium_undersampling": (
                "RandomUnderSampler reduces Medium Risk to the larger of the High Risk and Low Risk counts "
                "in that training fold/split."
            ),
            "class_weight": "per-row sample_weight from sklearn balanced inverse-frequency formula",
            "test_leakage_check": {
                "target_source_excluded_from_features": "Final_Exam_Score is dropped by pipeline.data.load_dataset",
                "validation_used_for_thresholds": True,
                "test_used_for_training_resampling_or_threshold_tuning": False,
                "test_probability_passes_per_seed_setting": 1,
                "preprocessor_fit_on_test": False,
            },
            "smoke_run": bool(smoke),
            "sample_rows": sample_size,
        },
        "models": {},
    }
    for model_name in models:
        output["models"][model_name] = {
            "label": label(model_name),
            "params": params[model_name],
            "settings": {},
        }
        for setting in SETTINGS:
            records = seed_records[model_name][setting]
            output["models"][model_name]["settings"][setting] = {
                "label": SETTING_LABELS[setting],
                "n_seeds": len(records),
                "cv_argmax": mean_std([r["cv_mean_over_folds_argmax"] for r in records]),
                "test_argmax": mean_std([r["test"]["argmax"] for r in records]),
                "test_tuned_thresholds": mean_std([r["test"]["tuned_thresholds"] for r in records]),
                "thresholds": mean_std([r["validation"]["thresholds"] for r in records]),
                "per_seed": {str(r["seed"]): r for r in records},
            }
    return output


def save_chart(result: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 5), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    positions = np.arange(len(SETTINGS))
    width = 0.34
    for ax, model_name in zip(axes, MODELS):
        settings = result["models"][model_name]["settings"]
        argmax = [settings[s]["test_argmax"]["high_risk_f1"] for s in SETTINGS]
        tuned = [settings[s]["test_tuned_thresholds"]["high_risk_f1"] for s in SETTINGS]
        ax.bar(
            positions - width / 2,
            [v["mean"] for v in argmax],
            width,
            yerr=[v["std"] for v in argmax],
            capsize=3,
            label="Argmax",
            color="#6574cd",
        )
        ax.bar(
            positions + width / 2,
            [v["mean"] for v in tuned],
            width,
            yerr=[v["std"] for v in tuned],
            capsize=3,
            label="Validation-tuned thresholds",
            color="#e87945",
        )
        ax.set_title(label(model_name))
        ax.set_xticks(positions, [SETTING_LABELS[s] for s in SETTINGS], rotation=24, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Test High-Risk F1 (mean ± std over seeds)")
    fig.suptitle("Class-imbalance settings: High-Risk F1", x=0.08, ha="left", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temp_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    os.replace(temp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--time-budget", type=float, default=None, help="Minutes; stops between fit units")
    parser.add_argument("--seeds", nargs="+", type=int, default=config.EXPERIMENT_SEEDS)
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    parser.add_argument("--sample", type=int, default=None, help="Stratified subsample, intended for smoke runs")
    parser.add_argument("--smoke", action="store_true", help="Use one seed, two folds and a small subsample")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "imbalance")
    parser.add_argument("--aggregate-path", type=Path, default=ROOT / "results" / "imbalance_experiment.json")
    parser.add_argument("--figure-path", type=Path, default=ROOT / "figures" / "imbalance_comparison.png")
    args = parser.parse_args()

    if args.time_budget is not None and args.time_budget > 0:
        global _DEADLINE
        _DEADLINE = time.time() + 60 * args.time_budget
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.sample is not None and args.sample < 500:
        parser.error("--sample must be at least 500 to retain all classes in the smoke run")

    device = resolve_device(args.device)
    seeds = [0] if args.smoke else list(args.seeds)
    folds = min(args.folds, 2) if args.smoke else args.folds
    sample_size = args.sample if args.sample is not None else (6000 if args.smoke else None)
    params = load_model_params()
    if args.smoke:
        params["catboost"]["iterations"] = min(25, int(params["catboost"].get("iterations", 25)))

    X, y = load_dataset(check_labels=True)
    if sample_size is not None:
        X, y = stratified_sample(X, y, sample_size, seed=config.SPLIT_SEED)
    splits = split_dataset(X, y)
    signature, signature_payload = signature_for(device, seeds, folds, sample_size, params, args.smoke)
    if not args.smoke and args.aggregate_path.exists():
        existing = read_json(args.aggregate_path)
        if existing.get("protocol", {}).get("run_signature") != signature:
            raise FileExistsError(
                f"Existing aggregate belongs to another run; refusing to overwrite: {args.aggregate_path}"
            )
    experiment_root = args.results_root / "smoke" if args.smoke else args.results_root
    checkpoint_root = experiment_root / "checkpoints" / device / signature
    manifest_path = checkpoint_root / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != signature_payload:
        raise ValueError(f"Checkpoint manifest mismatch: {manifest_path}")
    if not manifest_path.exists():
        write_json(manifest_path, signature_payload)

    log(
        f"Starting {'SMOKE ' if args.smoke else ''}imbalance run: device={device}, "
        f"seeds={seeds}, folds={folds}, rows={len(X)}, "
        f"split={len(splits['train'][1])}/{len(splits['val'][1])}/{len(splits['test'][1])}"
    )
    seed_records = {m: {s: [] for s in SETTINGS} for m in MODELS}
    try:
        for model_name in MODELS:
            for setting in SETTINGS:
                for seed in seeds:
                    record = run_seed(
                        model_name, setting, seed, device, params[model_name],
                        splits, folds, checkpoint_root,
                    )
                    seed_records[model_name][setting].append(record)
    except TimeBudgetReached as stop:
        log(f"Time budget reached before {stop}; checkpoints are ready to resume with the same command.")
        return 0

    split_counts = {
        split_name: {"rows": int(len(y_part)), "class_counts": class_counts(y_part)}
        for split_name, (_, y_part) in splits.items()
    }
    result = aggregate(
        seed_records, MODELS, seeds, device, params, folds, sample_size,
        args.smoke, signature, split_counts,
    )
    if args.smoke:
        smoke_path = experiment_root / f"smoke_summary_{signature}.json"
        write_json(smoke_path, result)
        log(f"Smoke complete. Training-only resampling audit saved to {smoke_path}")
        return 0

    # Final deliverables are created once, only after all seed checkpoints exist.
    if args.aggregate_path.exists():
        existing = read_json(args.aggregate_path)
        if existing.get("protocol", {}).get("run_signature") != signature:
            raise FileExistsError(f"Refusing to overwrite existing aggregate: {args.aggregate_path}")
        log(f"Keeping existing aggregate unchanged: {args.aggregate_path}")
    else:
        write_json(args.aggregate_path, result)
    if args.figure_path.exists():
        log(f"Keeping existing figure unchanged: {args.figure_path}")
    else:
        save_chart(result, args.figure_path)
    log(f"Wrote {args.aggregate_path} and {args.figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
