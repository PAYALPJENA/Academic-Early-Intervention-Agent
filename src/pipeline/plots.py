"""Figures written to figures/."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_ORDER

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

CLASS_COLORS = {"High Risk": "#dc2626", "Medium Risk": "#f59e0b", "Low Risk": "#16a34a"}
MODEL_COLORS = ["#d1d5db", "#f59e0b", "#9ca3af", "#6b7280", "#93c5fd", "#60a5fa", "#2563eb"]


def _save(fig, figures_dir, name):
    path = os.path.join(figures_dir, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def class_distribution(split_info, figures_dir):
    splits = ["train", "val", "test"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    left = np.zeros(len(splits))
    for cls in CLASS_ORDER:
        shares = np.array([split_info[s]["class_share"][cls] for s in splits]) * 100
        ax.barh(splits, shares, left=left, color=CLASS_COLORS[cls], label=cls)
        for i, v in enumerate(shares):
            if v > 4:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9)
        left += shares
    ax.set_yticks(range(len(splits)), [f"{s} (n={split_info[s]['n']:,})" for s in splits])
    ax.set_xlabel("Share of students (%)")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    return _save(fig, figures_dir, "class_distribution.png")


def metric_bars(summary, models, labels, metrics, title, figures_dir, name):
    """Grouped bars of mean ± std for several metrics and models."""
    metric_names = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "balanced_accuracy": "Balanced\naccuracy",
        "high_risk_precision": "High-Risk\nprecision",
        "high_risk_recall": "High-Risk\nrecall",
        "high_risk_f1": "High-Risk\nF1",
        "high_risk_pr_auc": "High-Risk\nPR-AUC",
    }
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        means = [summary[m][k]["mean"] for k in metrics]
        stds = [summary[m][k]["std"] for k in metrics]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, means, width, yerr=stds, capsize=2,
               label=labels[m], color=MODEL_COLORS[i % len(MODEL_COLORS)])
    ax.set_xticks(x, [metric_names.get(k, k) for k in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean ± std over seeds)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, ncol=min(len(models), 5), loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8.5)
    return _save(fig, figures_dir, name)


def confusion_matrices(cms, labels, figures_dir, name="confusion_matrices_test.png"):
    """Row-normalised confusion matrices, summed over seeds."""
    models = list(cms)
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.4))
    axes = np.atleast_1d(axes)
    short = ["High", "Medium", "Low"]
    for ax, m in zip(axes, models):
        cm = np.array(cms[m], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "#1f2937", fontsize=9)
        ax.set_xticks(range(3), short)
        ax.set_yticks(range(3), short if ax is axes[0] else [])
        ax.set_xlabel("Predicted")
        ax.set_title(labels[m], fontsize=9.5)
    axes[0].set_ylabel("Actual")
    return _save(fig, figures_dir, name)


def rf_tuning_heatmap(tuning, figures_dir, metric="high_risk_f1"):
    depths = sorted({r["params"]["max_depth"] for r in tuning})
    leaves = sorted({r["params"]["min_samples_leaf"] for r in tuning})
    grid = np.zeros((len(depths), len(leaves)))
    for r in tuning:
        grid[depths.index(r["params"]["max_depth"]), leaves.index(r["params"]["min_samples_leaf"])] = r["val"][metric]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(grid, cmap="Blues")
    for i in range(len(depths)):
        for j in range(len(leaves)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if grid[i, j] > grid.mean() else "#1f2937")
    ax.set_xticks(range(len(leaves)), leaves)
    ax.set_yticks(range(len(depths)), depths)
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("max_depth")
    ax.set_title(f"RF tuning: validation {metric.replace('_', ' ')}", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, figures_dir, "rf_tuning_validation.png")


def pr_curves_high_risk(is_high, scores, labels, pr_auc_means, seed, figures_dir):
    """High-Risk precision-recall curves on the test set for one seed."""
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    constant = []
    for i, (m, s) in enumerate(scores.items()):
        if np.ptp(s) == 0:  # constant score (majority baseline): its PR curve is the prevalence line
            constant.append(m)
            continue
        precision, recall, _ = precision_recall_curve(is_high, s)
        ax.plot(recall, precision, color=MODEL_COLORS[i % len(MODEL_COLORS)], lw=1.8, drawstyle="steps-post",
                label=f"{labels[m]} (PR-AUC {pr_auc_means[m]:.3f})")
    chance = " = ".join([labels[m] for m in constant] + [f"prevalence {is_high.mean():.3f}"])
    ax.axhline(is_high.mean(), color="#9ca3af", lw=1.2, ls="--", label=chance)
    ax.set_xlabel("High-Risk recall")
    ax.set_ylabel("High-Risk precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"High-Risk precision–recall on the test set (seed {seed}; PR-AUC = mean over seeds)",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=8, loc="lower left")
    return _save(fig, figures_dir, "pr_curves_high_risk_test.png")


def operating_points_plot(curve_val, curve_test, summary, model_label, seed, figures_dir):
    """Recall vs precision for one model, with the three validation-chosen operating points on test."""
    from sklearn.metrics import precision_recall_curve

    styles = {
        "f1_optimal": ("F1-optimal", "o", "#2563eb"),
        "recall_ge_0.70": ("Recall ≥ 0.70 (on validation)", "s", "#f59e0b"),
        "recall_ge_0.80": ("Recall ≥ 0.80 (on validation)", "^", "#dc2626"),
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for (y, p), ls, name in [(curve_val, "--", "validation"), (curve_test, "-", "test")]:
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, ls, color="#6b7280" if name == "validation" else "#1f2937", lw=1.3,
                label=f"{model_label}, {name} (seed {seed})")
    for key, (name, marker, color) in styles.items():
        s = summary[key]
        t, r, pr = s["threshold"], s["test"]["recall"], s["test"]["precision"]
        ax.errorbar(r["mean"], pr["mean"], xerr=r["std"], yerr=pr["std"], fmt=marker, ms=9, color=color,
                    mec="white", mew=1.2, capsize=3, zorder=5,
                    label=f"{name}: t = {t['mean']:.3f} → test P {pr['mean']:.3f}, R {r['mean']:.3f}")
    ax.set_xlabel("High-Risk recall")
    ax.set_ylabel("High-Risk precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("High-Risk operating points (thresholds chosen on validation; markers = test mean ± std over seeds)",
                 loc="left", fontsize=8.5)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.5, loc="lower left")
    return _save(fig, figures_dir, "operating_points_high_risk.png")


def importance_bars(names, means, stds, title, xlabel, figures_dir, name, top=15):
    order = np.argsort(means)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1))
    ax.barh([names[i] for i in order], [means[i] for i in order], xerr=[stds[i] for i in order],
            color="#2563eb", capsize=2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10)
    return _save(fig, figures_dir, name)
