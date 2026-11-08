"""
plots.py - the five mandatory figures (+ two data-overview charts).

Every figure has a title, labelled axes, and names the dataset, model and method.
Figures are saved as PNG files in results/ by make_all_figures().
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no display needed (works on servers)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from experiments import METHOD_NAMES

COLORS = {"SelectKBest": "#0072B2", "RFE": "#D55E00", "All features": "#555555"}
GREY = "#B8B8B8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _title(res: dict, what: str, with_method: bool = True) -> str:
    second = f"{res['label']}  |  {res['model']}"
    if with_method:
        second += f"  |  {res['method']} (k={res['k']}/{res['n_features']})"
    return f"{what}\n{second}"


# --------------------------------------------------------------------------- #
# Figure 1 - accuracy vs number of features
# --------------------------------------------------------------------------- #
def fig_accuracy_vs_k(res):
    sw, base = res["sweep"], res["baseline"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in METHOD_NAMES:
        s = sw[sw.method == m].sort_values("k")
        ax.plot(s.k, s.cv_mean, marker="o", ms=4, lw=1.8, color=COLORS[m], label=m)
        ax.fill_between(s.k, s.cv_mean - s.cv_std, s.cv_mean + s.cv_std, color=COLORS[m], alpha=0.12)
        kb = res["cv_best_k"][m]
        ax.scatter([kb], [s[s.k == kb].cv_mean.iloc[0]], s=170, marker="*", color=COLORS[m],
                   edgecolor="black", zorder=5)
    ax.axhline(base["cv_mean"], ls="--", color=COLORS["All features"],
               label=f"All features ({res['n_features']})")
    ax.axvline(res["k"], ls=":", color="black", alpha=0.6, label=f"Selected k = {res['k']}")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", ls="", color="white", markeredgecolor="black",
                          markerfacecolor="gold", ms=12, label="CV-best k (per method)"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Number of selected features (k)")
    ax.set_ylabel("Cross-validation accuracy (5-fold, training split)")
    ax.set_title(_title(res, "Figure 1: Accuracy vs number of features", with_method=False))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 - baseline vs optimized
# --------------------------------------------------------------------------- #
def fig_baseline_vs_optimized(res):
    base, opt = res["baseline"], res["optimized"]
    metrics = ["Accuracy", "Macro-F1"]
    b = [base["test_acc"], base["test_f1"]]
    o = [opt["test_acc"], opt["test_f1"]]
    x, w = np.arange(2), 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    r1 = ax.bar(x - w / 2, b, w, color=COLORS["All features"], label=f"All features ({res['n_features']})")
    r2 = ax.bar(x + w / 2, o, w, color=COLORS[res["method"]], label=f"{res['method']} ({res['k']})")
    for bars in (r1, r2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, f"{r.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score on held-out test set")
    ax.set_xlabel("Metric")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", visible=False)
    ax.set_title(_title(res, "Figure 2: Baseline vs optimized"))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 - feature importance / scores
# --------------------------------------------------------------------------- #
def fig_feature_scores(res):
    opt = res["optimized"]
    s = opt["scores"].sort_values()                       # ascending -> best bar on top
    colors = [COLORS[res["method"]] if f in opt["features"] else GREY for f in s.index]
    fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.3 * len(s) + 1.6)))
    ax.barh(s.index, s.values, color=colors)
    ax.set_xlabel(opt["score_label"])
    ax.set_ylabel("Input feature")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=COLORS[res["method"]], label="Selected"),
                       Line2D([], [], marker="s", ls="", color=GREY, label="Not selected")],
              fontsize=8, loc="lower right")
    ax.set_title(_title(res, "Figure 3: Feature scores"))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 - confusion matrix (held-out test predictions)
# --------------------------------------------------------------------------- #
def fig_confusion_matrix(res):
    cm, names = res["optimized"]["cm"], res["class_names"]
    size = max(4.8, 1.2 * len(names) + 2.4)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=names, yticklabels=names, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")
    ax.grid(False)
    ax.set_title(_title(res, "Figure 4: Confusion matrix (test set)"))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 5 - accuracy vs runtime
# --------------------------------------------------------------------------- #
def fig_accuracy_vs_runtime(res):
    sw, base = res["sweep"], res["baseline"]
    n = res["n_features"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in METHOD_NAMES:
        s = sw[sw.method == m]
        ax.scatter(s.fit_time, s.cv_mean, s=34, color=COLORS[m], alpha=0.85, label=m)
        for _, r in s.iterrows():
            show = n <= 13 or r.k in (res["k"], res["cv_best_k"][m])
            if show:
                ax.annotate(f"{int(r.k)}", (r.fit_time, r.cv_mean), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=COLORS[m])
    ax.scatter([base["fit_time"]], [base["cv_mean"]], marker="*", s=200, color=COLORS["All features"],
               edgecolor="black", zorder=5, label=f"All features ({n})")
    o = res["optimized"]
    ax.scatter([o["fit_time"]], [o["cv_mean"]], s=170, facecolor="none", edgecolor="black", lw=1.6,
               zorder=6, label=f"Selected: {res['method']} k={res['k']}")
    ax.set_xlabel("Mean training time per CV fold (seconds)")
    ax.set_ylabel("Cross-validation accuracy")
    ax.legend(fontsize=8, loc="best")
    ax.set_title(_title(res, "Figure 5: Accuracy vs runtime (labels = k)", with_method=False))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Extra data-overview charts (shown in the app's "Data overview" tab)
# --------------------------------------------------------------------------- #
def fig_class_distribution(ds):
    counts = ds.y.value_counts().sort_index()
    labels = [ds.class_names[i] for i in counts.index]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(labels, counts.values, color="#0072B2")
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{int(r.get_height())}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Class distribution\n{ds.label}")
    plt.setp(ax.get_xticklabels(), rotation=15 if len(labels) > 2 else 0, ha="right" if len(labels) > 2 else "center")
    fig.tight_layout()
    return fig


def fig_correlation(ds):
    corr = ds.X.corr()
    n = corr.shape[0]
    size = min(12, max(5, 0.45 * n + 2))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=n <= 13, fmt=".2f",
                annot_kws={"size": 7}, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.grid(False)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    ax.set_title(f"Feature correlation heatmap\n{ds.label}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Save everything
# --------------------------------------------------------------------------- #
FIGURES = {
    "fig1_accuracy_vs_k": fig_accuracy_vs_k,
    "fig2_baseline_vs_optimized": fig_baseline_vs_optimized,
    "fig3_feature_scores": fig_feature_scores,
    "fig4_confusion_matrix": fig_confusion_matrix,
    "fig5_accuracy_vs_runtime": fig_accuracy_vs_runtime,
}
_BOTH_METHODS = {"fig1_accuracy_vs_k", "fig5_accuracy_vs_runtime"}   # these show both methods, so no method in the file name


def make_all_figures(res: dict, outdir) -> dict:
    """Draw the 5 mandatory figures for one experiment and save them as PNGs. Returns {key: Path}."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, fn in FIGURES.items():
        if key in _BOTH_METHODS:
            name = f"{slug(res['label'])}__{slug(res['model'])}__{key}.png"
        else:
            name = f"{slug(res['label'])}__{slug(res['model'])}__{slug(res['method'])}_k{res['k']}__{key}.png"
        fig = fn(res)
        fig.savefig(outdir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths[key] = outdir / name
    return paths
