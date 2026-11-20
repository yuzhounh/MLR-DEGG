"""Figure of the ERP CORE N170 confirmation result: per-participant differences to group lasso.

One row per method.  Panel a: accuracy difference (percentage points) under the min rule; panel b: difference of
the split-half reproducibility R measured on fresh halves under the 1se-repro rule.  One dot per participant
(37 participants, mean over the five outer folds), the black vertical bar is the mean, and the label at the
right of each row gives the Holm-corrected p of the paired Wilcoxon test (09_selection_rules_report.py).  The
method names are written once, at the left of panel a.

Usage:
    python code/scripts/13_erp_figure.py [--name erpcore_n170_37]

Output: manuscript/figures/erp_differences.pdf (vector, Arial embedded).
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import ScaledTranslation

from mlrdegg.paths import PROJECT_ROOT, RESULTS_ROOT

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 8,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})

METHODS = [("l2_logreg", "Ridge", "#008300"), ("graphnet", "GraphNet", "#2a78d6"), ("graph_tv", "Graph TV", "#eb6834"),
           ("graph_tv_tm", "Temporal\ngraph TV", "#1baf7a"), ("superquadratic", "Superquadratic", "#eda100"),
           ("nonconvex", "Nonconvex", "#e87ba4")]
INK, MUTED = "#0b0b0b", "#52514e"


def panel_title(ax, letter: str, title: str, pad: float = 6.0, indent: float = 13.0) -> None:
    """Bold panel letter at the y axis (left edge of the frame) and the regular-weight title to its right."""
    t = ax.set_title(title, loc="left", fontsize=8, color=INK, pad=pad)
    t.set_transform(t.get_transform() + ScaledTranslation(indent / 72, 0, ax.figure.dpi_scale_trans))
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, pad), textcoords="offset points",
                fontsize=11, fontweight="bold", color=INK, ha="left", va="baseline")


def per_subject(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = RESULTS_ROOT / name
    acc = pd.read_csv(out / "outer_results.csv").groupby(["method", "unit"]).accuracy.mean().unstack(0)
    rep = pd.read_csv(out / "fresh_repro_seed5003.csv").groupby(["method", "unit"]).fresh_repro.mean().unstack(0)
    return acc, rep


def panel(ax, table: pd.DataFrame, suffix: str, scale: float, tests: pd.DataFrame, metric: str, xlabel: str,
          names: bool) -> None:
    rng = np.random.default_rng(0)
    ref = table["group_lasso" + suffix]
    allv = []
    for k, (m, label, color) in enumerate(METHODS):
        d = scale * (table[m + suffix] - ref).dropna()
        allv.append(d)
        ax.scatter(d, k + rng.uniform(-0.18, 0.18, len(d)), s=9, color=color, alpha=0.8,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.plot([d.mean()] * 2, [k - 0.28, k + 0.28], color=INK, lw=1.5, solid_capstyle="butt", zorder=4)   # mean: short vertical bar
    allv = pd.concat(allv)
    lo, hi = allv.min(), allv.max()
    span = hi - lo
    lo -= 0.04 * span
    hi += 0.04 * span                                     # data end; the p-value zone follows to the right
    zone0, zone1 = hi + 0.03 * span, hi + 0.03 * span + 0.25 * span
    ax.set_xlim(lo, zone1 + 0.03 * span)
    row = tests.set_index(["rule", "method", "metric"])
    rule = "1se_repro" if suffix else "min"
    box = FancyBboxPatch((zone0, -0.3), zone1 - zone0, len(METHODS) - 0.4, boxstyle="round,pad=0,rounding_size=0.1",
                         fill=False, edgecolor="#c9c8c4", lw=0.6, zorder=1)       # corner radius made circular in main()
    ax.add_patch(box)
    ax._pbox = box
    for k, (m, _, _) in enumerate(METHODS):
        p = row.loc[(rule, m, metric), "p_holm"]
        txt = "p < 0.001" if p < 1e-3 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"
        ax.text(zone0 + 0.03 * span, k, txt, ha="left", va="center", fontsize=7, color=INK)   # left-aligned in the box
    ax.plot([0, 0], [-0.5, len(METHODS) - 0.5], color=MUTED, lw=0.6, zorder=1)
    ax.set_yticks(range(len(METHODS)), [m[1] for m in METHODS] if names else [])
    ax.set_ylim(len(METHODS) - 0.5, -0.5)                 # first method at the top
    ax.tick_params(axis="y", length=0 if not names else 2, labelsize=7.5, colors=INK)
    ax.set_xlabel(xlabel, color=INK)                      # all four spines stay: a closed frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="erpcore_n170_37")
    args = ap.parse_args()
    acc, rep = per_subject(args.name)
    report = RESULTS_ROOT / args.name / "report"
    tests = pd.read_csv(report / ("tests_all.csv" if (report / "tests_all.csv").exists() else "tests_confirmation.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.125, right=0.995, top=0.88, bottom=0.15, wspace=0.06)
    panel(axes[0], acc, "", 100.0, tests, "accuracy", "Accuracy difference to group lasso (points)", names=True)
    panel(axes[1], rep, "@1se_repro", 1.0, tests, "R", "Reproducibility R difference to group lasso", names=False)
    fig.canvas.draw()
    for ax in axes:                                       # round corners of 4 pt radius, circular in screen space
        w, h = ax.get_window_extent().width, ax.get_window_extent().height
        xs, ys = np.diff(ax.get_xlim())[0], abs(np.diff(ax.get_ylim())[0])
        r_px = 4.0 * fig.dpi / 72
        ax._pbox.set_boxstyle("round,pad=0,rounding_size=%g" % (r_px * xs / w))
        ax._pbox.set_mutation_aspect((ys / h) / (xs / w))
    panel_title(axes[0], "a", "Accuracy, min rule")
    panel_title(axes[1], "b", "Map reproducibility, 1se-repro rule")
    dest = PROJECT_ROOT / "manuscript" / "figures"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / "erp_differences.pdf")
    print("wrote", dest / "erp_differences.pdf")


if __name__ == "__main__":
    main()
