"""Figures for the 2D shape simulation: true versus estimated class maps and temporal profiles.

Usage:
    python code/scripts/06_plot_simulation_maps.py --results results/sim_shapes [--seed 2026]
        [--rule 1se_sel_first] [--maps selected_maps.npz --summary summary.csv]

Reads ``config.yaml``, ``simulated_data_seed<seed>.npz``, ``selected_maps.npz`` and
``summary.csv`` written by 05_simulation_benchmark.py (``layout: shapes2d``) and writes
``maps_seed<seed>_<rule>.png`` (one row per method, one column per class plus the selected pixels)
and ``profiles_seed<seed>_<rule>.png`` (temporal profile of each class inside its true region).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from mlrdegg.paths import resolve  # noqa: E402

# Reference palette (dataviz skill): surface and ink, categorical slots in fixed order,
# blue <-> red diverging pair with a neutral gray midpoint.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f19a98", "#e34948", "#9e2a2a"])
SELECTED = ListedColormap([SURFACE, "#52514e"])
LABELS = {"l2_logreg": "L2 logistic", "group_lasso": "group lasso", "graph_tv": "graph TV  (p_g=1)",
          "graph_p15": "p_g=1.5", "graphnet": "GraphNet  (p_g=2)", "superquadratic": "p_g=3",
          "nonconvex": "nonconvex  (p=1/2)"}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "font.size": 9, "axes.titlesize": 9})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--rule", default="1se_sel_first")
    ap.add_argument("--maps", default="selected_maps.npz", help="maps file inside --results")
    ap.add_argument("--summary", default="summary.csv", help="per-fit statistics inside --results")
    args = ap.parse_args()

    res = resolve(args.results)
    cfg = _bootstrap.load_config(res / "config.yaml")
    summary = pd.read_csv(res / args.summary)
    seed = args.seed if args.seed is not None else int(summary.seed.min())
    data = np.load(res / f"simulated_data_seed{seed}.npz")
    maps = np.load(res / args.maps)
    grid = tuple(int(g) for g in data["grid"])
    patterns = data["patterns"]  # (K, V, T)
    K, V, T = patterns.shape
    class_support = data["class_support"].reshape(K, *grid)
    names = [sh.get("name", sh["kind"]) for sh in cfg["simulation"]["shapes"][:K]]
    methods = [m["name"] for m in cfg["experiment"]["methods"]
               if f"seed{seed}__{args.rule}__{m['name']}" in maps.files]
    sel = summary[(summary.seed == seed) & (summary.rule == args.rule)].set_index("method")

    # ---- Figure 1: maps -------------------------------------------------------------------
    truth = patterns - patterns.mean(axis=0, keepdims=True)  # centred over classes, like W
    rows = [("truth", np.transpose(truth, (1, 2, 0)))] + [
        (m, maps[f"seed{seed}__{args.rule}__{m}"]) for m in methods]
    fig, axes = plt.subplots(len(rows), K + 1, figsize=(1.75 * (K + 1) + 1.6, 1.75 * len(rows)),
                             squeeze=False, layout="constrained")
    for i, (name, W) in enumerate(rows):  # W: (V, T, K)
        avg = W.mean(axis=1)  # time-averaged class maps, (V, K)
        lim = float(np.abs(avg).max()) or 1.0
        for k in range(K):
            ax = axes[i, k]
            im = ax.imshow(avg[:, k].reshape(grid), cmap=DIVERGING, vmin=-lim, vmax=lim,
                           interpolation="nearest")
            ax.contour(class_support[k], levels=[0.5], colors=INK, linewidths=0.6)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(f"class {k}: {names[k]}")
        ax = axes[i, K]
        if name == "truth":
            active = class_support.any(0)
            ax.set_title("true / selected pixels")
        else:
            active = (np.abs(W).max(axis=(1, 2)) > 1e-12 * max(lim, 1e-300)).reshape(grid)
        ax.imshow(active, cmap=SELECTED, vmin=0, vmax=1, interpolation="nearest")
        for k in range(K):
            ax.contour(class_support[k], levels=[0.5], colors=SERIES[0], linewidths=0.8)
        ax.set_xticks([]), ax.set_yticks([])
        label = "truth" if name == "truth" else LABELS.get(name, name)
        if name != "truth" and name in sel.index:
            r = sel.loc[name]
            stats = f"acc {r.test_accuracy:.2f}"
            if not pd.isna(r.get("f1", np.nan)):
                stats += f"  F1 {r.f1:.2f}\n{int(r.n_selected)} px"
            if not pd.isna(r.get("map_corr", np.nan)):
                stats += f"  r {r.map_corr:.2f}"
            label += "\n" + stats
        axes[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.5, color=INK)
        fig.colorbar(im, ax=axes[i, K - 1], fraction=0.05, pad=0.02, ticks=[-lim, 0, lim],
                     format="%.2g").ax.tick_params(labelsize=7)
    fig.suptitle(f"Class coefficient maps, time-averaged (seed {seed}, rule {args.rule}); "
                 "outlines: true class regions; colour scale per row", fontsize=9, color=INK2)
    out1 = res / f"maps_seed{seed}_{args.rule}.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # ---- Figure 2: temporal profiles inside each true class region ---------------------------
    fig, axes = plt.subplots(1, K, figsize=(3.0 * K, 2.9), sharey=True, layout="constrained")
    t = np.arange(T)
    for k in range(K):
        ax = axes[k]
        region = data["class_support"][k]
        prof = truth[k][region].mean(0)
        ax.plot(t, prof / np.abs(prof).max(), color=INK, lw=2, ls="--", label="truth")
        for j, m in enumerate(methods):
            W = maps[f"seed{seed}__{args.rule}__{m}"]
            p = W[region, :, k].mean(0)
            if np.abs(p).max() == 0:
                continue
            p = p / np.abs(p).max()
            ax.plot(t, p, color=SERIES[j % len(SERIES)], lw=2, marker=MARKERS[j % len(MARKERS)],
                    ms=5, label=LABELS.get(m, m).split("  ")[0])
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_title(f"class {k}: {names[k]}")
        ax.set_xlabel("time point")
        ax.set_xticks(t)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("coefficient in true region\n(scaled to peak 1)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False)
    fig.suptitle(f"Temporal profiles (seed {seed}, rule {args.rule})", fontsize=9, color=INK2)
    out2 = res / f"profiles_seed{seed}_{args.rule}.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
