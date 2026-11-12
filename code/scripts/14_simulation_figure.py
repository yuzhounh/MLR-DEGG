"""Injected signal templates versus standardized class coefficients (seed 2026, 1se-repro rule); needs selected_maps_members.npz (17_simulation_member_maps.py).

One row per method, one column per class, each at the time point where the injected class template is
strongest.  The maps of every row are scaled to their own largest absolute value (diverging colour map
centred at zero); the black contour is the injected signal support of the class.  The grid points are those
selected with the 1se-repro rule on the validation set (``selected_maps.npz`` of the base setting).

Usage:
    python code/scripts/14_simulation_figure.py [--results sim_shapes_repro_wide] [--seed 2026] [--rule 1se_repro]

Output: manuscript/figures/simulation_maps.pdf (vector, Arial embedded).
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlrdegg.paths import PROJECT_ROOT, RESULTS_ROOT

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 8})

INK, MUTED = "#0b0b0b", "#52514e"
DIVERGING = plt.get_cmap("RdBu_r")
ROWS = [("l2_logreg", "Ridge"), ("group_lasso", "Group\nlasso"), ("graphnet", "GraphNet"), ("graph_tv", "Graph TV"),
        ("graph_tv_tm", "Temporal\ngraph TV"), ("superquadratic", "Super-\nquadratic"), ("nonconvex", "Nonconvex")]
CLASSES = ["Disk", "Gaussian bump", "Bar", "Two disks"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="sim_shapes_repro_wide")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rule", default="1se_repro")
    args = ap.parse_args()

    res = RESULTS_ROOT / args.results
    data = np.load(res / f"simulated_data_seed{args.seed}.npz")
    # temporal graph TV and nonconvex come from 17_simulation_member_maps.py (same data, same rule)
    maps = {**np.load(res / "selected_maps.npz"), **np.load(res / "selected_maps_members.npz")}
    grid = tuple(int(g) for g in data["grid"])
    patterns = data["patterns"]                      # (K, V, T)
    K, V, T = patterns.shape
    support = data["class_support"].reshape(K, *grid)
    truth = patterns - patterns.mean(axis=0, keepdims=True)
    t_peak = [int(np.argmax(np.abs(patterns[k]).sum(axis=0))) for k in range(K)]   # strongest time point of the class

    panels = [("Signal\ntemplate", np.stack([truth[k, :, t_peak[k]].reshape(grid) for k in range(K)]))]
    for name, label in ROWS:
        W = maps[f"seed{args.seed}__{args.rule}__{name}"]            # (V, T, K)
        panels.append((label, np.stack([W[:, t_peak[k], k].reshape(grid) for k in range(K)])))

    n = len(panels)
    fig, axes = plt.subplots(K + 1, n, figsize=(7.2, 4.1), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1] * K + [0.07]})
    for j, (label, M) in enumerate(panels):
        vmax = np.abs(M).max() or 1.0
        for k in range(K):
            ax = axes[k, j]
            im = ax.imshow(M[k], cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.contour(support[k], levels=[0.5], colors=INK, linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
                s.set_color(MUTED)
            if k == 0:
                ax.set_title(label, fontsize=7.5, color=INK)
            if j == 0:
                ax.set_ylabel(CLASSES[k], rotation=90, ha="center", va="bottom",
                              fontsize=7.5, color=INK, labelpad=4)
        cax = axes[K, j]                                  # one color bar under each column, scaled to that column
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([-vmax, 0, vmax])
        cb.set_ticklabels([f"{-vmax:.2g}", "0", f"{vmax:.2g}"])
        labels = cax.get_xticklabels()
        labels[0].set_ha("left")                          # keep the end labels inside the bar, clear of the neighbors
        labels[-1].set_ha("right")
        cax.tick_params(labelsize=6, length=2, width=0.4, color=MUTED)
        cb.outline.set_linewidth(0.4)
        cb.outline.set_edgecolor(MUTED)
    dest = PROJECT_ROOT / "manuscript" / "figures"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / "simulation_maps.pdf")
    print("wrote", dest / "simulation_maps.pdf", "peak time points", [t + 1 for t in t_peak])


if __name__ == "__main__":
    main()
