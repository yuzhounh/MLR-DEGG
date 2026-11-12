"""Figures of the features retained by the algorithm on the two real data sets.

The fitted model keeps a row group ``B[j, :]`` (one location-time feature, all classes) when its norm is
nonzero; the figures show this norm, which is the algorithm's own answer to which features matter.
The norm of every outer refit is normalized to a maximum of one and averaged over the outer folds
(and participants for EEG), as in 04_summarize_results.py, for the maps selected with the 1se-repro rule.

  * ERP CORE N170: scalp topographies at four 25 ms bins and the time profile (sum over electrodes).
  * HCP working memory: cortical surface (fsaverage4, inflated), maximum over the four time bins, lateral
    and medial views of both hemispheres.  Every row is scaled to its own maximum.

Needs data/derived/hcp_wm_surf_final/inflated.{L,R}.surf.gii (wb_command -surface-resample of the S1200
inflated surfaces to fsaverage4 with the sphere pair of the preprocessing, BARYCENTRIC).

Usage:
    python code/scripts/15_feature_figures.py [--rule 1se_repro]

Output: manuscript/figures/erp_features.pdf and manuscript/figures/hcp_features.pdf (vector, Arial embedded).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
from matplotlib.collections import PolyCollection

from mlrdegg.paths import PROJECT_ROOT, RESULTS_ROOT

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 8})
mne.set_log_level("ERROR")

INK, MUTED, WALL = "#0b0b0b", "#52514e", "#d9d8d4"


def turbo_light() -> matplotlib.colors.Colormap:
    """Very light gray -> very light blue -> turbo; low values (feature not retained) stay visible on white."""
    t = np.linspace(0, 1, 256)
    gray, blue = np.array(matplotlib.colors.to_rgb("#eeeeee")), np.array(matplotlib.colors.to_rgb("#dcebf9"))
    a = np.clip((t - 0.12) / 0.28, 0, 1) ** 1.5                        # blue -> turbo between 0.12 and 0.40
    low = gray + np.clip(t / 0.12, 0, 1)[:, None] * (blue - gray)       # gray -> blue between 0 and 0.12
    return matplotlib.colors.ListedColormap((1 - a[:, None]) * low + a[:, None] * plt.get_cmap("turbo")(t)[:, :3],
                                            "turbo_light")


# Sequential scale for nonnegative coefficient row norms (B101-R10).
SEQ = plt.get_cmap("YlGnBu")
LABELS = {"l2_logreg": "Ridge", "nonconvex": "Nonconvex", "group_lasso": "Group lasso", "graphnet": "GraphNet", "graph_tv": "Graph TV",
          "graph_tv_tm": "Temporal\ngraph TV", "superquadratic": "Superquadratic"}
ERP_ROWS = ["l2_logreg", "group_lasso", "graphnet", "graph_tv", "graph_tv_tm", "superquadratic", "nonconvex"]
HCP_ROWS = ["l2_logreg", "group_lasso", "graphnet", "graph_tv", "graph_tv_tm", "superquadratic", "nonconvex"]
DEST = PROJECT_ROOT / "manuscript" / "figures"


def mean_map(files, key: str) -> np.ndarray:
    """Fold-and-unit average of the per-fit maps normalized to maximum one."""
    acc = []
    for f in files:
        with np.load(f) as z:
            maps = z[key]
            acc.extend(maps / np.maximum(maps.reshape(len(maps), -1).max(axis=1), 1e-300)[:, None, None])
    return np.mean(acc, axis=0)


def erp_figure(rule: str, name: str = "erpcore_n170_37") -> None:
    res = RESULTS_ROOT / name
    space = np.load(PROJECT_ROOT / "data" / "derived" / "erpcore_n170" / "space.npz", allow_pickle=True)
    times_ms = space["times"] * 1000.0
    montage = mne.channels.make_standard_montage(str(space["montage"]))
    low = {n.lower(): n for n in montage.ch_names}
    info = mne.create_info([low[n.lower()] for n in space["ch_names"]], 1.0, "eeg")
    info.set_montage(montage)
    files = sorted((res / "units").glob("*_maps.npz"))
    suffix = f"@{rule}"
    maps = {m: mean_map(files, m + suffix) for m in ERP_ROWS}

    bins = [int(np.argmin(np.abs(times_ms - t))) for t in (112.5, 162.5, 212.5, 262.5)]
    nb, nm = len(bins), len(ERP_ROWS)
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(nb + 2, nm, height_ratios=[0.55] + [1] * nb + [0.07],
                          wspace=0.12, hspace=0.12, left=0.1, right=0.995, top=0.94, bottom=0.07)
    for j, m in enumerate(ERP_ROWS):                      # one column per method, scaled to its own maximum
        col_max = maps[m][:, bins].max()
        for i, b in enumerate(bins):
            ax = fig.add_subplot(gs[1 + i, j])
            im, _ = mne.viz.plot_topomap(maps[m][:, b], info, axes=ax, show=False, cmap=SEQ, vlim=(0, col_max),
                                         contours=0, sensors=True, res=96, size=1)
            if j == 0:
                ax.set_ylabel(f"{times_ms[b] - 12.5:.0f}–{times_ms[b] + 12.5:.0f} ms", rotation=0, ha="right",
                              va="center", fontsize=7.5, color=INK, labelpad=4)
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(LABELS[m], fontsize=7.5, color=INK)
        prof = maps[m].sum(axis=0)
        ax.plot(times_ms, prof / prof.max(), color="#2a78d6", lw=1.1)
        for b in bins:
            ax.axvspan(times_ms[b] - 12.5, times_ms[b] + 12.5, color="#2a78d6", alpha=0.12, lw=0)
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1] if j == 0 else [])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_xticks([])
        if j == 0:
            ax.set_ylabel("Time" + chr(10) + "profile", rotation=0, ha="right", va="center", fontsize=7.5, color=INK, labelpad=14)
        cax = fig.add_subplot(gs[nb + 1, j])               # color bar under each column
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=matplotlib.colors.Normalize(0, col_max)), cax=cax,
                          orientation="horizontal")
        cb.set_ticks([0, col_max])
        cb.set_ticklabels(["0", f"{col_max:.2f}"])         # the actual range of this column
        labels = cax.get_xticklabels()
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
        cax.tick_params(labelsize=6, length=2, width=0.4, color=MUTED)
        cb.outline.set_linewidth(0.4)
        cb.outline.set_edgecolor(MUTED)
    DEST.mkdir(exist_ok=True)
    fig.savefig(DEST / "erp_features.pdf")
    plt.close(fig)
    print("wrote", DEST / "erp_features.pdf", "bins", [f"{times_ms[b]:.1f}" for b in bins])


def hemi_views(h: str):
    """Screen projection (columns of (x, y, depth)) of lateral and medial views; depth grows toward the viewer."""
    sign = -1.0 if h == "L" else 1.0          # the viewer of the lateral surface sits on the side of the hemisphere
    return {"lateral": lambda p: (sign * p[:, 1], p[:, 2], sign * p[:, 0]),
            "medial": lambda p: (-sign * p[:, 1], p[:, 2], -sign * p[:, 0])}


def draw_surface(ax, coords, tri, face_val, view, vmax, valid_faces) -> None:
    x, y, depth = view(coords)
    xy = np.stack([x, y], axis=-1)
    d = depth[tri].mean(axis=1)
    order = np.argsort(d)                       # far faces first
    colors = np.where(valid_faces[:, None], SEQ(np.clip(np.nan_to_num(face_val) / vmax, 0, 1)),
                      matplotlib.colors.to_rgba(WALL))
    pc = PolyCollection(xy[tri[order]], facecolors=colors[order], edgecolors=colors[order], linewidths=0.25)
    ax.add_collection(pc)
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")


def hcp_figure(rule: str, name: str = "hcp181_merged/family",
               members=None, output_name: str = "hcp_features.pdf") -> None:
    res = RESULTS_ROOT / name
    derived = PROJECT_ROOT / "data" / "derived" / "hcp_wm_surf_final"
    space = np.load(derived / "space.npz", allow_pickle=True)
    suffix = f"@{rule}"
    members = HCP_ROWS if members is None else members
    maps = {m: mean_map([res / "units" / "pooled_maps.npz"], m + suffix).max(axis=1) for m in members}

    surf = {}
    for k, h in enumerate("LR"):
        coords, tri = (d.data for d in nib.load(str(derived / f"inflated.{h}.surf.gii")).darrays)
        surf[h] = (np.asarray(coords, float), np.asarray(tri, np.int64), space["hemi"] == k)
    # rows: left hemisphere lateral and medial, then right hemisphere medial and lateral
    views = [("LH lateral", "L", "lateral"), ("LH medial", "L", "medial"),
             ("RH medial", "R", "medial"), ("RH lateral", "R", "lateral")]
    nv, nm = len(views), len(members)
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(nv + 1, nm, height_ratios=[1] * nv + [0.07], wspace=0.03, hspace=0.04,
                          left=0.1, right=0.995, top=0.9, bottom=0.07)
    for j, m in enumerate(members):                      # one column per method, own color scale
        vmax = np.percentile(maps[m], 99.5)
        for i, (name, h, side) in enumerate(views):
            coords, tri, sel = surf[h]
            full = np.full(len(coords), np.nan)
            full[space["vertex"][sel]] = maps[m][sel]
            face = full[tri].mean(axis=1)                 # NaN if any vertex is medial wall
            ax = fig.add_subplot(gs[i, j])
            draw_surface(ax, coords, tri, face, hemi_views(h)[side], vmax, ~np.isnan(face))
            if i == 0:
                ax.set_title(LABELS[m], fontsize=7.5, color=INK)
            if j == 0:
                ax.annotate(name, xy=(0, 0.5), xycoords="axes fraction", xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=7.5, color=INK)   # 4 pt left of the image, as in the ERP figure
        cax = fig.add_subplot(gs[nv, j])
        pos = cax.get_position()                           # narrower than the column, so neighboring bars do not touch
        cax.set_position([pos.x0 + 0.08 * pos.width, pos.y0, 0.84 * pos.width, pos.height])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=matplotlib.colors.Normalize(0, vmax)), cax=cax,
                          orientation="horizontal")
        cb.set_ticks([0, vmax])
        cb.set_ticklabels(["0", f"{vmax:.2f}"])
        labels = cax.get_xticklabels()
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
        cax.tick_params(labelsize=6, length=2, width=0.4, color=MUTED)
        cb.outline.set_linewidth(0.4)
        cb.outline.set_edgecolor(MUTED)
    DEST.mkdir(exist_ok=True)
    fig.savefig(DEST / output_name)
    plt.close(fig)
    print("wrote", DEST / output_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default="1se_repro")
    ap.add_argument("--erp-only", action="store_true", help="Update ERP displays without changing HCP figures")
    ap.add_argument("--hcp-only", action="store_true", help="Update only HCP displays")
    ap.add_argument("--primary-only", action="store_true", help="Draw only the accepted four-member primary HCP fits")
    ap.add_argument("--hcp-name", default="hcp181_merged/family")
    ap.add_argument("--cmap", help="matplotlib colormap name (default YlGnBu; turbo_light retains the historical rainbow scale)")
    ap.add_argument("--dest", help="output directory (default manuscript/figures)")
    args = ap.parse_args()
    global SEQ, DEST
    if args.cmap == "turbo_light":
        SEQ = turbo_light()
    elif args.cmap:
        SEQ = plt.get_cmap(args.cmap)
    if args.dest:
        DEST = Path(args.dest)
    warnings.filterwarnings("ignore", category=FutureWarning)
    if args.primary_only:
        if args.erp_only:
            ap.error("--primary-only cannot be combined with --erp-only")
        hcp_figure(args.rule, "hcp181_merged/primary",
                   ["group_lasso", "graphnet", "graph_tv", "superquadratic"],
                   "hcp_primary_features.pdf")
        return
    if args.erp_only and args.hcp_only:
        ap.error("Choose at most one dataset-only option")
    if not args.hcp_only:
        erp_figure(args.rule)
    if not args.erp_only:
        hcp_figure(args.rule, args.hcp_name)
        if args.hcp_name == "hcp181_merged/family" and (
                RESULTS_ROOT / "hcp181_primary_refinement_b101/report_installation.json").exists():
            hcp_figure(args.rule, "hcp181_merged/primary",
                       ["group_lasso", "graphnet", "graph_tv", "superquadratic"],
                       "hcp_primary_features.pdf")


if __name__ == "__main__":
    main()
