"""Figure of the spatial graphs used in the three experiments.

  a  simulation: 4-neighbourhood lattice on the 32 x 32 grid; shaded are the true supports of the classes
     (the maps of the first class pair overlap where the supports intersect);
  b  ERP CORE N170: 30 electrodes, edges between electrodes up to 70 mm apart (template 10-05 positions);
  c  HCP working memory: cortical mesh graph (fsaverage4), left hemisphere, lateral view, inflated surface;
     the temporal graph (chain) is not drawn.

Usage:
    python code/scripts/16_graph_figure.py

Output: manuscript/figures/graphs.pdf (vector, Arial embedded).
"""

from __future__ import annotations

import warnings

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon

from mlrdegg.graphs import gaussian_graph
from mlrdegg.paths import PROJECT_ROOT, RESULTS_ROOT

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 8})
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=FutureWarning)

INK, MUTED, EDGE, NODE = "#0b0b0b", "#52514e", "#9a9892", "#2a78d6"
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


PENDING: list = []                     # (axes, letter, title); placed after the layout is fixed


def panel_title(ax, letter: str, title: str) -> None:
    PENDING.append((ax, letter, title))


def place_titles(fig, pad: float = 4.0, indent: float = 13.0) -> None:
    """Bold letter and regular title at the top left corner of each grid cell, independent of the image inside."""
    for ax, letter, title in PENDING:
        cell = ax.get_subplotspec().get_position(fig)
        xy = (cell.x0, cell.y1)
        n_lines = title.count(chr(10)) + 1
        ax.annotate(title, xy=xy, xycoords="figure fraction", xytext=(indent, pad), textcoords="offset points",
                    fontsize=8, color=INK, ha="left", va="bottom", annotation_clip=False)
        ax.annotate(letter, xy=xy, xycoords="figure fraction", xytext=(0, pad + 9.6 * n_lines - 1.5),
                    textcoords="offset points", fontsize=11, fontweight="bold", color=INK, ha="left", va="top",
                    annotation_clip=False)


def lattice_panel(ax) -> None:
    data = np.load(RESULTS_ROOT / "sim_shapes_repro_wide" / "simulated_data_seed2026.npz")
    grid = tuple(int(g) for g in data["grid"])
    n = grid[0]
    support = data["class_support"].reshape(4, *grid)
    for k in range(4):
        ax.contourf(support[k].astype(float), levels=[0.5, 1.5], colors=[CLASS_COLORS[k]], alpha=0.28)
    lines = [[(i, j), (i + 1, j)] for i in range(n - 1) for j in range(n)]
    lines += [[(i, j), (i, j + 1)] for i in range(n) for j in range(n - 1)]
    ax.add_collection(LineCollection(lines, colors=EDGE, linewidths=0.35))
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ax.scatter(jj.ravel(), ii.ravel(), s=1.2, color=NODE, linewidths=0)
    ax.set_xlim(-1, n)
    ax.set_ylim(n, -1)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_title(ax, "a", "Simulation: lattice\n1024 nodes, 1984 edges")


def eeg_panel(ax) -> None:
    space = np.load(PROJECT_ROOT / "data" / "derived" / "erpcore_n170" / "space.npz", allow_pickle=True)
    montage = mne.channels.make_standard_montage(str(space["montage"]))
    low = {n.lower(): n for n in montage.ch_names}
    info = mne.create_info([low[n.lower()] for n in space["ch_names"]], 1.0, "eeg")
    info.set_montage(montage)
    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks=np.arange(len(info["ch_names"])), sphere=(0.0, 0.0, 0.0, 0.095))
    g = gaussian_graph(space["coords"], 40.0, 70.0)
    ax.add_collection(LineCollection([[pos[i], pos[j]] for i, j in g.edges], colors=EDGE, linewidths=0.5))
    ax.scatter(pos[:, 0], pos[:, 1], s=14, color=NODE, zorder=3, linewidths=0)
    for (x, y), name in zip(pos, space["ch_names"]):
        ax.text(x, y + 0.006, name, fontsize=4.8, ha="center", va="bottom", color=INK, zorder=4)
    ax.add_patch(Circle((0, 0), 0.095, fill=False, color=MUTED, lw=0.7))
    ax.add_patch(Polygon([(-0.012, 0.094), (0, 0.108), (0.012, 0.094)], fill=False, color=MUTED, lw=0.7))
    for s in (-1, 1):
        ax.add_patch(Polygon([(s * 0.0945, 0.012), (s * 0.104, 0.008), (s * 0.104, -0.022), (s * 0.0945, -0.026)],
                             fill=False, color=MUTED, lw=0.7))
    ax.set_xlim(-0.115, 0.115)
    ax.set_ylim(-0.105, 0.115)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_title(ax, "b", f"ERP CORE N170: electrodes\n{g.n_nodes} nodes, {g.n_edges} edges")


def mesh_panel(ax, h: str, letter: str) -> None:
    """Mesh graph of one hemisphere on the inflated surface, lateral side (nodes and edges counted for the hemisphere)."""
    k = "LR".index(h)
    derived = PROJECT_ROOT / "data" / "derived" / "hcp_wm_surf_final"
    space = np.load(derived / "space.npz", allow_pickle=True)
    coords, tri = (d.data for d in nib.load(str(derived / f"inflated.{h}.surf.gii")).darrays)
    coords = np.asarray(coords, float)
    sel = space["hemi"] == k
    edges = space["edges"]
    edges = edges[(space["hemi"][edges] == k).all(axis=1)]
    vert = space["vertex"]
    a, b = vert[edges[:, 0]], vert[edges[:, 1]]
    sign = -1.0 if h == "L" else 1.0                               # lateral view, anterior left for LH and right for RH
    xy = np.column_stack([sign * coords[:, 1], coords[:, 2]])
    mid = (coords[a, 0] + coords[b, 0]) / 2                        # draw only edges on the lateral side
    keep = mid < np.median(coords[:, 0]) if h == "L" else mid > np.median(coords[:, 0])
    ax.add_collection(LineCollection([[xy[i], xy[j]] for i, j in zip(a[keep], b[keep])], colors=EDGE, linewidths=0.2))
    nodes = np.unique(np.concatenate([a[keep], b[keep]]))
    ax.scatter(xy[nodes, 0], xy[nodes, 1], s=0.8, color=NODE, linewidths=0)
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    name = "left" if h == "L" else "right"
    panel_title(ax, letter, f"HCP: {name} hemisphere" + chr(10) + f"{int(sel.sum())} nodes, {len(edges)} edges")


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for ax in axes.ravel():
        ax.set_anchor("C")                                    # every panel is centered in its cell
    lattice_panel(axes[0])
    eeg_panel(axes[1])
    mesh_panel(axes[2], "L", "c")
    mesh_panel(axes[3], "R", "d")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.8, bottom=0.02, wspace=0.05)
    place_titles(fig)
    dest = PROJECT_ROOT / "manuscript" / "figures"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / "graphs.pdf")
    print("wrote", dest / "graphs.pdf")


if __name__ == "__main__":
    main()
