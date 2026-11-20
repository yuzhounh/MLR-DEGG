"""Tables, statistics and figures from a finished decoding run.

Usage:  python code/scripts/04_summarize_results.py --config code/configs/erpcore_n170.yaml

Writes ``results/<name>/summary/``: ``summary_table.{csv,md}``, ``paired_tests.csv`` (subject
units only), ``accuracy.{png,pdf}`` and importance maps (heatmaps for EEG, NIfTI for fMRI).
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from mlrdegg.paths import RESULTS_ROOT, resolve  # noqa: E402

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier"]
STRUCT = ["sel_zero_fraction", "sp_zero_fraction", "tm_zero_fraction"]
INK, MUTED, GRID, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def unit_level(df: pd.DataFrame) -> pd.DataFrame:
    """Average outer folds within each unit (subject); pooled runs keep folds as rows."""
    cols = [c for c in METRICS + STRUCT + ["fit_time"] if c in df]
    if df["unit"].nunique() > 1:
        return df.groupby(["method", "unit"], as_index=False)[cols].mean()
    return df.rename(columns={"outer_fold": "fold"})[["method", "unit", "fold"] + cols]


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def accuracy_figure(u: pd.DataFrame, order: list[str], chance: float, path):
    fig, ax = plt.subplots(figsize=(1.1 * len(order) + 1.5, 3.4))
    data = [u.loc[u.method == m, "accuracy"].to_numpy() for m in order]
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, color=SERIES, alpha=0.45,
                   linewidths=0)
        ax.hlines(np.mean(vals), i - 0.28, i + 0.28, color=INK, linewidth=2)
    ax.axhline(chance, color=MUTED, linestyle="--", linewidth=1)
    ax.text(len(order) - 0.5, chance, "chance", color=MUTED, va="bottom", ha="right", fontsize=8)
    ax.set_xticks(range(len(order)), order, rotation=25, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy per unit (dots) and mean (bar)", color=INK, fontsize=10, loc="left")
    style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def mean_maps(out, methods) -> dict[str, np.ndarray]:
    """Per method: maps normalized to max 1 per fit, then averaged over folds and units."""
    acc: dict[str, list[np.ndarray]] = {}
    for f in sorted((out / "units").glob("*_maps.npz")):
        with np.load(f) as z:
            for m in methods:
                if m in z.files:
                    maps = z[m]
                    norm = maps.reshape(len(maps), -1).max(axis=1)
                    acc.setdefault(m, []).extend(maps / np.maximum(norm, 1e-300)[:, None, None])
    return {m: np.mean(v, axis=0) for m, v in acc.items()}


def eeg_maps_figure(maps, space, path):
    names = list(space["ch_names"])
    times = space["times"] * 1000.0
    ms = list(maps)
    ncol = min(3, len(ms))
    nrow = int(np.ceil(len(ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 5.0 * nrow), squeeze=False)
    for ax, m in zip(axes.ravel(), ms):
        im = ax.imshow(maps[m], aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
                       extent=[times[0] - 12.5, times[-1] + 12.5, len(names) - 0.5, -0.5])
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.set_xlabel("Time (ms)")
        ax.set_title(m, fontsize=10, loc="left")
    for ax in axes.ravel()[len(ms):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Normalized ||W_j,:||")
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def fmri_maps(maps, space, out_dir):
    import nibabel as nib

    shape = tuple(int(s) for s in space["coarse_shape"])
    f = int(space["factor"])
    affine = space["affine"].copy()
    affine[:3, :3] *= f
    affine[:3, 3] = space["affine"][:3, 3] + space["affine"][:3, :3] @ np.full(3, (f - 1) / 2.0)
    ijk = space["ijk"]
    for m, mp in maps.items():
        vol = np.zeros(shape + (mp.shape[1],), dtype=np.float32)
        vol[ijk[:, 0], ijk[:, 1], ijk[:, 2], :] = mp
        nib.save(nib.Nifti1Image(vol, affine), out_dir / f"importance_{m}.nii.gz")


def surface_maps(maps, space, out_dir):
    """One GIFTI metric per hemisphere on the full target mesh (medial wall = 0), T columns."""
    import nibabel as nib

    mesh = str(space["mesh"])
    for k, (h, structure) in enumerate((("L", "CortexLeft"), ("R", "CortexRight"))):
        sel = space["hemi"] == k
        n_full = space[f"valid_{h}"].size
        for m, mp in maps.items():
            full = np.zeros((n_full, mp.shape[1]), dtype=np.float32)
            full[space["vertex"][sel]] = mp[sel]
            img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(full[:, t]) for t in range(full.shape[1])])
            img.meta = nib.gifti.GiftiMetaData({"AnatomicalStructurePrimary": structure})
            nib.save(img, out_dir / f"importance_{m}.{h}.{mesh}.func.gii")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name")
    ap.add_argument("--results-root")
    args = ap.parse_args()
    cfg = _bootstrap.load_config(args.config)
    exp = cfg["experiment"]
    out = (resolve(args.results_root) if args.results_root else RESULTS_ROOT) / (args.name or exp["name"])
    sdir = out / "summary"
    sdir.mkdir(exist_ok=True)
    log = _bootstrap.setup_logging(sdir / "summary.log")

    df = pd.read_csv(out / "outer_results.csv")
    df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    order = [m["name"] for m in exp["methods"] if m["name"] in set(df.method)]
    u = unit_level(df)

    rows = []
    for m in order:
        g = u[u.method == m]
        row = {"method": m, "n": len(g)}
        for c in [c for c in METRICS + STRUCT + ["fit_time"] if c in g]:
            row[f"{c}_mean"], row[f"{c}_sd"] = g[c].mean(), g[c].std(ddof=1)
        row["converged_rate"] = df.loc[df.method == m, "converged"].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(sdir / "summary_table.csv", index=False)
    md = ["| method | n | accuracy | balanced acc. | macro-F1 | log-loss | row zeros | fit time (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda c, d=3: f"{r.get(c + '_mean', np.nan):.{d}f} ± {r.get(c + '_sd', np.nan):.{d}f}"  # noqa: E731
        md.append(f"| {r['method']} | {r['n']} | {f('accuracy')} | {f('balanced_accuracy')} | "
                  f"{f('macro_f1')} | {f('log_loss')} | {f('sel_zero_fraction', 2)} | {f('fit_time', 1)} |")
    (sdir / "summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(md))

    if df["unit"].nunique() > 1:
        tests = []
        for a, b in combinations(order, 2):
            x = u[u.method == a].set_index("unit")["accuracy"]
            y = u[u.method == b].set_index("unit")["accuracy"]
            x, y = x.align(y, join="inner")
            p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            tests.append({"method_a": a, "method_b": b, "mean_diff": float((x - y).mean()), "p": p})
        tests = pd.DataFrame(tests)
        tests["p_holm"] = holm(tests["p"].to_numpy())
        tests.to_csv(sdir / "paired_tests.csv", index=False)

    ds_dir = resolve(cfg["preprocess"]["out_dir"])
    with np.load(next(ds_dir.glob("sub-*.npz"))) as z:
        chance = 1.0 / len(np.unique(z["y"]))
    accuracy_figure(u, order, chance, sdir / "accuracy")

    with np.load(ds_dir / "space.npz") as z:
        space = {k: z[k] for k in z.files}
    maps = mean_maps(out, order)
    if maps:
        if "ch_names" in space:
            eeg_maps_figure(maps, space, sdir / "importance_maps")
        elif "ijk" in space:
            fmri_maps(maps, space, sdir)
        elif "triangles_L" in space:
            surface_maps(maps, space, sdir)
    log.info("summary written to %s", sdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
