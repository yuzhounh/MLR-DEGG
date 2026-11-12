"""Quick algorithm benchmark on simulated spatiotemporal data (no downloaded datasets needed).

Usage:
    python code/scripts/05_simulation_benchmark.py --config code/configs/simulation.yaml

Each class adds a signal with a fixed temporal profile on its own ball of voxels of a 3D lattice
(``layout: blobs3d``) or its own shape on a 2D lattice (``layout: shapes2d``: disk, Gaussian bump,
bar, two disks, with class-specific temporal profiles); the noise is smoothed in space (and
optionally time).  The ground-truth support is therefore known, and ``selected_maps.npz`` keeps the
class coefficient maps of every selected fit for plotting with ``06_plot_simulation_maps.py``.

Phase A fits every method at every grid point on the training set, selects the grid point by the
validation metric, and reports test metrics of the selected fit together with support recovery
(voxel-level precision / recall / F1 of the exact nonzeros and the AUC of the importance map).
Phase B refits the selected grid point of the MM-solved methods with several solver variants
(time, objective, iterations).  With ``--seeds`` the whole grid is repeated for
several simulated datasets, and the grid point is chosen by three rules (see ``select``).  Outputs in ``results/<name>/``:
``fits.csv`` (every fit), ``summary.csv`` (selected fits per seed and rule), ``aggregate.csv``
(mean and SD over seeds), ``solver_variants.csv``, ``simulated_data_seed*.npz``, ``run.log`` and
``status.json``.  A hard ``time_limit_min`` stops new fits from starting.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from mlrdegg.experiment.metrics import classification_metrics, selection_score
from mlrdegg.experiment.models import Context, build_graphs, fit_info, grid_points, importance_map, make_estimator
from mlrdegg.experiment.progress import Progress
from mlrdegg.graphs import vectorize
from mlrdegg.paths import RESULTS_ROOT, resolve
from mlrdegg.residual import summarize as summarize_residuals


def shape_patterns(shape2d: tuple[int, int], shapes: list[dict]) -> np.ndarray:
    """Spatial class patterns on a 2D lattice, one per entry of ``shapes`` (values in [0, 1])."""
    yy, xx = np.meshgrid(np.arange(shape2d[0]), np.arange(shape2d[1]), indexing="ij")
    out = []
    for sh in shapes:
        kind = sh["kind"]
        if kind == "disk":  # sharp edge, piecewise constant
            m = np.zeros(shape2d)
            for c in sh["centres"]:
                m = np.maximum(m, (np.hypot(yy - c[0], xx - c[1]) <= sh["radius"]).astype(float))
        elif kind == "gaussian":  # smooth bump, no edge
            c = sh["centre"]
            m = np.exp(-((yy - c[0]) ** 2 + (xx - c[1]) ** 2) / (2 * sh["sigma"] ** 2))
            m[m < sh.get("cutoff", 0.1)] = 0.0
        elif kind == "bar":  # thin elongated ridge
            (r0, c0), (r1, c1) = sh["corners"]
            m = ((yy >= r0) & (yy <= r1) & (xx >= c0) & (xx <= c1)).astype(float)
        else:
            raise ValueError(f"unknown shape {kind}")
        out.append(m.ravel())
    return np.stack(out)


def simulate(cfg: dict) -> dict:
    """Simulated trials.  ``layout: blobs3d`` (default): one ball per class on a 3D lattice.
    ``layout: shapes2d``: one shape per class (disk, Gaussian bump, bar, two disks...) on a 2D
    lattice with class-specific temporal profiles; the noise is smoothed in space and time."""
    rng = np.random.default_rng(cfg["seed"])
    T, K = cfg["n_time"], cfg["n_classes"]
    if cfg.get("layout", "blobs3d") == "shapes2d":
        shape = tuple(cfg["grid"])  # (rows, cols)
        coords = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 2)
        coords = np.column_stack([coords, np.zeros(len(coords))])
        spatial = shape_patterns(shape, cfg["shapes"][:K])
        profiles = np.array([sh.get("profile", cfg["temporal_profile"]) for sh in cfg["shapes"][:K]], float)
        support = spatial > 0
    else:
        shape = tuple(cfg["grid"])
        coords = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
        # Class centres spread over the lattice, away from the border.
        lo, hi = np.array([2, 2, 2]), np.array(shape) - 3
        centres = [lo + (hi - lo) * f for f in ([0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1])][:K]
        support = np.stack([np.linalg.norm(coords - c, axis=1) <= cfg["blob_radius"] for c in centres])
        spatial = support.astype(float)
        profiles = np.tile(np.asarray(cfg["temporal_profile"], dtype=float), (K, 1))
    V = len(coords)
    patterns = cfg["amplitude"] * spatial[:, :, None] * profiles[:, None, :]  # (K, V, T)
    s_sigma = cfg.get("noise_smoothing", 0.0)
    t_sigma = cfg.get("noise_time_smoothing", 0.0)

    def draw(n):
        y = np.arange(n) % K
        rng.shuffle(y)
        noise = rng.normal(size=(n, T) + shape)
        if s_sigma > 0 or t_sigma > 0:
            noise = gaussian_filter(noise, sigma=(0, t_sigma) + (s_sigma,) * len(shape))
            noise /= noise.std()
        Z = np.moveaxis(noise.reshape(n, T, V), 1, 2) + patterns[y]
        return vectorize(Z), y.astype(str)

    Xtr, ytr = draw(cfg["n_train"])
    Xva, yva = draw(cfg["n_val"])
    Xte, yte = draw(cfg["n_test"])
    return {"coords": coords.astype(float), "grid": np.array(shape), "support": support.any(0),
            "class_support": support, "patterns": patterns,
            "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte, "yte": yte}


def class_maps(pipe, ctx) -> np.ndarray:
    """Class coefficient maps ``W`` on the standardized scale, shape ``(V, T, K)``, centred over
    classes (the MLR-DEGG coefficients already sum to zero over classes)."""
    clf = pipe.named_steps["clf"]
    W = clf.coef_ if hasattr(clf, "split_") else clf.coef_.T  # (d, K)
    W = W - W.mean(axis=1, keepdims=True)
    return np.transpose(W.reshape(ctx.n_time, ctx.n_space, -1), (1, 0, 2)).astype(np.float32)


def support_metrics(imap: np.ndarray, support: np.ndarray, exact_nonzero: np.ndarray | None) -> dict:
    score = np.abs(imap).max(axis=1)  # voxel importance: largest class-effect norm over time
    out = {"support_auc": float(roc_auc_score(support, score)) if score.std() > 0 else 0.5}
    if exact_nonzero is not None:
        sel = exact_nonzero
        tp = int(np.sum(sel & support))
        out.update(n_selected=int(sel.sum()),
                   precision=tp / max(sel.sum(), 1), recall=tp / support.sum(),
                   f1=2 * tp / max(sel.sum() + support.sum(), 1))
    return out


def fit_one(method, params, data, ctx, solver_cfg, overrides, deadline, label, seed=None):
    if time.time() > deadline:
        return {"seed": seed, "method": method["name"], "label": label, **params,
                "error": "skipped: time limit"}
    t0 = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            pipe = make_estimator(method, params, ctx, {**solver_cfg, **overrides})
            pipe.fit(data["Xtr"], data["ytr"])
            p_val = pipe.predict_proba(data["Xva"])
            p_te = pipe.predict_proba(data["Xte"])
        classes = pipe.classes_
        row = {"seed": seed, "method": method["name"], "label": label, **params, "error": "",
               "fit_time": time.perf_counter() - t0}
        row.update({f"val_{k}": v for k, v in classification_metrics(data["yva"], p_val, classes).items()})
        true_idx = np.searchsorted(classes, data["yva"])
        per_sample = -np.log(np.clip(p_val[np.arange(len(true_idx)), true_idx], 1e-15, None))
        row["val_log_loss_se"] = float(per_sample.std(ddof=1) / np.sqrt(len(per_sample)))
        row.update({f"test_{k}": v for k, v in classification_metrics(data["yte"], p_te, classes).items()})
        row.update(fit_info(pipe))
        clf = pipe.named_steps["clf"]
        exact = None
        if hasattr(clf, "split_"):
            row["objective"] = clf.objective_
            exact = (np.linalg.norm(clf.B_, axis=1) > 0).reshape(ctx.n_time, ctx.n_space).any(0)
            auxiliary = (np.linalg.norm(clf.split_["sel"], axis=1) > 0).reshape(ctx.n_time, ctx.n_space).any(0)
            aux_metrics = support_metrics(importance_map(pipe, ctx), data["support"], auxiliary)
            row.update({f"aux_{key}": value for key, value in aux_metrics.items()})
        row.update(support_metrics(importance_map(pipe, ctx), data["support"], exact))
        W = class_maps(pipe, ctx)
        # Historical CSV key: standardized coefficients versus original-scale injected
        # class means, not recovery of population logistic coefficients.
        template = np.transpose(data["patterns"], (1, 2, 0))  # (V, T, K)
        template = template - template.mean(axis=2, keepdims=True)
        row["map_corr"] = float(np.corrcoef(W.ravel(), template.ravel())[0, 1]) if W.std() > 0 else 0.0
        row["_coef"] = W
        return row
    except Exception:
        return {"seed": seed, "method": method["name"], "label": label, **params,
                "error": traceback.format_exc(),
                "fit_time": time.perf_counter() - t0}


def split_halves(data: dict, seed: int) -> tuple[dict, dict]:
    """Two disjoint, class-balanced halves of the training set (validation and test unchanged)."""
    rng = np.random.default_rng(seed + 7919)
    idx = [[], []]
    for c in np.unique(data["ytr"]):
        members = rng.permutation(np.flatnonzero(data["ytr"] == c))
        idx[0] += list(members[::2])
        idx[1] += list(members[1::2])
    return tuple({**data, "Xtr": data["Xtr"][np.sort(i)], "ytr": data["ytr"][np.sort(i)]} for i in idx)


def grid_key(r: dict) -> tuple:
    """Identifies a grid point of a method for a seed, for matching split-half fits."""
    params = tuple(sorted((k, round(float(r[k]), 12)) for k in ("lambda_sel", "lambda_g", "C")
                          if k in r and r[k] is not None and not pd.isna(r[k])))
    return (r.get("seed"), r["method"], params)


def run_tasks(tasks, n_jobs, progress):
    rows = []
    for row in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(tasks):
        rows.append(row)
        progress.update()
    return rows


RULES = ("min", "1se_sparsest", "1se_sel_first", "npairs", "1se_repro")


def select(ok: pd.DataFrame, method: dict, rule: str) -> pd.Series:
    """Grid point chosen on the validation set.  ``"min"``: smallest validation log loss.  The
    one-standard-error rules consider the grid points within one standard error of that minimum
    (standard error of the per-sample validation losses of the best point) and take
    ``"1se_sparsest"``: the fewest selected voxels (this favours weak graph penalties, because
    graph smoothing spreads the support), or ``"1se_sel_first"``: the largest ``lambda_sel``, then
    the largest ``lambda_g``.  For L2 both take the smallest ``C``.

    With split-half reproducibility ``repro`` (correlation of the class maps fitted on two
    disjoint halves of the training set; Strother et al., NPAIRS; Rasmussen et al., 2012):
    ``"npairs"`` minimizes the distance ``sqrt((1 - P)^2 + (1 - R)^2)`` of (validation accuracy,
    reproducibility) to the ideal point (1, 1), and ``"1se_repro"`` takes the most reproducible
    grid point within one standard error of the smallest validation log loss."""
    best = ok.loc[ok.val_log_loss.idxmin()]
    if rule == "min":
        return best
    if rule == "npairs":
        dist = np.hypot(1 - ok.val_accuracy, 1 - ok.repro.fillna(0.0))
        return ok.loc[dist.idxmin()]
    cand = ok[ok.val_log_loss <= best.val_log_loss + best.val_log_loss_se]
    if rule == "1se_repro":
        return cand.loc[cand.repro.fillna(-1.0).idxmax()]
    if method["model"] == "logreg_l2":
        return cand.sort_values("C").iloc[0]
    keys = [k for k in ("lambda_sel", "lambda_g") if k in cand and cand[k].notna().any()]
    if rule == "1se_sparsest":
        return cand.sort_values(["n_selected"] + keys, ascending=[True] + [False] * len(keys)).iloc[0]
    return cand.sort_values(keys, ascending=False).iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name")
    ap.add_argument("--results-root", help="default: <project>/results")
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--time-limit-min", type=float)
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int, help="default: simulation.seed of the config")
    ap.add_argument("--skip-variants", action="store_true")
    args = ap.parse_args()

    cfg = _bootstrap.load_config(args.config)
    exp = cfg["experiment"]
    name = args.name or exp["name"]
    out = (resolve(args.results_root) if args.results_root else RESULTS_ROOT) / name
    out.mkdir(parents=True, exist_ok=True)
    log = _bootstrap.setup_logging(out / "run.log")
    shutil.copyfile(args.config, out / "config.yaml")
    n_jobs = args.n_jobs or exp.get("n_jobs", 1)
    limit = 60 * (args.time_limit_min or exp.get("time_limit_min", 25))
    deadline = time.time() + limit
    methods = [m for m in exp["methods"] if not args.methods or m["name"] in args.methods]
    seeds = args.seeds or [cfg["simulation"]["seed"]]

    datasets = {}
    for seed in seeds:
        datasets[seed] = simulate({**cfg["simulation"], "seed": seed})
        np.savez_compressed(out / f"simulated_data_seed{seed}.npz", **datasets[seed])
    data0 = datasets[seeds[0]]
    T = cfg["simulation"]["n_time"]
    spatial, temporal = build_graphs(cfg["graph"], {"coords": data0["coords"]}, T)
    ctx = Context(len(data0["coords"]), T, spatial, temporal)
    log.info("simulated data: V=%d, T=%d, d=%d, train/val/test=%d/%d/%d, support=%d voxels, seeds %s",
             ctx.n_space, T, ctx.n_space * T, len(data0["ytr"]), len(data0["yva"]), len(data0["yte"]),
             int(data0["support"].sum()), seeds)
    solver_cfg = exp.get("solver", {})

    # Phase A: every grid point for every seed, plus two split-half fits per grid point for the
    # reproducibility of the maps (experiment.split_half).
    tasks = [delayed(fit_one)(m, p, datasets[sd], ctx, solver_cfg, {}, deadline, "grid", sd)
             for sd in seeds for m in methods for p in grid_points(m)]
    if exp.get("split_half", False):
        halves = {sd: split_halves(datasets[sd], sd) for sd in seeds}
        tasks += [delayed(fit_one)(m, p, halves[sd][h], ctx, solver_cfg, {}, deadline, f"half_{h}", sd)
                  for sd in seeds for m in methods for p in grid_points(m) for h in (0, 1)]
    variants = {} if args.skip_variants else exp.get("solver_variants", {})
    vmethods = [m for m in methods if m["name"] in exp.get("solver_variant_methods", [])]
    total = len(tasks) + len(variants) * len(vmethods)
    progress = Progress(total, out / "status.json", log, log_every=30)
    progress.set_stage("grid")
    log.info("%d grid fits + %d solver-variant fits, %d parallel jobs, time limit %.0f min",
             len(tasks), total - len(tasks), n_jobs, limit / 60)
    rows = run_tasks(tasks, n_jobs, progress)
    half_maps = {}
    grid_rows = []
    for r in rows:
        if str(r.get("label", "")).startswith("half_") and "_coef" in r:
            half_maps.setdefault(grid_key(r), {})[r["label"]] = r.pop("_coef")
        elif not str(r.get("label", "")).startswith("half_"):
            grid_rows.append(r)
    coefs = {}
    for i, r in enumerate(grid_rows):
        pair = half_maps.get(grid_key(r), {})
        if len(pair) == 2:
            a, b = pair["half_0"].ravel(), pair["half_1"].ravel()
            r["repro"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        if "_coef" in r:
            coefs[i] = r.pop("_coef")
            r["fit_id"] = i
    fits = pd.DataFrame(grid_rows)
    fits.to_csv(out / "fits.csv", index=False)
    log.info("%s", summarize_residuals(grid_rows))

    # Selection on the validation set, per seed, by both rules.
    summary = []
    for sd in seeds:
        for m in methods:
            ok = fits[(fits.seed == sd) & (fits.method == m["name"]) & (fits.error == "")]
            if ok.empty:
                continue
            for rule in RULES:
                if rule in ("npairs", "1se_repro") and "repro" not in ok:
                    continue
                row = select(ok, m, rule).copy()
                row["rule"] = rule
                for k, vals in (m.get("grid") or {}).items():
                    row[f"{k}_at_edge"] = bool(row[k] in (min(vals), max(vals))) if len(vals) > 1 else False
                summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    if coefs:  # class coefficient maps of every selected fit, for 06_plot_simulation_maps.py
        np.savez_compressed(out / "selected_maps.npz", **{
            f"seed{int(r.seed)}__{r.rule}__{r.method}": coefs[int(r.fit_id)]
            for r in summary.itertuples() if not pd.isna(getattr(r, "fit_id", np.nan))})
    cols = [c for c in ["seed", "rule", "method", "lambda_sel", "lambda_g", "C", "test_accuracy",
                        "test_log_loss", "support_auc", "map_corr", "n_selected", "precision",
                        "recall", "f1", "status", "n_outer"] if c in summary]
    log.info("selected fits:\n%s", summary[cols].to_string(index=False))
    metrics = [c for c in ["test_accuracy", "test_log_loss", "support_auc", "map_corr", "repro",
                           "n_selected", "precision", "recall", "f1", "fit_time"] if c in summary]
    agg = summary.groupby(["rule", "method"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    edges = [c for c in summary if c.endswith("_at_edge")]
    agg = agg.join(summary.groupby(["rule", "method"])[edges].mean().add_prefix("frac_"))
    agg["n_seeds"] = summary.groupby(["rule", "method"]).size()
    agg.to_csv(out / "aggregate.csv")
    log.info("mean and SD over %d seeds:\n%s", len(seeds), agg.round(3).to_string())
    log.info("total fit time per method (s):\n%s", fits.groupby("method").fit_time.sum().round(1).to_string())

    # Phase B: solver variants at the grid point selected by "min" for the first seed.
    if variants and vmethods:
        progress.set_stage("solver variants")
        first = summary[(summary.seed == seeds[0]) & (summary.rule == "min")].set_index("method")
        tasks = [delayed(fit_one)(m, {k: float(first.loc[m["name"], k]) for k in m["grid"]},
                                  datasets[seeds[0]], ctx, solver_cfg, ov, deadline, vname, seeds[0])
                 for m in vmethods if m["name"] in first.index for vname, ov in variants.items()]
        vrows = pd.DataFrame([{k: v for k, v in r.items() if k != "_coef"}
                              for r in run_tasks(tasks, n_jobs, progress)])
        best_obj = vrows.groupby("method").objective.transform("min")
        vrows["objective_rel_gap"] = (vrows.objective - best_obj) / best_obj.abs()
        vrows.to_csv(out / "solver_variants.csv", index=False)
        vcols = [c for c in ["method", "label", "fit_time", "objective", "objective_rel_gap", "status",
                             "n_outer", "n_inner", "n_kkt_violations", "test_accuracy", "f1", "error"] if c in vrows]
        log.info("solver variants:\n%s", vrows[vcols].to_string(index=False))

    progress.write_status(state="completed")
    progress.close()
    n_err = int((fits.error != "").sum())
    log.info("finished: %d fits with errors or skipped", n_err)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
