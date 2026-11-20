"""Supplementary simulation studies: more seeds, SNR / training-set size, misspecified graphs.

Usage:
    python code/scripts/10_simulation_studies.py --config code/configs/simulation_studies.yaml --study snr --seeds 2026 2027 2028 [--dry-run]

Runs the grid of ``05_simulation_benchmark.py`` (same simulation, estimators and split-half fits)
for every condition of one study and every seed.  A condition overrides entries of the
``simulation`` section, replaces the spatial graph of the model, or perturbs it (``perturb``:
``drop`` a fraction of the edges, multiply the weights by lognormal ``weight_noise``, or add a
fraction of ``spurious`` edges between random non-adjacent pixels); the data are always generated
on the lattice of the base configuration.  Besides the metrics of ``05``, every fit records per
class (shape) the correlation of its class map with the true one (``corr_<shape>``), the fraction
of its map energy on pixels outside every true support (``bg_<shape>``), and the AUC of its
importance map for that shape against the background (``auc_<shape>``).

Outputs in ``results/sim_studies/<study>_<first seed>-<last seed>/``: ``fits.csv`` (every
full-training-set fit, with ``condition`` and split-half ``repro``), ``graphs.csv`` (the graph of
every condition and seed), ``run.log`` and ``status.json``.  Grid-point selection and the
comparison with the base setting are done by ``11_simulation_studies_report.py``.  The run stops
starting new fits after ``--time-limit-min`` minutes and exits with a nonzero code if any fit
failed or was skipped.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import shutil
import sys
import time
import zlib

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from mlrdegg.experiment.models import Context, build_graphs, grid_points
from mlrdegg.experiment.progress import Progress
from mlrdegg.graphs import Graph
from mlrdegg.paths import RESULTS_ROOT, resolve

bench = importlib.import_module("05_simulation_benchmark")

# Mean seconds per fit in sim_shapes_repro_wide (3 seeds, 20 jobs), for --dry-run estimates only.
SEC_PER_FIT = {"l2_logreg": 0.3, "group_lasso": 1.7, "graph_tv": 21.8, "graph_p15": 22.9,
               "graphnet": 1.8, "superquadratic": 1.7,
               # graph_tv_tm and nonconvex: grid and split-half fits of one seed at 20 jobs (2026-10-01)
               "graph_tv_tm": 7.9, "nonconvex": 43.3}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def study_setup(cfg: dict, study: dict) -> tuple[dict, list[dict]]:
    """Base configuration and the methods of a study (subset and grid override of the base)."""
    base = _bootstrap.load_config(resolve(cfg["base_config"]))
    methods = base["experiment"]["methods"]
    if study.get("methods"):
        methods = [m for m in methods if m["name"] in study["methods"]]
    if study.get("grid"):
        out = []
        for m in methods:
            m = copy.deepcopy(m)
            if m["model"] == "mlrdegg":
                m["grid"] = {k: list(v) for k, v in study["grid"].items()
                             if k in m["grid"]}  # group lasso has no lambda_g
            out.append(m)
        methods = out
    return base, methods


def perturb_graph(g: Graph, spec: dict, rng: np.random.Generator) -> Graph:
    kind, m = spec["kind"], g.n_edges
    edges, w = g.edges.copy(), g.weights.copy()
    if kind == "drop":
        keep = rng.random(m) >= spec["fraction"]
        edges, w = edges[keep], w[keep]
    elif kind == "weight_noise":
        w = w * np.exp(rng.normal(0.0, spec["sigma"], m))
    elif kind == "spurious":
        n_new = int(round(spec["fraction"] * m))
        existing = set(map(tuple, edges))
        new = set()
        while len(new) < n_new:
            i, j = sorted(rng.choice(g.n_nodes, 2, replace=False))
            if (i, j) not in existing:
                new.add((int(i), int(j)))
        edges = np.vstack([edges, np.array(sorted(new), dtype=np.int64).reshape(-1, 2)])
        w = np.concatenate([w, np.full(len(new), np.median(g.weights))])
    else:
        raise ValueError(f"unknown perturbation {kind}")
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    return Graph(g.n_nodes, edges[order], w[order])


def shape_slugs(sim_cfg: dict) -> list[str]:
    if sim_cfg.get("layout") == "shapes2d":
        return [sh["name"].replace(" ", "_") for sh in sim_cfg["shapes"][: sim_cfg["n_classes"]]]
    return [f"class{k}" for k in range(sim_cfg["n_classes"])]


def shape_metrics(W: np.ndarray, data: dict, slugs: list[str]) -> dict:
    """Per-class map quality of the centred class maps ``W`` (V, T, K)."""
    truth = np.transpose(data["patterns"], (1, 2, 0))
    truth = truth - truth.mean(axis=2, keepdims=True)
    cls_sup = data["class_support"]
    bg = ~cls_sup.any(0)
    out = {}
    for k, s in enumerate(slugs):
        w, t = W[:, :, k], truth[:, :, k]
        out[f"corr_{s}"] = float(np.corrcoef(w.ravel(), t.ravel())[0, 1]) if w.std() > 0 else 0.0
        energy = float(np.sum(w**2))
        out[f"bg_{s}"] = float(np.sum(w[bg] ** 2) / energy) if energy > 0 else np.nan
        score = np.abs(w).max(axis=1)
        mask = cls_sup[k] | bg
        out[f"auc_{s}"] = (float(roc_auc_score(cls_sup[k][mask], score[mask]))
                           if score[mask].std() > 0 else 0.5)
    return out


def fit_task(method, params, data, ctx, solver_cfg, deadline, label, seed, cond, slugs):
    r = bench.fit_one(method, params, data, ctx, solver_cfg, {}, deadline, label, seed)
    r["condition"] = cond
    if label == "grid" and "_coef" in r:
        r.update(shape_metrics(r.pop("_coef"), data, slugs))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--conditions", nargs="*", help="subset of the study's conditions")
    ap.add_argument("--name", help="default: <study>_<first seed>-<last seed>")
    ap.add_argument("--n-jobs", type=int, default=28)
    ap.add_argument("--time-limit-min", type=float, default=27.0)
    ap.add_argument("--dry-run", action="store_true", help="print fit counts and a CPU-time estimate")
    args = ap.parse_args()

    cfg = _bootstrap.load_config(args.config)
    study = cfg["studies"][args.study]
    base, methods = study_setup(cfg, study)
    conditions = [c for c in study["conditions"] if not args.conditions or c["name"] in args.conditions]
    seeds = args.seeds
    split = base["experiment"].get("split_half", False)
    n_grid = sum(len(grid_points(m)) for m in methods)
    n_fits = len(conditions) * len(seeds) * n_grid * (3 if split else 1)
    cpu = sum(SEC_PER_FIT.get(m["name"], 20.0) * len(grid_points(m)) for m in methods)
    cpu *= len(seeds) * (3 if split else 1) * sum(
        c.get("simulation", {}).get("n_train", base["simulation"]["n_train"]) / base["simulation"]["n_train"]
        for c in conditions)
    print(f"study {args.study}: {len(conditions)} conditions x {len(seeds)} seeds x {n_grid} grid points"
          f"{' x 3 (split halves)' if split else ''} = {n_fits} fits; estimated CPU {cpu / 60:.0f} min, "
          f"wall ~{cpu / 60 / args.n_jobs:.0f} min with {args.n_jobs} jobs (unknown graphs at the base rate)")
    if args.dry_run:
        return 0

    name = args.name or f"{args.study}_{seeds[0]}-{seeds[-1]}"
    out = RESULTS_ROOT / "sim_studies" / name
    out.mkdir(parents=True, exist_ok=True)
    log = _bootstrap.setup_logging(out / "run.log")
    shutil.copyfile(args.config, out / "config.yaml")
    shutil.copyfile(resolve(cfg["base_config"]), out / "base_config.yaml")
    deadline = time.time() + 60 * args.time_limit_min
    solver_cfg = base["experiment"].get("solver", {})

    tasks, graph_rows = [], []
    for c in conditions:
        sim_cfg = deep_update(base["simulation"], c.get("simulation"))
        graph_cfg = deep_update(base["graph"], c.get("graph"))
        slugs = shape_slugs(sim_cfg)
        for sd in seeds:
            data = bench.simulate({**sim_cfg, "seed": sd})
            spatial, temporal = build_graphs(graph_cfg, {"coords": data["coords"]}, sim_cfg["n_time"])
            if c.get("perturb"):
                rng = np.random.default_rng([sd, zlib.crc32(c["name"].encode())])
                spatial = perturb_graph(spatial, c["perturb"], rng)
            graph_rows.append({"condition": c["name"], "seed": sd, **spatial.summary(),
                               "weight_sum": float(spatial.weights.sum())})
            ctx = Context(len(data["coords"]), sim_cfg["n_time"], spatial, temporal)
            parts = [("grid", data)]
            if split:
                h0, h1 = bench.split_halves(data, sd)
                parts += [("half_0", h0), ("half_1", h1)]
            tasks += [delayed(fit_task)(m, p, d, ctx, solver_cfg, deadline, label, sd, c["name"], slugs)
                      for label, d in parts for m in methods for p in grid_points(m)]
    pd.DataFrame(graph_rows).to_csv(out / "graphs.csv", index=False)
    log.info("study %s, conditions %s, seeds %s: %d fits, %d jobs, time limit %.0f min\n%s",
             args.study, [c["name"] for c in conditions], seeds, len(tasks), args.n_jobs,
             args.time_limit_min, pd.DataFrame(graph_rows).to_string(index=False))

    progress = Progress(len(tasks), out / "status.json", log, log_every=60)
    progress.set_stage(args.study)
    grid_rows, halves = [], {}
    for r in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(tasks):
        progress.update()
        if r["label"] == "grid":
            grid_rows.append(r)
        elif "_coef" in r:  # keep only the flattened half maps until both halves are in
            halves.setdefault((r["condition"],) + bench.grid_key(r), {})[r["label"]] = r["_coef"].ravel()
        elif r.get("error"):
            grid_rows.append(r)  # failed or skipped half fit: record the error
    for r in grid_rows:
        pair = halves.get((r["condition"],) + bench.grid_key(r), {})
        if r["label"] == "grid" and len(pair) == 2:
            a, b = pair["half_0"], pair["half_1"]
            r["repro"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    fits = pd.DataFrame(grid_rows)
    fits.to_csv(out / "fits.csv", index=False)

    n_err = int((fits.error.fillna("") != "").sum())
    ok = fits[(fits.error.fillna("") == "") & (fits.label == "grid")]
    log.info("fit time per method (s):\n%s", ok.groupby(["condition", "method"]).fit_time
             .agg(["count", "mean", "max", "sum"]).round(1).to_string())
    log.info("status per method:\n%s", ok.groupby(["condition", "method"]).status.value_counts()
             .unstack(fill_value=0).to_string())
    progress.write_status(state="completed" if n_err == 0 else "completed_with_errors", failed_fits=n_err)
    progress.close()
    log.info("finished: %d fits with errors or skipped", n_err)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
