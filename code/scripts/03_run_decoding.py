"""Nested cross-validated decoding with MLR-DEGG family members and baselines.

Usage:
    python code/scripts/03_run_decoding.py --config code/configs/erpcore_n170.yaml
    python code/scripts/03_run_decoding.py --config code/configs/hcp_wm.yaml

Options ``--subjects``/``--methods`` restrict the run (e.g. for a smoke test) and ``--name``
writes to a different results directory.  Completed units are skipped unless ``--overwrite``.
Outputs (``results/<name>/``): ``outer_results.csv``, ``inner_scores.csv``, ``units/`` per-unit
files and importance maps, ``config.yaml``, ``run.log`` and ``status.json``.  The exit code is
nonzero if any fit failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from mlrdegg.datasets import load_derived
from mlrdegg.experiment.cv import count_fits, run_nested_cv
from mlrdegg.experiment.progress import Progress
from mlrdegg.experiment.models import Context, build_graphs
from mlrdegg.experiment.provenance import run_provenance, fingerprint
from mlrdegg.graphs import vectorize
from mlrdegg.paths import RESULTS_ROOT, resolve


def make_units(ds, unit: str):
    """Yield ``(unit_id, X, y, groups)`` with ``X`` vectorized column-major."""
    if unit == "subject":
        for s in ds.subjects:
            groups = {k: np.asarray(v) for k, v in s.meta.items()}
            yield s.subject, vectorize(s.X).astype(float), s.y, groups
    elif unit == "pooled":
        X = np.concatenate([s.X for s in ds.subjects])
        y = np.concatenate([s.y for s in ds.subjects])
        subj = np.concatenate([[s.subject] * len(s.y) for s in ds.subjects])
        groups = {"subject": subj}
        if all("run" in s.meta for s in ds.subjects):
            run = np.concatenate([s.meta["run"] for s in ds.subjects])
            groups["run"] = run
            groups["subject_run"] = np.char.add(np.char.add(subj.astype(str), "_"), run.astype(str))
        yield "pooled", vectorize(X).astype(float), y, groups
    else:
        raise ValueError(f"unknown unit {unit}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--name")
    ap.add_argument("--results-root", help="default: <project>/results; use tmp/ for smoke tests")
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="pipeline check: first grid point, 2 outer folds, at most 20 MM iterations")
    ap.add_argument("--dry-run", action="store_true", help="only count the fits and exit")
    ap.add_argument("--train-fraction", type=float,
                    help="keep this stratified fraction of every outer training set (learning curves)")
    args = ap.parse_args()

    cfg = _bootstrap.load_config(args.config)
    exp = cfg["experiment"]
    if args.train_fraction is not None:
        exp["train_fraction"] = args.train_fraction
    name = args.name or exp["name"]
    out = (resolve(args.results_root) if args.results_root else RESULTS_ROOT) / name
    (out / "units").mkdir(parents=True, exist_ok=True)
    log = _bootstrap.setup_logging(out / "run.log")
    n_jobs = args.n_jobs or exp.get("n_jobs", 1)

    subjects = args.subjects or (None if exp.get("subjects", "all") == "all" else exp["subjects"])
    ds = load_derived(resolve(cfg["preprocess"]["out_dir"]), subjects,
                      exclude=exp.get("exclude_subjects"))
    spatial, temporal = build_graphs(cfg["graph"], ds.space, ds.n_time)
    ctx = Context(ds.n_space, ds.n_time, spatial, temporal)
    methods = [m for m in exp["methods"] if not args.methods or m["name"] in args.methods]
    if args.quick:
        methods = [{**m, "grid": {k: v[:1] for k, v in (m.get("grid") or {}).items()}} for m in methods]
        exp = {**exp, "outer_cv": {**exp["outer_cv"], "max_folds": 2},
               "solver": {**exp.get("solver", {}), "max_outer": 20}}
    exp = {**exp, "name": name, "subjects": [s.subject for s in ds.subjects],
           "methods": methods, "n_jobs": n_jobs}
    effective = {**cfg, "experiment": exp}
    provenance = run_provenance(effective, ds, vars(args))
    stored = out / "provenance.json"
    if any((out / "units").glob("*_outer.csv")) and not args.overwrite:
        previous = json.loads(stored.read_text()) if stored.exists() else {}
        if previous.get("cache_fingerprint") != provenance["cache_fingerprint"]:
            raise ValueError("Completed outputs have different or unknown provenance; use a new --name or explicit --overwrite")
    exp["cache_provenance"] = provenance["cache_fingerprint"]
    if not args.dry_run:
        (out / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        stored.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    log.info("dataset %s: %d subjects, V=%d, T=%d, d=%d", ds.name, len(ds.subjects), ds.n_space,
             ds.n_time, ds.n_space * ds.n_time)
    log.info("spatial graph %s", spatial.summary() if spatial else None)
    log.info("methods: %s", [m["name"] for m in methods])
    if exp.get("train_fraction", 1.0) < 1.0:
        log.info("train_fraction %.3g (seed %d) of every outer training set", exp["train_fraction"],
                 exp.get("train_fraction_seed", 0))

    units = list(make_units(ds, exp["unit"]))
    todo = [u for u in units if args.overwrite or not (out / "units" / f"{u[0]}_outer.csv").exists()]
    total_fits = sum(count_fits(y, g, methods, exp) for _, _, y, g in todo)
    log.info("%d units (%d still to run), at most %d fit tasks, %d parallel jobs",
             len(units), len(todo), total_fits, n_jobs)
    if args.dry_run:
        return 0
    progress = Progress(total_fits, out / "status.json", log)
    n_failed, t0 = 0, time.time()
    for k, (unit, X, y, groups) in enumerate(units):
        unit_csv = out / "units" / f"{unit}_outer.csv"
        if unit_csv.exists() and not args.overwrite:
            log.info("[%d/%d] %s: already done, skipped", k + 1, len(units), unit)
            continue
        t_unit = time.time()
        predictions = []
        outer_rows, inner_rows, maps = run_nested_cv(unit, X, y, groups, methods, ctx, exp, n_jobs,
                                                     log, progress, cache_dir=out / "cache",
                                                     predictions=predictions)
        if predictions:  # one row per outer test sample and method
            pd.DataFrame([{"unit": unit, "outer_fold": p["outer_fold"], "method": p["method"],
                           "sample": int(i), **{g: groups[g][i] for g in groups},
                           "y_true": t, "y_pred": q}
                          for p in predictions for i, t, q in zip(p["index"], p["y_true"], p["y_pred"])]
                         ).to_csv(out / "units" / f"{unit}_predictions.csv", index=False)
        pd.DataFrame(inner_rows).to_csv(out / "units" / f"{unit}_inner.csv", index=False)
        np.savez_compressed(out / "units" / f"{unit}_maps.npz", **maps)
        df = pd.DataFrame(outer_rows)
        df.to_csv(unit_csv, index=False)
        errs = df["error"].astype(str).str.len().gt(0).sum() + sum(bool(r["error"]) for r in inner_rows)
        n_failed += int(errs)
        summary = df.groupby("method")["accuracy"].mean().round(3).to_dict() if "accuracy" in df else {}
        log.info("[%d/%d] %s done in %.0f s, failed fits %d, accuracy %s", k + 1, len(units), unit,
                 time.time() - t_unit, errs, summary)
    progress.close()

    frames = [pd.read_csv(out / "units" / f"{u[0]}_outer.csv") for u in units]
    pd.concat(frames).to_csv(out / "outer_results.csv", index=False)
    frames = [pd.read_csv(out / "units" / f"{u[0]}_inner.csv") for u in units]
    pd.concat(frames).to_csv(out / "inner_scores.csv", index=False)
    with open(out / "graph_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"spatial": spatial.summary() if spatial else None,
                        "temporal": temporal.summary() if temporal else None}, f)
    final_rows = pd.read_csv(out / "outer_results.csv")
    inner_final = pd.read_csv(out / "inner_scores.csv")
    all_rows = pd.concat([final_rows, inner_final], ignore_index=True)
    exceptions = all_rows.error.fillna("").astype(str).ne("")
    unfinished = ~exceptions & all_rows.converged.eq(False)
    half_log = out / "fit_diagnostics.csv"
    half_unfinished = 0
    if half_log.exists():
        quality = pd.read_csv(half_log).drop_duplicates("cache_key", keep="last")
        half_unfinished = int((quality.stage.eq("half") & quality.converged.eq(False)).sum())
    progress.write_status(state="failed" if exceptions.any() else "completed",
                          failed_fits=int(exceptions.sum()), unfinished_rows=int(unfinished.sum()),
                          unfinished_half_fits=half_unfinished, stationarity_certified=False)
    provenance["result_sha256"] = fingerprint([out / "outer_results.csv", out / "inner_scores.csv"] +
                                               sorted((out / "units").glob("*_maps.npz")))
    stored.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    log.info("finished: %d failed fits, %.0f s", n_failed, time.time() - t0)
    return 1 if exceptions.any() else 0


if __name__ == "__main__":
    sys.exit(main())
