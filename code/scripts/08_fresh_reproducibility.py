"""Re-measure the split-half reproducibility of the selected grid points on fresh halves.

The reproducibility-aware rules (``experiment.selection_rules``) pick the grid point with the
highest split-half reproducibility, so the reproducibility of the selected point is optimistic,
and more so for methods with larger grids.  This script refits every selected grid point of a
finished run on two new class-balanced halves of its outer training set (seed ``--seed`` + fold)
and writes ``results/<name>/fresh_repro_seed<seed>.csv`` (unit, outer_fold, method, fresh_repro).

Usage:
    python code/scripts/08_fresh_reproducibility.py --config code/configs/erpcore_n170_confirm_repro.yaml --seed 5003
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mlrdegg.datasets import load_derived
from mlrdegg.experiment.cv import _fit_coef, outer_splits, split_halves
from mlrdegg.experiment.models import Context, build_graphs
from mlrdegg.paths import RESULTS_ROOT, resolve

run_decoding = importlib.import_module("03_run_decoding")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", help="results directory (default: experiment.name)")
    ap.add_argument("--seed", type=int, default=5003, help="must differ from split_half_seed")
    ap.add_argument("--n-jobs", type=int)
    args = ap.parse_args()

    cfg = _bootstrap.load_config(args.config)
    exp = cfg["experiment"]
    if args.seed == exp.get("split_half_seed", 0):
        raise SystemExit("--seed must differ from experiment.split_half_seed")
    out = RESULTS_ROOT / (args.name or exp["name"])
    subjects = None if exp.get("subjects", "all") == "all" else exp["subjects"]
    ds = load_derived(resolve(cfg["preprocess"]["out_dir"]), subjects, exclude=exp.get("exclude_subjects"))
    spatial, temporal = build_graphs(cfg["graph"], ds.space, ds.n_time)
    ctx = Context(ds.n_space, ds.n_time, spatial, temporal)
    methods = {m["name"]: m for m in exp["methods"]}
    data = {unit: (X, y, groups) for unit, X, y, groups in run_decoding.make_units(ds, exp["unit"])}

    outer = pd.read_csv(out / "outer_results.csv")
    keys, jobs, job_of = [], [], {}
    for (unit, fold, name), g in outer.groupby(["unit", "outer_fold", "method"]):
        X, y, groups = data[unit]
        tr, _ = outer_splits(exp, y, groups)[fold]
        params = ast.literal_eval(g["selected"].iloc[0])
        half_groups = groups[exp["split_half_group"]] if exp.get("split_half_group") else None
        for h, idx in enumerate(split_halves(tr, y, args.seed + fold, half_groups)):
            keys.append((unit, fold, name, h))
            # Rules that selected the same point of a method in a fold share one fit.
            fit_key = (unit, fold, name.split("@")[0], str(sorted(params.items())), h)
            if fit_key not in job_of:
                job_of[fit_key] = len(jobs)
                jobs.append(delayed(_fit_coef)(methods[name.split("@")[0]], params, X, y, idx, ctx,
                                               exp.get("solver", {}), 1))
            keys[-1] = (*keys[-1], job_of[fit_key])
    print(f"{len(jobs)} fits", flush=True)
    res = Parallel(n_jobs=args.n_jobs or exp.get("n_jobs", 1))(jobs)

    pairs = {}
    for key in keys:
        fit, W = res[key[4]]
        pairs.setdefault(key[:3], {})[key[3]] = (fit, W)
    rows = []
    for (unit, fold, name), p in pairs.items():
        a, b = p[0][1], p[1][1]
        if a is None or b is None:  # failed fit
            r = np.nan
        elif a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
        else:  # an all-zero map, scored 0 as in the selection (cv.run_nested_cv)
            r = 0.0
        row = {"unit": unit, "outer_fold": fold, "method": name, "fresh_repro": r}
        for half, (info, _) in p.items():
            row.update({f"half{half}_{key}": value for key, value in info.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"fresh_repro_seed{args.seed}.csv", index=False)
    n_bad = sum(np.isnan(r["fresh_repro"]) for r in rows)
    n_unfinished = sum(not bool(r.get(f"half{h}_converged", False)) for r in rows for h in (0, 1))
    print(f"wrote fresh_repro_seed{args.seed}.csv ({len(rows)} rows, {n_bad} failed fits)")
    print(f"{n_unfinished} half-results did not report numerical stopping (rule duplicates included)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
