"""Refine both halves of selected ERP points with an unfinished cached half.

Combine these R updates with the previously computed selected-inner refinements.
Keep all confirmation outputs unchanged; write a separate numerical sensitivity.
"""
from __future__ import annotations

import hashlib
import importlib
import json

import _bootstrap
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import refine_selected_fits as previous
from mlrdegg.datasets import load_derived
from mlrdegg.experiment.cv import FitCache, _fit_coef, _fit_eval, outer_splits, split_halves
from mlrdegg.experiment.models import Context, build_graphs
from mlrdegg.paths import RESULTS_ROOT, resolve


def main():
    source = RESULTS_ROOT / "erpcore_n170_main"
    dest = RESULTS_ROOT / "paper_revision" / "numerical_quality"
    cfg = _bootstrap.load_config(source / "config.yaml")
    exp = cfg["experiment"]
    methods = {m["name"]: m for m in exp["methods"]}
    cache = FitCache(source / "cache", exp)
    audit = pd.read_csv(dest / "selected_half_quality.csv")
    prior = pd.read_csv(dest / "refinement.csv").fillna({"error": ""})
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ("config.yaml", "inner_scores.csv", "outer_results.csv")}
    bad = audit[(audit.dataset == "erpcore_n170_main") & (audit.converged != True)]
    points = bad.drop_duplicates(["unit", "outer_fold", "base", "params"])
    solver = {**exp["solver"], "max_outer": 5000, "max_inner": 1000,
              "tol_obj": 1e-7, "tol_x": 1e-4}
    protocol = {"scope": "both halves of selected unfinished ERP points", "points": len(points),
                "fits": 2 * len(points), "solver": solver, "split_seed": exp["split_half_seed"],
                "comparison": "combined inner-loss and selection-R sensitivity; original outputs retained"}
    (dest / "half_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol), flush=True)
    subjects = sorted(set(points.unit) | set(prior.unit))
    ds = load_derived(resolve(cfg["preprocess"]["out_dir"]), subjects=subjects)
    spatial, temporal = build_graphs(cfg["graph"], ds.space, ds.n_time)
    ctx = Context(ds.n_space, ds.n_time, spatial, temporal)
    units = {u: (X, y, g) for u, X, y, g in importlib.import_module("03_run_decoding").make_units(ds, "subject")}

    def task(row, half):
        X, y, groups = units[row.unit]
        tr, _ = outer_splits(exp, y, groups)[row.outer_fold]
        idx = split_halves(tr, y, exp["split_half_seed"] + row.outer_fold)[half]
        return _fit_coef(methods[row.base], json.loads(row.params), X, y, idx, ctx, solver, 1)

    keys = [(row, half) for row in points.itertuples() for half in (0, 1)]
    fitted = Parallel(n_jobs=4)(delayed(task)(row, half) for row, half in keys)
    frame = pd.read_csv(source / "inner_scores.csv")
    frame["error"] = frame["error"].astype(object)
    for r in prior[prior.stage == "inner"].to_dict("records"):
        rows = previous.matching(frame, r["unit"], r["outer_fold"], r["method"], json.loads(r["params"]))
        index = rows.index[rows.inner_fold == r["inner_fold"]]
        for column in frame.columns.intersection(prior.columns):
            frame.loc[index, column] = r[column]
    new_pairs, old_pairs, half_rows, r_rows = {}, {}, [], []
    for (row, half), (info, W) in zip(keys, fitted):
        if info.get("error"):
            raise RuntimeError(info["error"])
        key = (row.unit, row.outer_fold, row.base, row.params)
        new_pairs.setdefault(key, {})[half] = W
        meta = {"unit": row.unit, "stage": "half", "outer_fold": row.outer_fold,
                "method": row.base, "params": json.loads(row.params), "inner_fold": half}
        hit = cache.load(cache.path(meta, methods[row.base]))
        old_pairs.setdefault(key, {})[half] = hit[1]
        half_rows.append({"unit": row.unit, "outer_fold": row.outer_fold, "method": row.base,
                          "params": row.params, "half": half, **info})
    for key, pair in new_pairs.items():
        old = old_pairs[key]
        correlation = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        before, after = correlation(old[0], old[1]), correlation(pair[0], pair[1])
        rows = previous.matching(frame, key[0], key[1], key[2], json.loads(key[3]))
        frame.loc[rows.index, "repro"] = after
        r_rows.append({"unit": key[0], "outer_fold": key[1], "method": key[2],
                       "params": key[3], "old_R": before, "refined_R": after, "delta_R": after-before})
    pd.DataFrame(half_rows).to_csv(dest / "half_refinement.csv", index=False)
    pd.DataFrame(r_rows).to_csv(dest / "R_sensitivity.csv", index=False)
    original = pd.read_csv(source / "inner_scores.csv")
    affected = set(zip(prior.unit, prior.outer_fold, prior.method)) | set((k[0], k[1], k[2]) for k in new_pairs)
    selections, new_outer, done = [], [], set()
    for unit, fold, method in sorted(affected):
        old_frame = original[(original.unit == unit) & (original.outer_fold == fold) & (original.method == method)]
        updated = frame[(frame.unit == unit) & (frame.outer_fold == fold) & (frame.method == method)]
        for rule in ("min", "1se_repro"):
            before = previous.select(old_frame, methods[method], rule, "record")
            for policy in ("record", "require_converged"):
                after = previous.select(updated, methods[method], rule, policy)
                selections.append({"unit": unit, "outer_fold": fold, "method": method, "rule": rule,
                                   "policy": policy, "original": json.dumps(before), "revised": json.dumps(after),
                                   "changed": before != after})
                if before != after:
                    key = (unit, fold, method, json.dumps(after, sort_keys=True))
                    if key not in done:
                        X, y, groups = units[unit]
                        tr, te = outer_splits(exp, y, groups)[fold]
                        info, _ = _fit_eval(methods[method], after, X, y, tr, te, ctx, solver, True, 1)
                        new_outer.append({"unit": unit, "outer_fold": fold, "method": method,
                                          "params": json.dumps(after), **info})
                        done.add(key)
    pd.DataFrame(selections).to_csv(dest / "combined_selection_sensitivity.csv", index=False)
    pd.DataFrame(new_outer).to_csv(dest / "combined_outer_sensitivity.csv", index=False)
    for name, digest in hashes.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, "Frozen source changed"
    completion = {"state": "completed", "half_fits": len(fitted), "changed_outer_points": len(done),
                  "source_sha256": hashes}
    (dest / "half_completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
