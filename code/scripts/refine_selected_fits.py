"""Numerical sensitivity of selected unfinished ERP fits; never changes frozen outputs.

Plan first, then use --run. Each distinct selected unfinished inner/outer fit is refined
at fixed exponents, strengths, offsets and data splits. Refinement caps are 5000/1000
MM/ADMM iterations with tolerances 1e-7/1e-4. Compare historical record policy with
require_converged after refinement. Changed selected points receive an outer refit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import time
from pathlib import Path

import _bootstrap
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mlrdegg.datasets import load_derived
from mlrdegg.experiment.cv import _fit_eval, _pooled, inner_fit_quality, outer_splits, select_grid_point, split
from mlrdegg.experiment.models import Context, build_graphs, grid_points
from mlrdegg.paths import RESULTS_ROOT, resolve


def matching(frame, unit, fold, method, params):
    mask = (frame.unit == unit) & (frame.outer_fold == fold) & (frame.method == method)
    for name, value in params.items():
        mask &= np.isclose(frame[name], value, rtol=1e-12, atol=0)
    return frame[mask]


def select(frame, method, rule, policy):
    stats = []
    for gi, params in enumerate(grid_points(method)):
        rows = frame[frame.grid_index == gi].fillna({"error": ""}).to_dict("records")
        state = inner_fit_quality(rows, policy)
        if state["ok"]:
            state["score"] = -float(np.mean([r["log_loss"] for r in rows]))
            state["accuracy"], state["log_loss"], state["se"] = _pooled(rows)
            state["repro"] = float(rows[0]["repro"])
        stats.append(state)
    return grid_points(method)[select_grid_point(stats, rule)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    source = RESULTS_ROOT / "erpcore_n170_main"
    dest = RESULTS_ROOT / "paper_revision" / "numerical_quality"
    dest.mkdir(parents=True, exist_ok=True)
    cfg = _bootstrap.load_config(source / "config.yaml")
    exp = cfg["experiment"]
    methods = {m["name"]: m for m in exp["methods"]}
    outer = pd.read_csv(source / "outer_results.csv")
    inner = pd.read_csv(source / "inner_scores.csv")
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ("config.yaml", "outer_results.csv", "inner_scores.csv")}
    targets = {}
    affected = set()
    selected_quality = []
    for record in outer.to_dict("records"):
        unit, fold = record["unit"], int(record["outer_fold"])
        method = record["method"].split("@")[0]
        params = ast.literal_eval(record["selected"])
        rows = matching(inner, unit, fold, method, params)
        unfinished = rows[~rows.converged.astype(bool)]
        selected_quality.append({"unit": unit, "outer_fold": fold, "method": record["method"],
                                 "selected": record["selected"], "inner_unfinished": len(unfinished)})
        for row in unfinished.to_dict("records"):
            key = (unit, fold, method, json.dumps(params, sort_keys=True), int(row["inner_fold"]), "inner")
            targets[key] = {"unit": unit, "outer_fold": fold, "method": method, "params": params,
                            "inner_fold": int(row["inner_fold"]), "stage": "inner", "old": row}
            affected.add((unit, fold, method))
        if not bool(record["converged"]):
            key = (unit, fold, method, json.dumps(params, sort_keys=True), -1, "outer")
            targets[key] = {"unit": unit, "outer_fold": fold, "method": method, "params": params,
                            "inner_fold": -1, "stage": "outer", "old": record}
    solver = {**exp["solver"], "max_outer": 5000, "max_inner": 1000,
              "tol_obj": 1e-7, "tol_x": 1e-4}
    plan = {"scope": "selected unfinished ERP fits only", "solver": solver,
            "source_hashes": hashes,
            "targets": [{k: v for k, v in t.items() if k != "old"} for t in targets.values()],
            "selection_sensitivity": ["record", "require_converged"],
            "original_outputs": "preserved", "new_analysis": "numerical sensitivity, not a new confirmation sample"}
    (dest / "protocol.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    pd.DataFrame(selected_quality).to_csv(dest / "selected_quality.csv", index=False)
    print(json.dumps({"unique_inner_targets": sum(t["stage"] == "inner" for t in targets.values()),
                      "unique_outer_targets": sum(t["stage"] == "outer" for t in targets.values()),
                      "affected_selected_rule_rows": sum(x["inner_unfinished"] > 0 for x in selected_quality),
                      "protocol": str(dest / "protocol.json")}), flush=True)
    if not args.run:
        return
    started = time.time()
    units_needed = sorted({t["unit"] for t in targets.values()})
    ds = load_derived(resolve(cfg["preprocess"]["out_dir"]), subjects=units_needed)
    spatial, temporal = build_graphs(cfg["graph"], ds.space, ds.n_time)
    ctx = Context(ds.n_space, ds.n_time, spatial, temporal)
    units = {u: (X, y, groups) for u, X, y, groups in importlib.import_module("03_run_decoding").make_units(ds, "subject")}

    def task(target):
        X, y, groups = units[target["unit"]]
        tr, te = outer_splits(exp, y, groups)[target["outer_fold"]]
        if target["stage"] == "inner":
            inner_splits = split(exp["inner_cv"], y[tr], {k: v[tr] for k, v in groups.items()})
            a, b = inner_splits[target["inner_fold"]]
            tr, te = tr[a], tr[b]
        info, imap = _fit_eval(methods[target["method"]], target["params"], X, y, tr, te,
                               ctx, solver, True, 1)
        return info, imap

    ordered = list(targets.values())
    results = Parallel(n_jobs=args.n_jobs)(delayed(task)(t) for t in ordered)
    revised_inner = inner.copy()
    revised_inner["error"] = revised_inner["error"].astype(object)
    comparisons, revised, maps = [], {}, {}
    for target, (info, imap) in zip(ordered, results):
        key = (target["unit"], target["outer_fold"], target["method"],
               json.dumps(target["params"], sort_keys=True), target["stage"])
        revised[key] = info
        comparisons.append({k: target[k] for k in ("unit", "outer_fold", "method", "stage", "inner_fold")} |
                           {"params": json.dumps(target["params"]), "old_status": target["old"]["status"],
                            "old_accuracy": target["old"]["accuracy"], "old_log_loss": target["old"]["log_loss"],
                            "delta_accuracy": info.get("accuracy", np.nan) - target["old"]["accuracy"],
                            "delta_log_loss": info.get("log_loss", np.nan) - target["old"]["log_loss"], **info})
        if target["stage"] == "inner" and not info.get("error"):
            rows = matching(revised_inner, target["unit"], target["outer_fold"], target["method"], target["params"])
            index = rows.index[rows.inner_fold == target["inner_fold"]]
            for field, value in info.items():
                if field in revised_inner.columns:
                    revised_inner.loc[index, field] = value
        if target["stage"] == "outer" and imap is not None:
            maps["__".join(map(str, key[:3]))] = imap
    pd.DataFrame(comparisons).to_csv(dest / "refinement.csv", index=False)
    selections = []
    for unit, fold, method in sorted(affected):
        original = inner[(inner.unit == unit) & (inner.outer_fold == fold) & (inner.method == method)]
        frame = revised_inner[(revised_inner.unit == unit) & (revised_inner.outer_fold == fold) & (revised_inner.method == method)]
        for rule in ("min", "1se_repro"):
            old = select(original, methods[method], rule, "record")
            for policy in ("record", "require_converged"):
                new = select(frame, methods[method], rule, policy)
                selections.append({"unit": unit, "outer_fold": fold, "method": method, "rule": rule,
                                   "policy": policy, "original": json.dumps(old), "revised": json.dumps(new),
                                   "changed": old != new})
                if old != new:
                    target = {"unit": unit, "outer_fold": fold, "method": method, "params": new,
                              "inner_fold": -1, "stage": "outer"}
                    key = (unit, fold, method, json.dumps(new, sort_keys=True), "outer")
                    if key not in revised:
                        revised[key], imap = task(target)
                        comparisons.append({"unit": unit, "outer_fold": fold, "method": method,
                                            "stage": "selection_sensitivity", "rule": rule, "policy": policy,
                                            "params": json.dumps(new), **revised[key]})
    pd.DataFrame(comparisons).to_csv(dest / "refinement.csv", index=False)
    pd.DataFrame(selections).to_csv(dest / "selection_sensitivity.csv", index=False)
    np.savez_compressed(dest / "refined_outer_maps.npz", **maps)
    for name, digest in hashes.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, "Frozen source changed"
    completion = {"state": "completed", "seconds": time.time() - started,
                  "refined_fits": len(results), "sensitivity_outer_fits": len(comparisons)-len(results),
                  "source_hashes_unchanged": True}
    (dest / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
