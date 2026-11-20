"""Read-only numerical-quality audit of frozen ERP/HCP outputs; no model fits."""
from __future__ import annotations

import ast
import json

import _bootstrap
import numpy as np
import pandas as pd

from mlrdegg.experiment.cv import FitCache
from mlrdegg.paths import RESULTS_ROOT


def summarize(frame, dataset, stage):
    rows = []
    for method, group in frame.groupby("base"):
        failed = group["error"].fillna("").astype(str).ne("") if "error" in group else pd.Series(False, index=group.index)
        known = group["converged"].notna() if "converged" in group else pd.Series(False, index=group.index)
        stopped = group["converged"].eq(True) if "converged" in group else pd.Series(False, index=group.index)
        rows.append({"dataset": dataset, "stage": stage, "method": method, "unique_fits": len(group),
                     "exceptions": int(failed.sum()), "numerically_stopped": int((~failed & stopped).sum()),
                     "unfinished": int((~failed & known & ~stopped).sum()), "quality_unknown": int((~failed & ~known).sum()),
                     "iteration_limit_hit": int(group.get("status", pd.Series(index=group.index, dtype=str)).eq("max_outer").sum()),
                     "final_stationarity_unknown": int(group.get("final_stationarity_B", pd.Series(np.nan, index=group.index)).isna().sum())})
    return rows


def main():
    dest = RESULTS_ROOT / "paper_revision" / "numerical_quality"
    dest.mkdir(parents=True, exist_ok=True)
    summary, halves, accounting = [], [], []
    for dataset in ("erpcore_n170_main", "hcp_wm_surf_final"):
        source = RESULTS_ROOT / dataset
        exp = _bootstrap.load_config(source / "config.yaml")["experiment"]
        methods = {m["name"]: m for m in exp["methods"]}
        inner = pd.read_csv(source / "inner_scores.csv")
        outer = pd.read_csv(source / "outer_results.csv")
        inner["base"] = inner.method.str.split("@").str[0]
        outer["base"] = outer.method.str.split("@").str[0]
        outer["params"] = outer.selected.map(lambda s: json.dumps(ast.literal_eval(s), sort_keys=True))
        physical = outer.drop_duplicates(["unit", "outer_fold", "base", "params"])
        summary += summarize(inner, dataset, "inner") + summarize(physical, dataset, "selected_outer")
        cache = FitCache(source / "cache", exp)
        for record in physical.to_dict("records"):
            for half in (0, 1):
                meta = {"unit": record["unit"], "stage": "half", "outer_fold": int(record["outer_fold"]),
                        "method": record["base"], "params": json.loads(record["params"]), "inner_fold": half}
                path = cache.path(meta, methods[record["base"]])
                hit = cache.load(path)
                info = hit[0] if hit is not None else {}
                halves.append({"dataset": dataset, "unit": record["unit"], "outer_fold": record["outer_fold"],
                               "base": record["base"], "params": record["params"], "half": half,
                               "cache_key": path.stem, "cache_missing": hit is None,
                               **{k: info.get(k, np.nan) for k in ("status", "converged", "error", "solver",
                                                                  "final_stationarity_B", "final_stationarity_c", "final_feasibility_norm")}})
        fresh = pd.read_csv(source / "fresh_repro_seed5003.csv")
        fresh["base"] = fresh.method.str.split("@").str[0]
        # Both half states are required. Legacy fresh-R CSVs lack them entirely.
        fresh["converged"] = np.nan
        if {"half0_converged", "half1_converged"} <= set(fresh.columns):
            known = fresh[["half0_converged", "half1_converged"]].notna().all(axis=1)
            fresh.loc[known, "converged"] = fresh.loc[known, ["half0_converged", "half1_converged"]].eq(True).all(axis=1)
        summary += summarize(fresh, dataset, "fresh_R_rule_rows")
        log = pd.read_csv(source / "fits.csv")
        keys = ["unit", "stage", "outer_fold", "inner_fold", "method", "params"]
        for stage, group in log.groupby("stage"):
            accounting.append({"dataset": dataset, "stage": stage, "logged_events": len(group),
                               "distinct_logged_keys": len(group.drop_duplicates(keys)),
                               "log_scope": "cumulative; not the current run or a solver-call budget"})
    half_frame = pd.DataFrame(halves)
    for dataset, group in half_frame.groupby("dataset"):
        summary += summarize(group, dataset, "selected_half")
    pd.DataFrame(summary).to_csv(dest / "historical_quality_summary.csv", index=False)
    half_frame.to_csv(dest / "selected_half_quality.csv", index=False)
    pd.DataFrame(accounting).to_csv(dest / "fit_accounting.csv", index=False)
    print(json.dumps({"selected_half_keys": len(halves), "missing": int(half_frame.cache_missing.sum()),
                      "unfinished_selected_halves": int(half_frame.converged.eq(False).sum()),
                      "output": str(dest)}, indent=2))


if __name__ == "__main__":
    main()
