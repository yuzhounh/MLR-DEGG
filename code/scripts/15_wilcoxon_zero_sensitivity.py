"""Sensitivity of the reported paired Wilcoxon tests to the zero-difference convention.

Reads only the saved report inputs in manuscript/tables/inputs (no fitting) and repeats the
paper's tests with the opposite convention: 'wilcox' (drop zero differences) for the
simulation and 'zsplit' (split zero differences) for ERP/HCP. Holm families are as in
paper_report.py. Writes manuscript/tables/wilcoxon_zero_sensitivity.csv.
"""
from __future__ import annotations

import importlib

import _bootstrap  # noqa: F401  (adds the package to sys.path)
import pandas as pd

from mlrdegg.paths import PROJECT_ROOT

report = importlib.import_module("paper_report")

inputs = PROJECT_ROOT / "manuscript" / "tables" / "inputs"
sim, erp, hcp = (pd.read_csv(inputs / f"{n}.csv") for n in ("simulation", "erp", "hcp_accuracy"))
sim_metrics = ["accuracy", "log_loss", "map_corr", "support_auc", "f1"]
cases = {"simulation": (sim, sim_metrics, "zsplit", "wilcox"),
         "erp": (erp, ["accuracy", "R", "stability"], "wilcox", "zsplit"),
         "hcp": (hcp, ["accuracy"], "wilcox", "zsplit")}
rows = []
for name, (frame, metrics, reported, alternative) in cases.items():
    a = report.tests(frame, metrics, reported)
    b = report.tests(frame, metrics, alternative)
    merged = a.merge(b[["rule", "method", "metric", "p_holm"]], on=["rule", "method", "metric"],
                     suffixes=("_reported", "_alternative"))
    merged.insert(0, "dataset", name)
    merged["zero_differences"] = merged["n"] - merged["wins"] - merged["losses"]
    merged["reported_convention"], merged["alternative_convention"] = reported, alternative
    merged["same_decision_at_0.05"] = (merged.p_holm_reported < 0.05) == (merged.p_holm_alternative < 0.05)
    rows.append(merged)
out = pd.concat(rows, ignore_index=True)
target = PROJECT_ROOT / "manuscript" / "tables" / "wilcoxon_zero_sensitivity.csv"
target.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(target, index=False)
print(out.groupby("dataset")[["zero_differences"]].sum())
print("comparisons:", len(out), "decision changes at 0.05:", int((~out["same_decision_at_0.05"]).sum()))
print(out.loc[~out["same_decision_at_0.05"], ["dataset", "rule", "method", "metric", "p_holm_reported", "p_holm_alternative"]])
