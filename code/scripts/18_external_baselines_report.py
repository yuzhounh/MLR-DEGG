"""Compare the external baselines (scripts/16_external_baselines.py) with group lasso and GraphNet.

No fitting. Participant accuracy: ERP = mean of the five outer folds (as in the paper); HCP = share of a
participant's test segments classified correctly. Reference methods use the paper's saved report inputs
(manuscript/tables/inputs) under the *min* rule. Paired Wilcoxon tests over participants with zero
differences omitted, as for ERP/HCP in the paper; Holm over the two baselines per dataset and reference.
Writes manuscript/tables/external_baselines.csv.
"""
from __future__ import annotations

import importlib

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mlrdegg.paths import PROJECT_ROOT, RESULTS_ROOT

rules_report = importlib.import_module("09_selection_rules_report")
inputs = PROJECT_ROOT / "manuscript" / "tables" / "inputs"
rows = []
for dataset, run, ref_file in [("ERP", "erpcore_n170_external", "erp.csv"), ("HCP", "hcp_wm_surf_external", "hcp_accuracy.csv")]:
    run_dir = RESULTS_ROOT / run
    if not (run_dir / "outer_results.csv").exists():
        print(f"{dataset}: external results not available yet")
        continue
    ref = pd.read_csv(inputs / ref_file)
    ref = ref[ref.rule == "min"].pivot(index="unit", columns="method", values="accuracy")
    if dataset == "ERP":
        ext = pd.read_csv(run_dir / "outer_results.csv").groupby(["unit", "method"]).accuracy.mean().unstack()
    else:
        pred = pd.read_csv(run_dir / "test_predictions.csv")
        pred["correct"] = (pred.y_true == pred.y_pred).astype(float)
        ext = pred.groupby(["subject", "method"]).correct.mean().unstack()
    ext.index = ext.index.astype(str)
    common = ref.index.intersection(ext.index)
    if len(common) != len(ref):
        raise SystemExit(f"{dataset}: participant mismatch ({len(common)} of {len(ref)})")
    for reference in ["group_lasso", "graphnet"]:
        block = []
        for baseline in ["lda_shrinkage", "svm_linear"]:
            d = (ext.loc[common, baseline] - ref.loc[common, reference]).to_numpy()
            p = wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
            block.append({"dataset": dataset, "baseline": baseline, "reference": reference, "n": len(d),
                          "baseline_accuracy_pct": 100 * ext.loc[common, baseline].mean(),
                          "baseline_accuracy_sd_pct": 100 * ext.loc[common, baseline].std(ddof=1),
                          "reference_accuracy_pct": 100 * ref.loc[common, reference].mean(),
                          "diff_pp": 100 * d.mean(), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": p})
        for b, h in zip(block, rules_report.holm(np.array([x["p"] for x in block]))):
            b["p_holm"] = float(h)
        rows += block
out = pd.DataFrame(rows)
out.to_csv(PROJECT_ROOT / "manuscript" / "tables" / "external_baselines.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
