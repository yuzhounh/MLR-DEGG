"""Reconstruct final tables without fitting; --from-archive uses only the three retained dirs."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import platform
from pathlib import Path

import _bootstrap
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mlrdegg.experiment.provenance import fingerprint, sha256
from mlrdegg.experiment.reporting import prediction_support_from_maps
from mlrdegg.paths import PROJECT_ROOT, RESULTS_ROOT, resolve

rules_report = importlib.import_module("09_selection_rules_report")
ORDER = ["l2_logreg", "group_lasso", "graphnet", "graph_tv", "graph_tv_tm", "superquadratic", "nonconvex"]
LABEL = dict(zip(ORDER, ["Ridge", "Group lasso", "GraphNet", "Graph TV", "Temporal graph TV", "Superquadratic", "Nonconvex"]))
RULES = ["min", "1se_repro"]


def split_names(frame):
    frame = frame.copy()
    frame["rule"] = frame.method.map(lambda s: s.partition("@")[2] or "min")
    frame["method"] = frame.method.str.split("@").str[0]
    return frame


def summarize(frame, metrics):
    rows = []
    for (rule, method), group in frame.groupby(["rule", "method"]):
        row = {"rule": rule, "method": method, "n": len(group)}
        for metric in metrics:
            row[metric] = group[metric].mean()
            row[metric + "_sd"] = group[metric].std(ddof=1)
            row[metric + "_n"] = int(group[metric].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def tests(frame, metrics, zero_method):
    rows = []
    for rule, group in frame.groupby("rule"):
        for metric in metrics:
            pivot = group.pivot(index="unit", columns="method", values=metric)
            for method in ORDER:
                if method == "group_lasso" or method not in pivot:
                    continue
                if method == "l2_logreg" and metric == "f1":
                    continue  # Dense Ridge has no thresholded sparse-support F1.
                diff = (pivot[method] - pivot.group_lasso).dropna()
                if metric != "stability" and len(diff) != len(pivot):
                    raise ValueError("Missing paired outcomes")
                p = wilcoxon(diff, zero_method=zero_method).pvalue if diff.ne(0).any() else 1.0
                rows.append({"rule": rule, "method": method, "metric": metric, "n": len(diff),
                             "diff": diff.mean(), "wins": int(diff.gt(0).sum()), "losses": int(diff.lt(0).sum()),
                             "p": p, "zero_method": zero_method})
    output = pd.DataFrame(rows)
    for _, block in output.groupby(["rule", "metric"]):
        output.loc[block.index, "p_holm"] = rules_report.holm(block.p.to_numpy())
        output.loc[block.index, "family_size"] = len(block)
    return output


def decoding_inputs(name, pooled=False, erp_subjects=37):
    out = RESULTS_ROOT / name
    cfg = _bootstrap.load_config(out / "config.yaml")
    outer = pd.read_csv(out / "outer_results.csv")
    if outer.error.fillna("").ne("").any():
        raise ValueError("Final reporting does not silently drop exception rows")
    if outer.duplicated(["unit", "outer_fold", "method"]).any():
        raise ValueError("Duplicate final output row")
    support = prediction_support_from_maps(out, outer)
    frame = outer.drop(columns=["sel_zero_fraction"], errors="ignore").merge(
        support[["unit", "outer_fold", "method", "sel_zero_fraction"]], validate="one_to_one")
    fresh = pd.read_csv(out / "fresh_repro_seed5003.csv")
    frame = frame.merge(fresh[["unit", "outer_fold", "method", "fresh_repro"]],
                        on=["unit", "outer_fold", "method"], validate="one_to_one")
    frame = frame.merge(rules_report.map_stability(out), on=["unit", "method"], validate="many_to_one")
    smallest = {m["name"]: min(m["grid"]["lambda_g"]) for m in cfg["experiment"]["methods"]
                if "lambda_g" in m.get("grid", {})}
    frame["graph_above_min"] = [float(row.lambda_g > smallest[row.method.split("@")[0]])
                               if row.method.split("@")[0] in smallest else np.nan for row in frame.itertuples()]
    frame = frame.rename(columns={"fresh_repro": "R", "sel_zero_fraction": "zeros"})
    if frame[["accuracy", "log_loss", "R", "zeros"]].isna().any().any():
        raise ValueError("Missing final reported quantity")
    if not pooled:
        unit = frame.groupby(["unit", "method"]).agg({c: "mean" for c in
                    ("accuracy", "log_loss", "R", "stability", "zeros", "graph_above_min")}).reset_index()
        dev = {f"sub-{int(s):03d}" for s in cfg["experiment"].get("development_subjects", [])}
        if set(unit.unit) & dev or unit.unit.nunique() != erp_subjects:
            raise ValueError("ERP reporting cohort does not match the expected participant count")
        return split_names(unit), None, cfg
    pred = pd.read_csv(out / "units" / "pooled_predictions.csv")
    if pred.duplicated(["subject", "sample", "method"]).any() or pred.groupby("subject").outer_fold.nunique().ne(1).any():
        raise ValueError("Duplicated predictions or overlapping subject test folds")
    if pred.subject.nunique() != 50 or pred.groupby(["subject", "method"]).size().ne(64).any():
        raise ValueError("HCP confirmation cohort or test coverage changed")
    pred["correct"] = pred.y_true.eq(pred.y_pred)
    acc = pred.groupby(["subject", "method"]).correct.mean().rename("accuracy").reset_index().rename(columns={"subject": "unit"})
    check = pred.groupby(["outer_fold", "method"]).correct.mean()
    expected = frame.set_index(["outer_fold", "method"]).accuracy
    if not np.allclose(check.sort_index(), expected.sort_index(), rtol=0, atol=1e-12):
        raise ValueError("HCP predictions disagree with outer accuracy")
    return split_names(acc), split_names(frame), cfg


def assert_same_folds(first, second):
    """Both HCP runs must test every participant in the same outer fold."""
    folds = [pd.read_csv(d / "units" / "pooled_predictions.csv").groupby("subject").outer_fold.first() for d in (first, second)]
    if not folds[0].equals(folds[1]):
        raise ValueError("The HCP runs use different outer folds")


def tex_rows(dataset, table, paired):
    lines = []
    metrics = {"simulation": ["accuracy", "log_loss", "map_corr", "support_auc", "f1"],
               "erp": ["accuracy", "log_loss", "R", "stability", "graph_above_min"],
               "hcp": ["accuracy", "log_loss", "R", "stability"]}[dataset]
    for rule in RULES:
        if lines:
            lines.append(r"\midrule")
        count = 7 if dataset == "hcp" else 6
        lines.append(r"\multicolumn{" + str(count) + r"}{l}{\emph{" + rule.replace("_", "-") + r"} rule} \\")
        for method in ORDER:
            part = table[(table.rule == rule) & (table.method == method)]
            if part.empty:
                continue
            row = part.iloc[0]
            cells = [LABEL[method]]
            for metric in metrics:
                value = row[metric]
                if dataset == "simulation":
                    cells.append("--" if method == "l2_logreg" and metric == "f1" else f"{value:.3f} ({row[metric+'_sd']:.3f})")
                elif metric == "accuracy":
                    cells.append(f"{100*value:.1f} ({100*row.accuracy_sd:.1f})")
                elif metric == "graph_above_min":
                    cells.append("--" if np.isnan(value) else f"{100*value:.0f}" + r"\%")
                elif metric == "stability":
                    cells.append(f"{value:.2f}" + (r"$^*$" if method == "l2_logreg" else ""))
                else:
                    cells.append(f"{value:.3f}")
            if dataset == "hcp":
                test = paired[(paired.rule == rule) & (paired.method == method)]
                if method == "group_lasso":
                    cells += ["--", "--"]
                else:
                    t = test.iloc[0]
                    cells += ["$" + f"{100*t['diff']:+.1f}" + "$", "$<0.001$" if t.p_holm < .001 else f"{t.p_holm:.3f}"]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-archive", action="store_true")
    args = parser.parse_args()
    dest = PROJECT_ROOT / "manuscript" / "tables"
    current_manifest = dest / "report_manifest.json"
    if current_manifest.exists() and json.loads(current_manifest.read_text()).get("active_hcp_run") == "hcp181_merged":
        importlib.import_module("update_hcp181_report").main(from_archive=args.from_archive)
        return
    inputs = dest / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if args.from_archive:
        saved = json.loads((dest / "report_manifest.json").read_text())
        if saved["input_sha256"] != fingerprint(sorted(inputs.glob("*.csv"))):
            raise ValueError("Archived reporting inputs differ from their verified manifest")
    original_hashes = {}
    if not args.from_archive:
        original_paths = [RESULTS_ROOT / "sim_studies" / "report" / "selected.csv"]
        for name in ("erpcore_n170_37", "hcp_wm_surf_final", "hcp_wm_surf_final_nonconvex"):
            out = RESULTS_ROOT / name
            original_paths += [out / f for f in ("outer_results.csv", "inner_scores.csv", "config.yaml", "fresh_repro_seed5003.csv")]
            original_paths += sorted((out / "units").glob("*_maps.npz")) + sorted((out / "units").glob("*_predictions.csv"))
        original_hashes = fingerprint(original_paths)
        sim = pd.read_csv(original_paths[0])
        sim = sim[(sim.study == "seeds") & (sim.condition == "base") & sim.method.isin(ORDER) & sim.rule.isin(RULES)]
        if len(sim) != 140 or sim.duplicated(["seed", "method", "rule"]).any():
            raise ValueError("Expected the frozen 10-seed, 7-method, 2-rule simulation")
        sim = sim.rename(columns={"seed": "unit", "test_accuracy": "accuracy", "test_log_loss": "log_loss"})
        sim = sim[["unit", "method", "rule", "accuracy", "log_loss", "map_corr", "support_auc", "f1", "converged", "status"]]
        erp, _, erp_cfg = decoding_inputs("erpcore_n170_37")
        hcp, folds, hcp_cfg = decoding_inputs("hcp_wm_surf_final", pooled=True)
        # The nonconvex member was run afterwards in its own directory (same participants, folds and seeds).
        nc_hcp, nc_folds, nc_cfg = decoding_inputs("hcp_wm_surf_final_nonconvex", pooled=True)
        assert_same_folds(RESULTS_ROOT / "hcp_wm_surf_final", RESULTS_ROOT / "hcp_wm_surf_final_nonconvex")
        if set(nc_hcp.method) != {"nonconvex"} or set(hcp.method) & {"nonconvex"}:
            raise ValueError("Unexpected method sets in the HCP runs")
        hcp, folds = pd.concat([hcp, nc_hcp], ignore_index=True), pd.concat([folds, nc_folds], ignore_index=True)
        for key, frame in {"simulation": sim, "erp": erp, "hcp_accuracy": hcp, "hcp_folds": folds}.items():
            frame.to_csv(inputs / f"{key}.csv", index=False)
        historical = {}
        for key, cfg, subjects in (("erp", erp_cfg, sorted(erp.unit.unique())),
                                  ("hcp", hcp_cfg, sorted(hcp.unit.unique())),
                                  ("hcp_nonconvex", nc_cfg, sorted(hcp.unit.unique()))):
            cfg["experiment"]["subjects"] = subjects
            directory = resolve(cfg["preprocess"]["out_dir"])
            files = [directory / "space.npz"] + [directory / f"{s}.npz" for s in subjects]
            historical[key] = {"reconstructed_effective_config": cfg, "input_sha256": fingerprint(files),
                               "original_cli": "not retained", "original_dependencies": "not retained",
                               "fit_code_commit": "not certified from historical outputs"}
        (dest / "historical_provenance.json").write_text(json.dumps(historical, indent=2), encoding="utf-8")
        numerical = dest / "numerical"
        numerical.mkdir(exist_ok=True)
        quality = RESULTS_ROOT / "paper_revision" / "numerical_quality"
        for name in ("historical_quality_summary.csv", "fit_accounting.csv", "refinement.csv", "half_refinement.csv",
                     "R_sensitivity.csv", "combined_selection_sensitivity.csv", "combined_outer_sensitivity.csv",
                     "protocol.json", "half_protocol.json", "completion.json", "half_completion.json"):
            shutil.copy2(quality / name, numerical / name)
    sim, erp, hcp, folds = [pd.read_csv(inputs / f"{name}.csv") for name in ("simulation", "erp", "hcp_accuracy", "hcp_folds")]
    sim_metrics = ["accuracy", "log_loss", "map_corr", "support_auc", "f1"]
    tables = {"simulation": summarize(sim, sim_metrics),
              "erp": summarize(erp, ["accuracy", "log_loss", "R", "stability", "zeros", "graph_above_min"])}
    tables["hcp"] = summarize(hcp, ["accuracy"]).merge(
        summarize(folds, ["log_loss", "R", "stability", "zeros"]), on=["rule", "method"], suffixes=("", "_fold"))
    comparisons = {"simulation": tests(sim, sim_metrics, "zsplit"),
                   "erp": tests(erp, ["accuracy", "R", "stability"], "wilcox"),
                   "hcp": tests(hcp, ["accuracy"], "wilcox")}
    for key in tables:
        tables[key].to_csv(dest / f"{key}.csv", index=False)
        comparisons[key].to_csv(dest / f"{key}_tests.csv", index=False)
        macro = key.title() + "TableRows"
        (dest / f"{key}_rows.tex").write_text("\\newcommand{\\" + macro + "}{%\n" +
                                             tex_rows(key, tables[key], comparisons[key]) + "}\n", encoding="utf-8")
    if original_hashes and fingerprint([PROJECT_ROOT / p for p in original_hashes]) != original_hashes:
        raise ValueError("Frozen outputs changed during reporting")
    manifest = {"input_sha256": fingerprint(sorted(inputs.glob("*.csv"))),
                "original_result_sha256": original_hashes or json.loads((dest / "report_manifest.json").read_text())["original_result_sha256"],
                "report_code_sha256": sha256(Path(__file__)), "methods": ORDER, "rules": RULES,
                "report_environment": {"python": platform.python_version(), "dependencies": {
                    package: importlib.metadata.version(package) for package in ("numpy", "scipy", "pandas")}},
                "units": {"simulation": "10 seeds", "erp": f"{erp.unit.nunique()} participants; mean of five folds",
                          "hcp_accuracy": "50 participants", "hcp_other": "five folds; no fold-level tests"},
                "SD": "ddof=1", "Holm": "per dataset/rule/metric: 6 comparisons for simulation/ERP/HCP; simulation sparse-support F1 excludes dense Ridge (5); family_size retained in CSV",
                "SE": "pooled validation-sample SD / sqrt(N); segments may be correlated",
                "rounding": "fixed decimal Python formatting; ERP/HCP accuracy in percent",
                "failure_policy": "reject exceptions/missing prediction outcomes; undefined all-zero-map stability uses available participant pairs with metric-specific n; unfinished fits retained with separate audit",
                "fresh_R_quality": "ERP37: stopping records retained for ten participants; fresh-half states of the reused 27 participants remain unknown. Original six-member HCP half states unknown; HCP nonconvex states retained."}
    (dest / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": {k: len(v) for k, v in tables.items()}, "original_outputs_unchanged": True, "output": str(dest)}))


if __name__ == "__main__":
    main()
