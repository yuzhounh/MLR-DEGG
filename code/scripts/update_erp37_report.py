"""Install the completed ERP37 report without fitting or changing HCP outputs."""
from pathlib import Path
import csv
import hashlib
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import paper_report as report
from mlrdegg.paths import PROJECT_ROOT, RESULTS_ROOT


def replace_dataset(path, frame, dataset):
    """Preserve every other dataset's existing CSV rows verbatim."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    columns = next(csv.reader([lines[0]]))
    replacement = frame.reindex(columns=columns).to_csv(index=False, header=False, lineterminator="\n")
    result, inserted = [lines[0]], False
    for line in lines[1:]:
        if not line.strip():
            continue
        if next(csv.reader([line]))[0] == dataset:
            if not inserted:
                result.append(replacement)
                inserted = True
        else:
            result.append(line)
    assert inserted, (path, dataset)
    path.write_text("".join(result), encoding="utf-8", newline="\n")


def main():
    out = RESULTS_ROOT / "erpcore_n170_37"
    dest = PROJECT_ROOT / "manuscript/tables"
    validation = json.loads((out / "validation.json").read_text())
    assert validation["state"] == "validated" and validation["subjects"] == 37
    erp, _, cfg = report.decoding_inputs(out.name)
    summary = report.summarize(erp, ["accuracy", "log_loss", "R", "stability", "zeros", "graph_above_min"])
    paired = report.tests(erp, ["accuracy", "R", "stability"], "wilcox")
    saved = pd.read_csv(out / "report/table_all.csv")
    saved_tests = pd.read_csv(out / "report/tests_all.csv")
    keys = ["rule", "method"]
    a, b = summary.set_index(keys).sort_index(), saved.set_index(keys).sort_index()
    for metric in ["accuracy", "log_loss", "R", "stability", "zeros", "graph_above_min"]:
        assert np.allclose(a[metric], b[metric], atol=1e-12, rtol=0, equal_nan=True), metric
    keys += ["metric"]
    a, b = paired.set_index(keys).sort_index(), saved_tests.set_index(keys).sort_index()
    assert np.allclose(a.p_holm, b.p_holm, atol=1e-12, rtol=0)
    erp.to_csv(dest / "inputs/erp.csv", index=False)
    summary.to_csv(dest / "erp.csv", index=False)
    paired.to_csv(dest / "erp_tests.csv", index=False)
    (dest / "erp_rows.tex").write_text("\\newcommand{\\ErpTableRows}{%\n" + report.tex_rows("erp", summary, paired) + "}\n", encoding="utf-8")
    replace_dataset(dest / "external_baselines.csv", pd.read_csv(out / "report/external_baselines_37.csv"), "ERP")
    alternative = report.tests(erp, ["accuracy", "R", "stability"], "zsplit")
    zero = paired.merge(alternative[["rule", "method", "metric", "p_holm"]],
                        on=["rule", "method", "metric"], suffixes=("_reported", "_alternative"))
    zero.insert(0, "dataset", "erp")
    zero["zero_differences"] = zero.n - zero.wins - zero.losses
    zero["reported_convention"], zero["alternative_convention"] = "wilcox", "zsplit"
    zero["same_decision_at_0.05"] = (zero.p_holm_reported < .05) == (zero.p_holm_alternative < .05)
    replace_dataset(dest / "wilcoxon_zero_sensitivity.csv", zero, "erp")
    manifest_path = dest / "report_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_sha256"]["manuscript/tables/inputs/erp.csv"] = hashlib.sha256((dest / "inputs/erp.csv").read_bytes()).hexdigest()
    manifest["units"]["erp"] = "37 participants; mean of five folds"
    manifest["report_code_sha256"] = hashlib.sha256(Path(report.__file__).read_bytes()).hexdigest()
    manifest["erp37_reporting"] = {"run": out.name, "validated": True, "model_refitting": False,
                                  "source_sha256": validation["sha256"]}
    manifest["fresh_R_quality"] = "ERP37: stopping records retained for ten participants; fresh-half states of the reused 27 participants remain unknown. Original six-member HCP half states unknown; HCP nonconvex states retained."
    original = manifest["original_result_sha256"]
    for key in list(original):
        if key.startswith("results/erpcore_n170_main/"):
            del original[key]
    sources = [out / n for n in ["config.yaml", "outer_results.csv", "inner_scores.csv", "fresh_repro_seed5003.csv"]]
    sources += sorted((out / "units").glob("*_maps.npz"))
    for path in sources:
        original[path.relative_to(PROJECT_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    provenance_path = dest / "historical_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    cfg["experiment"]["subjects"] = sorted(erp.unit.unique())
    data = report.resolve(cfg["preprocess"]["out_dir"])
    paths = [data / "space.npz"] + [data / (unit + ".npz") for unit in sorted(erp.unit.unique())]
    provenance["erp"] = {"reconstructed_effective_config": cfg,
                         "input_sha256": report.fingerprint(paths),
                         "reporting_run": out.name,
                         "original_cli": "Reuse/merge and completion commands recorded in ERP37 source_manifest.json and logs",
                         "original_dependencies": "See retained run records",
                         "fit_code_commit": "Not certified for reused fits"}
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"participants": 37, "report_matches_completed_results": True,
                      "zero_convention_decision_changes": int((~zero["same_decision_at_0.05"]).sum()),
                      "accuracy_zero_differences": int(zero.loc[zero.metric.eq("accuracy"), "zero_differences"].sum())}))


if __name__ == "__main__":
    main()
