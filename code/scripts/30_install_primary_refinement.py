"""Install eight existing accepted B101 solutions and refresh only the primary report."""
from pathlib import Path
import importlib,json,shutil
import _bootstrap
import numpy as np
import pandas as pd
from mlrdegg.paths import PROJECT_ROOT as ROOT
from mlrdegg.experiment.provenance import fingerprint,sha256
report=importlib.import_module("paper_report")
rules=importlib.import_module("09_selection_rules_report")
installer=importlib.import_module("update_hcp181_report")

def main():
    ref=ROOT/"results/hcp181_primary_refinement_b101"
    base=ROOT/"results/hcp181_merged"
    primary=base/"primary"
    rpt=base/"report"
    dest=ROOT/"manuscript/tables"
    inputs=dest/"inputs"
    manifest=json.loads((ref/"manifest.json").read_text())
    outer=pd.read_csv(primary/"outer_results.csv")
    outer["error"]=outer["error"].astype(object)
    pred=pd.read_csv(primary/"units/pooled_predictions.csv")
    with np.load(primary/"units/pooled_maps.npz") as z:
        maps={k:z[k].copy() for k in z.files}
    unaffected_outer=outer[~outer.method.str.startswith("graph_tv")].copy()
    unaffected_pred=pred[~pred.method.str.startswith("graph_tv")].copy()
    provenance=[]
    replacements=0
    for task in manifest["tasks"]:
        source=ref/("report_export_"+task["host"])/task["id"]
        exported=json.loads((source/"export.json").read_text())
        assert exported["task"]==task
        assert exported["new_fit_calls"]==0 and exported["new_residual_calls"]==0
        metrics=exported["metrics"]
        assert metrics["common_residual_1e-03"]=="pass"
        newpred=pd.read_csv(source/"predictions.csv").sort_values("sample")
        with np.load(source/"importance.npz") as z:importance=z["importance"]
        assert importance.shape==(4801,4) and np.isfinite(importance).all() and (importance>=0).all()
        assert np.isclose(np.mean(importance==0),metrics["sel_zero_fraction"],atol=1e-12)
        for rule in task["rules"]:
            method="graph_tv" if rule=="min" else "graph_tv@"+rule
            mask=(outer.outer_fold==task["outer_fold"])&(outer.method==method)
            assert mask.sum()==1
            pmask=(pred.outer_fold==task["outer_fold"])&(pred.method==method)
            previous=pred.loc[pmask].sort_values("sample")
            for key in ["sample","subject","y_true","outer_fold"]:
                assert np.array_equal(previous[key],newpred[key]),(task["id"],key)
            lookup=newpred.set_index("sample").y_pred
            pred.loc[pmask,"y_pred"]=pred.loc[pmask,"sample"].map(lookup)
            observed=pred.loc[pmask,"y_pred"].eq(pred.loc[pmask,"y_true"]).mean()
            assert np.isclose(observed,metrics["accuracy"],atol=1e-12)
            # Clear unavailable old diagnostics rather than attach them to new coefficients.
            diagnostic_columns=[c for c in outer if c.startswith(("final_","common_residual","sp_","tm_","sel_","main_"))]
            for column in diagnostic_columns:
                if column not in metrics:outer.loc[mask,column]=np.nan
            for column,value in metrics.items():
                if column not in outer:
                    outer[column]=np.nan if isinstance(value,(int,float,bool)) else ""
                outer.loc[mask,column]=value
            outer.loc[mask,"fit_time"]=exported["metadata"]["elapsed_seconds"]
            outer.loc[mask,"error"]=""
            outer.loc[mask,"precision_accepted"]=True
            outer.loc[mask,"reported_solution_source"]=task["id"]+"/"+exported["metadata"]["solution_source"]
            maps[method][task["outer_fold"]]=importance
            replacements+=1
        provenance.append({"task":task,"source":str(source.relative_to(ROOT)),
            "solution_source":exported["metadata"],"common_residual":metrics["common_residual"],
            "common_residual_error_bound":metrics["common_residual_error_bound"]})
    assert replacements==10
    pd.testing.assert_frame_equal(
        outer.loc[unaffected_outer.index,unaffected_outer.columns],unaffected_outer,check_dtype=False)
    pd.testing.assert_frame_equal(pred.loc[unaffected_pred.index],unaffected_pred,check_dtype=False)
    assert pred.subject.nunique()==181 and pred.groupby(["subject","method"]).size().eq(64).all()
    assert pred.groupby("subject").outer_fold.nunique().eq(1).all()
    assert not pred.duplicated(["sample","method"]).any()
    outer.to_csv(primary/"outer_results.csv",index=False)
    outer.to_csv(primary/"units/pooled_outer.csv",index=False)
    pred.to_csv(primary/"units/pooled_predictions.csv",index=False)
    np.savez_compressed(primary/"units/pooled_maps.npz",**maps)
    acc=pred.assign(correct=pred.y_true.eq(pred.y_pred)).groupby(["subject","method"]).correct.mean().rename("accuracy").reset_index().rename(columns={"subject":"unit"})
    accuracy=report.split_names(acc)
    frame=outer.merge(rules.map_stability(primary),on=["unit","method"],validate="many_to_one")
    frame=frame.rename(columns={"sel_zero_fraction":"zeros"})
    frame=report.split_names(frame)
    tests=report.tests(accuracy,["accuracy"],"wilcox")
    rng=np.random.default_rng(2036)
    idx=rng.integers(0,181,size=(10000,181))
    for i,r in tests.iterrows():
        pivot=accuracy[accuracy.rule.eq(r.rule)].pivot(index="unit",columns="method",values="accuracy")
        delta=(pivot[r.method]-pivot.group_lasso).to_numpy()
        tests.loc[i,["ci_low","ci_high"]]=np.percentile(delta[idx].mean(axis=1),[2.5,97.5])
    for data,name in [(accuracy,"primary_accuracy.csv"),(frame,"primary_folds.csv"),(tests,"primary_tests.csv")]:
        data.to_csv(rpt/name,index=False)
        shutil.copy2(rpt/name,inputs/("hcp_"+name))
    summaries=report.summarize(frame,["log_loss","stability","zeros"]).drop(columns="n").merge(
        report.summarize(accuracy,["accuracy"]),on=["rule","method"],validate="one_to_one")
    summaries.to_csv(rpt/"primary_summary.csv",index=False)
    installer.render_primary(dest,inputs)
    selected=frame[frame.method.eq("graph_tv")]
    assert selected["common_residual_1e-03"].eq("pass").all()
    receipt={"date":"2026-11-12","independent_solutions":8,"replaced_rule_fold_rows":10,
        "participants":181,"all_primary_selected_graph_tv_pass_1e_3":True,
        "new_fits":0,"new_residual_evaluations":0,"selection_and_half_fits_unchanged":True,
        "full_family_and_external_unchanged":True,"sources":provenance}
    (ref/"report_installation.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    manifest_path=dest/"report_manifest.json"
    report_manifest=json.loads(manifest_path.read_text())
    report_manifest["input_sha256"]=fingerprint(sorted(inputs.glob("*.csv")))
    report_manifest["report_code_sha256"]=sha256(ROOT/"code/scripts/update_hcp181_report.py")
    for rel in ["outer_results.csv","units/pooled_maps.npz","units/pooled_predictions.csv"]:
        p=primary/rel
        report_manifest["original_result_sha256"].update(fingerprint([p]))
    report_manifest["hcp181_reporting"]["primary"]="common-grid, selected graph-TV outer refits accepted at 1e-3"
    report_manifest["hcp181_reporting"]["primary_outer_refinement"]=receipt
    report_manifest["hcp181_reporting"]["unfinished_outer_refits_scope"]="historical full-family comparison"
    report_manifest["hcp181_reporting"]["primary_without_strict_solver_stopping"]=1
    manifest_path.write_text(json.dumps(report_manifest,indent=2),encoding="utf-8")
    (dest/"primary_refinement_provenance.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(summaries[["rule","method","accuracy","log_loss","stability","zeros"]].to_string(index=False))
    print(tests[["rule","method","diff","p_holm","ci_low","ci_high"]].to_string(index=False))

if __name__=="__main__":main()
