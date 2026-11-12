"""Export existing B101 primary solutions for reporting; never fit or evaluate residuals."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse,json,sys,zipfile,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--host",choices=["partition_a","partition_b"],required=True)
    args=ap.parse_args()
    root=args.root
    run=root/"results/hcp181_primary_refinement_b101"
    sys.path[:0]=[str(run/"runtime/code/src"),str(run/"runtime/code/scripts")]
    import numpy as np,pandas as pd,joblib,importlib
    from scipy.special import softmax
    from mlrdegg.graphs import unvectorize
    from mlrdegg.datasets import load_derived
    from mlrdegg.experiment.cv import outer_splits,classification_metrics
    import _bootstrap
    manifest=json.loads((run/"manifest.json").read_text())
    tasks=[t for t in manifest["tasks"] if t["host"]==args.host]
    destination=run/("report_export_"+args.host)
    destination.mkdir(exist_ok=True)
    exported=[]
    for t in tasks:
        source=run/"tasks"/t["id"]
        out=destination/t["id"];out.mkdir(exist_ok=True)
        if (source/"result.json").exists():
            result=json.loads((source/"result.json").read_text())
            assert result["new"]["common_residual_1e-03"]=="pass"
            shutil.copy2(source/"new_predictions.csv",out/"predictions.csv")
            shutil.copy2(source/"refined_importance.npz",out/"importance.npz")
            metrics=dict(result["new"])
            metadata={"solution_source":"refined_result.pkl","strict_solver_stopped":True,
                      "elapsed_seconds":result["elapsed_seconds"]}
        else:
            assert t["id"]=="F03_S00025"
            checked=json.loads((source/"checkpoint_residual_check_20261112/diagnostic.json").read_text())
            assert checked["diagnostic"]["common_residual_1e-03"]=="pass"
            checkpoint=joblib.load(source/"refine_checkpoint.pkl")
            context=joblib.load(source/"fit_context.pkl")
            assert len(checkpoint["history"])==checked["checkpoint_iterations"]
            cfg=_bootstrap.load_config(run/"inputs/config.yaml");exp=cfg["experiment"]
            ds=load_derived(root/cfg["preprocess"]["out_dir"],exp["subjects"])
            _,X,y,groups=next(importlib.import_module("03_run_decoding").make_units(ds,"pooled"))
            tr,te=outer_splits(exp,y,groups)[t["outer_fold"]]
            Xte=context["scaler"].transform(X[te])
            classes=context["classes"];H=context["H"];state=checkpoint["state"]
            B,c=state.Theta[:-1],state.Theta[-1]
            probability=softmax(Xte@(B@H.T)+H@c,axis=1)
            predictions=classes[probability.argmax(axis=1)]
            old=pd.read_csv(run/"inputs/pooled_predictions.csv")
            selected=old[(old.outer_fold==t["outer_fold"])&(old.method=="graph_tv")].sort_values("sample").copy()
            assert np.array_equal(te,selected["sample"]) and np.array_equal(y[te],selected.y_true)
            selected["y_pred"]=predictions;selected.to_csv(out/"predictions.csv",index=False)
            importance=unvectorize(np.linalg.norm(B,axis=1),ds.n_space,ds.n_time)
            np.savez_compressed(out/"importance.npz",importance=importance)
            metrics=classification_metrics(y[te],probability,classes)
            actual=[br.apply(B) for br in context["branches"]]
            metrics.update(
                converged=False,numerical_stopping=False,status="budget_checkpoint_precision_pass",
                solver="mm_admm",main_solver="mm_admm",main_status="budget_checkpoint_precision_pass",
                main_converged=False,iteration_limit_hit=False,main_iteration_limit_hit=False,
                stationarity_certified=False,n_outer=len(checkpoint["history"]),
                n_inner=sum(h["inner_iterations"] for h in checkpoint["history"]),
                n_kkt_violations=checkpoint["kkt_violations"],
                objective=checkpoint["history"][-1]["objective"],lambda_max=context["lambda_max"],
                sel_zero_fraction=float(np.mean(importance==0)),
                sel_aux_zero_fraction=float(np.mean(np.linalg.norm(state.U[0],axis=1)==0)),
                final_feasibility_norm=float(np.sqrt(sum(np.sum((a-u)**2) for a,u in zip(actual,state.U)))),
                final_dual_residual=checkpoint["history"][-1]["s_dual"],
                **checked["diagnostic"])
            metadata={"solution_source":"refine_checkpoint.pkl","strict_solver_stopped":False,
                      "elapsed_seconds":checkpoint["elapsed_seconds"],
                      "checkpoint_iterations":len(checkpoint["history"])}
            del X,Xte,ds
        assert metrics["common_residual_1e-03"]=="pass"
        record={"task":t,"metrics":metrics,"metadata":metadata,
                "accepted_for_reporting":"common_residual_1e-03",
                "new_fit_calls":0,"new_residual_calls":0}
        (out/"export.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
        exported.append(t["id"])
    (destination/"export_manifest.json").write_text(json.dumps({"host":args.host,"tasks":exported,
        "new_fit_calls":0,"new_residual_calls":0,"purpose":"replacement of fixed selected primary refits"},indent=2),encoding="utf-8")
    archive=destination.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in destination.rglob("*"):
            if p.is_file():z.write(p,p.relative_to(destination))
    print(json.dumps({"host":args.host,"archive":str(archive),"bytes":archive.stat().st_size,
                      "exported":exported,"new_fit_calls":0,"new_residual_calls":0}))

if __name__=="__main__":main()
