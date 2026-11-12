"""Evaluate the retained F03_S00025 checkpoint without fitting or changing it."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse,json,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--budget",type=float,default=120)
    args=ap.parse_args()
    root=args.root
    run=root/"results/hcp181_primary_refinement_b101"
    task=run/"tasks/F03_S00025"
    out=task/"checkpoint_residual_check_20261112"
    out.mkdir(exist_ok=True)
    sys.path[:0]=[str(run/"runtime/code/src"),str(run/"runtime/code/scripts")]
    import joblib
    import numpy as np
    import importlib
    import _bootstrap
    from mlrdegg.datasets import load_derived
    from mlrdegg.experiment.cv import outer_splits
    from mlrdegg import likelihood as lik
    from mlrdegg.residual import common_residual
    def write(name,obj):
        p=out/name;t=p.with_suffix(".tmp")
        t.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if hasattr(v,"item") else str(v)),encoding="utf-8")
        os.replace(t,p)
    start=time.monotonic()
    checkpoint=joblib.load(task/"refine_checkpoint.pkl")
    context=joblib.load(task/"fit_context.pkl")
    t=context["task"]
    assert t["outer_fold"]==2 and t["lambda_sel"]==.0025 and t["lambda_g"]==.01
    assert checkpoint["stage"]=="refine"
    write("status.json",{"state":"running","phase":"load_existing_training_data","pid":os.getpid(),
          "checkpoint_iterations":len(checkpoint["history"]),"fit_calls":0})
    cfg=_bootstrap.load_config(run/"inputs/config.yaml")
    exp=cfg["experiment"]
    ds=load_derived(root/cfg["preprocess"]["out_dir"],exp["subjects"])
    _,X,y,groups=next(importlib.import_module("03_run_decoding").make_units(ds,"pooled"))
    tr,te=outer_splits(exp,y,groups)[2]
    Xtr=context["scaler"].transform(X[tr])
    classes=context["classes"]
    yi=np.searchsorted(classes,y[tr])
    assert np.array_equal(classes[yi],y[tr])
    Y=np.eye(len(classes))[yi]
    del X,ds
    Xa=lik.augment(Xtr)
    del Xtr
    write("status.json",{"state":"running","phase":"common_residual_only","pid":os.getpid(),"fit_calls":0})
    diagnostic=common_residual(Xa,Y,context["H"],context["branches"],
        checkpoint["state"].Theta,context["null_theta"],context["original_options"].epsilon,budget=args.budget)
    result={"task":t,"checkpoint_iterations":len(checkpoint["history"]),
       "checkpoint_solver_converged":checkpoint["converged"],
       "checkpoint_original_elapsed_seconds":checkpoint["elapsed_seconds"],
       "diagnostic":diagnostic,"new_fit_calls":0,"solver_state_unchanged":True,
       "other_seven_tasks_untouched":True,"total_seconds":time.monotonic()-start}
    write("diagnostic.json",result)
    write("status.json",{"state":"completed","precision_state":diagnostic["common_residual_1e-03"],"fit_calls":0})
    print(json.dumps(result,default=lambda v:v.item() if hasattr(v,"item") else str(v)),flush=True)
    if diagnostic.get("common_residual_status")=="error":
        raise RuntimeError("Residual evaluator returned an error; do not interpret as a precision failure")

if __name__=="__main__":main()
