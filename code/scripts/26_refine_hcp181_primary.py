"""B101: reconstruct and refine only eight fixed selected primary Graph-TV fits.

All output and accepted-iteration checkpoints live in a separate run directory.
No inner selection, half fits, new methods, or changes to frozen results.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import contextlib
import dataclasses
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

RUN = "hcp181_primary_refinement_b101"

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    os.replace(temporary, path)

def save_state(path, payload):
    import joblib
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    joblib.dump(payload, temporary)
    if path.exists():
        os.replace(path, path.with_suffix(".previous.pkl"))
    os.replace(temporary, path)

def load_state(path):
    import joblib
    for candidate in (Path(path), Path(path).with_suffix(".previous.pkl")):
        if candidate.exists():
            try:
                return joblib.load(candidate)
            except Exception:
                continue
    return None

def fit_scaled_training(pipe, X_scaled, y):
    """Keep the already fitted raw-input scaler intact."""
    pipe.named_steps["clf"].fit(X_scaled, y)

def predict_scaled_test(pipe, X_scaled):
    return pipe.named_steps["clf"].predict_proba(X_scaled)

def task_worker(root_string, task, budget_hours=4.0, reconstruct_hours=2.0):
    root = Path(root_string)
    run = root / "results" / RUN
    out = run / "tasks" / task["id"]
    out.mkdir(parents=True, exist_ok=True)
    completion = out / "completion.json"
    if completion.exists() and json.loads(completion.read_text())["state"] == "completed":
        return {"id": task["id"], "state": "completed", "reused": True}
    status_path = out / "status.json"
    prior = json.loads(status_path.read_text()) if status_path.exists() else {}
    spent = float(prior.get("elapsed_seconds", 0))
    start = time.monotonic()
    phase = "preparing"
    def elapsed():
        return spent + time.monotonic() - start
    def status(**fields):
        atomic_json(status_path, {"id": task["id"], "state": "running", "phase": phase,
                    "pid": os.getpid(), "elapsed_seconds": elapsed(), **fields})
    try:
        with (out / "worker.log").open("a", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print(time.strftime("%Y-%m-%dT%H:%M:%S"), "START/RESUME", task, flush=True)
            # Frozen runtime: later changes in the active project cannot alter this run.
            runtime = run / "runtime" / "code"
            sys.path[:0] = [str(runtime / "src"), str(runtime / "scripts")]
            import joblib
            import numpy as np
            import pandas as pd
            from scipy.special import softmax
            import _bootstrap
            from mlrdegg.datasets import load_derived
            from mlrdegg.experiment.cv import outer_splits, classification_metrics
            from mlrdegg.experiment.models import Context, build_graphs, make_estimator, fit_info, importance_map
            from mlrdegg.solver import SolverState, solve_mm_admm
            from mlrdegg.residual import common_residual
            source = run / "inputs"
            cfg = _bootstrap.load_config(source / "config.yaml")
            exp = cfg["experiment"]
            method = next(m for m in exp["methods"] if m["name"] == "graph_tv")
            assert method["fixed"] == {"p_s": 1.0, "p_g": 1.0, "eta": 0.5}
            params = {"lambda_sel": task["lambda_sel"], "lambda_g": task["lambda_g"]}
            rows = pd.read_csv(source / "outer_results.csv")
            selected = rows[(rows.outer_fold == task["outer_fold"]) &
                            np.isclose(rows.lambda_sel, params["lambda_sel"]) &
                            np.isclose(rows.lambda_g, params["lambda_g"])]
            assert len(selected) >= 1
            old = selected.iloc[0]
            pred = pd.read_csv(source / "pooled_predictions.csv")
            oldpred = pred[(pred.outer_fold == task["outer_fold"]) & (pred.method == old.method)].sort_values("sample")
            status()
            ds = load_derived(root / cfg["preprocess"]["out_dir"], exp["subjects"])
            spatial, temporal = build_graphs(cfg["graph"], ds.space, ds.n_time)
            ctx = Context(ds.n_space, ds.n_time, spatial, temporal)
            _, X, y, groups = next(importlib.import_module("03_run_decoding").make_units(ds, "pooled"))
            tr, te = outer_splits(exp, y, groups)[task["outer_fold"]]
            assert np.array_equal(te, oldpred["sample"]) and np.array_equal(y[te], oldpred.y_true)
            pipe = make_estimator(method, params, ctx, exp["solver"])
            clf = pipe.named_steps["clf"]
            clf.residual_diagnostic = False
            Xtr = pipe.named_steps["scale"].fit_transform(X[tr])
            Xte = pipe.named_steps["scale"].transform(X[te])
            ytr, yte = y[tr], y[te]
            del X, ds
            captured = {}

            def solve_phase(Xa, Y, H, branches, opts, initial, stage):
                nonlocal phase
                phase = stage
                checkpoint = out / (stage + "_checkpoint.pkl")
                saved = load_state(checkpoint)
                if saved:
                    initial = saved["state"]
                    opts = dataclasses.replace(opts, resume_history=saved["history"],
                          resume_converged=saved["converged"],
                          resume_rho_updates=saved["rho_updates"],
                          resume_kkt_violations=saved["kkt_violations"])
                    print("RESUMING", stage, "outer", len(saved["history"]), flush=True)
                last_save = [0.0]
                def monitor(theta, row):
                    frame = sys._getframe(1).f_locals
                    stopped = abs(frame["dF"]) <= opts.tol_obj * max(1.0, abs(frame["F"])) and frame["step"] <= opts.tol_x * max(np.linalg.norm(theta), 1e-12)
                    expired = elapsed() >= budget_hours * 3600 or (stage == "reconstruct" and elapsed() >= reconstruct_hours * 3600)
                    if time.monotonic() - last_save[0] >= 60 or stopped or expired:
                        payload = {
                            "state": SolverState(theta.copy(), [u.copy() for u in frame["U"]], [z.copy() for z in frame["Z"]], float(frame["rho"])),
                            "history": list(frame["history"]), "converged": bool(stopped),
                            "rho_updates": frame["n_rho_updates"], "kkt_violations": frame["n_kkt_violations"],
                            "elapsed_seconds": elapsed(), "task": task, "stage": stage,
                        }
                        save_state(checkpoint, payload)
                        status(outer_iteration=int(row["outer"]), objective=row["objective"], checkpoint=str(checkpoint))
                        print(stage, row, flush=True)
                        last_save[0] = time.monotonic()
                    if expired:
                        raise TimeoutError("Cumulative task/stage budget reached; restartable checkpoint retained")
                opts.monitor = monitor
                return solve_mm_admm(Xa, Y, H, branches, opts, initial)

            def wrapped(Xa, Y, H, branches, opts, initial):
                context = {"scaler": pipe.named_steps["scale"], "classes": clf.classes_,
                           "H": H, "branches": branches, "null_theta": initial.Theta,
                           "lambda_max": clf.lambda_max_, "task": task,
                           "original_options": opts}
                save_state(out / "fit_context.pkl", context)
                original_path = out / "reconstructed_result.pkl"
                original = load_state(original_path)
                if original is None:
                    original = solve_phase(Xa, Y, H, branches, opts, initial, "reconstruct")
                    save_state(original_path, original)
                proba = softmax(Xte @ (original.Theta[:-1] @ H.T) + H @ original.Theta[-1], axis=1)
                labels = clf.classes_[proba.argmax(axis=1)]
                scores = classification_metrics(yte, proba, clf.classes_)
                comparison = {"prediction_match": bool(np.array_equal(labels, oldpred.y_pred)),
                    "objective_match": bool(np.isclose(original.objective, old.objective, rtol=0, atol=1e-8)),
                    "log_loss_match": bool(np.isclose(scores["log_loss"], old.log_loss, rtol=0, atol=1e-8)),
                    "objective": original.objective, "log_loss": scores["log_loss"]}
                atomic_json(out / "reproduction.json", comparison)
                if not all(comparison[k] for k in ("prediction_match", "objective_match", "log_loss_match")):
                    raise RuntimeError("Original fixed fit did not reproduce; refinement not started")
                strict = dataclasses.replace(opts, max_outer=5000, max_inner=1000,
                     tol_obj=1e-8, tol_x=1e-5, cg_tol=1e-8,
                     inner_rel_tol0=1e-5, inner_rel_tol_min=1e-8,
                     resume_history=None, resume_converged=False,
                     resume_rho_updates=0, resume_kkt_violations=0)
                initial_refine = SolverState(original.Theta, original.U, original.Z, original.rho)
                refined_path = out / "refined_result.pkl"
                refined = load_state(refined_path)
                if refined is None:
                    refined = solve_phase(Xa, Y, H, branches, strict, initial_refine, "refine")
                    save_state(refined_path, refined)
                captured.update(Xa=Xa, Y=Y, H=H, branches=branches, null=initial.Theta, original=original)
                return refined

            clf._solve = wrapped
            fit_scaled_training(pipe, Xtr, ytr)
            # Remove the task-local callable before saving the fitted estimator.
            del clf.__dict__["_solve"]
            save_state(out / "fitted_pipeline.pkl", pipe)
            phase = "residual_evaluation"
            status()
            c = captured
            diagnostic = common_residual(c["Xa"], c["Y"], c["H"], c["branches"],
                         np.vstack([clf.B_, clf.c_[None, :]]), c["null"], clf.epsilon, budget=120)
            proba = predict_scaled_test(pipe, Xte)
            labels = clf.classes_[proba.argmax(axis=1)]
            scores = classification_metrics(yte, proba, clf.classes_)
            result = {"task": task, "old_accuracy": float(old.accuracy), "old_log_loss": float(old.log_loss),
                "old_objective": float(old.objective), "new": {**scores, **fit_info(pipe), **diagnostic},
                "changed_test_predictions": int(np.sum(labels != oldpred.y_pred.to_numpy())),
                "elapsed_seconds": elapsed()}
            atomic_json(out / "result.json", result)
            newpred = oldpred.copy()
            newpred["y_pred"] = labels
            newpred.to_csv(out / "new_predictions.csv", index=False)
            np.savez_compressed(out / "refined_importance.npz", importance=importance_map(pipe, ctx))
            state = "completed" if diagnostic.get("common_residual_1e-03") == "pass" else "attention"
            atomic_json(completion, {"id": task["id"], "state": state, "reason": "common_residual_1e-03",
                        "elapsed_seconds": elapsed(), "resumable": state != "completed"})
            status(state=state)
            return {"id": task["id"], "state": state}
    except Exception:
        error = traceback.format_exc()
        atomic_json(completion, {"id": task["id"], "state": "failed", "phase": phase,
                    "elapsed_seconds": elapsed(), "error": error, "resumable": True})
        status(state="failed", error=error)
        return {"id": task["id"], "state": "failed"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--host", choices=["partition_a", "partition_b"], required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    run = args.root / "results" / RUN
    manifest = json.loads((run / "manifest.json").read_text())
    tasks = [t for t in manifest["tasks"] if t["host"] == args.host]
    assert len(tasks) == 4 and args.workers >= 1
    if args.check_only:
        for p in [run/"inputs/config.yaml", run/"inputs/outer_results.csv",
                  run/"inputs/pooled_predictions.csv", run/"runtime/code/src/mlrdegg/solver.py",
                  args.root/"data/derived/hcp_wm_surf_unrelated/space.npz"]:
            assert p.exists(), str(p)
        print(json.dumps({"host":args.host,"workers":args.workers,"tasks":tasks,"mode":"check_only"}))
        return
    atomic_json(run / "host_status.json", {"host":args.host,"state":"running",
                "workers":args.workers,"tasks":len(tasks),"pid":os.getpid()})
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(task_worker,str(args.root),t):t for t in tasks}
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception:
                results.append({"id":pending[future]["id"],"state":"failed","error":traceback.format_exc()})
            atomic_json(run / "host_status.json", {"host":args.host,"state":"running",
                        "workers":args.workers,"tasks":len(tasks),"finished":results})
    state = "completed" if all(r["state"]=="completed" for r in results) else "attention"
    atomic_json(run / "host_completion.json", {"host":args.host,"state":state,"results":results})
    atomic_json(run / "host_status.json", {"host":args.host,"state":state,"results":results})
    raise SystemExit(0 if state=="completed" else 1)

if __name__ == "__main__":
    main()
