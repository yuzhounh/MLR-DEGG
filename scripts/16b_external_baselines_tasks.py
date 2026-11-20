"""Task-level parallel version of 16_external_baselines.py (same fits, same outputs; no model change).

``16_external_baselines.py`` runs one outer fold per process and fits the 14 models of a fold one after
another. The 12 inner-fold SVM fits (4 values of C x 3 inner folds), the two outer refits and the LDA refit are
independent once C is chosen, so here each fit is its own task and the tasks can be spread over many processes
and over several machines that share no file system. Every fit uses exactly the functions, data, folds and
seeds of 16_external_baselines.py, so the per-fold results are the same.

* ``plan``    list the tasks and their assignment to hosts (``--hosts name:slots:speed,...``);
* ``run``     run this host's tasks (``--host name``) with ``--n-jobs`` processes, longest first; a task whose result
              file exists is skipped, each finished task is saved at once to results/<name>_tasks/<id>.json;
* ``status``  count finished tasks;
* ``combine`` after the task files of all hosts are in one results/<name>_tasks/ directory: choose C by the smallest mean
              inner-fold log-loss (ties -> smaller C), take that C's speculative outer refit, and write
              results/<name>/folds/*, outer_results.csv, test_predictions.csv and run_info.json exactly like the
              original script.

The outer SVM refit is run for every C (speculatively) so that it does not wait for the inner fits; the three
unselected refits are discarded. Usage::

    python scripts/16b_external_baselines_tasks.py run --config configs/hcp181_family_apg.yaml --name hcp181_external \
        --hosts s120:20:1,s140:18:1.3,s141:18:1.3 --host s120 --n-jobs 20
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score

from mlrdegg.datasets import load_derived
from mlrdegg.experiment.cv import outer_splits, split
from mlrdegg.paths import RESULTS_ROOT, resolve

base = importlib.import_module("16_external_baselines")
decoding = importlib.import_module("03_run_decoding")
C_GRID = base.C_GRID
COST = {"inner": 1.0, "svm_refit": 2.5, "lda_refit": 0.3}  # relative run times, only used to order and assign


def build_folds(cfg):
    exp = cfg["experiment"]
    subjects = None if exp.get("subjects", "all") == "all" else exp["subjects"]
    ds = load_derived(resolve(cfg["preprocess"]["out_dir"]), subjects, exclude=exp.get("exclude_subjects"))
    folds = []
    for unit, X, y, groups in decoding.make_units(ds, exp["unit"]):
        labels = np.unique(y)
        for k, (train, test) in enumerate(outer_splits(exp, y, groups)):
            sub_groups = {g: v[train] for g, v in groups.items()}
            inner = split(exp["inner_cv"], y[train], sub_groups)
            folds.append({"unit": unit, "k": k, "X": X, "y": y, "groups": groups, "train": train, "test": test,
                          "inner": inner, "labels": labels})
    return ds, folds


def list_tasks(folds):
    tasks = []
    for f in folds:
        u, k = f["unit"], f["k"]
        for c in C_GRID:
            for j in range(len(f["inner"])):
                tasks.append({"id": f"{u}_fold{k}_inner_C{c:g}_j{j}", "kind": "inner", "unit": u, "k": k, "C": c, "j": j})
        for c in C_GRID:
            tasks.append({"id": f"{u}_fold{k}_svm_refit_C{c:g}", "kind": "svm_refit", "unit": u, "k": k, "C": c})
        tasks.append({"id": f"{u}_fold{k}_lda_refit", "kind": "lda_refit", "unit": u, "k": k})
    for t in tasks:
        t["cost"] = COST[t["kind"]]
    return sorted(tasks, key=lambda t: (-t["cost"], t["id"]))


def parse_hosts(spec):
    hosts = {}
    for part in spec.split(","):
        name, slots, speed = part.split(":")
        hosts[name] = (int(slots), float(speed))
    return hosts


def assign(tasks, hosts):
    """Greedy: each task (longest first) goes to the host that would finish it earliest."""
    load = {h: 0.0 for h in hosts}
    owner = {}
    for t in tasks:
        h = min(hosts, key=lambda h: (load[h] + t["cost"] / hosts[h][1]) / hosts[h][0])
        load[h] += t["cost"] / hosts[h][1]
        owner[t["id"]] = h
    return owner, {h: load[h] / hosts[h][0] for h in hosts}


def save_json(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def run_task(task, f, tasks_dir: Path):
    out = tasks_dir / f"{task['id']}.json"
    if out.exists():
        return task["id"]
    X, y, train, test, labels = f["X"], f["y"], f["train"], f["test"], f["labels"]
    progress = tasks_dir / "progress" / f"{task['id']}.log"
    tag = f"unit={task['unit']} outer_fold={task['k']} {task['id']}"
    t0 = time.time()
    if task["kind"] == "inner":
        tr, te = f["inner"][task["j"]]
        _, loss = base.fit_score(base.svm(task["C"]), X[train], y[train], tr, te, labels, progress, tag)
        payload = {"log_loss": loss}
    else:
        model = base.lda() if task["kind"] == "lda_refit" else base.svm(task["C"])
        pred, loss = base.fit_score(model, X, y, train, test, labels, progress, tag)
        payload = {"log_loss": loss, "accuracy": accuracy_score(y[test], pred), "pred": pred.tolist()}
    payload.update(seconds=time.time() - t0, host=os.environ.get("COMPUTERNAME", ""))
    save_json(out, payload)
    return task["id"]


def tasks_dir_of(name):
    d = RESULTS_ROOT / f"{name}_tasks"
    (d / "progress").mkdir(parents=True, exist_ok=True)
    return d


def cmd_plan(args, tasks):
    hosts = parse_hosts(args.hosts)
    owner, finish = assign(tasks, hosts)
    for h in hosts:
        mine = [t for t in tasks if owner[t["id"]] == h]
        kinds = {k: sum(t["kind"] == k for t in mine) for k in COST}
        print(f"{h}: {len(mine)} tasks {kinds}, relative finish time {finish[h]:.2f} (units of one inner fit)")


def cmd_run(args, tasks, folds):
    if args.hosts:
        owner, _ = assign(tasks, parse_hosts(args.hosts))
        tasks = [t for t in tasks if owner[t["id"]] == args.host]
    tdir = tasks_dir_of(args.name)
    todo = [t for t in tasks if not (tdir / f"{t['id']}.json").exists()]
    by_fold = {(f["unit"], f["k"]): f for f in folds}
    print(f"host={args.host} tasks={len(tasks)} todo={len(todo)} n_jobs={args.n_jobs}", flush=True)
    done = Parallel(n_jobs=args.n_jobs, batch_size=1)(
        delayed(run_task)(t, by_fold[(t["unit"], t["k"])], tdir) for t in todo)
    print(f"finished {len(done)} tasks", flush=True)


def cmd_status(args, tasks):
    tdir = tasks_dir_of(args.name)
    have = [t for t in tasks if (tdir / f"{t['id']}.json").exists()]
    print(f"{len(have)}/{len(tasks)} tasks finished")
    for k in COST:
        print(f"  {k}: {sum(t['kind'] == k for t in have)}/{sum(t['kind'] == k for t in tasks)}")


def cmd_combine(args, ds, folds, tasks):
    tdir = tasks_dir_of(args.name)
    out = RESULTS_ROOT / args.name
    (out / "folds").mkdir(parents=True, exist_ok=True)
    (out / "progress").mkdir(exist_ok=True)
    log = out / "progress.log"
    missing = [t["id"] for t in tasks if t["kind"] != "svm_refit" and not (tdir / f"{t['id']}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} required task files missing, e.g. {missing[:3]}")

    def load(task_id):
        return json.loads((tdir / f"{task_id}.json").read_text(encoding="utf-8"))

    for f in folds:
        u, k, y, groups, test = f["unit"], f["k"], f["y"], f["groups"], f["test"]
        if (out / "folds" / f"{u}_fold{k}_outer.csv").exists():
            continue
        losses = {c: np.mean([load(f"{u}_fold{k}_inner_C{c:g}_j{j}")["log_loss"] for j in range(len(f["inner"]))])
                  for c in C_GRID}
        best = min(C_GRID, key=lambda c: (losses[c], c))
        base.log_line(log, f"SELECT unit={u} outer_fold={k} svm_linear C={best:g} inner_log_loss={losses[best]:.4f}")
        rows, preds = [], []
        for name, task_id, chosen in (("lda_shrinkage", f"{u}_fold{k}_lda_refit", np.nan),
                                      ("svm_linear", f"{u}_fold{k}_svm_refit_C{best:g}", best)):
            r = load(task_id)
            rows.append({"unit": u, "outer_fold": k, "method": name, "accuracy": r["accuracy"], "log_loss": r["log_loss"],
                         "selected_C": chosen, "fit_seconds": r["seconds"]})
            for i, p in zip(test, r["pred"]):
                preds.append({"unit": u, "outer_fold": k, "method": name, "sample": int(i),
                              **{g: groups[g][i] for g in groups}, "y_true": y[i], "y_pred": p})
        pd.DataFrame(preds).to_csv(out / "folds" / f"{u}_fold{k}_predictions.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "folds" / f"{u}_fold{k}_outer.csv", index=False)  # written last = fold complete
        base.log_line(log, f"FOLD_DONE unit={u} outer_fold={k} (combined from task files)")
    outer = sorted((out / "folds").glob("*_outer.csv"))
    pd.concat([pd.read_csv(p) for p in outer]).to_csv(out / "outer_results.csv", index=False)
    pd.concat([pd.read_csv(str(p).replace("_outer.csv", "_predictions.csv")) for p in outer]).to_csv(
        out / "test_predictions.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"config": args.config, "outer_folds": len(outer), "C_grid": C_GRID,
                                                   "subjects": [s.subject for s in ds.subjects]}, indent=2))
    base.log_line(log, f"RUN done outer_folds={len(outer)} (task-level run, 16b_external_baselines_tasks.py)")
    print("done", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run", "status", "combine"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hosts", help="name:slots:relative_speed,... used to assign tasks to hosts")
    ap.add_argument("--host", help="this host's name in --hosts")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    if args.command == "run" and args.hosts and args.host not in parse_hosts(args.hosts):
        raise SystemExit("--host must be one of --hosts")
    cfg = _bootstrap.load_config(args.config)
    ds, folds = build_folds(cfg)
    tasks = list_tasks(folds)
    if args.command == "plan":
        if not args.hosts:
            raise SystemExit("plan needs --hosts")
        cmd_plan(args, tasks)
    elif args.command == "run":
        cmd_run(args, tasks, folds)
    elif args.command == "status":
        cmd_status(args, tasks)
    else:
        cmd_combine(args, ds, folds, tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
