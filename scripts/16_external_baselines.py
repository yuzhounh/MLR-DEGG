"""External baseline decoders on the frozen ERP CORE and HCP designs (B017; review item B015-E04-R04).

Standalone: imports the data loader, unit construction and fold functions of the main pipeline without
changing them, and fits two standard linear decoders on exactly the same participants, outer folds and
inner folds as the confirmation runs:

* shrinkage LDA (analytic Ledoit-Wolf shrinkage, ``solver="lsqr"``; no tuning);
* linear SVM with Platt-scaled probabilities; ``C`` in {1e-3, 1e-2, 1e-1, 1} chosen by the smallest mean
  inner-fold log-loss (the paper's *min* rule).

Features are standardized within each training set. Progress: every fit appends a timestamped line to
results/<name>/progress/<unit>_fold<k>.log (model, C, inner fold, seconds). Each finished outer fold is saved at
once to results/<name>/folds/; a rerun skips finished folds. When all folds are present the combined
outer_results.csv and test_predictions.csv are written. Usage::

    python scripts/16_external_baselines.py --config configs/erpcore_n170.yaml --name erpcore_n170_external
    python scripts/16_external_baselines.py --config configs/hcp_wm_surf_final.yaml --name hcp_wm_surf_external
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mlrdegg.datasets import load_derived
from mlrdegg.experiment.cv import outer_splits, split
from mlrdegg.paths import RESULTS_ROOT, resolve

decoding = importlib.import_module("03_run_decoding")
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def lda():
    return Pipeline([("scale", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def svm(c):
    return Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="linear", C=c, probability=True, random_state=0))])


def log_line(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}\n")
        f.flush()


def fit_score(model, X, y, train, test, labels, progress, tag):
    t0 = time.time()
    log_line(progress, f"START {tag}")
    model.fit(X[train], y[train])
    proba = model.predict_proba(X[test])
    loss = log_loss(y[test], proba, labels=labels)
    log_line(progress, f"DONE  {tag} seconds={time.time() - t0:.1f} log_loss={loss:.4f}")
    return model.predict(X[test]), loss


def outer_fold(unit, k, X, y, groups, train, test, inner_cfg, labels, out):
    # One progress file per fold: concurrent appends to a shared file can lose lines on Windows.
    progress = out / "progress" / f"{unit}_fold{k}.log"
    fold_tag = f"unit={unit} outer_fold={k}"
    t_fold = time.time()
    sub_groups = {g: v[train] for g, v in groups.items()}
    inner = split(inner_cfg, y[train], sub_groups)
    losses = {}
    for c in C_GRID:
        losses[c] = np.mean([fit_score(svm(c), X[train], y[train], tr, te, labels, progress,
                                       f"{fold_tag} svm_linear C={c:g} inner_fold={j}")[1]
                             for j, (tr, te) in enumerate(inner)])
    best = min(C_GRID, key=lambda c: (losses[c], c))
    log_line(progress, f"SELECT {fold_tag} svm_linear C={best:g} inner_log_loss={losses[best]:.4f}")
    rows, preds = [], []
    for name, model, chosen in (("lda_shrinkage", lda(), np.nan), ("svm_linear", svm(best), best)):
        t0 = time.time()
        pred, loss = fit_score(model, X, y, train, test, labels, progress, f"{fold_tag} {name} refit")
        rows.append({"unit": unit, "outer_fold": k, "method": name, "accuracy": accuracy_score(y[test], pred),
                     "log_loss": loss, "selected_C": chosen, "fit_seconds": time.time() - t0})
        for i, p in zip(test, pred):
            preds.append({"unit": unit, "outer_fold": k, "method": name, "sample": int(i),
                          **{g: groups[g][i] for g in groups}, "y_true": y[i], "y_pred": p})
    folds = out / "folds"
    pd.DataFrame(preds).to_csv(folds / f"{unit}_fold{k}_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(folds / f"{unit}_fold{k}_outer.csv", index=False)  # written last = fold complete
    log_line(progress, f"FOLD_DONE {fold_tag} seconds={time.time() - t_fold:.1f}")
    return k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    cfg = _bootstrap.load_config(args.config)
    exp = cfg["experiment"]
    subjects = None if exp.get("subjects", "all") == "all" else exp["subjects"]
    ds = load_derived(resolve(cfg["preprocess"]["out_dir"]), subjects, exclude=exp.get("exclude_subjects"))
    out = RESULTS_ROOT / args.name
    (out / "folds").mkdir(parents=True, exist_ok=True)
    (out / "progress").mkdir(exist_ok=True)
    progress = out / "progress.log"  # run-level start/end lines; per-fold lines are in progress/
    jobs, skipped, n_inner = [], 0, 0
    for unit, X, y, groups in decoding.make_units(ds, exp["unit"]):
        labels = np.unique(y)
        for k, (train, test) in enumerate(outer_splits(exp, y, groups)):
            if (out / "folds" / f"{unit}_fold{k}_outer.csv").exists():
                skipped += 1
                continue
            if not n_inner:
                n_inner = len(split(exp["inner_cv"], y[train], {g: v[train] for g, v in groups.items()}))
            jobs.append((unit, k, X, y, groups, train, test, exp["inner_cv"], labels, out))
    fits_per_fold = len(C_GRID) * n_inner + 2
    log_line(progress, f"RUN start config={args.config} outer_folds_to_run={len(jobs)} skipped_done={skipped} "
                       f"fits_per_fold={fits_per_fold} total_fits={fits_per_fold * len(jobs)} n_jobs={args.n_jobs}")
    Parallel(n_jobs=args.n_jobs)(delayed(outer_fold)(*j) for j in jobs)
    outer = sorted((out / "folds").glob("*_outer.csv"))
    pd.concat([pd.read_csv(f) for f in outer]).to_csv(out / "outer_results.csv", index=False)
    pd.concat([pd.read_csv(str(f).replace("_outer.csv", "_predictions.csv")) for f in outer]).to_csv(
        out / "test_predictions.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"config": args.config, "outer_folds": len(outer), "C_grid": C_GRID,
                                                   "subjects": [s.subject for s in ds.subjects]}, indent=2))
    log_line(progress, f"RUN done outer_folds={len(outer)}")
    print("done", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
