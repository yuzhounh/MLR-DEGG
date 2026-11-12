"""Nested cross-validation for one analysis unit (one subject, or pooled subjects).

Inner folds select one grid point per method by the configured selection metric; the selected
model is refitted on the full outer-training set and evaluated once on the outer test fold.
Standardization is fitted inside every training set, so no test information reaches training.

``selection_metric: neg_log_loss_ts`` ranks grid points by the temperature-scaled log loss of
the inner validation predictions, with the temperature of each inner fold fitted on the other
inner folds, so that overconfident members are not pushed towards over-regularized grid points.

``experiment.selection_rules`` (default ``[min]``) adds rules that also weigh the split-half
reproducibility of the weight maps (see ``select_grid_point``); each rule gets its own outer
refit, reported as method ``<name>@<rule>`` (the first rule keeps the plain name).
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from threadpoolctl import threadpool_limits

from ..residual import FIELDS as RESIDUAL_FIELDS
from ..residual import summarize as summarize_residuals
from .metrics import classification_metrics, cross_fitted_temperature, selection_score
from .models import Context, fit_info, grid_points, importance_map, make_estimator


def split(cv_cfg: dict, y: np.ndarray, groups: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    kind = cv_cfg["type"]
    if kind == "stratified_kfold":
        cv = StratifiedKFold(cv_cfg["n_splits"], shuffle=True, random_state=cv_cfg.get("seed", 0))
        return list(cv.split(np.zeros(len(y)), y))
    if kind == "leave_one_group_out":
        g = groups[cv_cfg["group"]]
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, g))
    if kind == "group_kfold":
        g = groups[cv_cfg["group"]]
        return list(GroupKFold(cv_cfg["n_splits"]).split(np.zeros(len(y)), y, g))
    raise ValueError(f"unknown cv type {kind}")


def stratified_subsample(idx: np.ndarray, y: np.ndarray, fraction: float,
                         rng: np.random.Generator) -> np.ndarray:
    """``round(fraction * n_c)`` indices of every class ``c`` of ``y[idx]`` (at least one), sorted."""
    keep = []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        k = max(1, int(round(fraction * len(members))))
        keep.append(rng.choice(members, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def outer_splits(exp_cfg: dict, y, groups):
    """Outer folds, optionally truncated by ``outer_cv.max_folds`` (used by quick runs).

    ``train_fraction < 1`` keeps a stratified random part of every outer training set (learning
    curves); the test folds are unchanged.
    """
    folds = split(exp_cfg["outer_cv"], y, groups)
    folds = folds[: exp_cfg["outer_cv"].get("max_folds") or len(folds)]
    fraction = exp_cfg.get("train_fraction", 1.0)
    if fraction < 1.0:
        rng = np.random.default_rng(exp_cfg.get("train_fraction_seed", 0))
        folds = [(stratified_subsample(tr, y, fraction, rng), te) for tr, te in folds]
    return folds


def _subset_groups(groups: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in groups.items()}


def inner_fit_quality(rows: list[dict], policy: str = "record") -> dict:
    """Keep exception, numerical-stop and unknown-quality states separate.

    The historical protocol is record: exclude exceptions, retain and flag numerical
    stops. require_converged is an explicitly separate selection policy.
    """
    if policy not in ("record", "require_converged"):
        raise ValueError("inner_fit_policy must be record or require_converged")
    errors = sum(bool(r.get("error")) for r in rows)
    unknown = sum(not r.get("error") and r.get("converged") is None for r in rows)
    unfinished = sum(not r.get("error") and r.get("converged") is not None
                     and not bool(r["converged"]) for r in rows)
    return {"ok": bool(rows) and errors == 0 and
            (policy == "record" or unfinished + unknown == 0),
            "inner_error_count": errors, "inner_nonconverged_count": unfinished,
            "inner_quality_unknown_count": unknown, "inner_fit_policy": policy}


def _fit_eval(method, params, X, y, train, test, ctx, solver_cfg, want_map, n_threads,
              want_proba=False, want_pred=False):
    t0 = time.perf_counter()
    try:
        with threadpool_limits(limits=n_threads):
            pipe = make_estimator(method, params, ctx, solver_cfg).fit(X[train], y[train])
            proba = pipe.predict_proba(X[test])
        out = classification_metrics(y[test], proba, pipe.classes_)
        if want_proba:  # for temperature-scaled selection; removed before the rows are written
            out["proba"] = proba.astype(np.float32)
            out["y_index"] = np.searchsorted(pipe.classes_, y[test])
        if want_pred:  # per-sample predictions of the outer refit; removed before the rows are written
            out["pred"] = np.asarray(pipe.classes_)[np.argmax(proba, axis=1)]
        # Spread of the per-sample log losses, for the standard error of one-SE rules.
        p_true = proba[np.arange(len(test)), np.searchsorted(pipe.classes_, y[test])]
        losses = -np.log(np.clip(p_true, 1e-15, None))
        out["n_test"] = len(test)
        out["log_loss_sd"] = float(losses.std(ddof=1)) if len(test) > 1 else 0.0
        out.update(fit_info(pipe))
        out["fit_time"] = time.perf_counter() - t0
        out["error"] = ""
        imap = importance_map(pipe, ctx) if want_map else None
        return out, imap
    except Exception:
        return {"error": traceback.format_exc(), "fit_time": time.perf_counter() - t0}, None


def coef_map(pipe) -> np.ndarray:
    """Class coefficients on the standardized scale, centred over classes, flattened ``(d * K,)``."""
    clf = pipe.named_steps["clf"]
    W = clf.coef_ if hasattr(clf, "split_") else clf.coef_.T  # (d, K)
    return (W - W.mean(axis=1, keepdims=True)).astype(np.float32).ravel()


def _fit_coef(method, params, X, y, train, ctx, solver_cfg, n_threads):
    """Fit on ``train`` only and return the class coefficient map (split-half reproducibility)."""
    t0 = time.perf_counter()
    try:
        with threadpool_limits(limits=n_threads):
            pipe = make_estimator(method, params, ctx, solver_cfg).fit(X[train], y[train])
        out = {"fit_time": time.perf_counter() - t0, "error": "", **fit_info(pipe)}
        return out, coef_map(pipe)
    except Exception:
        return {"error": traceback.format_exc(), "fit_time": time.perf_counter() - t0}, None


def split_halves(idx: np.ndarray, y: np.ndarray, seed: int,
                 groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint halves of ``idx``.

    Without ``groups`` the halves are class balanced (alternate members of each shuffled class).
    With ``groups`` (one label per sample of ``y``, e.g. the participant) whole groups go to a half
    (alternate members of the shuffled group labels), so that no group contributes to both halves;
    the halves are as class balanced as the groups are.
    """
    rng = np.random.default_rng(seed)
    if groups is not None:
        labels = rng.permutation(np.unique(groups[idx]))
        return tuple(np.sort(idx[np.isin(groups[idx], part)]) for part in (labels[::2], labels[1::2]))
    halves = [[], []]
    for c in np.unique(y[idx]):
        members = rng.permutation(idx[y[idx] == c])
        halves[0].append(members[::2])
        halves[1].append(members[1::2])
    return tuple(np.sort(np.concatenate(h)) for h in halves)


REPRO_RULES = ("npairs", "1se_repro")


def _rules(exp_cfg: dict) -> list[str]:
    rules = list(exp_cfg.get("selection_rules", ["min"]))
    unknown = set(rules) - {"min", *REPRO_RULES}
    if unknown:
        raise ValueError(f"unknown selection rules {sorted(unknown)}")
    return rules


def select_grid_point(stats: list[dict], rule: str) -> int:
    """Index of the grid point chosen by ``rule`` from per-point inner statistics.

    ``stats[i]`` holds ``ok`` (all inner fits succeeded), ``score`` (mean selection score, larger
    is better), ``accuracy`` and ``log_loss`` (pooled over the inner validation samples), ``se``
    (standard error of that log loss) and ``repro`` (correlation of the class maps fitted on two
    halves of the outer training set; Strother et al., NPAIRS; Rasmussen et al., 2012).

    * ``"min"``: best mean selection score (the configured ``selection_metric``).
    * ``"npairs"``: smallest distance ``sqrt((1 - accuracy)^2 + (1 - repro)^2)`` to (1, 1).
    * ``"1se_repro"``: most reproducible point whose log loss is within one standard error of the
      smallest log loss.
    """
    ok = [i for i, s in enumerate(stats) if s["ok"]]
    if not ok:
        raise ValueError("No eligible grid point: inspect fit exceptions and numerical-stop policy")
    if rule == "min":
        return max(ok, key=lambda i: stats[i]["score"])
    if rule == "npairs":
        return min(ok, key=lambda i: np.hypot(1 - stats[i]["accuracy"], 1 - stats[i]["repro"]))
    best = min(ok, key=lambda i: stats[i]["log_loss"])
    bound = stats[best]["log_loss"] + stats[best]["se"]
    cand = [i for i in ok if stats[i]["log_loss"] <= bound]
    return max(cand, key=lambda i: (stats[i]["repro"], -stats[i]["log_loss"]))


def _pooled(rows: list[dict]) -> tuple[float, float, float]:
    """Accuracy, log loss and its standard error pooled over inner validation folds."""
    n = np.array([r.get("n_test", 1) for r in rows], float)
    m = np.array([r["log_loss"] for r in rows])
    sd = np.array([r.get("log_loss_sd", 0.0) for r in rows])
    N = n.sum()
    mean = float(np.sum(n * m) / N)
    var = (np.sum((n - 1) * sd**2) + np.sum(n * (m - mean) ** 2)) / max(N - 1, 1)
    acc = float(np.sum(n * np.array([r["accuracy"] for r in rows])) / N)
    return acc, mean, float(np.sqrt(var / N))


def grid_statistics(inner_rows, fold, method, metric="neg_log_loss", policy="record"):
    stats = []
    for gi in range(len(grid_points(method))):
        rows = [r for r in inner_rows if r["outer_fold"] == fold and r["method"] == method["name"]
                and r["grid_index"] == gi]
        state = {**inner_fit_quality(rows, policy), "repro": 0.0}
        if state["ok"]:
            state["score"] = float(np.mean([selection_score(r, metric) for r in rows]))
            state["accuracy"], state["log_loss"], state["se"] = _pooled(rows)
        stats.append(state)
    return stats


def reproducibility_candidates(stats, rules):
    """Points needed for the specified rules, including min's selected-point R."""
    eligible = {i for i, state in enumerate(stats) if state["ok"]}
    if not eligible:
        raise ValueError("No eligible grid point: inspect fit exceptions and numerical-stop policy")
    if "npairs" in rules:
        return eligible
    candidates = set()
    if "min" in rules:
        candidates.add(select_grid_point(stats, "min"))
    if "1se_repro" in rules:
        best = min(eligible, key=lambda i: stats[i]["log_loss"])
        bound = stats[best]["log_loss"] + stats[best]["se"]
        candidates.update(i for i in eligible if stats[i]["log_loss"] <= bound)
    return candidates


def add_temperature_scaled_losses(inner_rows: list[dict], probas: dict) -> None:
    """Add ``temperature`` and ``log_loss_ts`` to the inner rows of every (outer fold, method,
    grid point) whose inner fits all succeeded; each fold's temperature is fitted on the other
    inner folds' validation predictions (:func:`cross_fitted_temperature`)."""
    groups = {}
    for r in inner_rows:
        groups.setdefault((r["outer_fold"], r["method"], r["grid_index"]), []).append(r)
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["inner_fold"])
        pairs = [probas.get((*key, r["inner_fold"])) for r in rows]
        if any(r["error"] for r in rows) or any(p is None for p in pairs):
            continue
        for r, (tau, loss) in zip(rows, cross_fitted_temperature([p for p, _ in pairs],
                                                                  [i for _, i in pairs])):
            r["temperature"], r["log_loss_ts"] = tau, loss


def _indexed(i, fn, args, kwargs):
    return i, fn(*args, **kwargs)


def count_fits(y: np.ndarray, groups: dict[str, np.ndarray], methods: list[dict], exp_cfg: dict) -> int:
    """Upper bound on CV fit tasks; candidate-only halves and shared refits reduce it.

    Cache restores are tasks, not new fits. Convex pilots/warm starts inside one task,
    and separately run fresh-R fits, are not included.
    """
    total = 0
    rules = _rules(exp_cfg)
    n_halves = 2 if set(rules) & set(REPRO_RULES) else 0
    n_points = sum(len(grid_points(m)) for m in methods)
    for tr, _ in outer_splits(exp_cfg, y, groups):
        n_inner = len(split(exp_cfg["inner_cv"], y[tr], _subset_groups(groups, tr)))
        total += n_points * (n_inner + n_halves) + len(methods) * len(rules)
    return total


class FitCache:
    """One file per finished fit, so an interrupted run resumes without repeating work.

    The key covers the unit, stage, folds, method definition, grid point and the solver/CV
    settings; changing any of them invalidates the cached result.  ``fits.csv`` (next to the
    cache directory) receives one line per newly finished fit for monitoring.
    """

    def __init__(self, directory: Path | None, exp_cfg: dict):
        self.dir = directory
        salt = {k: exp_cfg.get(k) for k in ("solver", "outer_cv", "inner_cv")}
        for k in ("train_fraction", "train_fraction_seed"):  # only when set: keeps older keys
            if k in exp_cfg:
                salt[k] = exp_cfg[k]
        # Provenance-aware new runs invalidate keys when inputs, code or half seeds change.
        # Historical configs remain readable with their original keys for result audits.
        if "cache_provenance" in exp_cfg:
            salt["cache_provenance"] = exp_cfg["cache_provenance"]
            salt["split_half_seed"] = exp_cfg.get("split_half_seed", 0)
            salt["split_half_group"] = exp_cfg.get("split_half_group")
        if exp_cfg.get("selection_metric") == "neg_log_loss_ts":  # inner fits keep probabilities
            salt["selection_metric"] = "neg_log_loss_ts"
        self.salt = json.dumps(salt, sort_keys=True, default=str)
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, meta: dict, method: dict) -> Path | None:
        if self.dir is None:
            return None
        text = json.dumps([meta, method, self.salt], sort_keys=True, default=str)
        return self.dir / (hashlib.sha1(text.encode()).hexdigest() + ".pkl")

    @staticmethod
    def load(path: Path | None):
        if path is None or not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception:  # truncated file from a hard kill: recompute
            return None

    def save(self, path: Path | None, result, meta: dict) -> None:
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        joblib.dump(result, tmp)
        tmp.replace(path)
        out = result[0]
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "unit": meta["unit"],
               "stage": meta["stage"], "outer_fold": meta["outer_fold"],
               "inner_fold": meta["inner_fold"], "method": meta["method"],
               "params": json.dumps(meta["params"]),
               "fit_time_s": round(out.get("fit_time", float("nan")), 1),
               "status": "error" if out.get("error") else out.get("status", ""),
               "n_outer": out.get("n_outer", ""), "accuracy": out.get("accuracy", "")}
        log_path = self.dir.parent / "fits.csv"
        new = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        # Keep the historical fits.csv schema intact when resuming older runs.
        fields = ("solver", "converged", "numerical_stopping", "stationarity_certified",
                  "iteration_limit_hit", "stationarity_measure", "final_stationarity_B",
                  "final_stationarity_c", "final_subgradient_distance", "final_feasibility_norm",
                  "final_dual_residual", "final_surrogate_error_bound", "main_status",
                  "main_converged", "convex_warm_start_status", "convex_warm_start_converged")
        diagnostics = {**{k: row[k] for k in ("time", "unit", "stage", "outer_fold", "inner_fold", "method", "params")},
                       "cache_key": path.stem, "error": out.get("error", ""),
                       **{k: out.get(k, "") for k in fields}}
        diagnostic_path = self.dir.parent / "fit_diagnostics.csv"
        new_diagnostics = not diagnostic_path.exists()
        with open(diagnostic_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(diagnostics))
            if new_diagnostics:
                writer.writeheader()
            writer.writerow(diagnostics)
        # Post-fit common residual (report only) in its own file, so older schemas stay intact.
        if "common_residual_1e-03" in out:
            residual = {**{k: diagnostics[k] for k in ("time", "unit", "stage", "outer_fold",
                                                         "inner_fold", "method", "params", "cache_key")},
                        **{k: out.get(k, "") for k in RESIDUAL_FIELDS}}
            residual_path = self.dir.parent / "fit_residuals.csv"
            new_residual = not residual_path.exists()
            with open(residual_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(residual))
                if new_residual:
                    writer.writeheader()
                writer.writerow(residual)


def _run(tasks, metas, mdefs, n_jobs, progress, cache: FitCache):
    """Run tasks, reusing cached results; progress advances as each fit completes (any order)."""
    results = [None] * len(tasks)
    paths = [cache.path(meta, m) for meta, m in zip(metas, mdefs)]
    pending = []
    for k, p in enumerate(paths):
        hit = cache.load(p)
        if hit is not None:
            results[k] = hit
        else:
            pending.append(k)
    if progress is not None and len(pending) < len(tasks):
        progress.skip(len(tasks) - len(pending))
    if pending:
        jobs = []
        for k in pending:
            fn, args, kwargs = tasks[k]
            jobs.append(delayed(_indexed)(k, fn, args, kwargs))
        for k, r in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs):
            results[k] = r
            cache.save(paths[k], r, metas[k])
            if progress is not None:
                progress.update(1)
    return results


def run_nested_cv(unit: str, X: np.ndarray, y: np.ndarray, groups: dict[str, np.ndarray],
                  methods: list[dict], ctx: Context, exp_cfg: dict, n_jobs: int, log=None,
                  progress=None, cache_dir: Path | None = None, predictions: list | None = None):
    """Returns ``(outer_rows, inner_rows, maps)``; ``maps[method]`` has shape ``(n_outer, V, T)``.

    With ``experiment.save_predictions`` the outer refits also record the predicted label of every
    test sample; one dict per (outer fold, method) with the sample indices, true and predicted
    labels is appended to ``predictions``."""
    solver_cfg = exp_cfg.get("solver", {})
    save_pred = bool(exp_cfg.get("save_predictions")) and predictions is not None
    metric = exp_cfg.get("selection_metric", "neg_log_loss")
    calibrate = metric == "neg_log_loss_ts"
    n_threads = 1 if n_jobs != 1 else None
    outer = outer_splits(exp_cfg, y, groups)
    cache = FitCache(cache_dir, exp_cfg)

    # Phase 1: inner validation of every grid point.
    tasks, keys, metas, mdefs = [], [], [], []
    for f, (tr, _) in enumerate(outer):
        inner = split(exp_cfg["inner_cv"], y[tr], _subset_groups(groups, tr))
        for m in methods:
            for gi, params in enumerate(grid_points(m)):
                for i, (itr, iva) in enumerate(inner):
                    keys.append((f, m["name"], gi, i))
                    metas.append({"unit": unit, "stage": "inner", "outer_fold": f,
                                  "method": m["name"], "params": params, "inner_fold": i})
                    mdefs.append(m)
                    tasks.append(delayed(_fit_eval)(m, params, X, y, tr[itr], tr[iva], ctx,
                                                    solver_cfg, False, n_threads,
                                                    want_proba=calibrate))
    if log:
        log.info("%s: %d inner fits", unit, len(tasks))
    if progress is not None:
        progress.set_stage(f"{unit} inner")
    res = _run(tasks, metas, mdefs, n_jobs, progress, cache)
    inner_rows, probas = [], {}
    for (f, name, gi, i), (out, _) in zip(keys, res):
        method = next(m for m in methods if m["name"] == name)
        out = dict(out)
        p, yi = out.pop("proba", None), out.pop("y_index", None)
        if p is not None:
            probas[(f, name, gi, i)] = (p, yi)
        inner_rows.append({"unit": unit, "outer_fold": f, "method": name, "grid_index": gi,
                           "inner_fold": i, **grid_points(method)[gi], **out})
    if calibrate:
        add_temperature_scaled_losses(inner_rows, probas)

    # Phase 1b: all points by default, or just points needed by the selection rules.
    rules = _rules(exp_cfg)
    repro = {}
    if set(rules) & set(REPRO_RULES):
        tasks, keys, metas, mdefs = [], [], [], []
        half_groups = groups[exp_cfg["split_half_group"]] if exp_cfg.get("split_half_group") else None
        for f, (tr, _) in enumerate(outer):
            halves = split_halves(tr, y, exp_cfg.get("split_half_seed", 0) + f, half_groups)
            for m in methods:
                candidates = None
                if exp_cfg.get("repro_candidates_only"):
                    stats = grid_statistics(inner_rows, f, m, metric, exp_cfg.get("inner_fit_policy", "record"))
                    candidates = reproducibility_candidates(stats, rules)
                    if progress is not None:
                        progress.omit(2 * (len(grid_points(m)) - len(candidates)))
                for gi, params in enumerate(grid_points(m)):
                    if candidates is not None and gi not in candidates:
                        continue
                    for h, idx in enumerate(halves):
                        keys.append((f, m["name"], gi, h))
                        metas.append({"unit": unit, "stage": "half", "outer_fold": f,
                                      "method": m["name"], "params": params, "inner_fold": h})
                        mdefs.append(m)
                        tasks.append(delayed(_fit_coef)(m, params, X, y, idx, ctx, solver_cfg,
                                                        n_threads))
        if progress is not None:
            progress.set_stage(f"{unit} split-half")
        res = _run(tasks, metas, mdefs, n_jobs, progress, cache)
        coefs = {}
        half_info = {}
        for key, (info, W) in zip(keys, res):
            coefs.setdefault(key[:3], {})[key[3]] = W
            half_info[key] = info
        for key, pair in coefs.items():
            a, b = pair.get(0), pair.get(1)
            ok = a is not None and b is not None and a.std() > 0 and b.std() > 0
            repro[key] = float(np.corrcoef(a, b)[0, 1]) if ok else 0.0
        for r in inner_rows:
            r["repro"] = repro.get((r["outer_fold"], r["method"], r["grid_index"]), np.nan)
            for half in (0, 1):
                info = half_info.get((r["outer_fold"], r["method"], r["grid_index"], half), {})
                for field in ("status", "converged", "solver", "iteration_limit_hit",
                              "final_stationarity_B", "final_stationarity_c", "final_feasibility_norm",
                              "common_residual", "common_residual_1e-03", "common_residual_1e-05"):
                    r[f"half{half}_{field}"] = info.get(field, np.nan)

    # Selection per outer fold, method and rule; failed inner fits exclude a grid point.
    selected = {}
    for f in range(len(outer)):
        for m in methods:
            stats = grid_statistics(inner_rows, f, m, metric, exp_cfg.get("inner_fit_policy", "record"))
            for gi, state in enumerate(stats):
                state["repro"] = repro.get((f, m["name"], gi), 0.0)
            for rule in rules:
                gi = select_grid_point(stats, rule)
                selected[(f, m["name"], rule)] = (gi, stats[gi])

    # Phase 2: refit each selected model on the outer-training set (once per distinct point).
    tasks, keys, metas, mdefs, n_dup = [], [], [], [], 0
    for f, (tr, te) in enumerate(outer):
        for m in methods:
            for gi in dict.fromkeys(selected[(f, m["name"], rule)][0] for rule in rules):
                keys.append((f, m["name"], gi))
                meta = {"unit": unit, "stage": "refit", "outer_fold": f, "method": m["name"],
                        "params": grid_points(m)[gi], "inner_fold": -1}
                if save_pred:  # only then: keeps the cache keys of older runs valid
                    meta["pred"] = True
                metas.append(meta)
                mdefs.append(m)
                tasks.append(delayed(_fit_eval)(m, grid_points(m)[gi], X, y, tr, te, ctx,
                                                solver_cfg, exp_cfg.get("save_maps", True),
                                                n_threads, want_pred=save_pred))
            n_dup += len(rules) - len({selected[(f, m["name"], r)][0] for r in rules})
    if progress is not None:
        progress.set_stage(f"{unit} refit")
        if n_dup:
            progress.omit(n_dup)
    refits = dict(zip(keys, _run(tasks, metas, mdefs, n_jobs, progress, cache)))
    outer_rows, maps = [], {}
    for f, (tr, te) in enumerate(outer):
        for m in methods:
            for k, rule in enumerate(rules):
                gi, s = selected[(f, m["name"], rule)]
                out, imap = refits[(f, m["name"], gi)]
                out = dict(out)
                pred = out.pop("pred", None)
                name = m["name"] if k == 0 else f"{m['name']}@{rule}"
                if pred is not None:
                    predictions.append({"outer_fold": f, "method": name, "index": te,
                                        "y_true": y[te], "y_pred": pred})
                params = grid_points(m)[gi]
                extra = {"rule": rule, "inner_accuracy": s.get("accuracy", np.nan),
                          "repro": repro.get((f, m["name"], gi), np.nan)} if len(rules) > 1 else {}
                extra.update({key: s.get(key) for key in (
                    "inner_error_count", "inner_nonconverged_count",
                    "inner_quality_unknown_count", "inner_fit_policy")})
                outer_rows.append({"unit": unit, "outer_fold": f, "method": name,
                                   "n_train": len(tr), "n_test": len(te), "selected": str(params),
                                   **params, **extra, **out})
                if imap is not None:
                    maps.setdefault(name, []).append(imap)
    maps = {k: np.stack(v) for k, v in maps.items()}
    if log:
        log.info("%s inner fits: %s", unit, summarize_residuals(inner_rows))
        log.info("%s outer refits: %s", unit, summarize_residuals(outer_rows))
    return outer_rows, inner_rows, maps
