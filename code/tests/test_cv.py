"""Outer cross-validation splits, learning-curve fractions and reproducibility-aware selection."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from mlrdegg.experiment.cv import (FitCache, _pooled, outer_splits, run_nested_cv, select_grid_point,
                                   split_halves)
from mlrdegg.experiment.metrics import cross_fitted_temperature, fit_temperature
from mlrdegg.experiment.models import Context


def test_select_grid_point_rules():
    s = lambda ll, se, acc, r: {"ok": True, "score": -ll, "log_loss": ll, "se": se,  # noqa: E731
                                "accuracy": acc, "repro": r}
    stats = [s(1.00, 0.05, 0.60, 0.30),   # best log loss, poorly reproducible
             s(1.04, 0.05, 0.58, 0.80),   # within one SE, most reproducible
             s(1.20, 0.05, 0.55, 0.95),   # outside one SE
             {"ok": False, "repro": 1.0}]  # failed fits are never selected
    assert select_grid_point(stats, "min") == 0
    assert select_grid_point(stats, "1se_repro") == 1
    # npairs distances to (1, 1): 0.806, 0.458, 0.453 -> the most reproducible point
    assert select_grid_point(stats, "npairs") == 2


def test_pooled_log_loss_matches_concatenated_samples():
    rng = np.random.default_rng(0)
    folds = [rng.exponential(1.0, size=n) for n in (30, 31, 29)]
    rows = [{"n_test": len(v), "log_loss": v.mean(), "log_loss_sd": v.std(ddof=1), "accuracy": 0.5}
            for v in folds]
    allv = np.concatenate(folds)
    acc, mean, se = _pooled(rows)
    assert mean == pytest.approx(allv.mean())
    assert se == pytest.approx(allv.std(ddof=1) / np.sqrt(len(allv)))


def test_split_halves_are_disjoint_and_balanced():
    y = np.repeat(np.array(["a", "b"]), [20, 11])
    idx = np.arange(3, 34)[: len(y)]
    yy = np.empty(40, dtype=object); yy[idx] = y
    h0, h1 = split_halves(idx, yy, seed=3)
    assert not set(h0) & set(h1) and set(h0) | set(h1) == set(idx)
    assert abs(np.sum(yy[h0] == "a") - np.sum(yy[h1] == "a")) <= 1


def test_split_halves_by_group_never_split_a_group():
    groups = np.repeat(np.arange(7), 8)
    y = np.tile(np.repeat(np.array(["a", "b"]), 4), 7)
    idx = np.flatnonzero(groups != 2)
    h0, h1 = split_halves(idx, y, seed=5, groups=groups)
    assert not set(h0) & set(h1) and set(h0) | set(h1) == set(idx)
    assert not set(groups[h0]) & set(groups[h1])
    assert abs(len(set(groups[h0])) - len(set(groups[h1]))) <= 1
    assert np.array_equal(split_halves(idx, y, seed=5, groups=groups)[0], h0)


def test_nested_cv_with_group_split_halves():
    rng = np.random.default_rng(2)
    V, T, n_subj, per = 6, 2, 6, 8
    subj = np.repeat(np.arange(n_subj), per)
    y = np.tile(np.repeat(np.array(["a", "b", "c", "d"]), per // 4), n_subj)
    X = rng.normal(size=(len(y), V * T)) + (np.searchsorted(np.unique(y), y)[:, None] == np.arange(V * T) % 4)
    methods = [{"name": "gl", "model": "mlrdegg", "fixed": {"p_s": 1.0, "p_g": 2.0, "lambda_g": 0.0},
                "grid": {"lambda_sel": [0.3, 0.05]}}]
    exp = {"outer_cv": {"type": "leave_one_group_out", "group": "subject", "max_folds": 2},
           "inner_cv": {"type": "group_kfold", "n_splits": 2, "group": "subject"},
           "selection_rules": ["min", "1se_repro"], "split_half_group": "subject",
           "solver": {"max_outer": 50}}
    with threadpool_limits(1):
        outer, inner, maps = run_nested_cv("u", X, y, {"subject": subj}, methods,
                                           Context(V, T, None, None), exp, 1)
    assert {r["method"] for r in outer} == {"gl", "gl@1se_repro"}
    assert all(np.isfinite(r["repro"]) for r in inner)


def test_nested_cv_with_reproducibility_rules():
    rng = np.random.default_rng(1)
    V, T, n = 6, 2, 80
    y = np.repeat(np.array(["a", "b", "c", "d"]), n // 4)
    X = rng.normal(size=(n, V * T)) + (np.searchsorted(np.unique(y), y)[:, None] == np.arange(V * T) % 4)
    methods = [{"name": "l2", "model": "logreg_l2", "grid": {"C": [0.01, 1.0]}},
               {"name": "gl", "model": "mlrdegg", "fixed": {"p_s": 1.0, "p_g": 2.0, "lambda_g": 0.0},
                "grid": {"lambda_sel": [0.3, 0.05]}}]
    exp = {"outer_cv": {"type": "stratified_kfold", "n_splits": 2, "seed": 0},
           "inner_cv": {"type": "stratified_kfold", "n_splits": 2, "seed": 0},
           "selection_rules": ["min", "npairs", "1se_repro"], "solver": {"max_outer": 50}}
    with threadpool_limits(1):  # n_jobs = 1 leaves BLAS threads unlimited; slow on a busy machine
        outer, inner, maps = run_nested_cv("u", X, y, {}, methods, Context(V, T, None, None), exp, 1)
    names = {r["method"] for r in outer}
    assert names == {"l2", "l2@npairs", "l2@1se_repro", "gl", "gl@npairs", "gl@1se_repro"}
    assert all(r["error"] == "" and 0 <= r["accuracy"] <= 1 for r in outer)
    assert all(-1 <= r["repro"] <= 1 for r in inner)
    assert maps["gl@npairs"].shape == (2, V, T)


def test_outer_predictions_are_saved_and_match_the_reported_accuracy():
    rng = np.random.default_rng(4)
    V, T, n = 6, 2, 80
    y = np.repeat(np.array(["a", "b", "c", "d"]), n // 4)
    X = rng.normal(size=(n, V * T)) + (np.searchsorted(np.unique(y), y)[:, None] == np.arange(V * T) % 4)
    methods = [{"name": "l2", "model": "logreg_l2", "grid": {"C": [0.01, 1.0]}}]
    exp = {"outer_cv": {"type": "stratified_kfold", "n_splits": 2, "seed": 0},
           "inner_cv": {"type": "stratified_kfold", "n_splits": 2, "seed": 0},
           "save_predictions": True, "solver": {}}
    preds = []
    with threadpool_limits(1):
        outer, _, _ = run_nested_cv("u", X, y, {}, methods, Context(V, T, None, None), exp, 1,
                                    predictions=preds)
    assert len(preds) == len(outer) == 2 and all("pred" not in r for r in outer)
    for p, r in zip(preds, outer):
        assert np.mean(p["y_true"] == p["y_pred"]) == pytest.approx(r["accuracy"])
        assert np.array_equal(y[p["index"]], p["y_true"])
    assert sorted(np.concatenate([p["index"] for p in preds])) == list(range(n))


def _overconfident(rng, n, K=4, tau=2.5):
    """Labels drawn from softmax(z) and predictions softmax(tau * z): overconfident by ``tau``."""
    z = rng.normal(scale=1.5, size=(n, K))
    p_true = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    idx = np.array([rng.choice(K, p=p) for p in p_true])
    p_model = np.exp(tau * z) / np.exp(tau * z).sum(1, keepdims=True)
    return p_model, idx


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    p, idx = _overconfident(rng, 4000, tau=2.5)
    assert fit_temperature(p, idx) == pytest.approx(2.5, rel=0.1)
    p1, idx1 = _overconfident(rng, 4000, tau=1.0)
    assert fit_temperature(p1, idx1) == pytest.approx(1.0, rel=0.1)


def test_cross_fitted_temperature_uses_other_folds_only():
    rng = np.random.default_rng(1)
    folds = [_overconfident(rng, 300, tau=3.0) for _ in range(3)]
    res = cross_fitted_temperature([p for p, _ in folds], [i for _, i in folds])
    for k, (tau, loss) in enumerate(res):
        others = [j for j in range(3) if j != k]
        expect = fit_temperature(np.concatenate([folds[j][0] for j in others]),
                                 np.concatenate([folds[j][1] for j in others]))
        assert tau == pytest.approx(expect)
        raw = -np.mean(np.log(folds[k][0][np.arange(300), folds[k][1]]))
        assert loss < raw  # softening overconfident predictions lowers the log loss


def test_nested_cv_with_temperature_scaled_selection():
    rng = np.random.default_rng(2)
    V, T, n = 6, 2, 80
    y = np.repeat(np.array(["a", "b", "c", "d"]), n // 4)
    X = rng.normal(size=(n, V * T)) + (np.searchsorted(np.unique(y), y)[:, None] == np.arange(V * T) % 4)
    methods = [{"name": "gl", "model": "mlrdegg", "fixed": {"p_s": 1.0, "p_g": 2.0, "lambda_g": 0.0},
                "grid": {"lambda_sel": [0.3, 0.05]}}]
    exp = {"outer_cv": {"type": "stratified_kfold", "n_splits": 2, "seed": 0},
           "inner_cv": {"type": "stratified_kfold", "n_splits": 3, "seed": 0},
           "selection_metric": "neg_log_loss_ts", "solver": {"max_outer": 50}}
    with threadpool_limits(1):
        outer, inner, _ = run_nested_cv("u", X, y, {}, methods, Context(V, T, None, None), exp, 1)
    assert all(r["error"] == "" for r in outer) and len(outer) == 2
    assert all(r["temperature"] > 0 and np.isfinite(r["log_loss_ts"]) for r in inner)
    assert not any("proba" in r or "y_index" in r for r in inner)  # arrays never reach the CSV
    assert FitCache(None, exp).salt != FitCache(None, {**exp, "selection_metric": "neg_log_loss"}).salt


def test_train_fraction_is_stratified_nested_and_reproducible():
    y = np.repeat(np.array(["a", "b", "c", "d"]), [40, 30, 20, 10])
    exp = {"outer_cv": {"type": "stratified_kfold", "n_splits": 5, "seed": 1}}
    full = outer_splits(exp, y, {})
    half = outer_splits({**exp, "train_fraction": 0.5}, y, {})
    again = outer_splits({**exp, "train_fraction": 0.5}, y, {})
    for (tr_f, te_f), (tr_h, te_h), (tr_a, _) in zip(full, half, again):
        np.testing.assert_array_equal(te_f, te_h)          # test folds unchanged
        assert set(tr_h) <= set(tr_f)                      # a subset of the full training set
        np.testing.assert_array_equal(tr_h, tr_a)          # reproducible
        _, n_full = np.unique(y[tr_f], return_counts=True)
        _, n_half = np.unique(y[tr_h], return_counts=True)
        np.testing.assert_array_equal(n_half, np.maximum(1, np.round(0.5 * n_full)))


def test_candidate_only_halves_preserve_selection_and_reduce_actual_tasks(monkeypatch):
    import mlrdegg.experiment.cv as cv
    half_calls = []
    def evaluate(method, params, X, y, tr, te, ctx, solver, save, threads, **kw):
        return ({"error": "", "converged": True, "accuracy": .5, "log_loss": params["x"],
                 "log_loss_sd": 0.0, "n_test": len(te)}, np.ones((2, 1)))
    def coefficient(method, params, X, y, idx, ctx, solver, threads):
        half_calls.append(params["x"])
        return {"error": "", "converged": True}, np.array([0., 1., 2.])
    monkeypatch.setattr(cv, "_fit_eval", evaluate)
    monkeypatch.setattr(cv, "_fit_coef", coefficient)
    y = np.tile([0, 1], 12)
    exp = {"outer_cv": {"type": "stratified_kfold", "n_splits": 2, "seed": 1},
           "inner_cv": {"type": "stratified_kfold", "n_splits": 2, "seed": 2},
           "selection_rules": ["min", "1se_repro"]}
    method = {"name": "m", "grid": {"x": [1, 2, 3]}}
    args = ("u", np.ones((24, 2)), y, {}, [method], Context(2, 1, None, None))
    full, _, _ = cv.run_nested_cv(*args, exp, 1)
    assert len(half_calls) == 12
    half_calls.clear()
    reduced, _, _ = cv.run_nested_cv(*args, {**exp, "repro_candidates_only": True}, 1)
    assert len(half_calls) == 4
    assert [(r["selected"], r["repro"]) for r in full] == [(r["selected"], r["repro"]) for r in reduced]
    assert cv.FitCache(None, {"cache_provenance": "a", "split_half_seed": 1}).salt != cv.FitCache(
        None, {"cache_provenance": "a", "split_half_seed": 2}).salt
