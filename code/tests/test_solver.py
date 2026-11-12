"""Solver-level tests: descent, cross-check against proximal gradient, structure, reductions."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlrdegg import MLRDEGGClassifier
from mlrdegg import likelihood as lik
from mlrdegg.contrast import contrast_matrix
from mlrdegg.proxgrad import solve_proxgrad
from mlrdegg.solver import SolverOptions, SolverState, solve_mm_admm


def make_model(tp, **kw):
    params = dict(n_space=tp["V"], n_time=tp["T"], spatial_graph=tp["spatial"],
                  temporal_graph=tp["temporal"], lambda_sel=0.2, lambda_g=0.2, eta=0.5,
                  lambda_scale="none")
    params.update(kw)
    return MLRDEGGClassifier(**params)


@pytest.mark.parametrize("solver", ["mm_admm", "auto"])
@pytest.mark.parametrize("p_s,p_g", [(1, 2), (1, 1), (0.5, 2), (1, 0.5), (0.5, 0.5), (1.5, 3.0)])
def test_monotone_descent_all_regions(toy_problem, p_s, p_g, solver):
    tp = toy_problem
    m = make_model(tp, p_s=p_s, p_g=p_g, max_outer=150, warm_start_convex=False, solver=solver)
    m.fit(tp["X"], tp["y"])
    obj = [h["objective"] for h in m.history_]
    assert m.status_ in ("converged", "max_outer")
    assert np.all(np.diff(obj) <= 1e-10 * (1 + np.abs(obj[:-1])))


def test_admm_agrees_with_proximal_gradient(toy_problem):
    tp = toy_problem
    m = make_model(tp, p_s=1.0, p_g=2.0, max_outer=3000, tol_obj=1e-12, tol_x=1e-9,
                   solver="mm_admm")
    m.fit(tp["X"], tp["y"])
    K = len(m.classes_)
    H = contrast_matrix(K)
    Xa = lik.augment(tp["X"])
    Y = np.eye(K)[tp["y"]]
    Theta0 = np.zeros((Xa.shape[1], K - 1))
    Theta_pg, hist = solve_proxgrad(Xa, Y, H, m.build_branches(), Theta0, eps=m.epsilon)
    assert abs(m.objective_ - hist[-1]) <= 1e-6 * abs(hist[-1])
    P_admm = m.predict_proba(tp["X"])
    P_pg = lik.softmax(lik.scores(Xa, Theta_pg, H))
    assert np.max(np.abs(P_admm - P_pg)) < 5e-3


def test_exact_auxiliary_thresholding_and_prediction_support(toy_problem):
    tp = toy_problem
    m = make_model(tp, p_s=0.5, p_g=0.5, lambda_sel=1.0, lambda_g=1.0).fit(tp["X"], tp["y"])
    s = m.structure_summary()
    assert s["sel_zero_fraction"] > 0.2
    assert s["sp_aux_zero_fraction"] > 0.2
    assert s["sp_feasibility_norm"] >= 0
    m2 = make_model(tp, p_s=1.5, p_g=3.0, lambda_sel=1.0, lambda_g=1.0).fit(tp["X"], tp["y"])
    assert m2.structure_summary()["sel_zero_fraction"] == 0.0


def test_sum_to_zero_and_prediction(toy_problem):
    tp = toy_problem
    m = make_model(tp).fit(tp["X"], tp["y"])
    assert np.allclose(m.coef_.sum(1), 0)
    assert np.isclose(m.intercept_.sum(), 0)
    assert np.allclose(m.predict_proba(tp["X"]).sum(1), 1)
    assert m.score(tp["X"], tp["y"]) > 0.8


@pytest.mark.parametrize("solver", ["apg", "mm_admm"])
def test_final_diagnostics_use_the_returned_iterate(toy_problem, solver):
    tp = toy_problem
    m = make_model(tp, solver=solver, max_outer=1, apg_max_iter=1).fit(tp["X"], tp["y"])
    Xa = lik.augment(tp["X"])
    Theta = np.vstack([m.B_, m.c_])
    Y = np.eye(len(m.classes_))[np.searchsorted(m.classes_, tp["y"])]
    _, gradient, _ = lik.loss_grad(Xa, Y, Theta, m.H_)
    assert m.solver_ == solver
    assert m.diagnostics_["iteration_limit_hit"] and not m.converged_
    assert m.diagnostics_["final_stationarity_c"] == pytest.approx(np.linalg.norm(gradient[-1]), abs=1e-12)
    assert m.diagnostics_["final_feasibility_norm"] >= 0


def test_binary_reduction_to_sparse_logistic_regression():
    """K=2, lambda_g=0, p_s=1: equals L1 logistic regression under (eq:binary-scaling)."""
    rng = np.random.default_rng(5)
    n, d = 120, 8
    X = rng.normal(size=(n, d))
    beta = np.array([2.0, -1.5, 0, 0, 1.0, 0, 0, 0])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ beta)))).astype(int)
    lam_sel = 0.4
    m = MLRDEGGClassifier(n_space=d, lambda_sel=lam_sel, lambda_g=0.0, p_s=1.0, lambda_scale="none",
                          max_outer=5000, tol_obj=1e-13, tol_x=1e-10).fit(X, y)
    lam1 = n * lam_sel / (np.sqrt(2) * d)
    ref = LogisticRegression(penalty="l1", C=1.0 / lam1, solver="liblinear", tol=1e-8,
                             intercept_scaling=1e4, max_iter=100000).fit(X, y)
    beta_mlr = m.coef_[:, 1] - m.coef_[:, 0]
    assert np.allclose(beta_mlr, ref.coef_[0], atol=5e-3)


def test_woodbury_path_matches_dense(toy_problem):
    tp = toy_problem
    m = make_model(tp)
    branches = m.build_branches()
    K = 3
    H = contrast_matrix(K)
    Xa = lik.augment(tp["X"])
    Y = np.eye(K)[tp["y"]]
    Th0 = np.zeros((Xa.shape[1], K - 1))
    r1 = solve_mm_admm(Xa, Y, H, branches, SolverOptions(max_outer=30, kkt_tol0=None), SolverState(Th0))
    r2 = solve_mm_admm(Xa, Y, H, branches,
                       SolverOptions(max_outer=30, dense_threshold=0, cg_tol=1e-10, kkt_tol0=None),
                       SolverState(Th0))
    assert np.isclose(r1.objective, r2.objective, rtol=1e-6)


def test_lambda_max_scaling(toy_problem):
    """At lambda_sel = lambda_max the group-lasso member is identically zero."""
    tp = toy_problem
    m = make_model(tp, lambda_scale="max", lambda_g=0.0, lambda_sel=1.0).fit(tp["X"], tp["y"])
    assert m.structure_summary()["sel_zero_fraction"] == 1.0
    m = make_model(tp, lambda_scale="max", lambda_g=0.0, lambda_sel=0.8).fit(tp["X"], tp["y"])
    assert m.structure_summary()["sel_zero_fraction"] < 1.0
    assert np.isclose(m.lambda_sel_eff_, 0.8 * m.lambda_max_)



@pytest.mark.parametrize("p_s,p_g,lambda_g", [(1, 2, 0.2), (1.5, 3.0, 0.2), (1, 2, 0.0)])
def test_apg_matches_mm_admm(toy_problem, p_s, p_g, lambda_g):
    """The APG path and MM-ADMM reach the same objective on every APG-eligible member."""
    tp = toy_problem
    kw = dict(p_s=p_s, p_g=p_g, lambda_g=lambda_g, max_outer=5000, tol_obj=1e-12, tol_x=1e-10)
    m_mm = make_model(tp, solver="mm_admm", inner_budget0=None, **kw).fit(tp["X"], tp["y"])
    m_apg = make_model(tp, solver="auto", **kw).fit(tp["X"], tp["y"])
    assert m_apg.fit_log_[-1][1].n_inner_total == 0  # APG was used
    assert m_apg.converged_
    assert abs(m_apg.objective_ - m_mm.objective_) <= 1e-7 * abs(m_mm.objective_)
    assert np.max(np.abs(m_apg.predict_proba(tp["X"]) - m_mm.predict_proba(tp["X"]))) < 1e-3


@pytest.mark.parametrize("inner_budget0", [1, 2, 5])
@pytest.mark.parametrize("p_s,p_g", [(1, 1), (0.5, 0.5), (1, 0.5)])
def test_inexact_inner_matches_exact(toy_problem, p_s, p_g, inner_budget0):
    """Inexact solves retain the objective cross-check and distinguish a numerical stall
    from the requested stopping criterion."""
    tp = toy_problem
    kw = dict(p_s=p_s, p_g=p_g, max_outer=3000, solver="mm_admm", warm_start_convex=False)
    exact = make_model(tp, inner_budget0=None, tol_obj=1e-12, tol_x=1e-10, **kw).fit(tp["X"], tp["y"])
    inexact = make_model(tp, inner_budget0=inner_budget0, tol_obj=1e-9, tol_x=1e-6, **kw)
    inexact.fit(tp["X"], tp["y"])
    assert inexact.status_ in ("converged", "descent_stalled")
    assert inexact.converged_ == (inexact.status_ == "converged")
    tol = 1e-5 if p_s >= 1 and p_g >= 1 else 1e-2  # nonconvex: possibly a different local minimum
    assert inexact.objective_ <= exact.objective_ + tol * abs(exact.objective_)
    obj = [h["objective"] for h in inexact.history_]
    assert np.all(np.diff(obj) <= 1e-10 * (1 + np.abs(obj[:-1])))


@pytest.mark.parametrize("p_s,p_g", [(1, 1), (0.5, 0.5)])
def test_kkt_acceptance_bounds_distance_to_surrogate_minimizer(toy_problem, p_s, p_g):
    """Accepted inexact steps satisfy their residual tolerance (Theorem thm:inexact), except the
    few forced at max_inner, and the objective still decreases monotonically."""
    tp = toy_problem
    m = make_model(tp, p_s=p_s, p_g=p_g, max_outer=60, solver="mm_admm", inner_budget0=2,
                   kkt_tol0=1e-2, kkt_enforce=True, warm_start_convex=False).fit(tp["X"], tp["y"])
    ok = [h["kkt_bound"] <= h["kkt_tol"] for h in m.history_]
    assert len(ok) - sum(ok) == m.n_kkt_violations_ <= 0.05 * len(ok)
    obj = [h["objective"] for h in m.history_]
    assert np.all(np.diff(obj) <= 1e-10 * (1 + np.abs(obj[:-1])))


def test_kkt_error_bound_dominates_true_error(toy_problem):
    """Lemma lem:error-bound: the residual bound is an upper bound on the distance to the exact
    surrogate minimizer, here obtained by a long ADMM run."""
    from mlrdegg.penalties import mm_weights, row_norms
    from mlrdegg.solver import _kkt_error_bound, _op_norm_bound, solve_mm_admm
    tp = toy_problem
    m = make_model(tp, p_s=1, p_g=1)
    branches = m.build_branches()
    K = 3
    H = contrast_matrix(K)
    Xa = lik.augment(tp["X"])
    Y = np.eye(K)[tp["y"]]
    Th0 = np.zeros((Xa.shape[1], K - 1))
    # One MM surrogate (max_outer=1) solved coarsely and near-exactly from the same anchor.
    coarse = solve_mm_admm(Xa, Y, H, branches, SolverOptions(max_outer=1, inner_budget0=3,
                           max_inner=3, kkt_tol0=None), SolverState(Th0))
    exact = solve_mm_admm(Xa, Y, H, branches, SolverOptions(max_outer=1, inner_budget0=None,
                          max_inner=20000, inner_rel_tol_min=1e-12, inner_rel_tol0=1e-12,
                          kkt_tol0=None), SolverState(Th0))
    _, G, _ = lik.loss_grad(Xa, Y, Th0, H)
    kappas, nus = [], []
    for br in branches:
        omega, nu = mm_weights(row_norms(br.apply(Th0[:-1])), br.p, 1e-3)
        kappas.append(br.mu * omega)
        nus.append(nu)
    bound, _ = _kkt_error_bound(Xa, G, coarse.Theta, Th0, 1e-3, branches, kappas, nus,
                                coarse.U, coarse.Z, coarse.rho,
                                [_op_norm_bound(br) for br in branches])
    assert np.linalg.norm(coarse.Theta - exact.Theta) <= bound


def test_ref_scaling_leaves_p1_members_unchanged(toy_problem):
    """Under lambda_scale="ref", group lasso and graph TV are exactly the "max" members."""
    tp = toy_problem
    for p_g, lam_g in ((1.0, 0.0), (1.0, 0.1)):
        kw = dict(lambda_sel=0.1, lambda_g=lam_g, p_s=1.0, p_g=p_g, max_outer=5)
        a = make_model(tp, lambda_scale="ref", **kw).fit(tp["X"], tp["y"])
        b = make_model(tp, lambda_scale="max", **kw).fit(tp["X"], tp["y"])
        for x, y in zip(a.build_branches(), b.build_branches()):
            assert np.allclose(x.mu, y.mu) and x.eps is None


@pytest.mark.parametrize("p_s,p_g", [(0.5, 0.5), (1.0, 2.0), (1.0, 3.0), (1.5, 0.5)])
def test_ref_scaling_matches_p1_penalty_at_reference_radius(toy_problem, p_s, p_g):
    """Every branch equals its p = 1 counterpart at the reference radius, and eps is relative."""
    from mlrdegg.penalties import h_penalty
    tp = toy_problem
    kw = dict(lambda_sel=0.1, lambda_g=0.1, lambda_scale="ref", epsilon_rel=0.2, max_outer=3)
    m = make_model(tp, p_s=p_s, p_g=p_g, **kw).fit(tp["X"], tp["y"])
    ref = {br.name: br for br in m.build_branches(1.0, 1.0)}
    for br in m.build_branches():
        t = m.t_ref_[br.name]
        assert t > 0
        assert br.p == 1.0 or np.isclose(br.eps, 0.2 * t)
        assert np.allclose(br.mu * h_penalty(np.array([t]), br.p, br.offset(1e-3))[0],
                           ref[br.name].mu * t)


@pytest.mark.parametrize("p_g", [1.0, 1.5])
def test_null_model_shortcut_matches_mm_admm(toy_problem, p_g):
    """With the reference-radius scaling (``lambda_scale="ref"``, the estimator default) and
    ``lambda_sel >= 1`` the MM-ADMM members are solved by B = 0 through the group-lasso pilot;
    the full MM-ADMM run reaches the same objective, and below the threshold nothing changes."""
    fast = make_model(toy_problem, p_s=1.0, p_g=p_g, lambda_sel=1.2, lambda_g=0.1, lambda_scale="ref")
    fast.fit(toy_problem["X"], toy_problem["y"])
    assert fast.status_ == "converged_null_model"
    assert np.all(fast.coef_ == 0.0)
    assert all(np.all(U == 0.0) for U in fast.split_.values())
    slow = make_model(toy_problem, p_s=1.0, p_g=p_g, lambda_sel=1.2, lambda_g=0.1, lambda_scale="ref")
    slow._null_model = lambda *a, **k: None
    slow.fit(toy_problem["X"], toy_problem["y"])
    assert fast.objective_ <= slow.objective_ + 1e-6 * abs(slow.objective_)
    assert np.allclose(fast.intercept_, slow.intercept_, atol=1e-3)
    inner = make_model(toy_problem, p_s=1.0, p_g=p_g, lambda_sel=0.2, lambda_g=0.1, lambda_scale="ref")
    inner.fit(toy_problem["X"], toy_problem["y"])
    assert inner.status_ != "converged_null_model"
    assert np.any(inner.coef_ != 0.0)


@pytest.mark.parametrize("p_s,p_g", [(0.5, 0.5), (1.5, 2.0)])
def test_reference_radii_remain_defined_for_zero_graph_differences(p_s, p_g):
    from mlrdegg.graphs import chain_graph
    model = MLRDEGGClassifier(n_space=2, n_time=2, spatial_graph=chain_graph(2),
                             temporal_graph=chain_graph(2), p_s=p_s, p_g=p_g,
                             lambda_sel=.1, lambda_g=.1, max_outer=2)
    with np.errstate(invalid="raise", divide="raise"):
        y = np.tile([0, 1], 12)
        X = np.tile((2*y-1)[:, None], (1, 4))
        model.fit(X, y)
    assert model.lambda_max_ > 0
    assert model.t_ref_["sp"] == model.t_ref_["tm"] == 1.0
    assert model.reference_radius_source_["sp"] == model.reference_radius_source_["tm"] == "unit_scale"
    assert np.isfinite(model.objective_)
    assert np.isfinite(model.predict_proba(X)).all()


def test_zero_gradient_relative_scale_fails_before_empty_radius_calibration():
    model = MLRDEGGClassifier(n_space=2, n_time=2, p_s=.5, p_g=.5)
    with np.errstate(invalid="raise", divide="raise"), pytest.raises(ValueError, match="feature gradient is zero"):
        model.fit(np.zeros((12, 4)), np.tile([0, 1, 2], 4))
