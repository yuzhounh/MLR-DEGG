"""Necessary behaviour checks of the B048 benchmark (plan Section 11): objective identity,
equivalent decomposition, proximal accuracy and exit states.  Small instances only."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from mlrdegg.penalties import row_norms
from mlrdegg.solver import objective as production_objective
from mlrdegg.solver_benchmark import instances as I
from mlrdegg.solver_benchmark import methods as M
from mlrdegg.solver_benchmark.core import (JointProx, Problem, RowFn, compensated_root,
                                           curvature_bound, h_shift, load_instance, nonconvex_prox)
from mlrdegg.solver_benchmark.evaluate import Evaluator, band_status, dual_lower_bound

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "code/configs/solver_benchmark.yaml", encoding="utf-8"))


@pytest.fixture(scope="module")
def inst_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("sb_instances")
    data = I.build_dataset(CFG, "A", 2026, ROOT)
    for point in ("C0", "C1", "C2", "C5", "N1", "N3"):
        I.build_instance(CFG, data, "A", 2026, point, "S1", out)
    return out


def problem(inst_dir, point):
    return Problem(load_instance(inst_dir / f"A_s2026_{point}_S1.npz"))


def test_templates_and_data():
    data = I.build_dataset(CFG, "A", 2026, ROOT)
    norms = np.linalg.norm(data["templates"].reshape(4, -1), axis=1)
    assert np.allclose(norms, CFG["simulation"]["template_fro_norm"])
    assert np.all((data["templates"] != 0).reshape(4, -1).any(axis=1))  # no shape vanished
    assert np.all(np.bincount(data["ytr"]) == 24) and np.all(np.bincount(data["yte"]) == 128)
    assert np.allclose(data["Xtr"].mean(axis=0), 0, atol=1e-12)
    assert np.allclose(data["Xtr"].std(axis=0), 1)
    assert abs(data["Xte"].mean()) > 0  # test split uses the training transform only


def test_objective_identity_and_decomposition(inst_dir):
    rng = np.random.default_rng(0)
    for point in ("C1", "C5", "N1", "N3"):
        p = problem(inst_dir, point)
        T = p.theta0 + 0.05 * rng.normal(size=p.shape)
        F = p.objective(T)
        assert F == pytest.approx(production_objective(p.Xa, p.Y, p.H, T, p.branches, 1e-3), abs=0)
        fS, _ = p.smooth_S(T)
        assert fS + p.P_value(T) == pytest.approx(F, rel=1e-13, abs=1e-13)


def test_redistribution_gradient(inst_dir):
    p = problem(inst_dir, "N3")
    rng = np.random.default_rng(1)
    T = p.theta0 + 0.02 * rng.normal(size=p.shape)
    f, G = p.smooth_S(T)
    D = rng.normal(size=p.shape)
    h = 1e-6
    fd = (p.smooth_S(T + h * D)[0] - p.smooth_S(T - h * D)[0]) / (2 * h)
    assert fd == pytest.approx(float(np.sum(G * D)), rel=1e-6, abs=1e-9)


def test_radial_proxes_match_grid():
    rng = np.random.default_rng(2)
    V = rng.normal(size=(300, 3)) * rng.uniform(0, 2, size=(300, 1))
    tau = rng.uniform(0.01, 1.0, size=300)
    p, eps = 0.5, 0.02
    s = row_norms(V)
    grid = np.linspace(0, 1, 100001)[None, :] * s[:, None]
    t = row_norms(nonconvex_prox(V, tau, p, eps))
    q = lambda r: 0.5 * (r - s[:, None] if r.ndim == 2 else r - s) ** 2  # noqa: E731
    best = (q(grid) + tau[:, None] * h_shift(grid, p, eps)).min(axis=1)
    assert np.all(q(t) + tau * h_shift(t, p, eps) <= best + 1e-12)
    k = curvature_bound(p, eps)
    tc = compensated_root(s, tau, p, eps, k)
    phi = lambda r: h_shift(r, p, eps) + 0.5 * k * r * r  # noqa: E731
    best_c = (q(grid) + tau[:, None] * phi(grid)).min(axis=1)
    assert np.all(q(tc) + tau * phi(tc) <= best_c + 1e-12)


@pytest.mark.parametrize("fn", [RowFn(np.full(50, 0.3), 1.0), RowFn(np.full(50, 0.3), 1.5),
                                RowFn(np.full(50, 0.3), 3.0),
                                RowFn(np.full(50, 0.3), 0.5, "compensated", 0.02)])
def test_moreau_identity_and_fenchel_young(fn):
    rng = np.random.default_rng(3)
    W = rng.normal(size=(50, 3))
    s = 0.7
    y = fn.dual_prox(W, s)
    u = fn.prox(W / s, 1 / s)
    assert np.allclose(W, y + s * u, atol=1e-12)
    assert np.all(fn.fy(u, y) <= 1e-9)            # (u, y) is a subgradient pair
    assert np.all(fn.fy(rng.normal(size=(50, 3)), y) >= -1e-12)


def test_joint_prox_certificate(inst_dir):
    p = problem(inst_dir, "C1")
    ev = Evaluator(p)
    rng = np.random.default_rng(4)
    v = p.theta0 + 0.1 * rng.normal(size=p.shape)
    jp = JointProx(p, ev.sel, ev.graphs)
    x, y, gap, _ = jp.solve(v, ev.alpha, 0.0, max_iter=3000)
    zeta, xh = ev.certify(v, x, y, 1e18, 1e-14)
    assert zeta < 1e-9
    # an independent long run of the same strongly convex prox agrees within the certified bound
    jp2 = JointProx(p, ev.sel, ev.graphs)
    x2, _, gap2, _ = jp2.solve(v, ev.alpha, 0.0, max_iter=20000)
    assert np.linalg.norm(x2 - xh) <= ev.alpha * zeta + np.sqrt(2 * ev.alpha * gap2) + 1e-12


def test_band_status():
    assert band_status(5e-6, 1e-8, 1e-5) == "pass"
    assert band_status(5e-6, 1e-6, 1e-5) == "unresolved"
    assert band_status(3e-5, 1e-6, 1e-5) == "fail"


def run(inst_dir, tmp_path, point, method, cap=1.0, **kw):
    iid = f"A_s2026_{point}_S1"
    job = {"run_id": f"{iid}_{method}", "phase": "test", "instance": iid,
           "instance_path": str(inst_dir / f"{iid}.npz"), "method": method, "cap": cap, **kw}
    if method in ("M20", "M21", "M22"):
        job["fine_ratio"] = M.fine_ratio(problem(inst_dir, point), method)[0]
    res = M.run_job(job, tmp_path / job["run_id"])
    return res, np.load(tmp_path / job["run_id"] / "final.npz")["theta"]


def test_exit_states_and_stationarity(inst_dir, tmp_path):
    res, th = run(inst_dir, tmp_path, "C1", "M04", cap=3.0)
    assert res["exit_status"] in ("native_stop", "time_limit")
    p = problem(inst_dir, "C1")
    info, y = Evaluator(p).residual(th, 3.0)
    assert info["residual_1e-05"] == "pass"
    lb = dual_lower_bound(p, th, y)
    F = p.objective(th)
    assert lb["dual_lower_bound"] <= F + 1e-12
    assert (F - lb["dual_lower_bound"]) / F < 1e-8
    info0, _ = Evaluator(p).residual(p.theta0, 3.0)
    assert info0["residual_1e-03"] == "fail"
    res, _ = run(inst_dir, tmp_path, "C1", "M02", cap=0.05)
    assert res["exit_status"] == "time_limit"
    res, _ = run(inst_dir, tmp_path, "C1", "M19")
    assert res["exit_status"] == "condition_not_met"


def test_routes_do_not_increase_monotone_objectives(inst_dir, tmp_path):
    for point, method in (("N3", "M10"), ("N3", "M01"), ("N1", "M12a"), ("C0", "M14"), ("C2", "M16")):
        res, th = run(inst_dir, tmp_path, point, method, cap=0.5)
        assert res["exit_status"] in ("native_stop", "time_limit")
        p = problem(inst_dir, point)
        assert p.objective(th) <= p.objective(p.theta0) + 1e-12


def test_smoothing_bias_bound(inst_dir, tmp_path):
    p = problem(inst_dir, "C1")
    ratio, bound = M.fine_ratio(p, "M20")
    assert bound <= M.BIAS_TARGET * (1 + 1e-9)
    ratio, bound = M.fine_ratio(p, "M21")
    assert bound <= M.BIAS_TARGET * (1 + 1e-9)
