"""Route registry and adapters M01-M24 (plan Section 4) plus the single-run driver.

Every adapter receives the frozen ``Problem``, the common initial value and the run ``Clock``.
It calls ``clock.observe`` after each accepted outer step and returns
``(theta, extra, arrays)``; ``extra["native_status"]`` names the route's own stopping reason.
The clock raises ``BudgetReached`` at the configured cap; the driver then keeps the last
observed iterate.  Native stopping, the hard cap, residual targets and reference accuracy are
separate states; only the common evaluator decides accuracy.
"""

from __future__ import annotations

import contextlib
import math
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

from .. import likelihood as lik
from .. import proxgrad as pg
from .. import solver as sv
from ..penalties import mm_weights, prox_rows, row_norms
from .core import (RUNTIME, BudgetReached, Clock, ConditionNotMet, JointProx, Problem, RowFn,
                   count, curvature_bound, dh_shift, h_shift, load_instance, nonconvex_prox,
                   plain, raw_objective, read_json, redistribution_slope, save_npz, write_json)

CONVEX = ("C0", "C1", "C2", "C3", "C4", "C5")
NONCONVEX = ("N1", "N2", "N3")
ALL = CONVEX + NONCONVEX
NOT_C4 = tuple(x for x in ALL if x != "C4")

# code: (name, objective, applicable points, conditional, notes / theory conditions)
METHODS = {
    "M01": ("existing default MM-ADMM", "original", ALL, False,
            "frozen baseline settings of the simulation configuration; finite-budget acceptance"),
    "M02": ("strict MM-ADMM", "original", ALL, False,
            "same route, stricter settings; achieved inner KKT bounds and violations are recorded"),
    "M03": ("existing APG", "original", ("C0", "C3", "C4", "C5"), False,
            "existing FISTA/backtracking/monotone restart; p_s >= 1 and every graph p >= 2"),
    "M04": ("Condat-Vu primal-dual", "original", CONVEX, False,
            "1/tau - sigma ||D||^2 > L/2 with sigma = 1/||D||, tau = 0.99/(L/2 + ||D||)"),
    "M05": ("PD3O", "original", CONVEX, False,
            "f = loss, g = selection prox, h = graph dual; gamma = 1/L, gamma delta ||DD^T|| = 0.99"),
    "M06": ("diagonally preconditioned primal-dual", "original", CONVEX, False,
            "Pock-Chambolle alpha=1 preconditioners with smooth loss: T^-1 - K'SK > L/2"),
    "M07": ("direct two-block ADMM, Newton-CG coefficient block", "original", CONVEX, False,
            "softmax + quadratic coefficient block solved inexactly (decreasing tolerance)"),
    "M08": ("single-level majorized ADMM", "original", CONVEX, False,
            "loss majorizer refreshed every update; fixed system C + rho A'A, gamma = 1e-3"),
    "M09": ("MM with primal-dual inner solver", "original", ALL, False,
            "M02 surrogate and KKT target schedule; Condat-Vu inner iterations"),
    "M10": ("nonconvexity redistribution + convex joint prox", "original", NONCONVEX, False,
            "S = loss + psi corrections, P convex; backtracking down to 0.99/L_S; inexact prox"),
    "M11": ("APG with inexact joint prox", "original", CONVEX, False,
            "selection + graph joint prox by dual FISTA; prox errors O(k^-4)"),
    "M12a": ("nonconvex selection proximal gradient", "original", ("N1",), False,
             "global radial prox of the offset power; sufficient-decrease backtracking"),
    "M12b": ("monotone nonconvex APG (Li-Lin)", "original", ("N1",), False,
             "accelerated and plain prox candidates, monotone choice"),
    "M13a": ("proximal Newton", "original", CONVEX, False,
             "exact Hessian + 1e-8 L shift; composite subproblem by Chambolle-Pock; Armijo"),
    "M13b": ("proximal quasi-Newton (damped BFGS)", "original", CONVEX, False,
             "dense damped BFGS metric; composite subproblem by Chambolle-Pock; Armijo"),
    "M13c": ("MM with proximal Newton subproblems", "original", NONCONVEX, False,
             "nonconvex branches tangent-majorized; exact loss in the subproblem"),
    "M14": ("row-block proximal coordinate descent", "original", ("C0", "C2", "C3", "C4", "C5", "N1"),
            True, "block prox-gradient with block backtracking (Tseng-Yun); N1 conditional"),
    "M15": ("Davis-Yin three-operator splitting", "original", CONVEX, False,
            "g = selection prox, h = graph prox by timed dual inner solve; gamma = 1/L"),
    "M16": ("universal fast gradient (Hoelder)", "original", ("C2",), False,
            "Nesterov universal fast gradient, eps = 1e-10, composite selection"),
    "M17": ("conic interior point (CLARABEL via CVXPY)", "original", CONVEX, False,
            "exponential/second-order/power cones; own dual lower bound checked separately"),
    "M18": ("DCA with an explicit DC split", "original", NONCONVEX, False,
            "h = a r - k(r); concave part linearized; convex subproblem by Condat-Vu"),
    "M19": ("original-objective quadratic IRLS", "original", NOT_C4, True,
            "needs nonzero radii for p < 2; no weight truncation (condition failure recorded)"),
    "M20": ("Moreau-smoothed graph + APG", "smoothed", ("C1", "C2", "C3", "C4", "C5"), False,
            "per-row Moreau envelope (Huber for TV); output re-evaluated on the original"),
    "M21": ("offset-preserving smoothed IRLS", "smoothed", NOT_C4, False,
            "h(sqrt(r^2 + delta^2)) - h(delta); uniform bias <= sum mu h(delta)"),
    "M22": ("unshifted-power smoothed IRLS", "smoothed", NOT_C4, False,
            "(r^2 + delta^2)^(p/2) - delta^p; tends to the unshifted power when p < 1"),
    "M23": ("original joint proximal gradient", "original", NONCONVEX, False,
            "alpha = 0.5/max(L, W, 1); compensated convex joint prox by dual FISTA"),
    "M24": ("direct nonconvex two-block ADMM (exploratory)", "original", NONCONVEX, False,
            "global nonconvex radial prox; frozen automatic rho; no convergence claim"),
}

M01_OPTIONS = dict(gamma=1e-3, max_outer=1000, max_inner=300, tol_obj=1e-6, tol_x=1e-3,
                   inner_budget0=5, cg_tol=1e-6)
M02_OPTIONS = dict(gamma=1e-3, max_outer=5000, max_inner=1000, tol_obj=1e-9, tol_x=1e-6,
                   inner_budget0=None, cg_tol=1e-10, inner_abs_tol=1e-10, inner_rel_tol0=1e-6,
                   inner_rel_tol_min=1e-10, kkt_tol0=1e-3, kkt_tol_min=1e-10, kkt_enforce=True)
M03_OPTIONS = dict(apg_tol=1e-9, tol_x=1e-6, apg_max_iter=20000)
SMOOTHING_RATIOS = {"coarse": 0.1, "medium": 0.01, "fine": 1e-3}
BIAS_TARGET = 1e-6


class Stopper:
    """Native stopping: ``value <= tol`` in ``patience`` consecutive iterations."""

    def __init__(self, tol: float, patience: int = 3):
        self.tol, self.patience, self.hits = tol, patience, 0

    def __call__(self, ok: bool) -> bool:
        self.hits = self.hits + 1 if ok else 0
        return self.hits >= self.patience


def _rel(a: float, b: float) -> float:
    return a / max(1.0, b)


def _norm(x) -> float:
    return float(np.linalg.norm(x))


def _sum_adj(problem: Problem, items) -> np.ndarray:
    out = np.zeros((problem.d, problem.k1))
    for br, y in items:
        out += problem.At(br, y)
    return out


def _pd_steps(problem: Problem, graphs, L: float):
    """Condat-Vu steps with balance 1: sigma = 1/||D||, tau = 0.99 / (L/2 + ||D||)."""
    if graphs:
        nD = math.sqrt(problem.graph_norm2([br for br, _ in graphs]))
        return 0.99 / (0.5 * L + nD), 1.0 / nD
    return 0.99 / (0.5 * L), 0.0


# ----------------------------------------------------------------------------------------------
# M01-M03: existing solvers with operation counters and the monitor hook
# ----------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _existing_solver_hooks():
    orig = {
        "loss_grad": lik.loss_grad, "loss": lik.loss, "sv_prox": sv.prox_rows,
        "pg_prox": pg.prox_rows, "apply": sv.Branch.apply, "adjoint": sv.Branch.adjoint,
        "set_rho": sv._CoefficientSystem.set_rho, "solve": sv._CoefficientSystem.solve,
        "pcg": sv._block_pcg,
    }

    def loss_grad(*a, **k):
        count("loss_gradient_calls")
        return orig["loss_grad"](*a, **k)

    def loss(*a, **k):
        count("loss_calls")
        return orig["loss"](*a, **k)

    def prox(*a, **k):
        count("row_prox_calls")
        return orig["sv_prox"](*a, **k)

    def apply(self, B):
        if self.op is not None:
            count("graph_mult")
        return orig["apply"](self, B)

    def adjoint(self, V):
        if self.op is not None:
            count("graph_transpose_mult")
        return orig["adjoint"](self, V)

    def set_rho(self, rho):
        count("factorizations")
        return orig["set_rho"](self, rho)

    def solve(self, R):
        count("linear_solves")
        return orig["solve"](self, R)

    def pcg(*a, **k):
        count("pcg_calls")
        return orig["pcg"](*a, **k)

    lik.loss_grad, lik.loss = loss_grad, loss
    sv.prox_rows, pg.prox_rows = prox, prox
    sv.Branch.apply, sv.Branch.adjoint = apply, adjoint
    sv._CoefficientSystem.set_rho, sv._CoefficientSystem.solve = set_rho, solve
    sv._block_pcg = pcg
    try:
        yield
    finally:
        lik.loss_grad, lik.loss = orig["loss_grad"], orig["loss"]
        sv.prox_rows, pg.prox_rows = orig["sv_prox"], orig["pg_prox"]
        sv.Branch.apply, sv.Branch.adjoint = orig["apply"], orig["adjoint"]
        sv._CoefficientSystem.set_rho = orig["set_rho"]
        sv._CoefficientSystem.solve = orig["solve"]
        sv._block_pcg = orig["pcg"]


def existing_solver(problem: Problem, T, clock: Clock, job: dict):
    method = job["method"]
    settings = {"M01": M01_OPTIONS, "M02": M02_OPTIONS, "M03": M03_OPTIONS}[method]
    opts = sv.SolverOptions(**settings)

    def monitor(Theta, h):
        count("outer_iterations")
        if "inner_iterations" in h:
            count("inner_iterations", h["inner_iterations"])
        info = {"iteration": h["outer"] + 1, "native_objective": h["objective"], "step": h["step"]}
        for key in ("inner_iterations", "r_pri", "s_dual", "rho", "kkt_bound", "kkt_tol", "L"):
            if key in h:
                info[key] = h[key]
        clock.observe(Theta, info)

    opts.monitor = monitor
    state = sv.SolverState(T.copy())
    extra = {"settings": settings}
    if job.get("initial_state"):
        saved = np.load(job["initial_state"])
        nb = len(problem.branches)
        if "rho" in saved.files and all(f"U_{j}" in saved.files for j in range(nb)):
            state = sv.SolverState(T.copy(), [saved[f"U_{j}"] for j in range(nb)],
                                   [saved[f"Z_{j}"] for j in range(nb)], float(saved["rho"]))
            extra["continuation_state"] = "theta_auxiliary_dual_rho"
        else:
            extra["continuation_state"] = "theta_only"
    with _existing_solver_hooks():
        solve = pg.solve_apg if method == "M03" else sv.solve_mm_admm
        res = solve(problem.Xa, problem.Y, problem.H, problem.branches, opts, state)
    extra.update({"native_status": res.status, "native_converged": bool(res.converged),
                  "n_outer": res.n_outer, "n_inner_total": res.n_inner_total,
                  "n_kkt_violations": res.n_kkt_violations, "native_diagnostics": res.diagnostics})
    arrays = {}
    if method != "M03":
        arrays = {"rho": np.array(res.rho)}
        arrays.update({f"U_{j}": u for j, u in enumerate(res.U)})
        arrays.update({f"Z_{j}": z for j, z in enumerate(res.Z)})
    return res.Theta, extra, arrays


# ----------------------------------------------------------------------------------------------
# M04-M06: primal-dual splittings (convex points)
# ----------------------------------------------------------------------------------------------


def condat_vu(problem: Problem, T, clock: Clock, job: dict):
    sel, graphs = problem.convex_terms()
    L = problem.lip_loss
    tau, sigma = _pd_steps(problem, graphs, L)
    Y = [np.zeros((br.op.shape[0], problem.k1)) for br, _ in graphs]
    stop = Stopper(1e-13)
    it = 0
    while True:
        it += 1
        _, G = problem.loss_grad(T)
        if graphs:
            G[:-1] += _sum_adj(problem, [(br, y) for (br, _), y in zip(graphs, Y)])
        Tn = T - tau * G
        Tn[:-1] = sel.prox(Tn[:-1], tau)
        Bbar = 2.0 * Tn[:-1] - T[:-1]
        Yn = [fn.dual_prox(y + sigma * problem.A(br, Bbar), sigma) for (br, fn), y in zip(graphs, Y)]
        dp = _norm(Tn - T)
        dd = math.sqrt(sum(_norm(a - b) ** 2 for a, b in zip(Yn, Y)))
        T, Y = Tn, Yn
        clock.observe(T, {"iteration": it, "primal_change": dp, "dual_change": dd})
        if stop(dp <= 1e-13 * max(1.0, _norm(T)) and dd <= 1e-13 * max(1.0, math.sqrt(
                sum(_norm(y) ** 2 for y in Y)))):
            break
    return T, {"native_status": "fixed_point_change_below_1e-13", "tau": tau, "sigma": sigma,
               "iterations": it}, {f"dual_{br.name}": y for (br, _), y in zip(graphs, Y)}


def pd3o(problem: Problem, T, clock: Clock, job: dict):
    sel, graphs = problem.convex_terms()
    L = problem.lip_loss
    gamma = 1.0 / L
    delta = 0.99 / (gamma * problem.graph_norm2([br for br, _ in graphs])) if graphs else 0.0
    S = [np.zeros((br.op.shape[0], problem.k1)) for br, _ in graphs]
    z = T.copy()
    x = z.copy()
    x[:-1] = sel.prox(z[:-1], gamma)
    stop = Stopper(1e-13)
    it = 0
    DtS = np.zeros((problem.d, problem.k1))
    while True:
        it += 1
        _, G = problem.loss_grad(x)
        if graphs:
            w = 2.0 * x[:-1] - z[:-1] - gamma * G[:-1] - gamma * DtS
            S = [fn.dual_prox(s + delta * problem.A(br, w), delta) for (br, fn), s in zip(graphs, S)]
            DtS = _sum_adj(problem, [(br, s) for (br, _), s in zip(graphs, S)])
        zn = x - gamma * G
        zn[:-1] -= gamma * DtS
        xn = zn.copy()
        xn[:-1] = sel.prox(zn[:-1], gamma)
        dx = _norm(xn - x)
        dz = _norm(zn - z)
        x, z = xn, zn
        clock.observe(x, {"iteration": it, "primal_change": dx, "z_change": dz})
        if stop(dz <= 1e-13 * max(1.0, _norm(z))):
            break
    return x, {"native_status": "z_change_below_1e-13", "gamma": gamma, "delta": delta,
               "iterations": it}, {f"dual_{br.name}": s for (br, _), s in zip(graphs, S)}


def preconditioned_pd(problem: Problem, T, clock: Clock, job: dict):
    sel, graphs = problem.convex_terms()
    L = problem.lip_loss
    colsum = np.zeros(problem.d)
    sig = []
    for br, _ in graphs:
        absop = abs(br.op)
        colsum += np.asarray(absop.sum(axis=0)).ravel()
        sig.append(1.0 / np.maximum(np.asarray(absop.sum(axis=1)).ravel(), 1e-12))
    tauB = 0.99 / (colsum + 0.5 * L)
    tauc = 0.99 / (0.5 * L)
    Y = [np.zeros((br.op.shape[0], problem.k1)) for br, _ in graphs]
    stop = Stopper(1e-13)
    it = 0
    while True:
        it += 1
        _, G = problem.loss_grad(T)
        if graphs:
            G[:-1] += _sum_adj(problem, [(br, y) for (br, _), y in zip(graphs, Y)])
        Tn = T.copy()
        Tn[:-1] = sel.prox(T[:-1] - tauB[:, None] * G[:-1], tauB)
        Tn[-1] = T[-1] - tauc * G[-1]
        Bbar = 2.0 * Tn[:-1] - T[:-1]
        Yn = [fn.dual_prox(y + s[:, None] * problem.A(br, Bbar), s)
              for (br, fn), y, s in zip(graphs, Y, sig)]
        dp = _norm(Tn - T)
        dd = math.sqrt(sum(_norm(a - b) ** 2 for a, b in zip(Yn, Y)))
        T, Y = Tn, Yn
        clock.observe(T, {"iteration": it, "primal_change": dp, "dual_change": dd})
        if stop(dp <= 1e-13 * max(1.0, _norm(T)) and dd <= 1e-13 * max(1.0, math.sqrt(
                sum(_norm(y) ** 2 for y in Y)))):
            break
    return T, {"native_status": "fixed_point_change_below_1e-13", "iterations": it,
               "tau_range": [float(tauB.min()), float(tauB.max()), tauc],
               "sigma": [float(s[0]) for s in sig]}, \
        {f"dual_{br.name}": y for (br, _), y in zip(graphs, Y)}


# ----------------------------------------------------------------------------------------------
# M07 / M24 direct ADMM and M08 single-level majorized ADMM
# ----------------------------------------------------------------------------------------------


def _newton_block(problem: Problem, T, W, rho, tol, max_newton=50):
    """``min_T loss(T) + rho/2 sum_a ||A_a T_B - W_a||^2`` by Newton-CG with Armijo search."""
    brs = problem.branches

    def value_grad(X):
        f, G = problem.loss_grad(X)
        B = X[:-1]
        for br, w in zip(brs, W):
            R = problem.A(br, B) - w
            f += 0.5 * rho * float(np.sum(R * R))
            G[:-1] += rho * problem.At(br, R)
        return f, G

    def value(X):
        f = problem.loss(X)
        B = X[:-1]
        for br, w in zip(brs, W):
            R = problem.A(br, B) - w
            f += 0.5 * rho * float(np.sum(R * R))
        return f

    f, G = value_grad(T)
    steps = 0
    for _ in range(max_newton):
        gn = _norm(G)
        if gn <= tol:
            break
        Tc = T

        def mv(vec, Tc=Tc):
            D = vec.reshape(Tc.shape)
            count("cg_iterations")
            out = problem.hvp(Tc, D)
            DB = D[:-1]
            for br in brs:
                out[:-1] += rho * problem.At(br, problem.A(br, DB))
            return out.ravel()

        op = LinearOperator((T.size, T.size), matvec=mv, dtype=float)
        d, _ = cg(op, -G.ravel(), rtol=min(0.5, math.sqrt(gn)), maxiter=200)
        d = d.reshape(T.shape)
        slope = float(np.sum(G * d))
        if slope >= 0:
            d, slope = -G, -gn * gn
        s = 1.0
        for _ in range(40):
            Xn = T + s * d
            if value(Xn) <= f + 1e-4 * s * slope:
                break
            s *= 0.5
        T = Xn
        f, G = value_grad(T)
        steps += 1
    return T, steps


def direct_admm(problem: Problem, T, clock: Clock, job: dict):
    nonconvex_split = job["method"] == "M24"
    rho = float(sv.default_rho(problem.Xa))
    brs = problem.branches
    U = [problem.A(br, T[:-1]) for br in brs]
    Z = [np.zeros_like(u) for u in U]
    tol_inner = 1e-3
    stop = Stopper(1e-10)
    it = 0
    while True:
        it += 1
        T, steps = _newton_block(problem, T, [u - z for u, z in zip(U, Z)], rho, tol_inner)
        B = T[:-1]
        r2, s_acc = 0.0, np.zeros((problem.d, problem.k1))
        for j, br in enumerate(brs):
            AB = problem.A(br, B)
            V = AB + Z[j]
            if br.p < 1 and nonconvex_split:
                Un = nonconvex_prox(V, br.mu / rho, br.p, br.eps)
            else:
                count("row_prox_calls")
                Un = prox_rows(V, br.mu / rho, br.p)
            s_acc += problem.At(br, Un - U[j])
            Z[j] = Z[j] + AB - Un
            r2 += float(np.sum((AB - Un) ** 2))
            U[j] = Un
        r_pri, s_dual = math.sqrt(r2), rho * _norm(s_acc)
        tol_inner = max(1e-12, min(tol_inner, 0.1 * max(r_pri, s_dual)))
        clock.observe(T, {"iteration": it, "newton_steps": steps, "r_pri": r_pri, "s_dual": s_dual,
                          "rho": rho})
        scale = max(1.0, _norm(T))
        if stop(r_pri <= 1e-10 * scale and s_dual <= 1e-10 * scale):
            break
    arrays = {"rho": np.array(rho)}
    arrays.update({f"U_{j}": u for j, u in enumerate(U)})
    arrays.update({f"Z_{j}": z for j, z in enumerate(Z)})
    return T, {"native_status": "admm_residuals_below_1e-10", "rho": rho, "iterations": it}, arrays


def majorized_admm(problem: Problem, T, clock: Clock, job: dict):
    gamma = 1e-3
    rho = float(sv.default_rho(problem.Xa))
    C = problem.Xa.T @ problem.Xa / (2.0 * problem.n)
    C[np.diag_indices_from(C)] += gamma
    M = C.copy()
    M[:-1, :-1] += rho * problem.gram()
    count("factorizations")
    fac = la.cho_factor(M, lower=True)
    brs = problem.branches
    U = [problem.A(br, T[:-1]) for br in brs]
    Z = [np.zeros_like(u) for u in U]
    stop = Stopper(1e-10)
    it = 0
    while True:
        it += 1
        _, G = problem.loss_grad(T)
        rhs = C @ T - G
        rhs[:-1] += rho * _sum_adj(problem, [(br, u - z) for br, u, z in zip(brs, U, Z)])
        count("linear_solves")
        T = la.cho_solve(fac, rhs)
        B = T[:-1]
        r2, s_acc = 0.0, np.zeros((problem.d, problem.k1))
        for j, br in enumerate(brs):
            AB = problem.A(br, B)
            count("row_prox_calls")
            Un = prox_rows(AB + Z[j], br.mu / rho, br.p)
            s_acc += problem.At(br, Un - U[j])
            Z[j] = Z[j] + AB - Un
            r2 += float(np.sum((AB - Un) ** 2))
            U[j] = Un
        r_pri, s_dual = math.sqrt(r2), rho * _norm(s_acc)
        clock.observe(T, {"iteration": it, "r_pri": r_pri, "s_dual": s_dual})
        scale = max(1.0, _norm(T))
        if stop(r_pri <= 1e-10 * scale and s_dual <= 1e-10 * scale):
            break
    return T, {"native_status": "admm_residuals_below_1e-10", "rho": rho, "gamma": gamma,
               "iterations": it}, {f"dual_{br.name}": rho * z for br, z in zip(brs, Z)}


# ----------------------------------------------------------------------------------------------
# M09: MM surrogate (as M02) with Condat-Vu inner iterations
# ----------------------------------------------------------------------------------------------


def mm_primal_dual(problem: Problem, T, clock: Clock, job: dict):
    o = M02_OPTIONS
    gamma, kkt_decay = o["gamma"], 0.95
    Xa, n = problem.Xa, problem.n
    XtX = Xa.T @ Xa / (2.0 * n)
    brs, graphs = problem.branches, problem.graph_branches
    Lq = problem.lip_loss + gamma
    nD = math.sqrt(problem.graph_norm2()) if graphs else 0.0
    tau = 0.99 / (0.5 * Lq + nD)
    sigma = 1.0 / nD if graphs else 0.0
    op_norms = [sv._op_norm_bound(br) for br in brs]
    Y = [np.zeros((br.op.shape[0], problem.k1)) for br in graphs]
    F = problem.objective(T)
    n_viol, total_inner = 0, 0
    status = "max_outer"
    for r in range(o["max_outer"]):
        Tr = T.copy()
        _, G = problem.loss_grad(Tr)
        kappas, nus = [], []
        for br in brs:
            omega, nu = mm_weights(row_norms(problem.A(br, Tr[:-1])), br.p, br.offset(1e-3))
            kappas.append(br.mu * omega)
            nus.append(nu)
        selfn = RowFn(kappas[0], nus[0])
        gfns = [RowFn(k, nu) for k, nu in zip(kappas[1:], nus[1:])]
        Y = [fn.dual_prox(y, 1.0) if fn.p == 1 else y for fn, y in zip(gfns, Y)]
        q_old = sv._surrogate(Xa, Tr, Tr, G, gamma, brs, kappas, nus)
        tau_r = max(o["kkt_tol0"] * kkt_decay ** r, o["kkt_tol_min"]) * max(1.0, _norm(Tr))
        x = Tr.copy()
        bound = math.inf
        it = 0
        for it in range(1, o["max_inner"] + 1):
            count("inner_iterations")
            gq = G + XtX @ (x - Tr) + gamma * (x - Tr)
            if graphs:
                gq[:-1] += _sum_adj(problem, list(zip(graphs, Y)))
            pre = x - tau * gq
            xn = pre.copy()
            xn[:-1] = selfn.prox(pre[:-1], tau)
            s_sel = (pre[:-1] - xn[:-1]) / tau
            Bbar = 2.0 * xn[:-1] - x[:-1]
            Ug, Yn = [], []
            for br, fn, y in zip(graphs, gfns, Y):
                arg = y + sigma * problem.A(br, Bbar)
                u = fn.prox(arg / sigma, 1.0 / sigma)
                Ug.append(u)
                Yn.append(arg - sigma * u)
            x, Y = xn, Yn
            if it % 5 == 0 or it == o["max_inner"]:
                bound, _ = sv._kkt_error_bound(Xa, G, x, Tr, gamma, brs, kappas, nus,
                                               [x[:-1]] + Ug, [s_sel] + Y, 1.0, op_norms)
                if bound <= tau_r:
                    q_try = sv._surrogate(Xa, x, Tr, G, gamma, brs, kappas, nus)
                    if q_try <= q_old + 1e-12 * (1.0 + abs(q_old)):
                        break
        total_inner += it
        q_new = sv._surrogate(Xa, x, Tr, G, gamma, brs, kappas, nus)
        decrease = q_new <= q_old + 1e-12 * (1.0 + abs(q_old))
        if not decrease:
            status = ("descent_stalled" if q_new - q_old <= o["tol_obj"] * max(1.0, abs(F))
                      else "acceptance_failed")
            T = Tr
            break
        if bound > tau_r:
            n_viol += 1
        F_new = problem.objective(x)
        step = _norm(x - Tr)
        T = x
        count("outer_iterations")
        clock.observe(T, {"iteration": r + 1, "inner_iterations": it, "kkt_bound": bound,
                          "kkt_tol": tau_r, "step": step})
        dF = F - F_new
        F = F_new
        if abs(dF) <= o["tol_obj"] * max(1.0, abs(F)) and step <= o["tol_x"] * max(_norm(T), 1e-12):
            status = "converged"
            break
    return T, {"native_status": status, "n_kkt_violations": n_viol, "inner_total": total_inner,
               "tau": tau, "sigma": sigma, "settings": o}, {}


# ----------------------------------------------------------------------------------------------
# M10 / M11: (accelerated) proximal gradient with the convex joint prox
# ----------------------------------------------------------------------------------------------


def redistributed_pg(problem: Problem, T, clock: Clock, job: dict):
    sel, graphs = problem.convex_terms()
    jp = JointProx(problem, sel, graphs)
    a_min = 0.99 / problem.lip_S()
    a = 1.0 / problem.lip_loss
    base = 1e-10 if job.get("reference") else 1e-8
    fS, gS = problem.smooth_S(T)
    stop = Stopper(1e-13)
    it, n_floor = 0, 0
    while True:
        it += 1
        tol = max(1e-15, base * (it ** -2.5))
        while True:
            x, _, gap, _ = jp.solve(T - a * gS, a, tol)
            dlt = x - T
            fx, gx = problem.smooth_S(x)
            dd = float(np.sum(dlt * dlt))
            if fx <= fS + float(np.sum(gS * dlt)) + (1 - 1e-4) * dd / (2 * a) + 1e-14 * max(1.0, abs(fS)):
                break
            if a <= a_min:
                n_floor += 1
                break
            a = max(0.5 * a, a_min)
        step = math.sqrt(dd)
        T, fS, gS = x, fx, gx
        clock.observe(T, {"iteration": it, "alpha": a, "prox_gap": gap, "step": step})
        if stop(step <= 1e-13 * max(1.0, _norm(T))):
            break
        a = a * 1.25
    return T, {"native_status": "step_below_1e-13", "alpha_min": a_min, "iterations": it,
               "accepted_at_floor": n_floor}, {}


def inexact_apg(problem: Problem, T, clock: Clock, job: dict):
    sel, graphs = problem.convex_terms()
    jp = JointProx(problem, sel, graphs)
    L = 1.0
    X, Yk, tk = T.copy(), T.copy(), 1.0
    F = problem.objective(X)
    stop = Stopper(1e-13)
    it, k = 0, 0
    status = "small_change"
    while True:
        k += 1
        tol = max(1e-15, 1e-8 * k ** -4.0)
        fY, GY = problem.loss_grad(Yk)
        while True:
            Xn, _, gap, _ = jp.solve(Yk - GY / L, 1.0 / L, tol)
            D = Xn - Yk
            fX = problem.loss(Xn)
            if fX <= fY + float(np.sum(GY * D)) + 0.5 * L * float(np.sum(D * D)) + 1e-15 * max(1.0, abs(fY)):
                break
            L *= 2.0
        Fn = fX + problem.P_value(Xn)
        if Fn > F:
            if tk == 1.0:
                status = "descent_stalled"
                break
            Yk, tk = X.copy(), 1.0
            continue
        it += 1
        tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
        Yk = Xn + ((tk - 1.0) / tn) * (Xn - X)
        step = _norm(Xn - X)
        dF = F - Fn
        X, tk, F = Xn, tn, Fn
        clock.observe(X, {"iteration": it, "L": L, "prox_gap": gap, "step": step})
        if stop(dF <= 1e-14 * max(1.0, abs(F)) and step <= 1e-12 * max(1.0, _norm(X))):
            break
        L *= 0.9
    return X, {"native_status": status, "iterations": it}, {}


# ----------------------------------------------------------------------------------------------
# Smooth graph part for routes that keep the graph term in the gradient (p_g > 1)
# ----------------------------------------------------------------------------------------------


def smooth_graph_value_grad(problem: Problem, T, value_only=False):
    """``loss + sum_g mu ||(D B)_e||^p`` (p > 1) and its gradient."""
    if value_only:
        f, G = problem.loss(T), None
    else:
        f, G = problem.loss_grad(T)
    B = T[:-1]
    for br in problem.graph_branches:
        if br.p <= 1:
            raise ConditionNotMet("graph exponent <= 1 is not smooth")
        U = problem.A(br, B)
        r = row_norms(U)
        f += float(np.dot(br.mu, r**br.p))
        if G is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                coef = np.where(r > 0, br.mu * br.p * r ** (br.p - 2.0), 0.0)
            G[:-1] += problem.At(br, coef[:, None] * U)
    return f, G


def _selection_prox(problem: Problem):
    br = problem.selection
    if br.p < 1:
        return lambda V, t: nonconvex_prox(V, t * br.mu, br.p, br.eps)
    fn = RowFn(br.mu, float(br.p))
    return fn.prox


def _selection_value(problem: Problem, T) -> float:
    br = problem.selection
    return float(np.dot(br.mu, h_shift(row_norms(T[:-1]), br.p, br.offset(1e-3))))


def nonconvex_pg(problem: Problem, T, clock: Clock, job: dict):
    """M12a: proximal gradient with the global nonconvex selection prox (N1)."""
    prox = _selection_prox(problem)
    L = 1.0
    f, G = smooth_graph_value_grad(problem, T)
    stop = Stopper(1e-13)
    it = 0
    while True:
        it += 1
        while True:
            x = T - G / L
            x[:-1] = prox(x[:-1], 1.0 / L)
            d = x - T
            fx, gx = smooth_graph_value_grad(problem, x)
            if fx <= f + float(np.sum(G * d)) + (1 - 1e-4) * 0.5 * L * float(np.sum(d * d)) + 1e-15 * max(1.0, abs(f)):
                break
            L *= 2.0
        step = _norm(d)
        T, f, G = x, fx, gx
        clock.observe(T, {"iteration": it, "L": L, "step": step})
        if stop(step <= 1e-13 * max(1.0, _norm(T))):
            break
        L *= 0.9
    return T, {"native_status": "step_below_1e-13", "iterations": it}, {}


def nonconvex_apg(problem: Problem, T, clock: Clock, job: dict):
    """M12b: monotone APG of Li and Lin (2015), Algorithm 1, with backtracking (N1)."""
    prox = _selection_prox(problem)

    def prox_step(x, fx, gx, L):
        while True:
            z = x - gx / L
            z[:-1] = prox(z[:-1], 1.0 / L)
            d = z - x
            fz, _ = smooth_graph_value_grad(problem, z, value_only=True)
            if fz <= fx + float(np.sum(gx * d)) + (1 - 1e-4) * 0.5 * L * float(np.sum(d * d)) + 1e-15 * max(1.0, abs(fx)):
                return z, fz, L
            L *= 2.0

    x_prev = x = z = T.copy()
    t_prev, t = 0.0, 1.0
    Ly = Lx = 1.0
    fx, gx = smooth_graph_value_grad(problem, x)
    Fx = fx + _selection_value(problem, x)
    stop = Stopper(1e-13)
    it = 0
    while True:
        it += 1
        y = x + (t_prev / t) * (z - x) + ((t_prev - 1.0) / t) * (x - x_prev)
        fy, gy = smooth_graph_value_grad(problem, y)
        z_new, fz, Ly = prox_step(y, fy, gy, Ly)
        v, fv, Lx = prox_step(x, fx, gx, Lx)
        Fz = fz + _selection_value(problem, z_new)
        Fv = fv + _selection_value(problem, v)
        t_prev, t = t, 0.5 * (math.sqrt(4.0 * t * t + 1.0) + 1.0)
        x_prev = x
        x, Fn = (z_new, Fz) if Fz <= Fv else (v, Fv)
        z = z_new
        fx, gx = smooth_graph_value_grad(problem, x)
        step = _norm(x - x_prev)
        dF = Fx - Fn
        Fx = Fn
        clock.observe(x, {"iteration": it, "Lx": Lx, "Ly": Ly, "step": step,
                          "accelerated_chosen": bool(Fz <= Fv)})
        if stop(step <= 1e-13 * max(1.0, _norm(x)) and dF <= 1e-14 * max(1.0, abs(Fx))):
            break
        Lx *= 0.9
        Ly *= 0.9
    return x, {"native_status": "small_change", "iterations": it}, {}


# ----------------------------------------------------------------------------------------------
# M13a-c: proximal Newton type methods
# ----------------------------------------------------------------------------------------------


def _quad_composite_cp(problem: Problem, Hm, g, T, sel: RowFn, graphs, tol, max_iter=5000,
                       y0=None):
    """``min <g, x-T> + 1/2 (x-T)' H (x-T) + sel(x_B) + sum_g G_g(D_g x_B)`` (Chambolle-Pock).

    All penalty rows are dualized (K = [I; D] on the coefficient rows); the primal step solves
    ``(I + tau H) x = ...`` with one Cholesky factorization."""
    nK2 = 1.0 + (problem.graph_norm2([br for br, _ in graphs]) if graphs else 0.0)
    tau = sigma = 0.99 / math.sqrt(nK2)
    M = np.eye(T.size) + tau * Hm
    count("factorizations")
    fac = la.cho_factor(M, lower=True)
    const = tau * (Hm @ T.ravel() - g.ravel())
    x = T.copy()
    if y0 is None:
        ys = np.zeros((problem.d, problem.k1))
        yg = [np.zeros((br.op.shape[0], problem.k1)) for br, _ in graphs]
    else:
        ys, yg = y0[0].copy(), [y.copy() for y in y0[1]]

    def KT(ys, yg):
        out = np.zeros(problem.shape)
        out[:-1] = ys + (_sum_adj(problem, [(br, y) for (br, _), y in zip(graphs, yg)])
                         if graphs else 0.0)
        return out

    KTy = KT(ys, yg)
    res = math.inf
    it = 0
    for it in range(1, max_iter + 1):
        count("inner_iterations")
        count("linear_solves")
        xn = la.cho_solve(fac, x.ravel() - tau * KTy.ravel() + const).reshape(T.shape)
        xbar = 2.0 * xn[:-1] - x[:-1]
        ysn = sel.dual_prox(ys + sigma * xbar, sigma)
        ygn = [fn.dual_prox(y + sigma * problem.A(br, xbar), sigma) for (br, fn), y in zip(graphs, yg)]
        KTyn = KT(ysn, ygn)
        dx = x - xn
        pres = dx / tau - (KTy - KTyn)
        dres = [(ys - ysn) / sigma - dx[:-1]] + [
            (y - yn) / sigma - problem.A(br, dx[:-1]) for (br, _), y, yn in zip(graphs, yg, ygn)]
        res = _norm(pres) + math.sqrt(sum(_norm(v) ** 2 for v in dres))
        x, ys, yg, KTy = xn, ysn, ygn, KTyn
        if res <= tol:
            break
    return x, (ys, yg), res, it


def _prox_newton_loop(problem: Problem, T, clock: Clock, variant: str, sel, graphs, P_value,
                      max_iter=None, info_prefix=None, F_fn=None):
    """Proximal (quasi-)Newton on ``loss + P``; returns the last iterate and statistics."""
    F_fn = F_fn or (lambda X: problem.loss(X) + P_value(X))
    F = F_fn(T)
    shift = 1e-8 * problem.lip_loss
    Bmat = None
    old = None
    y = None
    k = 0
    status = "small_step"
    stop = Stopper(1e-13, patience=2)
    while max_iter is None or k < max_iter:
        k += 1
        f, g = problem.loss_grad(T)
        if variant == "newton":
            Hm = problem.hessian(T)
            Hm[np.diag_indices_from(Hm)] += shift
        else:
            if Bmat is None:
                Bmat = problem.lip_loss * np.eye(T.size)
            elif old is not None:
                s_vec, y_vec = (T - old[0]).ravel(), (g - old[1]).ravel()
                Bs = Bmat @ s_vec
                sBs, sy = float(s_vec @ Bs), float(s_vec @ y_vec)
                if sBs > 1e-30:
                    if sy < 0.2 * sBs:  # Powell damping keeps the metric positive definite
                        theta = 0.8 * sBs / (sBs - sy)
                        y_vec = theta * y_vec + (1 - theta) * Bs
                        sy = float(s_vec @ y_vec)
                    Bmat = Bmat + np.outer(y_vec, y_vec) / sy - np.outer(Bs, Bs) / sBs
            Hm = Bmat
        tol = max(1e-12, 1e-4 * k ** -2.0) * max(1.0, _norm(g))
        x, y, res, inner = _quad_composite_cp(problem, Hm, g, T, sel, graphs, tol, y0=y)
        d = x - T
        Delta = float(np.sum(g * d)) + P_value(x) - P_value(T)
        s = 1.0
        accepted = False
        for _ in range(40):
            Tn = T + s * d
            Fn = F_fn(Tn)
            if Fn <= F + 1e-4 * s * min(Delta, 0.0) + 1e-15 * max(1.0, abs(F)):
                accepted = True
                break
            s *= 0.5
        if not accepted:
            status = "line_search_failed"
            break
        old = (T, g)
        step = _norm(s * d)
        dF = F - Fn
        T, F = Tn, Fn
        info = {"iteration": k, "step_size": s, "inner_iterations": inner, "inner_residual": res,
                "predicted_decrease": Delta}
        if info_prefix:
            info.update(info_prefix)
        clock.observe(T, info)
        if stop(step <= 1e-13 * max(1.0, _norm(T)) and dF <= 1e-14 * max(1.0, abs(F))):
            break
    return T, status, k


def prox_newton(problem: Problem, T, clock: Clock, job: dict):
    sel, graphs = problem.convex_terms()
    variant = "newton" if job["method"] == "M13a" else "bfgs"
    T, status, k = _prox_newton_loop(problem, T, clock, variant, sel, graphs, problem.P_value)
    return T, {"native_status": status, "iterations": k}, {}


def mm_prox_newton(problem: Problem, T, clock: Clock, job: dict):
    """M13c: tangent-majorize the p < 1 branches, keep the loss, prox-Newton inner steps."""
    brs = problem.branches
    F = problem.objective(T)
    stop = Stopper(1e-13, patience=2)
    r = 0
    status = "small_change"
    while True:
        r += 1
        fns = []
        for br in brs:
            omega, nu = mm_weights(row_norms(problem.A(br, T[:-1])), br.p, br.offset(1e-3))
            fns.append(RowFn(br.mu * omega, nu))
        sel, graphs = fns[0], list(zip(brs[1:], fns[1:]))

        def P_sur(X, sel=sel, graphs=graphs):
            B = X[:-1]
            return sel.value(B) + sum(fn.value(problem.A(br, B)) for br, fn in graphs)

        Tn, inner_status, k = _prox_newton_loop(problem, T, clock, "newton", sel, graphs, P_sur,
                                                max_iter=10, info_prefix={"mm_iteration": r})
        Fn = problem.objective(Tn)
        if Fn > F + 1e-12 * max(1.0, abs(F)):
            status = "mm_descent_failed"
            break
        step = _norm(Tn - T)
        dF = F - Fn
        T, F = Tn, Fn
        if stop(step <= 1e-13 * max(1.0, _norm(T)) and dF <= 1e-14 * max(1.0, abs(F))):
            break
    return T, {"native_status": status, "mm_iterations": r}, {}


# ----------------------------------------------------------------------------------------------
# M14: row-block proximal coordinate descent
# ----------------------------------------------------------------------------------------------


def coordinate_descent(problem: Problem, T, clock: Clock, job: dict):
    graphs = problem.graph_branches
    if any(br.p <= 1 for br in graphs):
        raise ConditionNotMet("row-block CD is registered only for smooth graph powers")
    selbr = problem.selection
    if selbr.p < 1:
        def sprox(v, t):
            return nonconvex_prox(v, t, selbr.p, selbr.eps)
    else:
        def sprox(v, t):
            count("row_prox_calls")
            return prox_rows(v, t, selbr.p)
    Xa, Y, H, n, d = problem.Xa, problem.Y, problem.H, problem.n, problem.d
    T = T.copy()
    S = Xa @ T @ H.T
    Ug = [problem.A(br, T[:-1]) for br in graphs]
    csc = [br.op.tocsc() for br in graphs]
    colnorm2 = np.einsum("ij,ij->j", Xa, Xa)
    Lb = np.maximum(colnorm2 / (2.0 * n), 1e-8)

    def loss_of(Sm):
        count("loss_calls")
        lse = np.logaddexp.reduce(Sm, axis=1)
        return float(np.mean(lse - np.einsum("ik,ik->i", Y, Sm)))

    stop = Stopper(1e-13, patience=2)
    sweep = 0
    while True:
        sweep += 1
        max_change = 0.0
        for j in range(d + 1):
            count("coordinate_updates")
            P = lik.softmax(S)
            xj = Xa[:, j]
            grad = xj @ ((P - Y) @ H) / n
            local = []
            if j < d:
                for gi, br in enumerate(graphs):
                    sl = slice(csc[gi].indptr[j], csc[gi].indptr[j + 1])
                    idx, val = csc[gi].indices[sl], csc[gi].data[sl]
                    Ue = Ug[gi][idx]
                    r = row_norms(Ue)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        coef = np.where(r > 0, br.mu[idx] * br.p * r ** (br.p - 2.0), 0.0)
                    grad = grad + (val[:, None] * coef[:, None] * Ue).sum(axis=0)
                    local.append((gi, idx, val, Ue, float(np.dot(br.mu[idx], r ** br.p))))
            f_old = loss_of(S) + sum(item[4] for item in local)
            Lj = Lb[j]
            bj = T[j]
            for _ in range(60):
                cand = bj - grad / Lj
                if j < d:
                    cand = sprox(cand[None, :], selbr.mu[j] / Lj)[0]
                delta = cand - bj
                Sn = S + np.outer(xj, delta @ H.T)
                f_new = loss_of(Sn)
                new_local = []
                for gi, idx, val, Ue, _ in local:
                    Un = Ue + val[:, None] * delta[None, :]
                    f_new += float(np.dot(graphs[gi].mu[idx], row_norms(Un) ** graphs[gi].p))
                    new_local.append((gi, idx, Un))
                dd = float(delta @ delta)
                if f_new <= f_old + float(grad @ delta) + (1 - 1e-4) * 0.5 * Lj * dd + 1e-15 * max(1.0, abs(f_old)):
                    break
                Lj *= 2.0
            T[j] = cand
            S = Sn
            for gi, idx, Un in new_local:
                Ug[gi][idx] = Un
            Lb[j] = max(Lj * 0.9, 1e-8)
            max_change = max(max_change, math.sqrt(dd))
        clock.observe(T, {"iteration": sweep, "max_block_change": max_change})
        if stop(max_change <= 1e-13 * max(1.0, _norm(T))):
            break
    return T, {"native_status": "sweep_change_below_1e-13", "sweeps": sweep}, {}


# ----------------------------------------------------------------------------------------------
# M15 Davis-Yin, M16 universal fast gradient, M17 conic interior point
# ----------------------------------------------------------------------------------------------


def davis_yin(problem: Problem, T, clock: Clock, job: dict):
    sel, graphs = problem.convex_terms()
    jp = JointProx(problem, None, graphs)
    gamma = 1.0 / problem.lip_loss
    z = T.copy()
    stop = Stopper(1e-13)
    it = 0
    while True:
        it += 1
        xg = z.copy()
        xg[:-1] = sel.prox(z[:-1], gamma)
        _, G = problem.loss_grad(xg)
        w = 2.0 * xg - z - gamma * G
        tol = max(1e-15, 1e-8 * it ** -2.5)
        xh, _, gap, _ = jp.solve(w, gamma, tol)
        dz = xh - xg
        z = z + dz
        clock.observe(xg, {"iteration": it, "prox_gap": gap, "z_change": _norm(dz)})
        if stop(_norm(dz) <= 1e-13 * max(1.0, _norm(z))):
            break
    return xg, {"native_status": "z_change_below_1e-13", "gamma": gamma, "iterations": it}, {}


def universal_fast_gradient(problem: Problem, T, clock: Clock, job: dict):
    """Nesterov (2015) universal fast gradient method; f = loss + graph power (Hoelder)."""
    eps = 1e-10
    selfn = RowFn(problem.selection.mu, float(problem.selection.p))

    def prox(v, a):
        out = v.copy()
        out[:-1] = selfn.prox(v[:-1], a)
        return out

    x0 = T.copy()
    y = T.copy()
    Aacc, L = 0.0, 1.0
    gsum = np.zeros_like(T)
    stop = Stopper(1e-13)
    it = 0
    while True:
        it += 1
        v = prox(x0 - gsum, Aacc) if Aacc > 0 else x0.copy()
        while True:
            a = (1.0 + math.sqrt(1.0 + 4.0 * L * Aacc)) / (2.0 * L)
            An = Aacc + a
            tk = a / An
            x = tk * v + (1.0 - tk) * y
            fx, gx = smooth_graph_value_grad(problem, x)
            xhat = prox(v - a * gx, a)
            yn = tk * xhat + (1.0 - tk) * y
            fy, _ = smooth_graph_value_grad(problem, yn, value_only=True)
            dlt = yn - x
            if fy <= fx + float(np.sum(gx * dlt)) + 0.5 * L * float(np.sum(dlt * dlt)) + 0.5 * eps * tk:
                break
            L *= 2.0
        step = _norm(yn - y)
        y, Aacc = yn, An
        gsum = gsum + a * gx
        L *= 0.5
        clock.observe(y, {"iteration": it, "L": L, "A": Aacc, "step": step})
        if stop(step <= 1e-13 * max(1.0, _norm(y))):
            break
    return y, {"native_status": "step_below_1e-13", "eps": eps, "iterations": it}, {}


def import_cvxpy(extra_site: str | None = None):
    try:
        import cvxpy as cp  # noqa: F401
    except ImportError:
        import sys
        if extra_site and Path(extra_site).exists() and extra_site not in sys.path:
            sys.path.append(extra_site)
        import cvxpy as cp  # noqa: F401
    return cp


def conic(problem: Problem, T, clock: Clock, job: dict):
    cp = import_cvxpy(job.get("cvxpy_site"))
    Theta = cp.Variable(problem.shape)
    S = problem.Xa @ Theta @ problem.H.T
    obj = (cp.sum(cp.log_sum_exp(S, axis=1)) - cp.sum(cp.multiply(problem.Y, S))) / problem.n
    cons, terms = [], []
    for br in problem.branches:
        m = br.mu.size
        U = cp.Variable((m, problem.k1))
        cons.append(U == (Theta[:-1] if br.op is None else br.op @ Theta[:-1]))
        if br.p == 1:
            obj = obj + br.mu @ cp.norm(U, 2, axis=1)
        elif br.p == 2:
            obj = obj + br.mu @ cp.sum(cp.square(U), axis=1)
        else:
            obj = obj + br.mu @ cp.power(cp.norm(U, 2, axis=1), br.p)
        terms.append(br)
    prob = cp.Problem(cp.Minimize(obj), cons)
    Theta.value = T
    count("conic_solves")
    remaining = max(0.1, clock.cap - clock.elapsed())
    prob.solve(solver=cp.CLARABEL, max_iter=500, time_limit=remaining, tol_gap_abs=1e-10,
               tol_gap_rel=1e-10, tol_feas=1e-10, verbose=False)
    if Theta.value is None:
        raise ConditionNotMet(f"conic solver returned no point: {prob.status}")
    X = np.asarray(Theta.value, dtype=float)
    arrays = {}
    for br, c in zip(terms, cons):
        if br.op is None or c.dual_value is None:
            continue
        y = np.asarray(c.dual_value, dtype=float)
        U = problem.A(br, X[:-1])
        r = row_norms(U)
        nz = r > 1e-8 * max(1.0, r.max(initial=0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = np.where(nz, br.mu * br.p * r ** (br.p - 2.0), 0.0)[:, None] * U
        if float(np.sum(y * grad)) < 0:  # CVXPY's sign convention for the equality multiplier
            y = -y
        arrays[f"dual_{br.name}"] = y
    stats = prob.solver_stats
    F = problem.objective(X)
    extra = {"native_status": prob.status, "cvxpy_value": float(prob.value),
             "cvxpy_version": cp.__version__, "cvxpy_path": str(Path(cp.__file__).parent),
             "representation_abs_difference": abs(float(prob.value) - F),
             "iterations": getattr(stats, "num_iters", None),
             "solver_seconds_reported": getattr(stats, "solve_time", None),
             "setup_seconds_reported": getattr(stats, "setup_time", None)}
    clock.observe(X, {"iteration": extra["iterations"] or 0, "native_status": prob.status},
                  check=False)
    if clock.elapsed() > clock.cap:
        extra["exit_status"] = "time_limit"
    return X, extra, arrays


# ----------------------------------------------------------------------------------------------
# M18 DCA
# ----------------------------------------------------------------------------------------------


def dca(problem: Problem, T, clock: Clock, job: dict):
    sel, graphs = problem.convex_terms()
    L = problem.lip_loss
    tau, sigma = _pd_steps(problem, graphs, L)
    Y = [np.zeros((br.op.shape[0], problem.k1)) for br, _ in graphs]
    F = problem.objective(T)
    stop = Stopper(1e-13, patience=2)
    k = 0
    status = "small_change"
    while True:
        k += 1
        _, corr = problem.psi_value_grad(T)  # linearized concave part -mu k(||.||), as a gradient
        corrfull = problem.lift(corr)
        sur_old = problem.loss(T) + float(np.sum(corrfull * T)) + problem.P_value(T)
        tol = max(1e-13, 1e-6 * k ** -2.0)
        x = T.copy()
        inner = 0
        for inner in range(1, 20001):
            count("inner_iterations")
            _, G = problem.loss_grad(x)
            G += corrfull
            if graphs:
                G[:-1] += _sum_adj(problem, [(br, y) for (br, _), y in zip(graphs, Y)])
            xn = x - tau * G
            xn[:-1] = sel.prox(xn[:-1], tau)
            Bbar = 2.0 * xn[:-1] - x[:-1]
            Y = [fn.dual_prox(y + sigma * problem.A(br, Bbar), sigma) for (br, fn), y in zip(graphs, Y)]
            dx = _norm(xn - x)
            x = xn
            if inner >= 2 and dx <= tol * max(1.0, _norm(x)):
                sur_new = problem.loss(x) + float(np.sum(corrfull * x)) + problem.P_value(x)
                if sur_new <= sur_old + 1e-14 * (1.0 + abs(sur_old)):
                    break
        Fn = problem.objective(x)
        if Fn > F + 1e-12 * max(1.0, abs(F)):
            status = "dc_subproblem_not_descending"
            break
        step = _norm(x - T)
        dF = F - Fn
        T, F = x, Fn
        count("outer_iterations")
        clock.observe(T, {"iteration": k, "inner_iterations": inner, "step": step})
        if stop(step <= 1e-13 * max(1.0, _norm(T)) and dF <= 1e-14 * max(1.0, abs(F))):
            break
    return T, {"native_status": status, "dc_iterations": k},  \
        {f"dual_{br.name}": y for (br, _), y in zip(graphs, Y)}


# ----------------------------------------------------------------------------------------------
# Smoothing routes M20-M22 and the original IRLS M19
# ----------------------------------------------------------------------------------------------


def smoothing_levels(job: dict) -> list[tuple[str, float]]:
    mode = job.get("smoothing")
    fine = job.get("fine_ratio", SMOOTHING_RATIOS["fine"])
    ratios = {"coarse": SMOOTHING_RATIOS["coarse"], "medium": SMOOTHING_RATIOS["medium"], "fine": fine}
    if mode == "continuation":
        return [(k, ratios[k]) for k in ("coarse", "medium", "fine")]
    return [(mode, ratios[mode])]


def fine_ratio(problem: Problem, method: str) -> tuple[float, float | None]:
    """Fine smoothing ratio and its certified bias bound (plan Section 5)."""
    radii = problem.meta["smoothing_radii"]
    fine = SMOOTHING_RATIOS["fine"]
    if method == "M20" and all(br.p == 1 for br in problem.graph_branches):
        total = sum(radii[br.name] * float(br.mu.sum()) for br in problem.graph_branches)
        bound = lambda c: 0.5 * c * total  # noqa: E731
        if bound(fine) > BIAS_TARGET:
            fine = 2.0 * BIAS_TARGET / total
        return fine, bound(fine)
    if method == "M21":
        def bound(c):
            return sum(float(np.dot(br.mu, h_shift(np.full(br.mu.size, c * radii[br.name]),
                                                   br.p, br.offset(1e-3)))) for br in problem.branches)
        if bound(fine) > BIAS_TARGET:
            lo, hi = 0.0, fine
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if bound(mid) <= BIAS_TARGET else (lo, mid)
            fine = lo
        return fine, bound(fine)
    return fine, None


def smoothed_apg(problem: Problem, T, clock: Clock, job: dict):
    """M20: per-row Moreau envelope of each convex graph power (Huber for TV) + APG."""
    radii = problem.meta["smoothing_radii"]
    graphs = problem.graph_branches
    selfn = RowFn(problem.selection.mu, float(problem.selection.p))
    stages = smoothing_levels(job)

    def native_value_grad(X, lams, value_only=False):
        if value_only:
            f, G = problem.loss(X), None
        else:
            f, G = problem.loss_grad(X)
        B = X[:-1]
        for br, lam in zip(graphs, lams):
            U = problem.A(br, B)
            if br.p == 1:
                r = row_norms(U)
                inside = r <= lam
                f += float(np.dot(br.mu, np.where(inside, r * r / (2.0 * lam), r - 0.5 * lam)))
                if G is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        coef = np.where(inside, 1.0 / lam, 1.0 / np.where(r > 0, r, 1.0))
                    G[:-1] += problem.At(br, (br.mu * coef)[:, None] * U)
            else:
                count("row_prox_calls")
                V = prox_rows(U, lam, br.p)
                f += float(np.dot(br.mu, row_norms(V) ** br.p + np.sum((U - V) ** 2, axis=1) / (2.0 * lam)))
                if G is not None:
                    G[:-1] += problem.At(br, (br.mu / lam)[:, None] * (U - V))
        return f, G

    stage_info = []
    for si, (name, ratio) in enumerate(stages):
        lams = [ratio * radii[br.name] ** (2.0 - br.p) for br in graphs]
        last = si == len(stages) - 1
        apg_tol, x_tol = (1e-14, 1e-11) if last else (1e-9, 1e-6)
        L, X, Yk, tk = 1.0, T.copy(), T.copy(), 1.0
        f0, _ = native_value_grad(X, lams, value_only=True)
        F = f0 + selfn.value(X[:-1])
        n_small, it = 0, 0
        while True:
            fY, GY = native_value_grad(Yk, lams)
            while True:
                Xn = Yk - GY / L
                Xn[:-1] = selfn.prox(Xn[:-1], 1.0 / L)
                D = Xn - Yk
                fX, _ = native_value_grad(Xn, lams, value_only=True)
                if fX <= fY + float(np.sum(GY * D)) + 0.5 * L * float(np.sum(D * D)) + 1e-15 * max(1.0, abs(fY)):
                    break
                L *= 2.0
            Fn = fX + selfn.value(Xn[:-1])
            if Fn > F:
                if tk == 1.0:
                    break
                Yk, tk = X.copy(), 1.0
                continue
            it += 1
            tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            Yk = Xn + ((tk - 1.0) / tn) * (Xn - X)
            step = _norm(Xn - X)
            small = F - Fn <= apg_tol * max(1.0, abs(Fn)) and step <= x_tol * max(_norm(Xn), 1e-12)
            n_small = n_small + 1 if small else 0
            X, tk, F = Xn, tn, Fn
            clock.observe(X, {"iteration": it, "stage": si, "stage_name": name, "smoothing_ratio": ratio,
                              "native_objective": F, "L": L})
            if n_small >= 3:
                break
            L *= 0.9
        T = X
        stage_info.append({"stage": name, "ratio": ratio, "lambdas": lams, "iterations": it,
                           "native_objective": F})
    bias = None
    if all(br.p == 1 for br in graphs):
        bias = sum(0.5 * lam * float(br.mu.sum()) for br, lam in zip(graphs, stage_info[-1]["lambdas"]))
    return T, {"native_status": "final_stage_native_stop", "stages": stage_info,
               "uniform_bias_bound": bias}, {}


def irls(problem: Problem, T, clock: Clock, job: dict):
    method = job["method"]
    radii = problem.meta["smoothing_radii"]
    C = problem.Xa.T @ problem.Xa / (2.0 * problem.n)
    brs = problem.branches
    stages = [("original", 0.0)] if method == "M19" else smoothing_levels(job)

    def native(X, deltas):
        f = problem.loss(X)
        B = X[:-1]
        for br, dl in zip(brs, deltas):
            r = row_norms(problem.A(br, B))
            eps = br.offset(1e-3)
            if method == "M19":
                v = h_shift(r, br.p, eps)
            elif method == "M21":
                v = h_shift(np.sqrt(r * r + dl * dl), br.p, eps) - h_shift(dl, br.p, eps)
            else:
                v = (r * r + dl * dl) ** (br.p / 2.0) - dl**br.p
            f += float(np.dot(br.mu, v))
        return f

    stage_info = []
    status = "final_stage_native_stop"
    for si, (name, ratio) in enumerate(stages):
        deltas = [ratio * radii[br.name] for br in brs]
        last = si == len(stages) - 1
        obj_tol, x_tol = (1e-14, 1e-11) if last else (1e-9, 1e-6)
        Fn = native(T, deltas)
        n_small, it = 0, 0
        while True:
            _, G = problem.loss_grad(T)
            M = C.copy()
            B = T[:-1]
            for br, dl in zip(brs, deltas):
                r = row_norms(problem.A(br, B))
                eps = br.offset(1e-3)
                if method == "M19":
                    if br.p < 2 and np.any(r == 0):
                        raise ConditionNotMet(f"zero radius in branch {br.name} (p={br.p}): the "
                                              "exact quadratic majorizer weight is undefined")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        w = br.mu * dh_shift(r, br.p, eps) / r if br.p != 2 else 2.0 * br.mu
                elif method == "M21":
                    rt = np.sqrt(r * r + dl * dl)
                    w = br.mu * dh_shift(rt, br.p, eps) / rt
                else:
                    w = br.mu * br.p * (r * r + dl * dl) ** (br.p / 2.0 - 1.0)
                if not np.all(np.isfinite(w)):
                    raise ConditionNotMet(f"nonfinite IRLS weight in branch {br.name}")
                if br.op is None:
                    M[np.arange(problem.d), np.arange(problem.d)] += w
                else:
                    M[:-1, :-1] += (br.op.T @ sp.diags(w) @ br.op).toarray()
            count("factorizations")
            count("linear_solves")
            try:
                Tn = la.cho_solve(la.cho_factor(M, lower=True), C @ T - G)
            except la.LinAlgError as err:
                raise ConditionNotMet(f"IRLS system not positive definite: {err}") from err
            F_old, Fn = Fn, native(Tn, deltas)
            if Fn > F_old + 1e-12 * (1.0 + abs(F_old)):
                status = "majorizer_descent_failed"
                break
            it += 1
            step = _norm(Tn - T)
            T = Tn
            clock.observe(T, {"iteration": it, "stage": si, "stage_name": name,
                              "smoothing_ratio": ratio, "native_objective": Fn})
            small = F_old - Fn <= obj_tol * max(1.0, abs(Fn)) and step <= x_tol * max(1.0, _norm(T))
            n_small = n_small + 1 if small else 0
            if n_small >= 3:
                break
        stage_info.append({"stage": name, "ratio": ratio, "deltas": deltas, "iterations": it,
                           "native_objective": Fn})
        if status != "final_stage_native_stop":
            break
    bias = None
    if method == "M21":
        bias = sum(float(np.dot(br.mu, h_shift(np.full(br.mu.size, dl), br.p, br.offset(1e-3))))
                   for br, dl in zip(brs, stage_info[-1]["deltas"]))
    return T, {"native_status": status, "stages": stage_info, "uniform_bias_bound": bias}, {}


# ----------------------------------------------------------------------------------------------
# M23: original joint proximal gradient with weak-convexity compensation
# ----------------------------------------------------------------------------------------------


class CompensatedProx:
    """``argmin_x 1/(2 alpha)||x - v||^2 + sum_a mu_a h_a(||A_a x_B||)`` (original penalty).

    With ``alpha <= 1/(2W)`` each p < 1 radial term plus ``kappa r^2/2`` is convex, and the
    remaining quadratic matrix ``M = I/alpha - sum A' diag(mu kappa) A`` stays positive definite.
    Solved by FISTA on all branch duals; ``x(y) = M^{-1}(v/alpha - A'y)``."""

    def __init__(self, problem: Problem, alpha: float):
        self.problem, self.alpha = problem, alpha
        self.terms = []
        MB = np.eye(problem.d) / alpha
        for br in problem.branches:
            if br.p < 1:
                kap = curvature_bound(br.p, br.eps)
                if br.op is None:
                    MB[np.diag_indices(problem.d)] -= br.mu * kap
                else:
                    MB -= (br.op.T @ sp.diags(br.mu * kap) @ br.op).toarray()
                self.terms.append((br, RowFn(br.mu, br.p, "compensated", br.eps)))
            else:
                self.terms.append((br, RowFn(br.mu, float(br.p))))
        count("factorizations")
        self.fac = la.cho_factor(MB, lower=True)
        G = problem.gram()
        self.lip = float(la.eigh(G, MB, eigvals_only=True, subset_by_index=[problem.d - 1, problem.d - 1])[0])
        self.y = None

    def x_of(self, v, y):
        x = v.copy()
        rhs = v[:-1] / self.alpha - _sum_adj(self.problem, [(br, yy) for (br, _), yy in zip(self.terms, y)])
        count("linear_solves")
        x[:-1] = la.cho_solve(self.fac, rhs)
        return x

    def gap(self, x, y):
        B = x[:-1]
        return float(sum(np.sum(fn.fy(self.problem.A(br, B), yy)) for (br, fn), yy in zip(self.terms, y)))

    def solve(self, v, tol, max_iter=100000, check_every=10):
        p = self.problem
        y = self.y or [np.zeros((br.mu.size, p.k1)) for br, _ in self.terms]
        s = 1.0 / self.lip
        z = [yy.copy() for yy in y]
        t = 1.0
        best = (math.inf, None, None)
        it = 0
        for it in range(1, max_iter + 1):
            count("inner_iterations")
            xz = self.x_of(v, z)
            yn = [fn.dual_prox(zz + s * p.A(br, xz[:-1]), s) for (br, fn), zz in zip(self.terms, z)]
            if sum(float(np.sum((zz - a) * (a - b))) for zz, a, b in zip(z, yn, y)) > 0:
                t, z = 1.0, [a.copy() for a in yn]
            else:
                t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                z = [a + (t - 1.0) / t_new * (a - b) for a, b in zip(yn, y)]
                t = t_new
            y = yn
            if it % check_every == 0:
                x = self.x_of(v, y)
                g = self.gap(x, y)
                if g < best[0]:
                    best = (g, x, [yy.copy() for yy in y])
                if g <= tol:
                    break
        if best[1] is None:
            x = self.x_of(v, y)
            best = (self.gap(x, y), x, y)
        self.y = best[2]
        return best[1], best[0], it


def original_joint_pg(problem: Problem, T, clock: Clock, job: dict):
    W = problem.weak_bound()
    alpha = 0.5 / max(problem.lip_loss, W, 1.0)
    cpx = CompensatedProx(problem, alpha)
    F = problem.objective(T)
    stop = Stopper(1e-13)
    it, n_increase = 0, 0
    while True:
        it += 1
        _, G = problem.loss_grad(T)
        tol = max(1e-15, 1e-8 * it ** -2.5)
        x, gap, inner = cpx.solve(T - alpha * G, tol)
        Fn = problem.objective(x)
        if Fn > F + 1e-12 * max(1.0, abs(F)):
            n_increase += 1
        step = _norm(x - T)
        T, F = x, Fn
        clock.observe(T, {"iteration": it, "prox_gap": gap, "inner_iterations": inner, "step": step})
        if stop(step <= 1e-13 * max(1.0, _norm(T))):
            break
    return T, {"native_status": "step_below_1e-13", "alpha": alpha, "weak_bound": W,
               "objective_increases": n_increase, "iterations": it}, {}


ADAPTERS = {
    "M01": existing_solver, "M02": existing_solver, "M03": existing_solver,
    "M04": condat_vu, "M05": pd3o, "M06": preconditioned_pd, "M07": direct_admm,
    "M08": majorized_admm, "M09": mm_primal_dual, "M10": redistributed_pg, "M11": inexact_apg,
    "M12a": nonconvex_pg, "M12b": nonconvex_apg, "M13a": prox_newton, "M13b": prox_newton,
    "M13c": mm_prox_newton, "M14": coordinate_descent, "M15": davis_yin,
    "M16": universal_fast_gradient, "M17": conic, "M18": dca, "M19": irls, "M20": smoothed_apg,
    "M21": irls, "M22": irls, "M23": original_joint_pg, "M24": direct_admm,
}


# ----------------------------------------------------------------------------------------------
# Single-run driver (executed in its own process by the worker script)
# ----------------------------------------------------------------------------------------------


def initial_theta(problem: Problem, job: dict) -> np.ndarray:
    kind = job.get("init", "common")
    if job.get("initial_state"):
        return np.load(job["initial_state"])["theta"].copy()
    if kind not in ("common", "zero", "perturbed"):
        raise ValueError(f"unknown initial value {kind!r}")
    if kind == "zero":
        return problem.theta_null.copy()
    theta = problem.theta0.copy()
    if kind == "perturbed":
        rng = np.random.default_rng(job.get("perturbation_seed", 303))
        E = rng.normal(size=theta[:-1].shape)
        E *= 0.01 * max(_norm(theta[:-1]), 1.0) / _norm(E)
        theta[:-1] += E
    return theta


def _peak_memory_mb() -> float | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "peak_rss", None)
        if peak is None:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        return peak / 2**20
    except Exception:  # pragma: no cover - diagnostics only
        return None


def run_job(job: dict, run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    inst = load_instance(Path(job["instance_path"]))
    problem = Problem(inst)
    method = job["method"]
    theta = initial_theta(problem, job)
    import_seconds = None
    if method == "M17":
        t0 = time.perf_counter()
        import_cvxpy(job.get("cvxpy_site"))
        import_seconds = time.perf_counter() - t0
    RUNTIME.reset()
    clock = Clock(job["cap"], run_dir, objective=lambda th: raw_objective(problem, th))
    clock.start()
    clock.observe(theta, {"iteration": 0}, check=False)
    extra, arrays = {}, {}
    try:
        theta, extra, arrays = ADAPTERS[method](problem, theta, clock, job)
        exit_status = extra.pop("exit_status", "native_stop")
    except BudgetReached:
        exit_status = "time_limit"
        theta = clock.current[0]
    except ConditionNotMet as err:
        # At the initial value: the route is not applicable there.  After accepted steps: the
        # condition failed during the run (e.g. an IRLS radius reached exactly zero).
        exit_status = "condition_not_met" if clock.n_observed <= 1 else "condition_failed_during_run"
        theta = clock.current[0]
        extra = {"reason": str(err)}
    except Exception as err:  # recorded, never converted into a result
        exit_status = "error"
        theta = clock.current[0]
        extra = {"reason": f"{type(err).__name__}: {err}", "traceback": traceback.format_exc()}
    seconds = clock.finish(theta, {"exit_status": exit_status,
                                   "native_status": extra.get("native_status")})
    save_npz(run_dir / "final.npz", theta=theta, **arrays)
    result = {
        "run_id": job["run_id"], "method": method, "method_name": METHODS[method][0],
        "objective_kind": METHODS[method][1], "instance": job["instance"], "phase": job["phase"],
        "variant": job.get("variant", ""), "cap_seconds": job["cap"], "exit_status": exit_status,
        "solver_seconds": seconds, "monitoring_seconds": clock.overhead,
        "observations": clock.n_observed, "checkpoints": len(clock.records),
        "objective_monitor": clock.monitor,
        "operations": dict(RUNTIME.counts), "peak_memory_mb": _peak_memory_mb(),
        "cvxpy_import_seconds": import_seconds,
        "process_driver_seconds": time.perf_counter() - started,
        "stationarity_certified_by_route": False, **extra,
    }
    write_json(run_dir / "result.json", plain(result))
    return result


__all__ = ["ADAPTERS", "ALL", "CONVEX", "METHODS", "NONCONVEX", "fine_ratio", "run_job",
           "read_json", "write_json"]
