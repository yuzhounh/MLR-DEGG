"""Region-wise proximal MM with two-block ADMM (Algorithm "alg:mlrdegg").

Outer loop (MM): the softmax loss is replaced by its global quadratic majorizer
(eq:likelihood-majorizer); every branch with exponent ``p < 1`` is replaced by a weighted group
norm (eq:scalar-majorization); branches with ``p >= 1`` keep their convex power.  A proximal term
``gamma/2 ||Theta - Theta^(r)||^2`` makes each surrogate strongly convex (eq:mm-surrogate).

Inner loop (ADMM): the first block is ``Theta = [B; c^T]`` (a quadratic problem whose matrix is
constant for fixed ``rho``); the second block stacks one splitting variable per branch
(eq:explicit-splitting), each updated by a row-wise radial proximal map.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import likelihood as lik
from .penalties import h_penalty, mm_weights, prox_rows, row_norms, subgradient_distance

log = logging.getLogger(__name__)


@dataclass
class Branch:
    """One penalty branch ``sum_j mu_j h_{p,eps}(||(A B)_j||)`` (eq:active-objective).

    ``op=None`` denotes the identity (row-group selection branch).  ``eps`` is the branch's own
    offset (used when ``p < 1``); ``None`` falls back to ``SolverOptions.epsilon``.
    """

    name: str
    op: sp.csr_matrix | None
    mu: np.ndarray
    p: float
    eps: float | None = None

    def offset(self, default: float) -> float:
        return default if self.eps is None else self.eps

    def apply(self, B: np.ndarray) -> np.ndarray:
        return B if self.op is None else self.op @ B

    def adjoint(self, V: np.ndarray) -> np.ndarray:
        return V if self.op is None else self.op.T @ V


@dataclass
class SolverOptions:
    epsilon: float = 1e-3
    gamma: float = 1e-3
    rho: float | str = "auto"
    max_outer: int = 200
    max_inner: int = 500
    tol_obj: float = 1e-7
    tol_x: float = 1e-4
    inner_abs_tol: float = 1e-8
    inner_rel_tol0: float = 1e-3
    inner_rel_tol_min: float = 1e-6
    adapt_rho: bool = True
    adapt_rho_outer: int = 1  # residual balancing only in the first outer iterations (each change
    #                           refactorizes the (B, c) system, which is costly for large d)
    rho_check_every: int = 10
    max_rho_updates: int = 20
    # Inexact MM: each surrogate starts with ``inner_budget0`` ADMM iterations; if the candidate
    # fails the acceptance test (eq:acceptance), ADMM continues with the budget multiplied by
    # ``inner_budget_growth`` and a tighter tolerance, up to ``max_inner``.  ``None`` starts at
    # ``max_inner`` (near-exact surrogate solves).
    inner_budget0: int | None = 5
    inner_budget_growth: float = 4.0
    # Surrogate-error diagnostic (Proposition prop:inexact, Remark rem:inexact-scope).  At every
    # candidate that decreases the surrogate, the row-wise error bound (see _kkt_error_bound) on its
    # distance to the exact surrogate minimizer is compared with the geometric schedule
    #     tau_r = max(kkt_tol0 * kkt_decay**r, kkt_tol_min) * max(1, ||Theta^(r)||_F).
    # With kkt_enforce=True ADMM continues within the growing finite budget when the bound
    # exceeds tau_r; a descent candidate can still be accepted at max_inner.  Otherwise the
    # bound is only recorded.
    # Either way, accepted steps whose bound exceeds tau_r are counted in n_kkt_violations
    # (enforce mode: those accepted at max_inner).  kkt_tol0=None skips the computation.
    # The floor kkt_tol_min keeps the target above what the CG solves (cg_tol) can deliver, so the
    # vanishing-error assumption of the proposition is not certified by this finite schedule.
    # Enforcing is off by default: on a d >> n problem (simulation, d=4608, n=160) ADMM did not
    # reach the tolerance within max_inner=300 at any gamma in [1e-3, 1], so enforcing cost about
    # 4x the time for a 2e-5 relative objective improvement without certifying the steps.
    kkt_tol0: float | None = 1e-2
    kkt_decay: float = 0.95
    kkt_tol_min: float = 1e-6
    kkt_enforce: bool = False
    dense_threshold: int = 4000
    cg_tol: float = 1e-6
    cg_maxiter: int = 500
    # Accelerated proximal gradient (mlrdegg.proxgrad), used for smooth-graph convex members.
    apg_tol: float = 1e-9
    apg_max_iter: int = 20000
    # Optional accepted-outer-iteration restart; unused by ordinary fits.
    resume_history: list[dict] | None = None
    resume_converged: bool = False
    resume_rho_updates: int = 0
    resume_kkt_violations: int = 0


@dataclass
class SolverState:
    """Warm-startable state: parameters plus splitting and scaled dual variables."""

    Theta: np.ndarray
    U: list[np.ndarray] | None = None
    Z: list[np.ndarray] | None = None
    rho: float | None = None


@dataclass
class SolverResult:
    Theta: np.ndarray
    U: list[np.ndarray]
    Z: list[np.ndarray]
    rho: float
    objective: float
    n_outer: int
    n_inner_total: int
    converged: bool
    status: str
    history: list[dict] = field(default_factory=list)
    runtime: float = 0.0
    n_kkt_violations: int = 0  # accepted steps whose KKT error bound exceeded the tolerance
    solver: str = "unknown"
    diagnostics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------
# Linear system of the (B, c) update: [(1/2n) Xa^T Xa + gamma I + rho blkdiag(P, 0)] Theta = R,
# with P = I + sum_graph D^T D (eq:bc-hvp).  The matrix does not depend on the MM weights.
# ----------------------------------------------------------------------------------------------


def _block_pcg(matvec, Bm, X0, dinv, tol, maxiter):
    X = X0.copy()
    R = Bm - matvec(X)
    bnorm = np.linalg.norm(Bm, axis=0)
    active = np.linalg.norm(R, axis=0) > tol * np.maximum(bnorm, 1e-300)
    if not np.any(active):
        return X
    Zr = R * dinv[:, None]
    Pm = Zr.copy()
    rz = np.einsum("ij,ij->j", R, Zr)
    for _ in range(maxiter):
        AP = matvec(Pm)
        pap = np.einsum("ij,ij->j", Pm, AP)
        alpha = np.where(active & (pap > 0), rz / np.where(pap > 0, pap, 1.0), 0.0)
        X += Pm * alpha
        R -= AP * alpha
        active = np.linalg.norm(R, axis=0) > tol * np.maximum(bnorm, 1e-300)
        if not np.any(active):
            break
        Zr = R * dinv[:, None]
        rz_new = np.einsum("ij,ij->j", R, Zr)
        beta = np.where(active, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
        Pm = Zr + Pm * beta
        rz = rz_new
    return X


class _CoefficientSystem:
    def __init__(self, Xa: np.ndarray, branches: list[Branch], gamma: float, opts: SolverOptions):
        self.Xa = Xa
        self.n, d1 = Xa.shape
        self.d = d1 - 1
        self.gamma = gamma
        self.opts = opts
        L = sp.csr_matrix((self.d, self.d))
        for br in branches:
            if br.op is not None:
                L = L + (br.op.T @ br.op)
        self.L = L.tocsr()
        self.dense = d1 <= opts.dense_threshold
        self._x_prev = None
        self.rho = None

    def set_rho(self, rho: float) -> None:
        self.rho = rho
        n, d, g = self.n, self.d, self.gamma
        if self.dense:
            M = (self.Xa.T @ self.Xa) / (2.0 * n)
            M[np.diag_indices_from(M)] += g
            M[:d, :d] += rho * (np.eye(d) + self.L.toarray())
            self._chol = sla.cho_factor(M, lower=True)
            return
        # Woodbury: M = Q + (1/2n) Xa^T Xa with Q = blkdiag(gamma I + rho (I + L), gamma).
        self._qdiag_inv = 1.0 / (g + rho + rho * self.L.diagonal())
        QiXt = self._q_solve(self.Xa.T.copy(), None)
        C = 2.0 * n * np.eye(n) + self.Xa @ QiXt
        self._QiXt = QiXt
        self._capf = sla.cho_factor(C, lower=True)
        self._x_prev = None

    def _q_matvec_B(self, X):
        return (self.gamma + self.rho) * X + self.rho * (self.L @ X)

    def _q_solve(self, R: np.ndarray, X0) -> np.ndarray:
        d = self.d
        out = np.empty_like(R)
        out[d] = R[d] / self.gamma
        x0 = np.zeros_like(R[:d]) if X0 is None else X0[:d]
        out[:d] = _block_pcg(
            self._q_matvec_B, R[:d], x0, self._qdiag_inv, self.opts.cg_tol, self.opts.cg_maxiter
        )
        return out

    def solve(self, R: np.ndarray) -> np.ndarray:
        if self.dense:
            return sla.cho_solve(self._chol, R)
        Y = self._q_solve(R, self._x_prev)
        self._x_prev = Y
        return Y - self._QiXt @ sla.cho_solve(self._capf, self.Xa @ Y)


# ----------------------------------------------------------------------------------------------
# Objective and surrogate
# ----------------------------------------------------------------------------------------------


def penalty_value(B: np.ndarray, branches: list[Branch], eps: float) -> float:
    return float(sum(np.dot(br.mu, h_penalty(row_norms(br.apply(B)), br.p, br.offset(eps)))
                     for br in branches))


def objective(Xa, Y, H, Theta, branches, eps) -> float:
    """Fixed-offset objective ``F_eps`` (eq:main-model-w) in contrast coordinates."""
    return lik.loss(Xa, Y, Theta, H) + penalty_value(Theta[:-1], branches, eps)


def _surrogate(Xa, Theta, Theta_r, G, gamma, branches, kappas, nus) -> float:
    """``Q_r`` up to an additive constant, evaluated from ``A_a B`` (eq:acceptance)."""
    n = Xa.shape[0]
    D = Theta - Theta_r
    XD = Xa @ D
    val = np.sum(G * D) + np.sum(XD * XD) / (4.0 * n) + 0.5 * gamma * np.sum(D * D)
    B = Theta[:-1]
    for br, kap, nu in zip(branches, kappas, nus):
        val += np.dot(kap, row_norms(br.apply(B)) ** nu)
    return float(val)


def _best_candidate(Xa, cand, Theta_r, G, gamma, branches, kappas, nus, U):
    """Candidate with the lower surrogate value: the ADMM coefficient block, or the same block
    with the rows set to zero that the selection splitting variable has thresholded exactly."""
    q = _surrogate(Xa, cand, Theta_r, G, gamma, branches, kappas, nus)
    for br, u in zip(branches, U):
        if br.op is None:
            zero = row_norms(u) == 0
            if np.any(zero):
                alt = cand.copy()
                alt[:-1][zero] = 0.0
                q_alt = _surrogate(Xa, alt, Theta_r, G, gamma, branches, kappas, nus)
                if q_alt <= q:
                    return alt, q_alt
    return cand, q


def _op_norm_bound(branch: Branch) -> float:
    """Upper bound on the spectral norm of the branch operator (Gershgorin on ``A^T A``)."""
    if branch.op is None:
        return 1.0
    AtA = (branch.op.T @ branch.op).tocsr()
    return float(np.sqrt(np.max(np.asarray(abs(AtA).sum(axis=1)).ravel())))


def _kkt_error_bound(Xa, G, Theta, Theta_r, gamma, branches, kappas, nus, U, Z, rho, op_norms):
    """Row-wise sharpened form of the error bound (eq:error-bound) on ``||Theta - Theta_hat||_F``.

    With ``delta = Theta - Theta_hat`` and ``Lambda = rho Z``, strong convexity gives
    ``gamma ||delta||^2 <= e_st ||delta|| - sum_j <Lambda_j - Lambda_hat_j, (A delta)_j>``.  Each
    row is bounded by monotonicity of the convex row subdifferential, either at ``(A Theta)_j``
    (``>= -dist(Lambda_j, d g(A Theta)_j) ||(A delta)_j||``, linear in ``||delta||``) or, for rows
    thresholded to ``U_j = 0`` with ``nu = 1``, at ``U_j`` (``>= -d_U,j (f_j + ||(A delta)_j||) -
    (||Lambda_j|| + kappa_j) f_j`` with ``f_j = ||(A Theta - U)_j||``, using ``||Lambda_hat_j|| <=
    kappa_j``).  Solving ``gamma x^2 <= a x + b`` gives ``x <= a / gamma + sqrt(b / gamma)``.
    """
    n = Xa.shape[0]
    D = Theta - Theta_r
    grad = G + Xa.T @ (Xa @ D) / (2.0 * n) + gamma * D
    B = Theta[:-1]
    lin2, b = [], 0.0
    for br, kap, nu, u, z, a_norm in zip(branches, kappas, nus, U, Z, op_norms):
        lam = rho * z
        grad[:-1] += br.adjoint(lam)
        AB = br.apply(B)
        d_at_ab = subgradient_distance(lam, AB, kap, nu)
        if nu == 1.0:
            thr = row_norms(u) == 0
            d_at_u = subgradient_distance(lam, u, kap, nu)
            f = row_norms(AB - u)
            kap_rows = np.broadcast_to(np.asarray(kap, dtype=float), f.shape)
            row_lin = np.where(thr, d_at_u, d_at_ab)
            b += float(np.sum((d_at_u[thr] + row_norms(lam[thr]) + kap_rows[thr]) * f[thr]))
        else:
            row_lin = d_at_ab
        lin2.append(a_norm * float(np.linalg.norm(row_lin)))
    a = float(np.linalg.norm(grad)) + float(sum(lin2))
    return a / gamma + np.sqrt(b / gamma), (a, b)


def default_rho(Xa: np.ndarray) -> float:
    n = Xa.shape[0]
    return float(max(np.mean(np.einsum("ij,ij->j", Xa, Xa)) / (2.0 * n), 1e-8))


# ----------------------------------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------------------------------


def solve_mm_admm(
    Xa: np.ndarray,
    Y: np.ndarray,
    H: np.ndarray,
    branches: list[Branch],
    opts: SolverOptions,
    state: SolverState,
) -> SolverResult:
    t_start = time.perf_counter()
    n = Xa.shape[0]
    eps, gamma = opts.epsilon, opts.gamma
    Theta = state.Theta.copy()
    B = Theta[:-1]

    U = [br.apply(B).copy() for br in branches] if state.U is None else [u.copy() for u in state.U]
    Z = [np.zeros_like(u) for u in U] if state.Z is None else [z.copy() for z in state.Z]
    rho = state.rho or (default_rho(Xa) if opts.rho == "auto" else float(opts.rho))

    system = _CoefficientSystem(Xa, branches, gamma, opts)
    system.set_rho(rho)
    op_norms = [_op_norm_bound(br) for br in branches] if opts.kkt_tol0 is not None else None
    n_rho_updates = opts.resume_rho_updates
    XtX_over_2n = None  # only needed for the dense right-hand side
    if system.dense:
        XtX_over_2n = (Xa.T @ Xa) / (2.0 * n)

    F = objective(Xa, Y, H, Theta, branches, eps)
    history: list[dict] = list(opts.resume_history or [])
    status = "converged" if opts.resume_converged else "max_outer"
    converged = opts.resume_converged
    n_inner_total = sum(h["inner_iterations"] for h in history)
    n_kkt_violations = opts.resume_kkt_violations
    size_pri = np.sqrt(sum(u.size for u in U))
    size_dual = np.sqrt(B.size)

    r = 0
    for r in range(len(history), len(history) if converged else opts.max_outer):
        Theta_r = Theta.copy()
        B_r = Theta_r[:-1]
        _, G, _ = lik.loss_grad(Xa, Y, Theta_r, H)

        kappas, nus = [], []
        for br in branches:
            omega, nu = mm_weights(row_norms(br.apply(B_r)), br.p, br.offset(eps))
            kappas.append(br.mu * omega)
            nus.append(nu)

        # Constant part of the right-hand side: (1/2n) Xa^T Xa Theta_r + gamma Theta_r - G.
        if XtX_over_2n is not None:
            rhs0 = XtX_over_2n @ Theta_r + gamma * Theta_r - G
        else:
            rhs0 = Xa.T @ (Xa @ Theta_r) / (2.0 * n) + gamma * Theta_r - G
        q_old = _surrogate(Xa, Theta_r, Theta_r, G, gamma, branches, kappas, nus)
        rel_tol = max(opts.inner_rel_tol_min, opts.inner_rel_tol0 / (r + 1) ** 2)

        accepted = False
        inner_its = 0
        kkt_bound, tau_r = np.nan, np.nan
        if opts.kkt_tol0 is not None:
            tau_r = max(opts.kkt_tol0 * opts.kkt_decay**r, opts.kkt_tol_min) * max(
                1.0, float(np.linalg.norm(Theta_r))
            )
        cand = Theta_r
        budget = opts.max_inner if opts.inner_budget0 is None else min(opts.inner_budget0, opts.max_inner)
        chasing_kkt = False  # after a failed residual check, run ADMM to the full budget
        while True:
            while inner_its < budget:
                inner_its += 1
                # (B, c) update (eq:bc-update)
                rhs = rhs0.copy()
                acc = np.zeros_like(B_r)
                for br, u, z in zip(branches, U, Z):
                    acc += br.adjoint(u - z)
                rhs[:-1] += rho * acc
                cand = system.solve(rhs)
                Bc = cand[:-1]
                # Splitting-variable updates (eq:u-update)-(eq:r-update) and duals
                r_pri2, dual_acc = 0.0, np.zeros_like(Bc)
                ab_norm2 = u_norm2 = 0.0
                zt_acc = np.zeros_like(Bc)
                for k, (br, kap, nu) in enumerate(zip(branches, kappas, nus)):
                    AB = br.apply(Bc)
                    U_old = U[k]
                    U[k] = prox_rows(AB + Z[k], kap / rho, nu)
                    res = AB - U[k]
                    Z[k] = Z[k] + res
                    r_pri2 += np.sum(res * res)
                    dual_acc += br.adjoint(U[k] - U_old)
                    ab_norm2 += np.sum(AB * AB)
                    u_norm2 += np.sum(U[k] * U[k])
                    zt_acc += br.adjoint(Z[k])
                r_pri = np.sqrt(r_pri2)
                s_dual = rho * np.linalg.norm(dual_acc)
                eps_pri = size_pri * opts.inner_abs_tol + rel_tol * np.sqrt(max(ab_norm2, u_norm2))
                eps_dual = size_dual * opts.inner_abs_tol + rel_tol * rho * np.linalg.norm(zt_acc)
                if r_pri <= eps_pri and s_dual <= eps_dual and not chasing_kkt:
                    break
                # Residual balancing (Boyd et al., 2011, Sec. 3.4.1)
                if (
                    opts.adapt_rho
                    and r < opts.adapt_rho_outer
                    and n_rho_updates < opts.max_rho_updates
                    and inner_its % opts.rho_check_every == 0
                ):
                    factor = 0.0
                    if r_pri > 10.0 * s_dual * eps_pri / max(eps_dual, 1e-300):
                        factor = 2.0
                    elif s_dual > 10.0 * r_pri * eps_dual / max(eps_pri, 1e-300):
                        factor = 0.5
                    if factor:
                        rho *= factor
                        Z = [z / factor for z in Z]
                        system.set_rho(rho)
                        n_rho_updates += 1
            cand, q_new = _best_candidate(Xa, cand, Theta_r, G, gamma, branches, kappas, nus, U)
            decrease = q_new <= q_old + 1e-12 * (1.0 + abs(q_old))
            if decrease and opts.kkt_tol0 is not None:
                kkt_bound, _ = _kkt_error_bound(
                    Xa, G, cand, Theta_r, gamma, branches, kappas, nus, U, Z, rho, op_norms
                )
                if kkt_bound > tau_r and opts.kkt_enforce and budget < opts.max_inner:
                    decrease = False  # descent point, but not yet accurate enough: keep going
                    chasing_kkt = True
                elif kkt_bound > tau_r:
                    n_kkt_violations += 1
            if decrease:
                accepted = True
                break
            if budget >= opts.max_inner:
                break
            # Not yet an acceptable point: continue the same ADMM run with a larger budget.
            budget = min(opts.max_inner, int(np.ceil(budget * opts.inner_budget_growth)))
            rel_tol = max(rel_tol * 1e-2, 1e-12)
        n_inner_total += inner_its

        if not accepted:
            # A small failed-descent gap is a numerical stall, not a stationarity certificate.
            if q_new - q_old <= opts.tol_obj * max(1.0, abs(F)):
                status = "descent_stalled"
            else:
                status = "acceptance_failed"
                log.warning("outer %d: surrogate acceptance failed; keeping previous iterate", r)
            break

        F_new = objective(Xa, Y, H, cand, branches, eps)
        step = np.linalg.norm(cand - Theta_r)
        Theta = cand
        history.append(
            {
                "outer": r,
                "objective": F_new,
                "step": float(step),
                "inner_iterations": inner_its,
                "r_pri": float(r_pri),
                "s_dual": float(s_dual),
                "rho": float(rho),
                "kkt_bound": float(kkt_bound),
                "kkt_tol": float(tau_r),
            }
        )
        dF = F - F_new
        F = F_new
        monitor = getattr(opts, "monitor", None)
        if monitor is not None:
            monitor(Theta, history[-1])
        if abs(dF) <= opts.tol_obj * max(1.0, abs(F)) and step <= opts.tol_x * max(
            np.linalg.norm(Theta), 1e-12
        ):
            converged, status = True, "converged"
            break

    actual = [br.apply(Theta[:-1]) for br in branches]
    fixed_kkt = surrogate_kkt_residual(
        Xa, Y, H, Theta, Theta, branches, actual, Z, rho, eps, gamma)
    last = history[-1] if history else {}
    diagnostics = {
        "stationarity_measure": "fixed_objective_dual_kkt",
        "final_stationarity_B": fixed_kkt["stationarity_B"],
        "final_stationarity_c": fixed_kkt["stationarity_c"],
        "final_subgradient_distance": fixed_kkt["subgradient"],
        "final_feasibility_norm": float(np.sqrt(sum(np.sum((a-u)**2) for a, u in zip(actual, U)))),
        "final_dual_residual": last.get("s_dual", np.nan),
        "final_surrogate_error_bound": last.get("kkt_bound", np.nan),
        "iteration_limit_hit": status == "max_outer",
    }
    return SolverResult(
        Theta=Theta,
        U=U,
        Z=Z,
        rho=rho,
        objective=F,
        n_outer=len(history),
        n_inner_total=n_inner_total,
        converged=converged,
        status=status,
        history=history,
        runtime=time.perf_counter() - t_start,
        n_kkt_violations=n_kkt_violations,
        solver="mm_admm",
        diagnostics=diagnostics,
    )


def surrogate_kkt_residual(
    Xa, Y, H, Theta, Theta_r, branches, U, Z, rho, eps, gamma
) -> dict[str, float]:
    """Surrogate KKT residuals at an inner iterate (Section "Dual updates and stopping rules")."""
    n = Xa.shape[0]
    _, G, _ = lik.loss_grad(Xa, Y, Theta_r, H)
    gradM = G + Xa.T @ (Xa @ (Theta - Theta_r)) / (2.0 * n) + gamma * (Theta - Theta_r)
    B_r = Theta_r[:-1]
    stat_B = gradM[:-1].copy()
    subgrad, primal = 0.0, 0.0
    for br, u, z in zip(branches, U, Z):
        Lam = rho * z
        stat_B += br.adjoint(Lam)
        omega, nu = mm_weights(row_norms(br.apply(B_r)), br.p, br.offset(eps))
        subgrad = max(subgrad, float(np.max(subgradient_distance(Lam, u, br.mu * omega, nu), initial=0)))
        primal = max(primal, float(np.linalg.norm(br.apply(Theta[:-1]) - u)))
    return {
        "stationarity_B": float(np.linalg.norm(stat_B)),
        "stationarity_c": float(np.linalg.norm(gradM[-1])),
        "primal": primal,
        "subgradient": subgrad,
    }


__all__ = [
    "Branch",
    "SolverOptions",
    "SolverResult",
    "SolverState",
    "objective",
    "penalty_value",
    "solve_mm_admm",
    "surrogate_kkt_residual",
]
