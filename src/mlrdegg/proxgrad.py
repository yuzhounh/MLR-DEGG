"""Accelerated proximal gradient (APG) for the smooth-graph convex members.

Section "Convex model-family members", (eq:convex-proximal-gradient).  Applies when the selection
exponent is ``p_s >= 1`` and every graph branch has ``p_g >= 2`` (or there is no graph branch):
the graph penalty ``sum_e mu_e ||(D B)_e||^p`` is then continuously differentiable and joins the
likelihood in the smooth part, and the only nonsmooth term is the row-wise selection penalty, whose
proximal map is closed form (eq:soft-threshold)-(eq:radial-root).  FISTA with backtracking and a
monotone restart; for these members it is one to two orders of magnitude faster than MM-ADMM,
which remains the general solver (graph total variation and nonconvex exponents).
"""

from __future__ import annotations

import time

import numpy as np

from . import likelihood as lik
from .penalties import h_penalty, prox_rows, row_norms
from .solver import Branch, SolverOptions, SolverResult, SolverState, default_rho, objective


def apg_applicable(branches: list[Branch]) -> bool:
    sel = [br for br in branches if br.op is None]
    graph = [br for br in branches if br.op is not None]
    return len(sel) == 1 and sel[0].p >= 1.0 and all(br.p >= 2.0 for br in graph)


def solve_apg(
    Xa: np.ndarray,
    Y: np.ndarray,
    H: np.ndarray,
    branches: list[Branch],
    opts: SolverOptions,
    state: SolverState,
) -> SolverResult:
    """Minimizes ``F_eps`` by FISTA.  Stops when the relative objective change is at most
    ``opts.apg_tol`` and the relative step at most ``opts.tol_x`` in three consecutive iterations."""
    if not apg_applicable(branches):
        raise ValueError("APG requires p_s >= 1 and p_g >= 2 for every graph branch")
    t_start = time.perf_counter()
    sel = next(br for br in branches if br.op is None)
    graph = [br for br in branches if br.op is not None]
    eps = opts.epsilon

    def smooth(Theta):
        f, G, _ = lik.loss_grad(Xa, Y, Theta, H)
        B = Theta[:-1]
        for br in graph:
            DB = br.op @ B
            t = row_norms(DB)
            f += float(np.dot(br.mu, t ** br.p))
            coef = br.mu * br.p * (t ** (br.p - 2.0) if br.p != 2.0 else 1.0)
            G[:-1] += br.op.T @ (coef[:, None] * DB)
        return f, G

    def prox(Theta, a):
        out = Theta.copy()
        out[:-1] = prox_rows(Theta[:-1], a * sel.mu, sel.p)
        return out

    def sel_penalty(Theta):
        return float(np.dot(sel.mu, h_penalty(row_norms(Theta[:-1]), sel.p, sel.offset(eps))))

    L = 1.0
    X = state.Theta.copy()
    Yk = X.copy()
    tk = 1.0
    F = objective(Xa, Y, H, X, branches, eps)
    history: list[dict] = []
    status, converged, n_small = "max_outer", False, 0
    for it in range(opts.apg_max_iter):
        fY, GY = smooth(Yk)
        while True:
            Xn = prox(Yk - GY / L, 1.0 / L)
            D = Xn - Yk
            fX, _ = smooth(Xn)
            if fX <= fY + np.sum(GY * D) + 0.5 * L * np.sum(D * D) + 1e-15 * max(1.0, abs(fY)):
                break
            L *= 2.0
        Fn = fX + sel_penalty(Xn)
        if Fn > F:
            if tk == 1.0:
                # A plain proximal-gradient step from X failed to descend, which backtracking
                # rules out except through rounding: X is stationary to machine precision.
                converged, status = True, "converged"
                break
            Yk, tk = X.copy(), 1.0  # monotone restart
            continue
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        Yk = Xn + ((tk - 1.0) / tn) * (Xn - X)
        step = np.linalg.norm(Xn - X)
        small = (F - Fn <= opts.apg_tol * max(1.0, abs(Fn))
                 and step <= opts.tol_x * max(np.linalg.norm(Xn), 1e-12))
        n_small = n_small + 1 if small else 0
        X, tk, F = Xn, tn, Fn
        history.append({"outer": it, "objective": F, "step": float(step), "L": L})
        monitor = getattr(opts, "monitor", None)
        if monitor is not None:
            monitor(X, history[-1])
        L *= 0.9
        if n_small >= 3:
            converged, status = True, "converged"
            break

    B = X[:-1]
    _, final_gradient = smooth(X)
    mapping = L * (X - prox(X - final_gradient / L, 1.0 / L))
    return SolverResult(
        Theta=X,
        U=[br.apply(B).copy() for br in branches],
        Z=[np.zeros((br.mu.size, B.shape[1])) for br in branches],
        rho=state.rho or default_rho(Xa),
        objective=F,
        n_outer=len(history),
        n_inner_total=0,
        converged=converged,
        status=status,
        history=history,
        runtime=time.perf_counter() - t_start,
        solver="apg",
        diagnostics={
            "stationarity_measure": "proximal_gradient_mapping",
            "final_stationarity_B": float(np.linalg.norm(mapping[:-1])),
            "final_stationarity_c": float(np.linalg.norm(mapping[-1])),
            "final_feasibility_norm": 0.0,
            "iteration_limit_hit": status == "max_outer",
        },
    )


def solve_proxgrad(
    Xa: np.ndarray,
    Y: np.ndarray,
    H: np.ndarray,
    branches: list[Branch],
    Theta0: np.ndarray,
    eps: float = 1e-3,
    max_iter: int = 20000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Cross-check interface: returns the solution and its objective history."""
    opts = SolverOptions(epsilon=eps, apg_max_iter=max_iter, apg_tol=tol, tol_x=np.inf)
    res = solve_apg(Xa, Y, H, branches, opts, SolverState(Theta=Theta0))
    return res.Theta, [objective(Xa, Y, H, Theta0, branches, eps)] + [h["objective"] for h in res.history]
