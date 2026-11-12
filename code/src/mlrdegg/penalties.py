"""Radial penalties, region-wise MM weights and row-wise proximal maps.

Every function acts row-wise on a matrix whose rows are groups (a feature's class effects, or
one graph difference across classes).
"""

from __future__ import annotations

import numpy as np


def row_norms(A: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", A, A))


def h_penalty(t: np.ndarray, p: float, eps: float) -> np.ndarray:
    """Fixed-offset radial penalty ``h_{p,eps}`` (eq:shifted-penalty)."""
    t = np.asarray(t, dtype=float)
    if p < 1:
        return (t + eps) ** p - eps**p
    return t**p


def mm_weights(t: np.ndarray, p: float, eps: float) -> tuple[np.ndarray, float]:
    """Region-wise MM weights ``omega`` and exponent ``nu``.

    (eq:selection-mm-weight)/(eq:graph-mm-weight): tangent slope for ``p < 1``; the convex
    power is retained exactly (``omega = 1``, ``nu = p``) for ``p >= 1``.
    """
    t = np.asarray(t, dtype=float)
    if p < 1:
        return p * (t + eps) ** (p - 1.0), 1.0
    return np.ones_like(t), float(p)


def mm_constant(t: np.ndarray, p: float, eps: float, omega: np.ndarray) -> np.ndarray:
    """Per-group tangency constants ``h(t0) - omega * t0`` (zero when ``p >= 1``)."""
    if p < 1:
        return h_penalty(t, p, eps) - omega * t
    return np.zeros_like(np.asarray(t, dtype=float))


def radial_root(s: np.ndarray, tau: np.ndarray, nu: float, n_iter: int = 60) -> np.ndarray:
    """Unique root ``t in [0, s]`` of ``t + tau * nu * t^(nu-1) = s`` for ``nu > 1`` (eq:radial-root).

    Vectorized safeguarded Newton iteration in ``x = log t`` with a bisection bracket.  One of
    the two terms is at least ``s/2`` at the root, which gives the lower bracket end, so roots
    spanning many orders of magnitude (``1 < nu < 2`` with small ``s``) are resolved accurately.
    """
    s = np.asarray(s, dtype=float)
    tau = np.broadcast_to(np.asarray(tau, dtype=float), s.shape)
    out = np.zeros_like(s)
    m = (s > 0) & (tau > 0)
    out[(s > 0) & (tau <= 0)] = s[(s > 0) & (tau <= 0)]
    if not np.any(m):
        return out
    sm, a = s[m], tau[m] * nu
    tiny = 1e-300
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        pw = 1.0 / (nu - 1.0)
        lo_t = np.minimum(0.5 * sm, (0.5 * sm / a) ** pw)
        hi_t = np.minimum(sm, (sm / a) ** pw)
    lo = np.log(np.maximum(lo_t, tiny))
    hi = np.log(np.maximum(hi_t, tiny))
    x = 0.5 * (lo + hi)
    for _ in range(n_iter):
        t = np.exp(x)
        tp = a * t ** (nu - 1.0)
        g = t + tp - sm
        if np.all(np.abs(g) <= 1e-14 * sm):
            break
        lo = np.where(g < 0, x, lo)
        hi = np.where(g > 0, x, hi)
        # Newton on log(t + tp) - log(s), which is nearly linear in x.
        x_new = x - (np.log1p(g / sm)) * (t + tp) / (t + (nu - 1.0) * tp)
        bad = ~np.isfinite(x_new) | (x_new < lo) | (x_new > hi)
        x = np.where(bad, 0.5 * (lo + hi), x_new)
    out[m] = np.minimum(np.exp(x), sm)
    return out


def prox_rows(V: np.ndarray, tau: np.ndarray, nu: float) -> np.ndarray:
    """Row-wise ``argmin_u 0.5||u - v||^2 + tau ||u||^nu`` for ``nu >= 1``.

    Implements (eq:soft-threshold), (eq:quadratic-shrinkage) and (eq:radial-root).
    """
    tau = np.asarray(tau, dtype=float)
    s = row_norms(V)
    if nu == 1.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(s > 0, np.maximum(1.0 - tau / s, 0.0), 0.0)
        return V * scale[:, None]
    if nu == 2.0:
        return V / (1.0 + 2.0 * tau)[:, None] if tau.ndim else V / (1.0 + 2.0 * tau)
    t = radial_root(s, tau, nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(s > 0, t / s, 0.0)
    return V * scale[:, None]


def subgradient_distance(Lam: np.ndarray, U: np.ndarray, kappa: np.ndarray, nu: float) -> np.ndarray:
    """Row-wise ``dist(lambda, d phi(u))`` for ``phi(u) = kappa ||u||^nu`` (eq:subgradient-distance)."""
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), (U.shape[0],))
    t = row_norms(U)
    nz = t > 0
    out = np.empty(U.shape[0])
    lam_norm = row_norms(Lam)
    if nu == 1.0:
        out[~nz] = np.maximum(lam_norm[~nz] - kappa[~nz], 0.0)
        g = kappa[nz, None] * U[nz] / t[nz, None]
    else:
        out[~nz] = lam_norm[~nz]
        g = kappa[nz, None] * nu * (t[nz] ** (nu - 2.0))[:, None] * U[nz]
    out[nz] = row_norms(Lam[nz] - g)
    return out
