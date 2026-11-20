"""First-level GLM for HCP task runs: one z-statistic map per condition (or per block).

The design follows the HCP task analyses (Barch et al., 2013): block regressors convolved with
a double-gamma HRF, each with its temporal derivative as a confound of no interest, and a
high-pass filter (200 s) applied identically to data and design.  FSL FILM prewhitening is
approximated by an AR(1) noise model: the AR coefficient is estimated per location from the OLS
residuals, rounded to 0.01, and the model is refitted on the whitened data of every group of
locations that share a coefficient (the ``ar1`` model of nilearn).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

OVERSAMPLE = 16


def double_gamma_hrf(dt: float, length_s: float = 32.0) -> np.ndarray:
    """Canonical double-gamma HRF (peak 6 s, undershoot 16 s, ratio 1/6), unit sum."""
    t = np.arange(0.0, length_s, dt)
    h = stats.gamma.pdf(t, 6.0) - stats.gamma.pdf(t, 16.0) / 6.0
    return h / h.sum()


def block_regressors(events: list[tuple[float, float]], n_frames: int, tr: float) -> tuple[np.ndarray, np.ndarray]:
    """HRF-convolved boxcar of ``events`` ``[(onset, duration), ...]`` and its temporal derivative
    (orthogonalized with respect to the regressor, as FSL does), sampled at the frame onsets."""
    dt = tr / OVERSAMPLE
    n_hi = n_frames * OVERSAMPLE
    box = np.zeros(n_hi)
    t_hi = np.arange(n_hi) * dt
    for onset, duration in events:
        box[(t_hi >= onset) & (t_hi < onset + duration)] = 1.0
    h = double_gamma_hrf(dt)
    reg = np.convolve(box, h)[:n_hi]
    der = np.convolve(box, np.gradient(h, dt))[:n_hi]
    reg, der = reg[::OVERSAMPLE], der[::OVERSAMPLE]
    der = der - (der @ reg) / (reg @ reg) * reg
    return reg, der


def dct_basis(n: int, tr: float, cutoff_s: float) -> np.ndarray:
    """Constant plus DCT drift regressors with periods longer than ``cutoff_s``, orthonormal."""
    k = int(np.floor(2.0 * n * tr / cutoff_s))
    t = np.arange(n)
    basis = [np.ones(n)] + [np.cos(np.pi * j * (2 * t + 1) / (2 * n)) for j in range(1, k + 1)]
    return np.linalg.qr(np.column_stack(basis))[0]


def design(conditions: dict[str, list[tuple[float, float]]], n_frames: int, tr: float,
           cutoff_s: float) -> tuple[np.ndarray, list[str], np.ndarray]:
    """High-pass-filtered design ``(n_frames, p)``: condition regressors first, then derivatives.

    Returns the design, the condition names and the orthonormal drift basis ``Q``; the data must
    be filtered the same way, ``Y - Q (Q^T Y)``."""
    names = list(conditions)
    regs, ders = zip(*(block_regressors(conditions[c], n_frames, tr) for c in names))
    X = np.column_stack(list(regs) + list(ders))
    Q = dct_basis(n_frames, tr, cutoff_s)
    X = X - Q @ (Q.T @ X)
    return X, names, Q


def _ols(X: np.ndarray, Y: np.ndarray):
    pinv = np.linalg.pinv(X)
    beta = pinv @ Y
    resid = Y - X @ beta
    return beta, resid, pinv @ pinv.T


def _ar1(resid: np.ndarray) -> np.ndarray:
    num = np.sum(resid[1:] * resid[:-1], axis=0)
    den = np.maximum(np.sum(resid**2, axis=0), 1e-12)
    return np.clip(num / den, -0.99, 0.99)


def _whiten(A: np.ndarray, rho: float) -> np.ndarray:
    W = A.copy()
    W[1:] = A[1:] - rho * A[:-1]
    W[0] = A[0] * np.sqrt(1.0 - rho**2)
    return W


def t_to_z(t: np.ndarray, df: float) -> np.ndarray:
    """t -> z with equal tail probability, via the log tail (|z| is capped near 37)."""
    logp = np.maximum(stats.t.logsf(np.abs(t), df), -700.0)
    return np.sign(t) * stats.norm.isf(np.exp(logp))


def fit_contrasts(Y: np.ndarray, X: np.ndarray, contrasts: np.ndarray, n_drift: int,
                  ar1: bool = True, rho_decimals: int = 2) -> np.ndarray:
    """z statistics ``(n_contrasts, V)`` for high-passed data ``Y (n_frames, V)``.

    ``n_drift`` drift regressors were projected out of ``X`` and ``Y``; they cost degrees of
    freedom: ``df = n - rank(X) - n_drift`` (one fewer with the AR(1) coefficient).
    """
    n, V = Y.shape
    df = n - np.linalg.matrix_rank(X) - n_drift - (1 if ar1 else 0)
    C = np.atleast_2d(contrasts)
    beta, resid, XtXi = _ols(X, Y)
    if not ar1:
        groups = {0.0: np.arange(V)}
    else:
        rho = np.round(_ar1(resid), rho_decimals)
        groups = {r: np.flatnonzero(rho == r) for r in np.unique(rho)}
    z = np.empty((C.shape[0], V))
    for r, idx in groups.items():
        if ar1:
            beta_g, resid_g, XtXi_g = _ols(_whiten(X, r), _whiten(Y[:, idx], r))
        else:
            beta_g, resid_g, XtXi_g = beta[:, idx], resid[:, idx], XtXi
        sigma2 = np.sum(resid_g**2, axis=0) / df
        eff = C @ beta_g
        var = np.einsum("ij,jk,ik->i", C, XtXi_g, C)[:, None] * sigma2[None, :]
        z[:, idx] = t_to_z(eff / np.sqrt(np.maximum(var, 1e-30)), df)
    return z
