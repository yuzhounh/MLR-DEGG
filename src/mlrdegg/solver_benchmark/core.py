"""Frozen problem, convex row functions, joint proximal maps and the per-run algorithm clock.

Only NumPy/SciPy and the light solver modules are imported here, so every timed worker process
starts quickly.  The single evaluation target is the original fixed-offset objective

    F_eps(Theta) = L_softmax(Theta) + sum_a sum_j mu_aj h_{p_a,eps_a}(||(A_a B)_j||),

with the intercept unpenalized (plan Section 2).  ``Problem.objective`` delegates to the
production ``solver.objective`` so that every route is judged by the same implementation.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, svds
from scipy.special import binom

from .. import likelihood as lik
from ..penalties import prox_rows, row_norms
from ..solver import Branch
from ..solver import objective as _original_objective


class BudgetReached(Exception):
    """The configured algorithm-time cap was reached (hard limit, not numerical stopping)."""


class ConditionNotMet(Exception):
    """A route's applicability condition fails at the given state (no fictitious run time)."""


class Runtime:
    """Operation counters and the algorithm clock of the run executing in this process."""

    def __init__(self):
        self.counts: dict[str, int] = {}
        self.clock = None

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(n)
        if self.clock is not None:
            self.clock.tick()

    def reset(self) -> None:
        self.counts = {}
        self.clock = None


RUNTIME = Runtime()


def count(key: str, n: int = 1) -> None:
    RUNTIME.count(key, n)


# ----------------------------------------------------------------------------------------------
# JSON / array helpers
# ----------------------------------------------------------------------------------------------


def plain(x):
    """JSON-safe copy: arrays to lists, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(plain(obj), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(pending, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    pending = path.with_name(path.stem + ".pending.npz")
    np.savez_compressed(pending, **arrays)
    os.replace(pending, path)


def scale_rows(V: np.ndarray, t: np.ndarray, s: np.ndarray | None = None) -> np.ndarray:
    """Rows of ``V`` rescaled to radii ``t`` (``s`` = current row norms)."""
    s = row_norms(V) if s is None else s
    factor = np.divide(t, s, out=np.zeros_like(s), where=s > 0)
    return V * factor[:, None]


# ----------------------------------------------------------------------------------------------
# Radial functions
# ----------------------------------------------------------------------------------------------


def h_shift(t, p: float, eps: float) -> np.ndarray:
    """``h_{p,eps}(t)``; the p < 1 branch is evaluated without cancellation near t = 0."""
    t = np.asarray(t, dtype=float)
    if p < 1:
        return eps**p * np.expm1(p * np.log1p(t / eps))
    return t**p


def dh_shift(t, p: float, eps: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if p < 1:
        return p * (t + eps) ** (p - 1.0)
    if p == 1:
        return np.ones_like(t)
    return p * t ** (p - 1.0)


_RAW_LOSS = lik.loss  # captured before the M01-M03 counting hooks can wrap it


def raw_objective(problem: "Problem", T: np.ndarray) -> float:
    """Uncounted original objective for monitoring (never part of a route's work)."""
    B = T[:-1]
    val = _RAW_LOSS(problem.Xa, problem.Y, T, problem.H)
    for br in problem.branches:
        U = B if br.op is None else br.op @ B
        val += float(np.dot(br.mu, h_shift(row_norms(U), br.p, br.offset(1e-3))))
    return float(val)


def redistribution_slope(p: float, eps: float) -> float:
    """``a = p eps^(p-1)``: slope of h_{p,eps} at zero (plan Section 6)."""
    return p * eps ** (p - 1.0)


def curvature_bound(p: float, eps: float) -> float:
    """``kappa = p (1-p) eps^(p-2)``: bound on the negative curvature of h_{p,eps}, p < 1."""
    return p * (1.0 - p) * eps ** (p - 2.0)


def increasing_convex_root(g, dg, lo, hi, start=None, n_iter: int = 100) -> np.ndarray:
    """Root of an increasing convex function on ``[lo, hi]`` with ``g(lo) <= 0 < g(hi)``.

    Newton's method started to the right of the root (``start``, default ``hi``) stays there
    and converges monotonically; a bisection step guards against a vanishing derivative."""
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    t = hi.copy() if start is None else np.clip(np.asarray(start, dtype=float), lo, hi)
    for _ in range(n_iter):
        gt = g(t)
        lo = np.where(gt <= 0, t, lo)
        hi = np.where(gt > 0, t, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            tn = t - gt / dg(t)
        bad = ~np.isfinite(tn) | (tn < lo) | (tn > hi)
        tn = np.where(bad, 0.5 * (lo + hi), tn)
        done = (np.abs(tn - t) <= 1e-15 * np.maximum(np.abs(t), 1e-300)) | (gt == 0)
        t = tn
        if np.all(done):
            break
    return t


def nonconvex_prox(V: np.ndarray, tau, p: float, eps: float) -> np.ndarray:
    """Global row-wise minimizer of ``0.5||u - v||^2 + tau h_{p,eps}(||u||)`` for 0 < p < 1.

    The radial objective q has q''' > 0, so q' is convex and at most one interior local
    minimizer exists: the larger root of q'(t) = t - s + tau p (t + eps)^(p-1), beyond the
    inflection point where q'' = 0.  It is compared with the boundary candidate t = 0 (ties go to
    zero); the original offset is never replaced.
    """
    s = row_norms(V)
    tau = np.broadcast_to(np.asarray(tau, dtype=float), s.shape)
    count("row_prox_calls")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tc = np.maximum((tau * p * (1.0 - p)) ** (1.0 / (2.0 - p)) - eps, 0.0)
    tc = np.minimum(tc, s)
    g_tc = tc - s + tau * p * (tc + eps) ** (p - 1.0)
    active = (s > 0) & (g_tc < 0)
    t = np.zeros_like(s)
    if np.any(active):
        sa, ta = s[active], tau[active]
        # One fixed-point step from s stays right of the root: q'(t1) >= 0 when t1 >= tc.
        t1 = sa - ta * p * (sa + eps) ** (p - 1.0)
        root = increasing_convex_root(
            lambda x: x - sa + ta * p * (x + eps) ** (p - 1.0),
            lambda x: 1.0 - ta * p * (1.0 - p) * (x + eps) ** (p - 2.0),
            tc[active], sa, start=np.where(t1 >= tc[active], t1, sa))
        q_t = 0.5 * (root - sa) ** 2 + ta * h_shift(root, p, eps)
        t[active] = np.where(q_t < 0.5 * sa * sa, root, 0.0)
    return scale_rows(V, t, s)


def compensated_root(s, tau, p: float, eps: float, kappa: float) -> np.ndarray:
    """Radius of the prox of ``tau (h_{p,eps}(r) + kappa r^2 / 2)`` (convex, nondecreasing)."""
    s = np.asarray(s, dtype=float)
    tau = np.broadcast_to(np.asarray(tau, dtype=float), s.shape)
    active = s > tau * redistribution_slope(p, eps)
    t = np.zeros_like(s)
    if np.any(active):
        sa, ta = s[active], tau[active]
        # The tangent at 0 has slope exactly 1 (kappa cancels h''(0)); for a convex increasing
        # function its root s - tau a lies right of the true root.
        t[active] = increasing_convex_root(
            lambda x: x + ta * (p * (x + eps) ** (p - 1.0) + kappa * x) - sa,
            lambda x: 1.0 + ta * (p * (p - 1.0) * (x + eps) ** (p - 2.0) + kappa),
            np.zeros_like(sa), sa, start=sa - ta * redistribution_slope(p, eps))
    return t


def bregman_power(r: np.ndarray, t: np.ndarray, p: float) -> np.ndarray:
    """``r^p - t^p - p t^(p-1) (r - t)`` for p > 1, accurate when r is close to t."""
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)
    out = r**p
    pos = t > 0
    if np.any(pos):
        delta = r[pos] / t[pos] - 1.0
        g = (1.0 + delta) ** p - 1.0 - p * delta
        small = np.abs(delta) < 1e-2
        if np.any(small):
            ds = delta[small]
            series = np.zeros_like(ds)
            for k in range(6, 1, -1):
                series = (series + binom(p, k)) * ds
            g[small] = series * ds
        out[pos] = t[pos] ** p * np.maximum(g, 0.0)
    return out


def _angle_term(U, W, r, s):
    """``r s (1 - cos angle(U, W))`` row-wise, from the normalized difference (no cancellation)."""
    both = (r > 0) & (s > 0)
    out = np.zeros_like(r)
    if np.any(both):
        diff = U[both] / r[both, None] - W[both] / s[both, None]
        out[both] = 0.5 * r[both] * s[both] * np.einsum("ij,ij->i", diff, diff)
    return out


@dataclass
class RowFn:
    """Convex row term ``sum_j mu_j phi(||u_j||)``.

    ``kind="power"``: phi(r) = r^p, p >= 1.  ``kind="compensated"``: phi(r) = h_{p,eps}(r) +
    kappa r^2/2 with p < 1 and kappa = p(1-p)eps^(p-2), convex and nondecreasing (route M23).
    """

    mu: np.ndarray
    p: float
    kind: str = "power"
    eps: float = 0.0

    @property
    def kappa(self) -> float:
        return curvature_bound(self.p, self.eps) if self.kind == "compensated" else 0.0

    def phi(self, r):
        if self.kind == "power":
            return r if self.p == 1 else r**self.p
        return h_shift(r, self.p, self.eps) + 0.5 * self.kappa * r * r

    def dphi(self, r):
        if self.kind == "power":
            return dh_shift(r, self.p, 0.0)
        return dh_shift(r, self.p, self.eps) + self.kappa * r

    def value(self, U: np.ndarray) -> float:
        return float(np.dot(self.mu, self.phi(row_norms(U))))

    def prox(self, V: np.ndarray, t) -> np.ndarray:
        """Row-wise ``argmin_u 0.5||u - v||^2 + t mu phi(||u||)`` (``t`` scalar or per row)."""
        tau = np.asarray(t, dtype=float) * self.mu
        count("row_prox_calls")
        if self.kind == "power":
            return prox_rows(V, tau, self.p)
        s = row_norms(V)
        return scale_rows(V, compensated_root(s, tau, self.p, self.eps, self.kappa), s)

    def dual_prox(self, W: np.ndarray, s) -> np.ndarray:
        """``prox_{s (mu phi)^*}(W)`` by the Moreau identity; ball projection when p = 1."""
        if self.kind == "power" and self.p == 1:
            count("row_prox_calls")
            nw = row_norms(W)
            factor = np.minimum(1.0, np.divide(self.mu, nw, out=np.ones_like(nw), where=nw > 0))
            return W * factor[:, None]
        s = np.asarray(s, dtype=float)
        sc = s[:, None] if s.ndim else s
        return W - sc * self.prox(W / sc, 1.0 / s)

    def conj_rows(self, snorm: np.ndarray) -> np.ndarray:
        """Row-wise conjugate ``(mu phi)^*`` as a function of the dual row norm."""
        snorm = np.asarray(snorm, dtype=float)
        if self.kind == "power":
            if self.p == 1:
                bad = snorm > self.mu * (1.0 + 1e-12) + 1e-300
                return np.where(bad, np.inf, 0.0)
            q = self.p / (self.p - 1.0)
            with np.errstate(divide="ignore"):
                return self.mu * (self.p - 1.0) * (snorm / (self.mu * self.p)) ** q
        kap, p, eps = self.kappa, self.p, self.eps
        mu = np.broadcast_to(self.mu, snorm.shape)
        active = snorm > mu * redistribution_slope(p, eps)
        t = np.zeros_like(snorm)
        if np.any(active):
            sa, ma = snorm[active], mu[active]
            t[active] = increasing_convex_root(
                lambda x: ma * (p * (x + eps) ** (p - 1.0) + kap * x) - sa,
                lambda x: ma * (p * (p - 1.0) * (x + eps) ** (p - 2.0) + kap),
                np.zeros_like(sa), sa / (ma * kap))
        return np.maximum(snorm * t - self.mu * self.phi(t), 0.0)

    def fy(self, U: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Row-wise Fenchel-Young gaps ``mu phi(||u||) + (mu phi)^*(w) - <w, u>`` (>= 0).

        Written as sums of nonnegative terms so that small gaps keep relative precision."""
        r, s = row_norms(U), row_norms(W)
        ang = _angle_term(U, W, r, s)
        if self.kind == "power" and self.p == 1:
            if np.any(s > self.mu * (1.0 + 1e-9) + 1e-300):
                return np.full_like(r, np.inf)
            return r * np.maximum(self.mu - s, 0.0) + ang
        if self.kind == "power":
            tstar = (s / (self.mu * self.p)) ** (1.0 / (self.p - 1.0))
            return self.mu * bregman_power(r, tstar, self.p) + ang
        dot = np.einsum("ij,ij->i", U, W)
        return np.maximum(self.mu * self.phi(r) + self.conj_rows(s) - dot, 0.0)


# ----------------------------------------------------------------------------------------------
# Frozen problem
# ----------------------------------------------------------------------------------------------

# Dense eigenvalue computations up to this size (all benchmark instances); Lanczos beyond it.
DENSE_EIG_LIMIT = 2000


def _lambda_max(M: sp.spmatrix) -> float:
    """Largest eigenvalue of a sparse symmetric positive semidefinite matrix."""
    if M.shape[0] <= DENSE_EIG_LIMIT:
        return float(la.eigvalsh(M.toarray())[-1])
    # Small relative safety margin: an upper estimate keeps the derived step sizes valid.
    return float(eigsh(M.tocsc(), k=1, which="LA", tol=1e-8, return_eigenvectors=False)[0]) * (1 + 1e-6)


class Problem:
    """One frozen instance: training data, contrast matrix, branches and stored initial values."""

    def __init__(self, inst: dict):
        self.inst = inst
        self.Xa, self.Y, self.H = inst["Xa"], inst["Y"], inst["H"]
        self.n, self.d1 = self.Xa.shape
        self.d = self.d1 - 1
        self.k1 = self.H.shape[1]
        self.branches: list[Branch] = inst["branches"]
        self.theta0 = inst["theta0"]
        self.theta_null = inst["theta_null"]
        self.meta = inst.get("meta", {})
        self._cache: dict = {}

    # --- shapes and operators ---------------------------------------------------------------
    @property
    def shape(self):
        return (self.d1, self.k1)

    @property
    def graph_branches(self) -> list[Branch]:
        return [br for br in self.branches if br.op is not None]

    @property
    def selection(self) -> Branch:
        return self.branches[0]

    def A(self, br: Branch, B: np.ndarray) -> np.ndarray:
        if br.op is None:
            return B
        count("graph_mult")
        return br.op @ B

    def At(self, br: Branch, V: np.ndarray) -> np.ndarray:
        if br.op is None:
            return V
        count("graph_transpose_mult")
        return br.op.T @ V

    def lift(self, GB: np.ndarray) -> np.ndarray:
        """Coefficient-row array to a full Theta-shaped array with a zero intercept row."""
        out = np.zeros(self.shape)
        out[:-1] = GB
        return out

    @property
    def lip_loss(self) -> float:
        """Global Lipschitz bound ``lambda_max(Xa^T Xa) / (2n)`` of the softmax loss gradient."""
        if "lip_loss" not in self._cache:
            if min(self.n, self.d1) <= DENSE_EIG_LIMIT:
                G = self.Xa @ self.Xa.T
                lam = float(la.eigvalsh(G)[-1])
            else:  # large problems (e.g. the post-fit diagnostic on HCP): Lanczos estimate
                lam = float(svds(self.Xa, k=1, return_singular_vectors=False)[0]) ** 2 * (1 + 1e-6)
            self._cache["lip_loss"] = lam / (2.0 * self.n)
        return self._cache["lip_loss"]

    def op_norm2(self, br: Branch) -> float:
        if br.op is None:
            return 1.0
        key = ("op_norm2", br.name)
        if key not in self._cache:
            self._cache[key] = _lambda_max(br.op.T @ br.op)
        return self._cache[key]

    def graph_norm2(self, branches: list[Branch] | None = None) -> float:
        """``lambda_max(sum_g D_g^T D_g)`` over the given (default: all) graph branches."""
        branches = self.graph_branches if branches is None else branches
        key = ("graph_norm2",) + tuple(br.name for br in branches)
        if key not in self._cache:
            if not branches:
                self._cache[key] = 0.0
            else:
                M = sum((br.op.T @ br.op for br in branches), sp.csr_matrix((self.d, self.d)))
                self._cache[key] = _lambda_max(M)
        return self._cache[key]

    def gram(self, branches: list[Branch] | None = None) -> np.ndarray:
        """Dense ``sum_a A_a^T A_a`` on the coefficient rows."""
        branches = self.branches if branches is None else branches
        M = np.zeros((self.d, self.d))
        for br in branches:
            if br.op is None:
                M[np.diag_indices(self.d)] += 1.0
            else:
                M += (br.op.T @ br.op).toarray()
        return M

    # --- loss ------------------------------------------------------------------------------
    def loss(self, T: np.ndarray) -> float:
        count("loss_calls")
        return lik.loss(self.Xa, self.Y, T, self.H)

    def loss_grad(self, T: np.ndarray) -> tuple[float, np.ndarray]:
        count("loss_gradient_calls")
        f, G, _ = lik.loss_grad(self.Xa, self.Y, T, self.H)
        return f, G

    def hvp(self, T: np.ndarray, D: np.ndarray) -> np.ndarray:
        count("hessian_vector_calls")
        return lik.hvp(self.Xa, T, self.H, D)

    def hessian(self, T: np.ndarray) -> np.ndarray:
        """Dense softmax-loss Hessian in C-ordered ``vec(Theta)`` coordinates."""
        count("hessian_builds")
        P = lik.softmax(lik.scores(self.Xa, T, self.H))
        mom = np.einsum("nk,ka,kb->nab", P, self.H, self.H)
        mean = P @ self.H
        mom -= np.einsum("na,nb->nab", mean, mean)
        Hm = np.einsum("ni,nj,nab->iajb", self.Xa, self.Xa, mom, optimize=True) / self.n
        Hm = Hm.reshape(T.size, T.size)
        return 0.5 * (Hm + Hm.T)

    # --- original objective -------------------------------------------------------------------
    def objective(self, T: np.ndarray) -> float:
        """The shared original objective (production implementation)."""
        count("objective_calls")
        return float(_original_objective(self.Xa, self.Y, self.H, T, self.branches, 1e-3))

    def penalty_value(self, T: np.ndarray) -> float:
        B = T[:-1]
        return float(sum(np.dot(br.mu, h_shift(row_norms(self.A(br, B)), br.p, br.offset(1e-3)))
                         for br in self.branches))

    # --- fixed S + P decomposition (plan Section 6) -----------------------------------------
    def nonconvex_branches(self) -> list[Branch]:
        return [br for br in self.branches if br.p < 1]

    def weak_bound(self) -> float:
        """``sum_{a: p_a<1} kappa_a max_j mu_aj ||A_a||^2`` (conservative)."""
        return float(sum(curvature_bound(br.p, br.eps) * float(br.mu.max()) * self.op_norm2(br)
                         for br in self.nonconvex_branches()))

    def lip_S(self) -> float:
        return self.lip_loss + self.weak_bound()

    def psi_value_grad(self, T: np.ndarray) -> tuple[float, np.ndarray]:
        """Smooth nonconvex corrections ``sum mu psi(A B)`` and their gradient (B rows)."""
        B = T[:-1]
        f, G = 0.0, np.zeros_like(B)
        for br in self.nonconvex_branches():
            U = self.A(br, B)
            r = row_norms(U)
            a = redistribution_slope(br.p, br.eps)
            f += float(np.dot(br.mu, h_shift(r, br.p, br.eps) - a * r))
            # h'(r) - a = a * expm1((p-1) log1p(r/eps)), computed without cancellation.
            slope_gap = a * np.expm1((br.p - 1.0) * np.log1p(r / br.eps))
            coef = np.divide(br.mu * slope_gap, r, out=np.zeros_like(r), where=r > 0)
            G += self.At(br, coef[:, None] * U)
        return f, G

    def smooth_S(self, T: np.ndarray) -> tuple[float, np.ndarray]:
        f, G = self.loss_grad(T)
        if self.nonconvex_branches():
            fp, GB = self.psi_value_grad(T)
            f += fp
            G[:-1] += GB
        return f, G

    def convex_fn(self, br: Branch) -> RowFn:
        """Convex part of a branch in the fixed decomposition: ``mu a ||.||`` for p < 1, else
        the original convex power."""
        if br.p < 1:
            return RowFn(br.mu * redistribution_slope(br.p, br.eps), 1.0)
        return RowFn(br.mu, float(br.p))

    def convex_terms(self) -> tuple[RowFn, list[tuple[Branch, RowFn]]]:
        sel = self.convex_fn(self.selection)
        return sel, [(br, self.convex_fn(br)) for br in self.graph_branches]

    def P_value(self, T: np.ndarray) -> float:
        sel, graphs = self.convex_terms()
        B = T[:-1]
        return sel.value(B) + sum(fn.value(self.A(br, B)) for br, fn in graphs)

    # --- prediction ---------------------------------------------------------------------------
    def probabilities(self, Xa: np.ndarray, T: np.ndarray) -> np.ndarray:
        return lik.softmax(lik.scores(Xa, T, self.H))


# ----------------------------------------------------------------------------------------------
# Joint proximal map of convex selection + graph terms
# ----------------------------------------------------------------------------------------------


class JointProx:
    """``argmin_x 1/(2a)||x - v||^2 + sel(x_B) + sum_g G_g(D_g x_B)``; the intercept row is free.

    Accelerated proximal gradient on the graph duals (FISTA with gradient restart).  The
    selection map is kept exact, so ``x(y)`` has exact zero rows.  Because ``x(y)`` minimizes the
    Lagrangian exactly, the primal-dual gap equals the sum of graph Fenchel-Young gaps.
    """

    def __init__(self, problem: Problem, sel: RowFn | None, graphs: list[tuple[Branch, RowFn]]):
        self.problem, self.sel, self.graphs = problem, sel, graphs
        self.norm2 = problem.graph_norm2([br for br, _ in graphs]) if graphs else 0.0
        self.y: list[np.ndarray] | None = None

    def x_of(self, v: np.ndarray, a: float, y) -> np.ndarray:
        x = v.copy()
        if self.graphs:
            x[:-1] -= a * sum(self.problem.At(br, yg) for (br, _), yg in zip(self.graphs, y))
        if self.sel is not None:
            x[:-1] = self.sel.prox(x[:-1], a)
        return x

    def gap(self, x: np.ndarray, y) -> float:
        B = x[:-1]
        return float(sum(np.sum(fn.fy(self.problem.A(br, B), yg))
                         for (br, fn), yg in zip(self.graphs, y)))

    def solve(self, v: np.ndarray, a: float, tol: float, max_iter: int = 200000,
              deadline: float | None = None, check_every: int = 10, y0=None):
        """Returns ``(x, y, gap, iterations)`` for the best gap found."""
        if not self.graphs:
            return self.x_of(v, a, []), [], 0.0, 0
        p = self.problem
        y = y0 if y0 is not None else self.y
        if y is None:
            y = [np.zeros((br.op.shape[0], p.k1)) for br, _ in self.graphs]
        y = [fn.dual_prox(yg, 1.0) if fn.p == 1 and fn.kind == "power" else yg.copy()
             for (_, fn), yg in zip(self.graphs, y)]
        s = 1.0 / (a * max(self.norm2, 1e-300))
        z = [yg.copy() for yg in y]
        t = 1.0
        best_gap, best_x, best_y = math.inf, None, None
        it = 0
        for it in range(1, max_iter + 1):
            count("inner_iterations")
            xz = self.x_of(v, a, z)
            Bz = xz[:-1]
            y_new = [fn.dual_prox(zg + s * p.A(br, Bz), s)
                     for (br, fn), zg in zip(self.graphs, z)]
            restart = sum(float(np.sum((zg - yn) * (yn - yo)))
                          for zg, yn, yo in zip(z, y_new, y)) > 0
            if restart:
                t = 1.0
                z = [yn.copy() for yn in y_new]
            else:
                t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                beta = (t - 1.0) / t_new
                z = [yn + beta * (yn - yo) for yn, yo in zip(y_new, y)]
                t = t_new
            y = y_new
            if it % check_every == 0 or it == max_iter:
                x = self.x_of(v, a, y)
                g = self.gap(x, y)
                if g < best_gap:
                    best_gap, best_x, best_y = g, x, [yg.copy() for yg in y]
                if g <= tol:
                    break
                if deadline is not None and time.perf_counter() > deadline:
                    break
        if best_x is None:
            best_x = self.x_of(v, a, y)
            best_gap, best_y = self.gap(best_x, y), y
        self.y = best_y
        return best_x, best_y, best_gap, it


# ----------------------------------------------------------------------------------------------
# Per-run algorithm clock with geometric checkpoints
# ----------------------------------------------------------------------------------------------


def checkpoint_times(cap: float, first: float = 1e-3, per_decade: int = 4) -> list[float]:
    """0 and ``first * 10^(k/per_decade)`` up to the cap (plan Section 6: geometric checkpoints)."""
    times, k = [0.0], 0
    while first * 10 ** (k / per_decade) <= cap * (1 + 1e-12):
        times.append(first * 10 ** (k / per_decade))
        k += 1
    return times


class Clock:
    """Algorithm time = wall time minus monitoring overhead; raises ``BudgetReached`` at the cap.

    Iterates are observed after every accepted outer step.  Geometric checkpoints are written
    immediately (so a hard kill keeps them); the last five observations and the final state are
    written by ``finish``.
    """

    def __init__(self, cap: float, directory: Path, times: list[float] | None = None,
                 objective=None, monitor_fraction: float = 0.3):
        self.cap = float(cap)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.times = checkpoint_times(cap) if times is None else times
        self.next_k = 0
        self.seq = 0
        self.overhead = 0.0
        self.t0 = None
        self.current: tuple[np.ndarray, dict] | None = None
        self.ring: list[tuple[np.ndarray, dict]] = []
        self.records: list[dict] = []
        self.n_observed = 0
        self.trace_file = self.directory / "trace.jsonl"
        if self.trace_file.exists():
            self.trace_file.unlink()
        # Original-objective monitoring between observations (increase count and size).  It is
        # monitoring overhead, skipped while that overhead exceeds ``monitor_fraction`` of the
        # algorithm time; the number of skipped observations is reported.
        self.objective = objective
        self.monitor_fraction = monitor_fraction
        self.monitor = {"monitored": 0, "skipped": 0, "increases": 0, "max_increase": 0.0,
                        "max_relative_increase": 0.0}
        self._last_F = None

    def start(self) -> None:
        self.t0 = time.perf_counter()
        RUNTIME.clock = self

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0 - self.overhead

    def tick(self) -> None:
        if self.t0 is not None and self.elapsed() >= self.cap:
            raise BudgetReached()

    def _record(self, theta, info, kind):
        name = f"cp_{self.seq:04d}.npy"
        np.save(self.directory / name, theta)
        rec = {**info, "kind": kind, "file": name}
        self.records.append(rec)
        with open(self.trace_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(plain(rec)) + "\n")
        self.seq += 1

    def observe(self, theta: np.ndarray, info: dict | None = None, check: bool = True) -> None:
        t = self.elapsed()
        t1 = time.perf_counter()
        self.n_observed += 1
        theta = np.array(theta, dtype=float, copy=True)
        rec = {"solver_seconds": t, "observation": self.n_observed, **(info or {}),
               "operations": dict(RUNTIME.counts)}
        if self.objective is not None:
            if self.overhead <= self.monitor_fraction * max(t, 1e-3) or self._last_F is None:
                F = float(self.objective(theta))
                self.monitor["monitored"] += 1
                if self._last_F is not None and F > self._last_F + 1e-13 * max(1.0, abs(self._last_F)):
                    inc = F - self._last_F
                    self.monitor["increases"] += 1
                    self.monitor["max_increase"] = max(self.monitor["max_increase"], inc)
                    self.monitor["max_relative_increase"] = max(
                        self.monitor["max_relative_increase"], inc / max(1.0, abs(self._last_F)))
                self._last_F = F
                rec["monitored_objective"] = F
            else:
                self.monitor["skipped"] += 1
        self.current = (theta, rec)
        self.ring.append((theta, rec))
        if len(self.ring) > 5:
            self.ring.pop(0)
        if self.next_k < len(self.times) and t >= self.times[self.next_k]:
            self._record(theta, rec, "geometric")
            while self.next_k < len(self.times) and self.times[self.next_k] <= t:
                self.next_k += 1
        self.overhead += time.perf_counter() - t1
        if check:
            self.tick()

    def finish(self, theta: np.ndarray, info: dict) -> float:
        """Write the last five observations and the final state; returns algorithm seconds."""
        RUNTIME.clock = None
        elapsed = self.elapsed()
        written = {r["observation"] for r in self.records}
        for th, rec in self.ring:
            if rec["observation"] not in written:
                self._record(th, rec, "last_five")
            else:
                for r in self.records:
                    if r["observation"] == rec["observation"]:
                        r["last_five"] = True
        final = {"solver_seconds": elapsed, "observation": self.n_observed + 1, **info,
                 "operations": dict(RUNTIME.counts)}
        self._record(np.asarray(theta, dtype=float), final, "final")
        write_json(self.directory / "trace.json", self.records)
        return elapsed


# ----------------------------------------------------------------------------------------------
# Instance files
# ----------------------------------------------------------------------------------------------


def branch_arrays(branches: list[Branch]) -> dict:
    out = {"n_branches": np.array(len(branches))}
    for j, br in enumerate(branches):
        out[f"br{j}_name"] = np.array(br.name)
        out[f"br{j}_p"] = np.array(float(br.p))
        out[f"br{j}_eps"] = np.array(np.nan if br.eps is None else float(br.eps))
        out[f"br{j}_mu"] = np.asarray(br.mu, dtype=float)
        out[f"br{j}_has_op"] = np.array(br.op is not None)
        if br.op is not None:
            op = br.op.tocsr()
            out[f"br{j}_data"] = op.data
            out[f"br{j}_indices"] = op.indices
            out[f"br{j}_indptr"] = op.indptr
            out[f"br{j}_shape"] = np.array(op.shape)
    return out


def load_instance(path: Path) -> dict:
    """Frozen instance: arrays from ``<id>.npz`` and metadata from ``<id>.json``."""
    path = Path(path)
    z = np.load(path)
    branches = []
    for j in range(int(z["n_branches"])):
        op = None
        if bool(z[f"br{j}_has_op"]):
            op = sp.csr_matrix((z[f"br{j}_data"], z[f"br{j}_indices"], z[f"br{j}_indptr"]),
                               shape=tuple(z[f"br{j}_shape"]))
        eps = float(z[f"br{j}_eps"])
        branches.append(Branch(str(z[f"br{j}_name"]), op, z[f"br{j}_mu"].copy(),
                               float(z[f"br{j}_p"]), None if math.isnan(eps) else eps))
    inst = {k: z[k] for k in ("Xa", "Y", "H", "Xva", "yva", "Xte", "yte", "theta0", "theta_null")}
    inst["branches"] = branches
    meta_path = path.with_suffix(".json")
    inst["meta"] = read_json(meta_path) if meta_path.exists() else {}
    return inst
