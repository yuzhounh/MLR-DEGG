"""Common original-objective evaluator, convex dual lower bounds and offline test metrics.

Residual (plan Section 6).  With the fixed decomposition F = S + P (smooth nonconvex corrections
of every p < 1 branch in S, the convex group norms ``mu a ||.||`` and all convex powers in P),
the step ``alpha = 1 / (2 L_S)`` and ``v = Theta - alpha grad S(Theta)``,

    G_alpha(Theta) = (Theta - prox_{alpha P}(v)) / alpha,
    r_orig = ||G_alpha||_F / max(1, ||grad S(Theta_null)||_F).

The joint prox is solved independently of the candidate route.  Its error is bounded in two
ways and the smaller bound is used: the duality gap g gives ``e_G <= sqrt(2 g / alpha)``; and for
any point x_hat, strong convexity gives ``||x_hat - x*|| <= alpha dist(0, d Phi(x_hat))``, so
``e_G <= dist(0, d Phi(x_hat))``.  The second certificate fuses graph differences only through
connected clusters (never edge-by-edge truncation), refines the cluster values by Newton's
method and chooses the free dual rows by a ball-constrained least-squares problem; any feasible
choice gives a valid bound.  A residual band passes only when the estimate is below the band and
the error bound is at most 1% of it; it fails when the estimate minus the bound exceeds it;
otherwise it is ``unresolved``.  A small mapping is a first-order accuracy measure, not a local
or global optimality certificate.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.special import xlogy

from .. import likelihood as lik
from ..penalties import row_norms
from .core import JointProx, Problem, RowFn, load_instance, read_json, write_json

RESIDUAL_BANDS = (1e-3, 1e-5)
# The certificate refines cluster values with a dense Newton system; beyond this many unknowns
# (large post-fit diagnostics) only the duality-gap bound is used.
POLISH_LIMIT = 6000


def band_status(r: float, e: float, band: float) -> str:
    if r is None or e is None or not math.isfinite(e):
        return "unresolved"
    if r <= band and e <= 0.01 * band:
        return "pass"
    if r - e > band:
        return "fail"
    return "unresolved"


def _ball_project(W, mu):
    nw = row_norms(W)
    factor = np.minimum(1.0, np.divide(mu, nw, out=np.ones_like(nw), where=nw > 0))
    return W * factor[:, None]


def _incidence_pairs(op: sp.csr_matrix):
    """Endpoints (i, j) of every row of an incidence-type operator (two nonzeros per row)."""
    op = op.tocsr()
    counts = np.diff(op.indptr)
    if not np.all(counts == 2):
        raise ValueError("graph operator rows must have exactly two nonzeros")
    idx = op.indices.reshape(-1, 2)
    return idx[:, 0], idx[:, 1]


class Evaluator:
    """Common evaluator of one frozen instance (independent of every candidate route)."""

    def __init__(self, problem: Problem):
        self.p = problem
        self.alpha = 0.5 / problem.lip_S()
        _, g0 = problem.smooth_S(problem.theta_null)
        self.denom = max(1.0, float(np.linalg.norm(g0)))
        self.sel, self.graphs = problem.convex_terms()
        self.jp = JointProx(problem, self.sel, self.graphs)
        self.pairs = {br.name: _incidence_pairs(br.op) for br, _ in self.graphs}

    # ------------------------------------------------------------------------------------------
    def _r0(self, xB, vB):
        """Fixed part of an element of d Phi(x_hat) and the free (ball-constrained) rows."""
        a = self.alpha
        r0 = (xB - vB) / a
        sel = self.sel
        rx = row_norms(xB)
        zero_sel = np.zeros(len(rx), dtype=bool)
        if sel.p == 1:
            zero_sel = rx == 0
            nz = ~zero_sel
            r0[nz] += (sel.mu[nz] / rx[nz])[:, None] * xB[nz]
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                coef = np.where(rx > 0, sel.mu * sel.p * rx ** (sel.p - 2.0), 0.0)
            r0 += coef[:, None] * xB
        free = []
        for br, fn in self.graphs:
            U = br.op @ xB
            r = row_norms(U)
            if fn.p == 1:
                fr = r == 0
                nz = ~fr
                coef = np.zeros_like(r)
                coef[nz] = fn.mu[nz] / r[nz]
                if np.any(fr):
                    free.append((br, np.flatnonzero(fr), fn.mu[fr]))
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    coef = np.where(r > 0, fn.mu * fn.p * r ** (fn.p - 2.0), 0.0)
            r0 += br.op.T @ (coef[:, None] * U)
        return r0, zero_sel, free

    def _ball_lsq(self, r0, zero_sel, free, y_init, deadline, target, max_iter=3000):
        """``min sqrt(sum_Z dist(w_j, B(mu_j))^2 + sum_notZ ||w_j||^2)``, ``w = r0 + D_F' y``."""
        mu_sel = self.sel.mu

        def residual(ys):
            w = r0.copy()
            for (br, rows, _), y in zip(free, ys):
                w += br.op[rows].T @ y
            res = w.copy()
            if np.any(zero_sel):
                res[zero_sel] = w[zero_sel] - _ball_project(w[zero_sel], mu_sel[zero_sel])
            return res

        if not free:
            return float(np.linalg.norm(residual([]))), []
        subs = [br.op[rows] for br, rows, _ in free]
        ys = [_ball_project(y0, mu) for (br, rows, mu), y0 in zip(free, y_init)]
        res = residual(ys)
        best = (float(np.linalg.norm(res)), [y.copy() for y in ys])
        lip = 2.0 * max(self.p.graph_norm2([br for br, _, _ in free]), 1e-300)
        step = 1.0 / lip
        z = [y.copy() for y in ys]
        t = 1.0
        for it in range(max_iter):
            resz = residual(z)
            yn = [_ball_project(zz - step * 2.0 * (S @ resz), mu)
                  for zz, S, (_, _, mu) in zip(z, subs, free)]
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            z = [a + (t - 1.0) / t_new * (a - b) for a, b in zip(yn, ys)]
            ys, t = yn, t_new
            if it % 20 == 0 or it == max_iter - 1:
                val = float(np.linalg.norm(residual(ys)))
                if val < best[0]:
                    best = (val, [y.copy() for y in ys])
                if val <= target or time.perf_counter() > deadline:
                    break
        return best

    def _polish(self, xB0, vB, labels, zero_cluster, deadline, max_newton=30):
        """Newton refinement of the cluster values (fused rows equal, zero clusters fixed)."""
        a, k1 = self.alpha, self.p.k1
        d = xB0.shape[0]
        clusters = np.unique(labels)
        keep = clusters[~zero_cluster[clusters]]
        newid = -np.ones(labels.max() + 1, dtype=int)
        newid[keep] = np.arange(keep.size)
        col = newid[labels]
        rows = np.flatnonzero(col >= 0)
        kc = keep.size
        if kc == 0:
            return np.zeros_like(xB0)
        if kc * k1 > POLISH_LIMIT:
            return None
        P = sp.csr_matrix((np.ones(rows.size), (rows, col[rows])), shape=(d, kc))
        sizes = np.asarray(P.sum(axis=0)).ravel()
        Pv = P.T @ vB
        z = (P.T @ xB0) / sizes[:, None]
        terms = []
        if self.sel.p == 1 or self.sel.p > 1:
            terms.append((P[rows], self.sel.mu[rows], self.sel.p))
        for br, fn in self.graphs:
            Dr = (br.op @ P).tocsr()
            Dr.eliminate_zeros()
            nz = np.flatnonzero(np.diff(Dr.indptr) > 0)
            if nz.size:
                terms.append((Dr[nz], fn.mu[nz], fn.p))

        def value(z):
            val = (float(np.sum(sizes[:, None] * z * z)) - 2.0 * float(np.sum(z * Pv))) / (2.0 * a)
            for A, mu, p in terms:
                r = row_norms(A @ z)
                val += float(np.dot(mu, r**p))
            return val

        def grad_hess(z):
            g = (sizes[:, None] * z - Pv) / a
            H4 = np.zeros((kc, kc, k1, k1))
            H4[np.arange(kc), np.arange(kc)] += (sizes / a)[:, None, None] * np.eye(k1)
            for A, mu, p in terms:
                U = A @ z
                r = np.maximum(row_norms(U), 1e-300)
                d1 = mu * p * r ** (p - 1.0)
                coef = d1 / r
                g += A.T @ (coef[:, None] * U)
                uh = U / r[:, None]
                d2 = mu * p * (p - 1.0) * r ** (p - 2.0)
                M = coef[:, None, None] * np.eye(k1) + (d2 - coef)[:, None, None] * \
                    np.einsum("ia,ib->iab", uh, uh)
                A = A.tocsr()
                cnt = np.diff(A.indptr)
                one = np.flatnonzero(cnt == 1)
                if one.size:
                    c = A.indices[A.indptr[one]]
                    w = A.data[A.indptr[one]] ** 2
                    np.add.at(H4, (c, c), w[:, None, None] * M[one])
                two = np.flatnonzero(cnt == 2)
                if two.size:
                    s0 = A.indptr[two]
                    c1, c2 = A.indices[s0], A.indices[s0 + 1]
                    a1, a2 = A.data[s0], A.data[s0 + 1]
                    Mt = M[two]
                    np.add.at(H4, (c1, c1), (a1 * a1)[:, None, None] * Mt)
                    np.add.at(H4, (c2, c2), (a2 * a2)[:, None, None] * Mt)
                    np.add.at(H4, (c1, c2), (a1 * a2)[:, None, None] * Mt)
                    np.add.at(H4, (c2, c1), (a1 * a2)[:, None, None] * Mt)
            Hm = H4.transpose(0, 2, 1, 3).reshape(kc * k1, kc * k1)
            return g, Hm

        f = value(z)
        for _ in range(max_newton):
            if time.perf_counter() > deadline:
                break
            g, Hm = grad_hess(z)
            gn = float(np.linalg.norm(g))
            if gn <= 1e-15 * max(1.0, float(np.linalg.norm(z))) / a:
                break
            try:
                step = -la.cho_solve(la.cho_factor(Hm, lower=True), g.ravel()).reshape(z.shape)
            except la.LinAlgError:
                step = -np.linalg.lstsq(Hm, g.ravel(), rcond=None)[0].reshape(z.shape)
            slope = float(np.sum(g * step))
            if slope >= 0:
                break
            s, improved = 1.0, False
            for _ in range(50):
                zn = z + s * step
                fn_ = value(zn)
                if fn_ <= f + 1e-4 * s * slope:
                    improved = True
                    break
                s *= 0.5
            if not improved:
                break
            z, f = zn, fn_
        return np.asarray(P @ z)

    def certify(self, v, x, y, deadline, target):
        """Smallest subgradient-distance bound over a few fusion thresholds."""
        vB, xB = v[:-1], x[:-1]
        d = xB.shape[0]
        rx = row_norms(xB)
        scale_sel = max(float(rx.max(initial=0.0)), 1e-300)
        graph_r = {br.name: row_norms(br.op @ xB) for br, _ in self.graphs}
        best = (math.inf, None)
        for thr in (1e-9, 1e-6, 1e-3):
            if time.perf_counter() > deadline:
                break
            zero_rows = rx <= (thr * scale_sel if self.sel.p == 1 else 0.0)
            if self.sel.p != 1:
                zero_rows = np.zeros(d, dtype=bool)
            ii, jj = [], []
            for br, fn in self.graphs:
                if fn.p != 1:
                    continue
                r = graph_r[br.name]
                fused = r <= thr * max(float(r.max(initial=0.0)), 1e-300)
                i, j = self.pairs[br.name]
                ii.append(i[fused])
                jj.append(j[fused])
            if ii and sum(len(a) for a in ii):
                i_all, j_all = np.concatenate(ii), np.concatenate(jj)
                adj = sp.csr_matrix((np.ones(i_all.size), (i_all, j_all)), shape=(d, d))
                _, labels = connected_components(adj, directed=False)
            else:
                labels = np.arange(d)
            zero_cluster = np.zeros(labels.max() + 1, dtype=bool)
            zero_cluster[labels[zero_rows]] = True
            xhB = self._polish(xB, vB, labels, zero_cluster, deadline)
            if xhB is None:
                continue
            r0, zero_sel, free = self._r0(xhB, vB)
            y_init = []
            for br, rows, _ in free:
                yy = None
                for (gbr, _), yg in zip(self.graphs, y):
                    if gbr.name == br.name:
                        yy = yg[rows]
                y_init.append(yy if yy is not None else np.zeros((rows.size, self.p.k1)))
            zeta, _ = self._ball_lsq(r0, zero_sel, free, y_init, deadline, 0.1 * target)
            if zeta < best[0]:
                xh = x.copy()
                xh[:-1] = xhB
                xh[-1] = v[-1]
                best = (zeta, xh)
        return best

    # ------------------------------------------------------------------------------------------
    def residual(self, theta: np.ndarray, budget: float, bands=RESIDUAL_BANDS):
        t0 = time.perf_counter()
        deadline = t0 + budget
        a, denom = self.alpha, self.denom
        _, gS = self.p.smooth_S(theta)
        v = theta - a * gS
        info = {"alpha": a, "denominator": denom}
        y = []
        if not self.graphs:
            x = self.jp.x_of(v, a, [])
            e = 1e-15 * max(1.0, float(np.linalg.norm(v))) / a
            kind, gap, iters = "closed_form_rounding", 0.0, 0
        else:
            gap_target = a * (0.01 * min(bands) * denom) ** 2 / 2.0
            chunk, iters, y = 200, 0, None
            best = (math.inf, None, None)
            while True:
                x_it, y, gap, it = self.jp.solve(v, a, gap_target, max_iter=chunk, deadline=deadline,
                                                 y0=y)
                iters += it
                e_gap = math.sqrt(2.0 * max(gap, 0.0) / a)
                if e_gap < best[0]:
                    best = (e_gap, x_it, "duality_gap")
                if self._resolved(theta, best, bands):
                    break
                if time.perf_counter() > deadline:
                    break
                remaining = deadline - time.perf_counter()
                zeta, xh = self.certify(v, x_it, y, time.perf_counter() + 0.5 * remaining,
                                        0.01 * min(bands) * denom)
                if xh is not None and zeta < best[0]:
                    best = (zeta, xh, "subgradient_certificate")
                if self._resolved(theta, best, bands) or time.perf_counter() > deadline:
                    break
                chunk *= 2
            e, x, kind = best
        r = float(np.linalg.norm(theta - x)) / a / denom
        en = e / denom
        info.update({"r_orig_estimate": r, "mapping_error_bound": en, "bound_type": kind,
                     "prox_gap": gap, "prox_iterations": iters,
                     "evaluation_seconds": time.perf_counter() - t0})
        for b in bands:
            info[f"residual_{b:.0e}"] = band_status(r, en, b)
        info["residual_status"] = ("resolved" if all(info[f"residual_{b:.0e}"] != "unresolved"
                                                     for b in bands) else "residual_unresolved")
        return info, y

    def _resolved(self, theta, best, bands):
        e, x, _ = best
        if x is None or not math.isfinite(e):
            return False
        r = float(np.linalg.norm(theta - x)) / self.alpha / self.denom
        return all(band_status(r, e / self.denom, b) != "unresolved" for b in bands)


# ----------------------------------------------------------------------------------------------
# Convex dual lower bound (Fenchel dual of the original objective)
# ----------------------------------------------------------------------------------------------


def dual_lower_bound(problem: Problem, theta: np.ndarray, graph_duals=None) -> dict:
    """Valid lower bound on min F for convex instances from a feasible dual point.

    Dual: max -(1/n) sum pi log pi - sum_a (mu phi)^*(y_a) subject to
    Xa^T (Pi - Y) H / n + sum_a [A_a^T y_a; 0] = 0, Pi row-stochastic.  Pi comes from the softmax
    at ``theta`` adjusted to the observed class totals (intercept condition); the selection dual
    absorbs the coefficient condition exactly and a common scaling keeps p = 1 duals feasible.
    """
    if problem.nonconvex_branches():
        return {"dual_status": "not_applicable"}
    Xa, Y, H, n = problem.Xa, problem.Y, problem.H, problem.n
    Pi = lik.softmax(lik.scores(Xa, theta, H))
    Pi = np.maximum(Pi, 1e-300)
    target = Y.sum(axis=0)
    for _ in range(10000):
        Pi *= (target / Pi.sum(axis=0))[None, :]
        Pi /= Pi.sum(axis=1, keepdims=True)
        if np.max(np.abs(Pi.sum(axis=0) - target)) <= 1e-13 * n:
            break
    sel, graphs = problem.convex_terms()
    B = theta[:-1]
    ys = []
    for gi, (br, fn) in enumerate(graphs):
        if graph_duals is not None and gi < len(graph_duals) and graph_duals[gi] is not None:
            y = np.asarray(graph_duals[gi], dtype=float).copy()
        else:
            U = br.op @ B
            r = row_norms(U)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.where(r > 0, fn.mu * fn.p * r ** (fn.p - 2.0), 0.0)[:, None] * U
        if fn.p == 1:
            y = _ball_project(y, fn.mu)
        ys.append(y)
    R = Xa[:, :-1].T @ ((Pi - Y) @ H) / n
    for (br, _), y in zip(graphs, ys):
        R = R + br.op.T @ y
    factor = 1.0
    if sel.p == 1:
        nr = row_norms(R)
        ratio = np.divide(sel.mu, nr, out=np.full_like(nr, np.inf), where=nr > 0)
        factor = min(1.0, float(ratio.min(initial=np.inf))) * (1.0 - 1e-12)
        if factor < 1.0:
            Pi = Y + factor * (Pi - Y)
            ys = [factor * y for y in ys]
            R = factor * R
    y_sel = -R
    conj = float(np.sum(sel.conj_rows(row_norms(y_sel))))
    conj += float(sum(np.sum(fn.conj_rows(row_norms(y))) for (_, fn), y in zip(graphs, ys)))
    entropy = -float(np.sum(xlogy(Pi, Pi))) / n
    lower = entropy - conj
    intercept_residual = float(np.linalg.norm(np.ones(n) @ ((Pi - Y) @ H))) / n
    valid = bool(np.isfinite(lower) and intercept_residual <= 1e-11 and Pi.min() >= 0)
    return {"dual_status": "numerically_feasible" if valid else "dual_unresolved",
            "dual_lower_bound": lower if valid else None, "dual_scaling": factor,
            "dual_intercept_residual": intercept_residual}


def test_metrics(problem: Problem, theta: np.ndarray, which: str = "te") -> dict:
    Xa = problem.inst["Xte"] if which == "te" else problem.inst["Xva"]
    y = problem.inst["yte"] if which == "te" else problem.inst["yva"]
    P = problem.probabilities(Xa, theta)
    p_true = np.clip(P[np.arange(len(y)), y], 1e-300, None)
    return {"test_log_loss": float(-np.mean(np.log(p_true))),
            "test_accuracy": float(np.mean(np.argmax(P, axis=1) == y))}


# ----------------------------------------------------------------------------------------------
# Evaluation of one stored run
# ----------------------------------------------------------------------------------------------


def load_records(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir)
    if (run_dir / "trace.json").exists():
        return read_json(run_dir / "trace.json")
    out = []
    if (run_dir / "trace.jsonl").exists():
        import json
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def evaluate_run(run_dir: Path, instance_path: Path, budget_final: float, budget_point: float,
                 wall_deadline: float | None = None) -> dict:
    """Original objective, certified residual and test metrics at every stored checkpoint.

    ``wall_deadline`` (``time.time()``) bounds the whole evaluation stage; later states are then
    reported ``residual_unresolved`` with an ``evaluation budget`` note, never filled with zeros."""
    run_dir = Path(run_dir)
    problem = Problem(load_instance(instance_path))
    ev = Evaluator(problem)
    records = load_records(run_dir)
    order = sorted(range(len(records)),
                   key=lambda i: (records[i].get("kind") != "final", records[i].get("solver_seconds", 0)))
    rows, final_duals = [None] * len(records), None
    for i in order:
        rec = records[i]
        theta = np.load(run_dir / rec["file"])
        row = {"file": rec["file"], "kind": rec.get("kind"), "solver_seconds": rec.get("solver_seconds"),
               "iteration": rec.get("iteration"), "native_objective": rec.get("native_objective"),
               "observation": rec.get("observation"), "last_five": bool(rec.get("last_five", False)),
               "stage": rec.get("stage_name"), "operations": rec.get("operations", {})}
        finite = bool(np.all(np.isfinite(theta)))
        row["finite"] = finite
        if not finite:
            row["residual_status"] = "residual_unresolved"
            rows[i] = row
            continue
        row["objective"] = problem.objective(theta)
        row.update(test_metrics(problem, theta))
        out_of_time = wall_deadline is not None and time.time() > wall_deadline
        if out_of_time:
            row.update({"residual_status": "residual_unresolved", "evaluation_note": "evaluation budget"})
        else:
            budget = budget_final if rec.get("kind") == "final" else budget_point
            info, y = ev.residual(theta, budget)
            row.update(info)
            if rec.get("kind") == "final":
                final_duals = y
        if rec.get("kind") == "final" and not problem.nonconvex_branches():
            row.update(dual_lower_bound(problem, theta, final_duals))
        rows[i] = row
    out = {"rows": rows, "alpha": ev.alpha, "denominator": ev.denom}
    write_json(run_dir / "eval.json", out)
    return out
