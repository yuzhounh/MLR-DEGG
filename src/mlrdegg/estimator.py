"""Scikit-learn compatible MLR-DEGG classifier."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from . import likelihood as lik
from .contrast import contrast_matrix
from .graphs import Graph, spatiotemporal_operators
from .penalties import h_penalty, row_norms
from .proxgrad import apg_applicable, solve_apg
from .residual import common_residual
from .solver import Branch, SolverOptions, SolverResult, SolverState, objective, solve_mm_admm


class MLRDEGGClassifier(ClassifierMixin, BaseEstimator):
    """Spatiotemporal double-exponent group-regularized multinomial logistic regression.

    Minimizes (eq:main-model-w)::

        L(W, b) + lambda_sel R_sel(W) + lambda_g [(1 - eta) R_sp(W) + eta R_tm(W)]

    subject to ``W 1 = 0`` and ``1^T b = 0``.

    Parameters
    ----------
    n_space, n_time : int
        ``V`` and ``T``; each input row is ``vec(Z)`` of a ``V x T`` trial (column-major, see
        :func:`mlrdegg.graphs.vectorize`).
    spatial_graph, temporal_graph : Graph or None
        Graphs on the ``V`` locations and ``T`` time points.
    p_s, p_g : float
        Selection and graph exponents, any finite positive value.
    lambda_scale : {"ref", "max", "none"}
        How the dimensionless ``lambda_sel`` and ``lambda_g`` map to model strengths
        (Section "Calibration of the regularization strengths").  ``lambda_max = d * max_j
        ||grad_B L(0, c0)_j||`` is the smallest selection strength at which the group-lasso member
        (``p_s = 1``, ``lambda_g = 0``) is identically zero.

        * ``"ref"`` (default): every branch ``a`` with exponent ``p_a`` uses
          ``lambda_a * lambda_max * t_a * h_{p_a,eps_a}(t) / h_{p_a,eps_a}(t_a)`` with offset
          ``eps_a = epsilon_rel * t_a``, where the reference radius ``t_a`` is the median nonzero
          row radius (selection) or edge-difference radius (graph branches) of the group-lasso
          pilot solution at the same ``lambda_sel``.  Branches with ``p = 1`` are exactly
          ``lambda_a * lambda_max * t`` and do not depend on the pilot; every other exponent
          agrees with them at the typical radius ``t_a``.
        * ``"max"`` (legacy, v0.4 experiments): ``lambda * lambda_max / s(p)`` with
          ``s(p) = p * epsilon^(p-1)`` and the absolute ``epsilon``.  For ``p > 1`` this inflates
          the strength by ``epsilon^(1-p) / p`` (500x for ``p = 2``, 3.3e5x for ``p = 3``).
        * ``"none"``: ``lambda_sel`` and ``lambda_g`` are used as model strengths directly.
    epsilon_rel : float
        Offset of the nonconvex penalties relative to the reference radius (``"ref"`` only);
        smaller values make ``p < 1`` penalties more strongly nonconvex.
    warm_start_convex : bool
        For ``p_s < 1`` or ``p_g < 1``, first solve the convex member with exponents clipped to
        ``max(p, 1)`` and start the nonconvex problem from it (Section "Prediction and discussion").
    solver : {"auto", "apg", "mm_admm"}
        ``"auto"`` uses accelerated proximal gradient when ``p_s >= 1`` and ``p_g >= 2`` (or there
        is no graph branch) and MM-ADMM otherwise.
    inner_budget0 : int or None
        Initial ADMM iterations per MM surrogate, enlarged only when the candidate fails the
        acceptance test; ``None`` solves every surrogate to ``max_inner``/tolerance.
    cg_tol : float
        Relative tolerance of the conjugate-gradient solves in the ``(B, c)`` update.
    apg_tol, apg_max_iter : float, int
        APG stopping tolerance on the relative objective change and iteration cap.
    kkt_tol0, kkt_decay, kkt_enforce : float or None, float, bool
        Surrogate-error monitoring associated with supplement Proposition
        ``prop:inexact`` and Remark ``rem:inexact-scope``.  The geometric target has a
        numerical floor and is relative to ``max(1, ||Theta||)``.  ``kkt_enforce=True``
        requests additional inner iterations within the finite budget; steps accepted
        at the cap can still violate the target.  Violations are counted in
        ``n_kkt_violations_``; ``kkt_tol0=None`` disables monitoring.  This finite-budget
        diagnostic is not a certificate of the proposition's vanishing-error assumption.
    residual_diagnostic, residual_budget : bool, float
        After the fit, record the common original-objective residual (:mod:`mlrdegg.residual`)
        within ``residual_budget`` seconds in ``common_residual_`` and ``diagnostics_``.  It is a
        report only: stopping rules and fitted coefficients do not depend on it.
    """

    def __init__(
        self,
        n_space: int,
        n_time: int = 1,
        spatial_graph: Graph | None = None,
        temporal_graph: Graph | None = None,
        lambda_sel: float = 1e-2,
        lambda_g: float = 1e-2,
        eta: float = 0.5,
        p_s: float = 1.0,
        p_g: float = 2.0,
        epsilon: float = 1e-3,
        gamma: float = 1e-3,
        rho: float | str = "auto",
        max_outer: int = 200,
        max_inner: int = 500,
        tol_obj: float = 1e-7,
        tol_x: float = 1e-4,
        lambda_scale: str = "ref",
        epsilon_rel: float = 0.1,
        warm_start_convex: bool = True,
        solver: str = "auto",
        inner_budget0: int | None = 5,
        cg_tol: float = 1e-6,
        apg_tol: float = 1e-9,
        apg_max_iter: int = 20000,
        kkt_tol0: float | None = 1e-2,
        kkt_decay: float = 0.95,
        kkt_enforce: bool = False,
        residual_diagnostic: bool = True,
        residual_budget: float = 2.0,
    ):
        self.n_space = n_space
        self.n_time = n_time
        self.spatial_graph = spatial_graph
        self.temporal_graph = temporal_graph
        self.lambda_sel = lambda_sel
        self.lambda_g = lambda_g
        self.eta = eta
        self.p_s = p_s
        self.p_g = p_g
        self.epsilon = epsilon
        self.gamma = gamma
        self.rho = rho
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol_obj = tol_obj
        self.tol_x = tol_x
        self.lambda_scale = lambda_scale
        self.epsilon_rel = epsilon_rel
        self.warm_start_convex = warm_start_convex
        self.solver = solver
        self.inner_budget0 = inner_budget0
        self.cg_tol = cg_tol
        self.apg_tol = apg_tol
        self.apg_max_iter = apg_max_iter
        self.kkt_tol0 = kkt_tol0
        self.kkt_decay = kkt_decay
        self.kkt_enforce = kkt_enforce
        self.residual_diagnostic = residual_diagnostic
        self.residual_budget = residual_budget

    # ------------------------------------------------------------------------------------------
    def build_branches(self, p_s: float | None = None, p_g: float | None = None) -> list[Branch]:
        """Active branches with normalized row weights (eq:active-weights)."""
        p_s = self.p_s if p_s is None else p_s
        p_g = self.p_g if p_g is None else p_g
        lam_sel, lam_g = self._effective_lambdas(p_s, p_g)
        if not lam_sel > 0:
            raise ValueError("lambda_sel must be positive (Proposition prop:existence)")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must lie in [0, 1]")
        d = self.n_space * self.n_time
        f, e = self._ref_factor("sel", p_s)
        branches = [Branch("sel", None, np.full(d, f * lam_sel / d), p_s, e)]
        if lam_g > 0:
            sp_op, tm_op = spatiotemporal_operators(
                self.spatial_graph, self.temporal_graph, self.n_space, self.n_time
            )
            for name, op, xi in (("sp", sp_op, 1.0 - self.eta), ("tm", tm_op, self.eta)):
                if op is None or xi <= 0:
                    continue
                D, w = op
                c = w.sum()
                if c <= 0:
                    continue
                f, e = self._ref_factor(name, p_g)
                branches.append(Branch(name, D, f * lam_g * xi * w / c, p_g, e))
        return branches

    def _ref_factor(self, name: str, p: float) -> tuple[float, float | None]:
        """Reference-radius factor ``t / h_{p,eps}(t)`` and offset ``eps = epsilon_rel * t`` of a
        branch under ``lambda_scale="ref"``; ``(1, None)`` otherwise and for ``p = 1``."""
        if self.lambda_scale != "ref" or p == 1.0:
            return 1.0, None
        t = self.t_ref_[name]
        eps = self.epsilon_rel * t
        return float(t / h_penalty(np.array([t]), p, eps)[0]), eps

    def _effective_lambdas(self, p_s: float | None = None, p_g: float | None = None) -> tuple[float, float]:
        """Model-scale strengths before the per-branch reference factor (``"ref"``)."""
        p_s = self.p_s if p_s is None else p_s
        p_g = self.p_g if p_g is None else p_g
        if self.lambda_scale == "none":
            return self.lambda_sel, self.lambda_g
        scale = getattr(self, "lambda_max_", None)
        if scale is None:
            raise RuntimeError("lambda_max_ is set during fit")
        if self.lambda_scale == "ref":
            return self.lambda_sel * scale, self.lambda_g * scale
        slope = lambda p: p * self.epsilon ** (p - 1.0)  # noqa: E731
        return self.lambda_sel * scale / slope(p_s), self.lambda_g * scale / slope(p_g)

    def _reference_radii(self, Xa, Y, H, Theta_null, G0) -> dict[str, float]:
        """Median nonzero radii of the group-lasso pilot at the same ``lambda_sel`` (APG)."""
        d = self.n_space * self.n_time
        pilot = [Branch("sel", None, np.full(d, self.lambda_sel * self.lambda_max_ / d), 1.0)]
        opts = SolverOptions(apg_tol=1e-7, apg_max_iter=self.apg_max_iter, tol_x=self.tol_x)
        B = solve_apg(Xa, Y, H, pilot, opts, SolverState(Theta=Theta_null)).Theta[:-1]
        # Fallback when the pilot is identically zero: one gradient step from the null model.
        B_step = -G0[:-1] * (2.0 * Xa.shape[0]) / np.linalg.norm(Xa, 2) ** 2
        ops = {"sel": None}
        sp_op, tm_op = spatiotemporal_operators(
            self.spatial_graph, self.temporal_graph, self.n_space, self.n_time
        )
        ops.update({k: v[0] for k, v in (("sp", sp_op), ("tm", tm_op)) if v is not None})
        radii = {}
        self.reference_radius_source_ = {}
        for name, op in ops.items():
            r = row_norms(B if op is None else op @ B)
            r = r[r > 1e-12 * max(1.0, r.max(initial=0.0))]
            if r.size == 0:
                r = row_norms(B_step if op is None else op @ B_step)
                r = r[r > 0]
                source = "gradient_step"
            else:
                source = "pilot"
            # No informative radius exists if both pilot and null-gradient branch are zero.
            # Use a declared positive coefficient-unit convention, never median(empty)/NaN.
            radii[name] = float(np.median(r)) if r.size else 1.0
            self.reference_radius_source_[name] = source if r.size else "unit_scale"
        return radii

    def _options(self, max_outer=None) -> SolverOptions:
        return SolverOptions(
            epsilon=self.epsilon,
            gamma=self.gamma,
            rho=self.rho,
            max_outer=self.max_outer if max_outer is None else max_outer,
            max_inner=self.max_inner,
            tol_obj=self.tol_obj,
            tol_x=self.tol_x,
            inner_budget0=self.inner_budget0,
            cg_tol=self.cg_tol,
            apg_tol=self.apg_tol,
            apg_max_iter=self.apg_max_iter,
            kkt_tol0=self.kkt_tol0,
            kkt_decay=self.kkt_decay,
            kkt_enforce=self.kkt_enforce,
        )

    def _solve(self, Xa, Y, H, branches, opts, state):
        """APG for smooth-graph convex members (``solver="auto"`` or ``"apg"``), else MM-ADMM."""
        if self.solver not in ("auto", "apg", "mm_admm"):
            raise ValueError("solver must be 'auto', 'apg' or 'mm_admm'")
        use_apg = self.solver == "apg" or (self.solver == "auto" and apg_applicable(branches))
        return (solve_apg if use_apg else solve_mm_admm)(Xa, Y, H, branches, opts, state)

    def _null_model(self, Xa, Y, H, branches, state) -> SolverResult | None:
        """Shortcut for MM-ADMM fits whose solution is ``B = 0``.  If the group-lasso part alone
        (selection branch, ``p_s = 1``) is minimized at ``B = 0`` with intercept ``c*``, then
        ``(0, c*)`` minimizes the full objective as well: every graph branch is nonnegative and zero
        at ``B = 0``, so the full objective is at least the group-lasso one and equal to it there.
        The group-lasso part is solved by APG (a few iterations at ``B = 0``); MM-ADMM would need
        hundreds of outer iterations to reach the same point.  ``None`` if the shortcut does not
        apply (APG path anyway, ``p_s != 1``, or a nonzero group-lasso solution)."""
        sel = branches[0]
        use_apg = self.solver == "apg" or (self.solver == "auto" and apg_applicable(branches))
        if use_apg or len(branches) == 1 or sel.p != 1.0:
            return None
        _, G, _ = lik.loss_grad(Xa, Y, state.Theta, H)
        if np.any(row_norms(G[:-1]) > 1.01 * sel.mu):  # zero rows fail the soft-threshold test
            return None
        pilot = solve_apg(Xa, Y, H, [sel], self._options(), state)
        if not pilot.converged or np.any(pilot.Theta[:-1] != 0.0):
            return None
        B = pilot.Theta[:-1]
        return SolverResult(
            Theta=pilot.Theta, U=[br.apply(B).copy() for br in branches],
            Z=[np.zeros((br.mu.size, B.shape[1])) for br in branches], rho=pilot.rho,
            objective=objective(Xa, Y, H, pilot.Theta, branches, self.epsilon),
            n_outer=pilot.n_outer, n_inner_total=0, converged=True, status="converged_null_model",
            history=pilot.history, runtime=pilot.runtime,
            solver="null_model", diagnostics=pilot.diagnostics,
        )

    def fit(self, X, y, init: dict | None = None):
        X = np.asarray(X, dtype=float)
        d = self.n_space * self.n_time
        if X.shape[1] != d:
            raise ValueError(f"X has {X.shape[1]} features, expected n_space*n_time = {d}")
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        yi = self.label_encoder_.transform(y)
        K = len(self.classes_)
        Y = np.eye(K)[yi]
        H = contrast_matrix(K)
        Xa = lik.augment(X)

        # Null model: zero coefficients, class-frequency intercept.
        prior = np.bincount(yi, minlength=K) / len(yi)
        logp = np.log(np.maximum(prior, 1e-12))
        Th_null = np.zeros((d + 1, K - 1))
        Th_null[-1] = H.T @ (logp - logp.mean())
        Theta0 = np.vstack([init["B"], init["c"][None, :]]) if init is not None else Th_null.copy()

        self.t_ref_, self.reference_radius_source_ = {}, {}
        if self.lambda_scale in ("ref", "max"):
            _, G0, _ = lik.loss_grad(Xa, Y, Th_null, H)
            self.lambda_max_ = float(d * np.max(row_norms(G0[:-1])))
            if self.lambda_max_ <= 0:
                raise ValueError("Null-model feature gradient is zero; relative regularization scale is undefined. "
                                 "Use lambda_scale='none' with positive absolute selection strength.")
            needs_ref = self.p_s != 1.0 or (self.lambda_g > 0 and self.p_g != 1.0)
            if self.lambda_scale == "ref":
                self.t_ref_ = self._reference_radii(Xa, Y, H, Th_null, G0) if needs_ref else {}
        elif self.lambda_scale != "none":
            raise ValueError("lambda_scale must be 'ref', 'max' or 'none'")
        self.lambda_sel_eff_, self.lambda_g_eff_ = self._effective_lambdas()

        self.fit_log_ = []
        state = SolverState(Theta=Theta0)
        nonconvex = self.p_s < 1 or self.p_g < 1
        if nonconvex and self.warm_start_convex and init is None:
            res0 = self._solve(
                Xa, Y, H, self.build_branches(max(self.p_s, 1.0), max(self.p_g, 1.0)),
                self._options(), state,
            )
            self.fit_log_.append(("convex_warm_start", res0))
            state = SolverState(Theta=res0.Theta, rho=res0.rho)

        branches = self.build_branches()
        res = None
        if init is None and not nonconvex:
            res = self._null_model(Xa, Y, H, branches, state)
        if res is None:
            res = self._solve(Xa, Y, H, branches, self._options(), state)
        self.fit_log_.append(("main", res))

        self.H_ = H
        self.B_ = res.Theta[:-1]
        self.c_ = res.Theta[-1]
        self.coef_ = self.B_ @ H.T  # W = B H^T, (d, K)
        self.intercept_ = H @ self.c_  # b = H c
        self.split_ = {br.name: u for br, u in zip(branches, res.U)}
        self.branches_ = branches
        self.objective_ = res.objective
        self.converged_ = res.converged
        self.status_ = res.status
        self.n_outer_ = res.n_outer
        self.n_inner_ = res.n_inner_total
        self.n_kkt_violations_ = sum(r.n_kkt_violations for _, r in self.fit_log_)
        self.history_ = res.history
        self.runtime_ = sum(r.runtime for _, r in self.fit_log_)
        self.solver_ = res.solver
        self.diagnostics_ = res.diagnostics
        self.common_residual_ = {}
        if self.residual_diagnostic:  # report only; the fit above is final
            self.common_residual_ = common_residual(Xa, Y, H, branches, res.Theta, Th_null,
                                                    self.epsilon, self.residual_budget)
            self.diagnostics_ = {**res.diagnostics, **self.common_residual_}
        return self

    # ------------------------------------------------------------------------------------------
    def decision_function(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        return lik.softmax(self.decision_function(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    # ------------------------------------------------------------------------------------------
    def structure_summary(self, tol: float = 0.0) -> dict:
        """Prediction-coefficient zeros, auxiliary zeros and split feasibility.

        Each branch_zero_fraction uses B (or DB for graph branches).
        Auxiliary thresholding is separate; tol=0 means exact zeros.
        """
        check_is_fitted(self, "coef_")
        if tol < 0:
            raise ValueError("tol must be nonnegative")
        out = {}
        for branch in self.branches_:
            name = branch.name
            actual = branch.apply(self.B_)
            auxiliary = self.split_[name]
            zero_B, zero_U = row_norms(actual) <= tol, row_norms(auxiliary) <= tol
            out[f"{name}_zero_fraction"] = float(np.mean(zero_B))
            out[f"{name}_aux_zero_fraction"] = float(np.mean(zero_U))
            out[f"{name}_support_disagreement_fraction"] = float(np.mean(zero_B != zero_U))
            out[f"{name}_feasibility_norm"] = float(np.linalg.norm(actual - auxiliary))
        out["row_norm_B"] = row_norms(self.B_)
        return out

    def feature_importance(self) -> np.ndarray:
        """Row norms ``||W_j,:||`` reshaped to ``(V, T)``."""
        from .graphs import unvectorize

        return unvectorize(row_norms(self.coef_), self.n_space, self.n_time)
