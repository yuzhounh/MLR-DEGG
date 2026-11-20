"""Model factory: MLR-DEGG family members and baselines, each preceded by standardization."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..estimator import MLRDEGGClassifier
from ..graphs import Graph, chain_graph, gaussian_graph, mesh_graph, unvectorize


@dataclass
class Context:
    """Problem geometry shared by all fits of one dataset."""

    n_space: int
    n_time: int
    spatial_graph: Graph | None
    temporal_graph: Graph | None


def build_graphs(graph_cfg: dict, space: dict, n_time: int) -> tuple[Graph | None, Graph | None]:
    """Spatial graph from ``space`` (``coords`` for "gaussian", ``edges``/``edge_dist`` for "mesh")."""
    spatial = temporal = None
    s = graph_cfg.get("spatial")
    if s:
        if s["type"] == "gaussian":
            spatial = gaussian_graph(space["coords"], s["delta"], s["kappa"])
        elif s["type"] == "mesh":
            spatial = mesh_graph(len(space["coords"]), space["edges"], space["edge_dist"], s["delta"])
        else:
            raise ValueError(f"unknown spatial graph type {s['type']}")
    t = graph_cfg.get("temporal")
    if t and n_time > 1:
        if t["type"] != "chain":
            raise ValueError(f"unknown temporal graph type {t['type']}")
        temporal = chain_graph(n_time, t.get("weight", 1.0))
    return spatial, temporal


def grid_points(method: dict) -> list[dict]:
    grid = method.get("grid", {}) or {}
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def make_estimator(method: dict, params: dict, ctx: Context, solver_cfg: dict) -> Pipeline:
    kind = method["model"]
    if kind == "logreg_l2":
        clf = LogisticRegression(C=params["C"], max_iter=5000, tol=1e-6)
    elif kind == "mlrdegg":
        kw = {**solver_cfg, **(method.get("fixed") or {}), **params}
        clf = MLRDEGGClassifier(n_space=ctx.n_space, n_time=ctx.n_time,
                                spatial_graph=ctx.spatial_graph, temporal_graph=ctx.temporal_graph,
                                **kw)
    else:
        raise ValueError(f"unknown model {kind}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def fit_info(pipe: Pipeline) -> dict:
    clf = pipe.named_steps["clf"]
    if isinstance(clf, MLRDEGGClassifier):
        info = {"converged": bool(clf.converged_), "status": clf.status_, "n_outer": clf.n_outer_,
                "n_inner": clf.n_inner_, "n_kkt_violations": clf.n_kkt_violations_,
                "objective": clf.objective_,
                "lambda_max": clf.lambda_max_ if clf.lambda_scale in ("ref", "max") else np.nan}
        info.update({k: v for k, v in clf.structure_summary().items() if np.isscalar(v)})
        info.update({"solver": clf.solver_, "numerical_stopping": clf.converged_,
                     "stationarity_certified": False, **clf.diagnostics_})
        for branch, radius in clf.t_ref_.items():
            info[f"reference_radius_{branch}"] = radius
            info[f"reference_radius_source_{branch}"] = clf.reference_radius_source_[branch]
        for phase, result in clf.fit_log_:
            info.update({f"{phase}_status": result.status, f"{phase}_solver": result.solver,
                         f"{phase}_converged": result.converged,
                         f"{phase}_iteration_limit_hit": result.diagnostics.get("iteration_limit_hit", False)})
        return info
    stopped = bool(np.all(clf.n_iter_ < clf.max_iter))
    return {"converged": stopped, "numerical_stopping": stopped,
            "status": "sklearn", "main_status": "sklearn", "main_converged": stopped,
            "solver": "sklearn_lbfgs", "stationarity_certified": False,
            "iteration_limit_hit": bool(np.any(clf.n_iter_ >= clf.max_iter))}


def importance_map(pipe: Pipeline, ctx: Context) -> np.ndarray:
    """Class-effect row norms ``||W_j,:||`` on the standardized scale, as a ``(V, T)`` map."""
    clf = pipe.named_steps["clf"]
    if isinstance(clf, MLRDEGGClassifier):
        return clf.feature_importance()
    W = clf.coef_.T - clf.coef_.T.mean(axis=1, keepdims=True)
    return unvectorize(np.linalg.norm(W, axis=1), ctx.n_space, ctx.n_time)
