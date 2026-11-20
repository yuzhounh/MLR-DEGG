"""Spatial and temporal graphs and their incidence operators.

A graph stores every undirected edge once together with a nonnegative weight.  The incidence
matrix ``D`` is unweighted (+1/-1 entries); edge weights stay outside the powered norm, as in
(eq:graph-generalization).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class Graph:
    """Undirected weighted graph on ``n_nodes`` nodes."""

    n_nodes: int
    edges: np.ndarray  # (m, 2) int, i < j
    weights: np.ndarray  # (m,) nonnegative

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if len(self.edges) != len(self.weights):
            raise ValueError("edges and weights must have the same length")
        if np.any(self.weights < 0):
            raise ValueError("edge weights must be nonnegative")

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def incidence(self) -> sp.csr_matrix:
        """Arbitrarily oriented incidence matrix of shape ``(m, n_nodes)``."""
        m = self.n_edges
        rows = np.repeat(np.arange(m), 2)
        cols = self.edges.ravel()
        vals = np.tile([1.0, -1.0], m)
        return sp.csr_matrix((vals, (rows, cols)), shape=(m, self.n_nodes))

    def adjacency(self) -> sp.csr_matrix:
        i, j = self.edges.T
        N = sp.coo_matrix((self.weights, (i, j)), shape=(self.n_nodes, self.n_nodes))
        return (N + N.T).tocsr()

    def n_components(self) -> int:
        return connected_components(self.adjacency(), directed=False)[0]

    def summary(self) -> dict:
        deg = np.bincount(self.edges.ravel(), minlength=self.n_nodes)
        return {
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "mean_degree": float(deg.mean()) if self.n_nodes else 0.0,
            "max_degree": int(deg.max()) if self.n_nodes else 0,
            "n_components": int(self.n_components()),
        }


def gaussian_graph(coords: np.ndarray, delta: float, kappa: float) -> Graph:
    """Distance-weighted graph (eq:gaussian-adjacency).

    ``N_jk = exp(-d_jk^2 / (2 delta^2))`` for ``0 < d_jk <= kappa``.
    """
    coords = np.asarray(coords, dtype=float)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=kappa, output_type="ndarray")
    if len(pairs) == 0:
        return Graph(len(coords), np.zeros((0, 2), int), np.zeros(0))
    pairs = np.sort(pairs, axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    dist = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    keep = dist > 0
    pairs, dist = pairs[keep], dist[keep]
    weights = np.exp(-(dist**2) / (2.0 * delta**2))
    return Graph(len(coords), pairs, weights)


def mesh_graph(n_nodes: int, edges: np.ndarray, dist: np.ndarray, delta: float) -> Graph:
    """Graph on given edges (e.g. a cortical mesh) with weights ``exp(-d^2 / (2 delta^2))``,
    the same kernel as :func:`gaussian_graph` but on geodesic neighbours instead of a radius."""
    edges = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    dist = np.asarray(dist, dtype=float).reshape(-1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    return Graph(n_nodes, edges[order], np.exp(-(dist[order] ** 2) / (2.0 * delta**2)))


def chain_graph(n_nodes: int, weight: float = 1.0) -> Graph:
    """Open first-order chain, e.g. adjacent time points within one trial."""
    edges = np.column_stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)])
    return Graph(n_nodes, edges, np.full(n_nodes - 1, float(weight)))


def spatiotemporal_operators(
    spatial: Graph | None, temporal: Graph | None, n_space: int, n_time: int
) -> tuple[tuple[sp.csr_matrix, np.ndarray] | None, tuple[sp.csr_matrix, np.ndarray] | None]:
    """Kronecker operators (eq:spatiotemporal-operators) for column-major ``vec(Z)``.

    Features are ordered as ``j = v + V * t`` (``Z`` is ``V x T``).  Returns
    ``(D_sp, a)`` and ``(D_tm, b)``; a branch is ``None`` when its graph is absent or empty.
    """
    sp_op = tm_op = None
    if spatial is not None and spatial.n_edges > 0:
        if spatial.n_nodes != n_space:
            raise ValueError("spatial graph size does not match n_space")
        D_sp = sp.kron(sp.identity(n_time, format="csr"), spatial.incidence(), format="csr")
        sp_op = (D_sp, np.tile(spatial.weights, n_time))
    if temporal is not None and temporal.n_edges > 0 and n_time > 1:
        if temporal.n_nodes != n_time:
            raise ValueError("temporal graph size does not match n_time")
        D_tm = sp.kron(temporal.incidence(), sp.identity(n_space, format="csr"), format="csr")
        tm_op = (D_tm, np.repeat(temporal.weights, n_space))
    return sp_op, tm_op


def vectorize(Z: np.ndarray) -> np.ndarray:
    """Column-major vectorization of trials ``(n, V, T) -> (n, V*T)`` (eq:vectorization)."""
    Z = np.asarray(Z)
    n = Z.shape[0]
    return np.transpose(Z, (0, 2, 1)).reshape(n, -1)


def unvectorize(x: np.ndarray, n_space: int, n_time: int) -> np.ndarray:
    """Inverse of :func:`vectorize` for a single vector or a ``(d, ...)`` array."""
    x = np.asarray(x)
    return np.swapaxes(x.reshape((n_time, n_space) + x.shape[1:]), 0, 1)
