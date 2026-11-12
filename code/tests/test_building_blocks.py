"""Unit tests for contrast coordinates, graphs, penalties and likelihood derivatives."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from mlrdegg import likelihood as lik
from mlrdegg.contrast import contrast_matrix
from mlrdegg.graphs import chain_graph, gaussian_graph, spatiotemporal_operators, unvectorize, vectorize
from mlrdegg.penalties import h_penalty, mm_weights, prox_rows, radial_root, row_norms


@pytest.mark.parametrize("K", [2, 3, 4, 7])
def test_contrast_matrix(K):
    H = contrast_matrix(K)
    assert np.allclose(H.T @ H, np.eye(K - 1))
    assert np.allclose(H.T @ np.ones(K), 0)


def test_vectorization_and_kronecker_operators():
    rng = np.random.default_rng(1)
    V, T = 4, 3
    Zs = rng.normal(size=(2, V, T))
    x = vectorize(Zs)
    assert np.allclose(x[0], Zs[0].flatten(order="F"))
    assert np.allclose(unvectorize(x[0], V, T), Zs[0])
    gV = gaussian_graph(rng.normal(size=(V, 3)), delta=1.0, kappa=10.0)
    gT = chain_graph(T)
    (Dsp, a), (Dtm, b) = spatiotemporal_operators(gV, gT, V, T)
    DV, DT = gV.incidence().toarray(), gT.incidence().toarray()
    Z = Zs[0]
    assert np.allclose(Dsp @ x[0], (DV @ Z).flatten(order="F"))
    assert np.allclose(Dtm @ x[0], (Z @ DT.T).flatten(order="F"))
    assert len(a) == Dsp.shape[0] and len(b) == Dtm.shape[0]


def test_laplacian_identity():
    rng = np.random.default_rng(2)
    g = gaussian_graph(rng.normal(size=(8, 2)), delta=0.7, kappa=1.5)
    w = rng.normal(size=8)
    N = g.adjacency().toarray()
    L = np.diag(N.sum(1)) - N
    D = g.incidence()
    assert np.isclose(w @ L @ w, np.sum(g.weights * (D @ w) ** 2))


@pytest.mark.parametrize("p", [0.3, 0.7])
def test_scalar_majorization(p):
    eps = 0.05
    t = np.linspace(0, 3, 301)
    for t0 in [0.0, 0.2, 1.5]:
        omega, _ = mm_weights(np.array([t0]), p, eps)
        bound = h_penalty(t0, p, eps) + omega[0] * (t - t0)
        assert np.all(h_penalty(t, p, eps) <= bound + 1e-12)


@pytest.mark.parametrize("nu", [1.0, 1.3, 2.0, 2.5, 4.0])
def test_prox_rows_matches_scalar_minimization(nu):
    rng = np.random.default_rng(3)
    V = rng.normal(size=(20, 3)) * rng.uniform(0, 2, size=(20, 1))
    V[0] = 0
    tau = rng.uniform(0.01, 1.0, size=20)
    U = prox_rows(V, tau, nu)
    s = row_norms(V)
    for i in range(20):
        if s[i] == 0:
            assert np.allclose(U[i], 0)
            continue
        res = minimize_scalar(lambda t: 0.5 * (t - s[i]) ** 2 + tau[i] * t**nu, bounds=(0, s[i]),
                              method="bounded", options={"xatol": 1e-12})
        assert np.isclose(np.linalg.norm(U[i]), res.x, atol=1e-6)
        if np.linalg.norm(U[i]) > 0:
            assert np.allclose(U[i] / np.linalg.norm(U[i]), V[i] / s[i])


def test_radial_root_equation():
    s = np.array([0.0, 1e-6, 0.3, 2.0, 50.0])
    for nu in [1.2, 1.9, 3.0]:
        t = radial_root(s, 0.4, nu)
        assert np.all((t >= 0) & (t <= s))
        mask = s > 0
        assert np.allclose(t[mask] + 0.4 * nu * t[mask] ** (nu - 1), s[mask], rtol=1e-9, atol=1e-12)


def test_likelihood_gradient_hvp_and_majorizer():
    rng = np.random.default_rng(4)
    n, d, K = 30, 6, 4
    H = contrast_matrix(K)
    Xa = lik.augment(rng.normal(size=(n, d)))
    Y = np.eye(K)[rng.integers(0, K, n)]
    Th = rng.normal(size=(d + 1, K - 1))
    f, G, _ = lik.loss_grad(Xa, Y, Th, H)
    Dl = rng.normal(size=Th.shape)
    h = 1e-6
    fd = (lik.loss(Xa, Y, Th + h * Dl, H) - lik.loss(Xa, Y, Th - h * Dl, H)) / (2 * h)
    assert np.isclose(fd, np.sum(G * Dl), rtol=1e-6)
    _, Gp, _ = lik.loss_grad(Xa, Y, Th + h * Dl, H)
    _, Gm, _ = lik.loss_grad(Xa, Y, Th - h * Dl, H)
    assert np.allclose((Gp - Gm) / (2 * h), lik.hvp(Xa, Th, H, Dl), atol=1e-6)
    for scale in [0.1, 1.0, 10.0]:
        Th2 = Th + scale * rng.normal(size=Th.shape)
        Dd = Th2 - Th
        M = f + np.sum(G * Dd) + np.sum((Xa @ Dd) ** 2) / (4 * n)
        assert lik.loss(Xa, Y, Th2, H) <= M + 1e-12
