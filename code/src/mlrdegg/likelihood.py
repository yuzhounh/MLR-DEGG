"""Multinomial softmax likelihood in contrast coordinates (Section "Likelihood derivatives").

Code convention: samples are rows, ``Xa = [X, 1]`` has shape ``(n, d+1)``, i.e. ``Xa = Xbar^T``
in the manuscript.  ``Theta = [B; c^T]`` has shape ``(d+1, K-1)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def augment(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])


def scores(Xa: np.ndarray, Theta: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Score matrix ``S = Xbar^T Theta H^T`` (eq:score-matrix)."""
    return (Xa @ Theta) @ H.T


def softmax(S: np.ndarray) -> np.ndarray:
    return np.exp(S - logsumexp(S, axis=1, keepdims=True))


def loss(Xa: np.ndarray, Y: np.ndarray, Theta: np.ndarray, H: np.ndarray) -> float:
    """Average negative log-likelihood (eq:multinomial-loss)."""
    S = scores(Xa, Theta, H)
    return float(np.mean(logsumexp(S, axis=1) - np.einsum("ik,ik->i", Y, S)))


def loss_grad(
    Xa: np.ndarray, Y: np.ndarray, Theta: np.ndarray, H: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Loss, gradient (eq:likelihood-gradient) and probabilities."""
    n = Xa.shape[0]
    S = scores(Xa, Theta, H)
    lse = logsumexp(S, axis=1, keepdims=True)
    P = np.exp(S - lse)
    value = float(np.mean(lse[:, 0] - np.einsum("ik,ik->i", Y, S)))
    G = Xa.T @ ((P - Y) @ H) / n
    return value, G, P


def hvp(Xa: np.ndarray, Theta: np.ndarray, H: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    """Hessian-vector product (eq:likelihood-hvp)."""
    n = Xa.shape[0]
    P = softmax(scores(Xa, Theta, H))
    E = (Xa @ Delta) @ H.T
    Jm = P * (E - np.sum(P * E, axis=1, keepdims=True))
    return Xa.T @ (Jm @ H) / n
