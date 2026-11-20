"""Orthonormal contrast coordinates (Section "Orthonormal contrast coordinates")."""

from __future__ import annotations

import numpy as np


def contrast_matrix(n_classes: int) -> np.ndarray:
    """Return a Helmert-type matrix ``H`` of shape ``(K, K-1)``.

    ``H`` satisfies ``H.T @ H = I`` and ``H.T @ 1 = 0``, so ``W = B H^T`` and ``b = H c``
    automatically satisfy the sum-to-zero constraints (eq:sum-zero).
    """
    if n_classes < 2:
        raise ValueError("At least two classes are required.")
    K = n_classes
    H = np.zeros((K, K - 1))
    for j in range(1, K):
        H[:j, j - 1] = 1.0
        H[j, j - 1] = -float(j)
        H[:, j - 1] /= np.sqrt(j * (j + 1))
    return H
