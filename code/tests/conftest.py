import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlrdegg.graphs import chain_graph, gaussian_graph, vectorize  # noqa: E402


@pytest.fixture
def toy_problem():
    """Small spatiotemporal 3-class problem with a class signal on a connected patch."""
    rng = np.random.default_rng(0)
    V, T, n, K = 12, 5, 90, 3
    coords = np.column_stack([np.arange(V) % 4, np.arange(V) // 4, np.zeros(V)]).astype(float)
    y = np.repeat(np.arange(K), n // K)
    Z = rng.normal(size=(n, V, T))
    for k in range(K):
        Z[y == k, k * 4 : k * 4 + 3, 1:4] += 1.0
    X = vectorize(Z)
    X = (X - X.mean(0)) / X.std(0)
    return {
        "X": X,
        "y": y,
        "V": V,
        "T": T,
        "spatial": gaussian_graph(coords, delta=1.0, kappa=1.01),
        "temporal": chain_graph(T),
    }
