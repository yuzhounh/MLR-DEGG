"""First-level GLM: regressors, AR(1) prewhitening and z-statistic calibration."""

import numpy as np
import pytest
from scipy import stats

from mlrdegg.datasets import hcp, hcp_glm

TR = 0.72
N = 405


def _ar1_noise(rng, n, V, rho):
    e = rng.normal(size=(n, V))
    x = np.empty_like(e)
    x[0] = e[0] / np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def test_hrf_and_block_regressor():
    h = hcp_glm.double_gamma_hrf(0.1)
    assert h.sum() == pytest.approx(1.0) and 4.5 < np.argmax(h) * 0.1 < 5.5
    reg, der = hcp_glm.block_regressors([(20.0, 27.5)], N, TR)
    t = np.arange(N) * TR
    assert reg[(t > 30) & (t < 47)].min() > 0.95  # plateau inside a long block
    assert abs(reg[t < 19]).max() < 1e-12      # nothing before the onset
    assert abs(der @ reg) < 1e-8               # derivative orthogonalized


def test_t_to_z():
    t = np.array([-3.0, 0.0, 2.0, 60.0])
    z = hcp_glm.t_to_z(t, df=380)
    np.testing.assert_allclose(z[:3], stats.norm.isf(stats.t.sf(np.abs(t[:3]), 380)) * np.sign(t[:3]))
    assert np.isfinite(z[3]) and z[3] > 25  # t = 60 with 380 df is z ~ 29.9 (heavier t tail)


@pytest.mark.parametrize("ar1", [True, False])
def test_null_z_is_calibrated_under_ar1_noise(ar1):
    rng = np.random.default_rng(1)
    conds = {"a": [(10, 27.5), (150, 27.5)], "b": [(60, 27.5), (220, 27.5)]}
    X, names, Q = hcp_glm.design(conds, N, TR, 200.0)
    Y = _ar1_noise(rng, N, 3000, rho=0.4)
    Y = Y - Q @ (Q.T @ Y)
    z = hcp_glm.fit_contrasts(Y, X, np.eye(2, X.shape[1]), n_drift=Q.shape[1], ar1=ar1)
    assert abs(z.mean()) < 0.1
    if ar1:
        assert 0.9 < z.std() < 1.1
    else:
        assert z.std() > 1.15  # ignoring the autocorrelation inflates z, which is why FILM exists


def test_effect_is_recovered():
    rng = np.random.default_rng(2)
    conds = {"a": [(10, 27.5), (150, 27.5)], "b": [(60, 27.5), (220, 27.5)]}
    X, names, Q = hcp_glm.design(conds, N, TR, 200.0)
    Y = _ar1_noise(rng, N, 200, rho=0.3)
    Y[:, :100] += 1.0 * X[:, [names.index("a")]]
    Y = Y - Q @ (Q.T @ Y)
    z = hcp_glm.fit_contrasts(Y, X, np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), n_drift=Q.shape[1])
    assert z[0, :100].mean() > 5 and abs(z[0, 100:].mean()) < 0.3 and abs(z[1].mean()) < 0.3


def _blocks():
    onsets = [8.0, 42.0, 80.0, 115.0, 160.0, 195.0, 235.0, 270.0]
    cats = ["body", "faces", "places", "tools"] * 2
    loads = ["2bk"] * 4 + ["0bk"] * 4
    return [{"onset": o, "duration": 27.5, "category": c, "load": l} for o, c, l in zip(onsets, cats, loads)]


@pytest.mark.parametrize("unit,n_maps", [("run", 4), ("block", 8)])
def test_glm_maps_labels(unit, n_maps):
    rng = np.random.default_rng(3)
    blocks = _blocks()
    ts = rng.normal(size=(50, N))
    faces = [(b["onset"], b["duration"]) for b in blocks if b["category"] == "faces"]
    ts[:10] += 2.0 * hcp_glm.block_regressors(faces, N, TR)[0]
    maps = hcp.glm_maps(ts, blocks, {"glm": {"unit": unit}})
    assert len(maps) == n_maps and all(m["Z"].shape == (50, 1) for m in maps)
    face = [m for m in maps if m["category"] == "faces"]
    other = [m for m in maps if m["category"] != "faces"]
    assert min(m["Z"][:10].mean() for m in face) > max(m["Z"][:10].mean() for m in other) + 2
    if unit == "run":
        assert [m["category"] for m in maps] == list(hcp.CATEGORIES) and {m["load"] for m in maps} == {"all"}
