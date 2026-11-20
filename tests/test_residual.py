"""Post-fit common residual: report-only behaviour and fields."""

import numpy as np
import pytest

from mlrdegg import MLRDEGGClassifier
from mlrdegg.residual import FIELDS, summarize


def make_model(tp, **kw):
    params = dict(n_space=tp["V"], n_time=tp["T"], spatial_graph=tp["spatial"],
                  temporal_graph=tp["temporal"], lambda_sel=0.2, lambda_g=0.2, eta=0.5,
                  lambda_scale="none")
    params.update(kw)
    return MLRDEGGClassifier(**params)


@pytest.mark.parametrize("p_s,p_g", [(1, 2), (1, 1), (0.5, 0.5)])
def test_residual_is_report_only(toy_problem, p_s, p_g):
    tp = toy_problem
    on = make_model(tp, p_s=p_s, p_g=p_g, max_outer=100).fit(tp["X"], tp["y"])
    off = make_model(tp, p_s=p_s, p_g=p_g, max_outer=100, residual_diagnostic=False).fit(tp["X"], tp["y"])
    assert np.array_equal(on.coef_, off.coef_) and np.array_equal(on.intercept_, off.intercept_)
    assert off.common_residual_ == {} and "common_residual" not in off.diagnostics_
    assert set(FIELDS) <= set(on.diagnostics_)
    assert on.common_residual_["common_residual_status"] in ("resolved", "residual_unresolved")
    assert np.isfinite(on.common_residual_["common_residual"])
    assert "1e-03" in summarize([on.diagnostics_])


def test_residual_small_at_accurate_fit(toy_problem):
    tp = toy_problem
    m = make_model(tp, p_s=1, p_g=2, solver="apg", apg_tol=1e-14, tol_x=1e-12, apg_max_iter=50000)
    m.fit(tp["X"], tp["y"])
    assert m.common_residual_["common_residual_1e-05"] == "pass"
