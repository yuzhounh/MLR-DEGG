"""Reporting regressions for finite-precision B/U disagreement."""
import numpy as np
import pandas as pd
import pytest

from mlrdegg import MLRDEGGClassifier
from mlrdegg.contrast import contrast_matrix
from mlrdegg.experiment.reporting import prediction_support_from_maps
from mlrdegg.solver import Branch
from mlrdegg.experiment.cv import inner_fit_quality, select_grid_point


def test_zero_auxiliary_rows_are_not_reported_as_zero_prediction_coefficients():
    # Reproduce the observed ERP case: 67 thresholded U rows, 720 nonzero B rows.
    model = MLRDEGGClassifier(n_space=720, lambda_g=0, lambda_scale="none")
    model.B_ = np.ones((720, 1))
    model.B_[:67] = 1.57e-8
    model.H_ = contrast_matrix(2)
    model.coef_ = model.B_ @ model.H_.T
    model.intercept_ = np.zeros(2)
    model.classes_ = np.array(["a", "b"])
    model.split_ = {"sel": model.B_.copy()}
    model.split_["sel"][:67] = 0
    model.branches_ = model.build_branches()
    summary = model.structure_summary()
    assert summary["sel_zero_fraction"] == 0
    assert summary["sel_aux_zero_fraction"] == pytest.approx(67 / 720)
    assert summary["sel_support_disagreement_fraction"] == pytest.approx(67 / 720)
    assert summary["sel_feasibility_norm"] > 0
    X = np.zeros((1, 720))
    X[0, 0] = 1e8
    assert not np.allclose(model.predict_proba(X), [[0.5, 0.5]])
    assert model.feature_importance()[0, 0] > 0


def test_saved_map_audit_preserves_tiny_nonzero_coefficients(tmp_path):
    (tmp_path / "units").mkdir()
    importance = np.ones((1, 30, 24))
    importance.ravel()[:67] = 1.57e-8
    np.savez(tmp_path / "units/sub-015_maps.npz", **{"graph_tv_tm@1se_repro": importance})
    outer = pd.DataFrame([{"unit": "sub-015", "outer_fold": 0,
                           "method": "graph_tv_tm@1se_repro", "sel_zero_fraction": 67 / 720}])
    row = prediction_support_from_maps(tmp_path, outer).iloc[0]
    assert row.sel_zero_fraction == 0
    assert row.sel_aux_zero_fraction == pytest.approx(67 / 720)
    assert row.zero_fraction_difference == pytest.approx(-67 / 720)


def test_zero_graph_auxiliary_does_not_certify_fused_prediction_coefficients():
    model = MLRDEGGClassifier(n_space=3, lambda_scale="none")
    model.B_ = np.array([[1.0], [1.0 + 1e-8], [2.0]])
    model.coef_ = model.B_ @ contrast_matrix(2).T
    D = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    branch = Branch("sp", D, np.ones(2), 1.0)
    model.branches_ = [branch]
    model.split_ = {"sp": D @ model.B_}
    model.split_["sp"][0] = 0
    summary = model.structure_summary()
    assert summary["sp_zero_fraction"] == 0
    assert summary["sp_aux_zero_fraction"] == 0.5
    assert summary["sp_support_disagreement_fraction"] == 0.5
    assert summary["sp_feasibility_norm"] == pytest.approx(1e-8)


def test_exception_free_grid_point_can_still_be_numerically_unfinished():
    rows = [{"error": "", "converged": True}, {"error": "", "converged": False}]
    historical = inner_fit_quality(rows)
    assert historical["ok"] and historical["inner_nonconverged_count"] == 1
    assert not inner_fit_quality(rows, "require_converged")["ok"]
    assert not inner_fit_quality([{"error": "fit failed", "converged": False}])["ok"]
    assert inner_fit_quality([{"error": ""}])["inner_quality_unknown_count"] == 1


def test_no_eligible_grid_point_cannot_silently_select_a_failed_fit():
    with pytest.raises(ValueError, match="No eligible"):
        select_grid_point([{"ok": False}], "min")


def test_resumed_fit_log_keeps_legacy_schema_and_separate_diagnostics(tmp_path):
    from mlrdegg.experiment.cv import FitCache
    cache = FitCache(tmp_path / "cache", {})
    meta = {"unit": "u", "stage": "half", "outer_fold": 0, "inner_fold": 0,
            "method": "m", "params": {}}
    info = {"error": "", "status": "max_outer", "converged": False,
            "solver": "mm_admm", "final_stationarity_B": 0.3}
    cache.save(cache.path(meta, {}), (info, None), meta)
    first_columns = pd.read_csv(tmp_path / "fits.csv").columns.tolist()
    info["final_feasibility_norm"] = 0.02
    cache.save(cache.path(meta, {}), (info, None), meta)
    assert pd.read_csv(tmp_path / "fits.csv").columns.tolist() == first_columns
    diagnostics = pd.read_csv(tmp_path / "fit_diagnostics.csv")
    assert len(diagnostics) == 2
    assert not diagnostics.converged.any()
    assert diagnostics.final_feasibility_norm.iloc[1] == 0.02


def test_provenance_cache_key_ignores_its_own_saved_key_but_detects_input_changes(tmp_path):
    from types import SimpleNamespace
    from mlrdegg.experiment.provenance import run_provenance
    (tmp_path / "space.npz").write_bytes(b"geometry")
    data = tmp_path / "sub-001.npz"
    data.write_bytes(b"first")
    ds = SimpleNamespace(subjects=[SimpleNamespace(subject="sub-001", y=[0, 1])])
    cfg = {"preprocess": {"out_dir": str(tmp_path)}, "experiment": {"subjects": ["sub-001"]}}
    first = run_provenance(cfg, ds, {})
    cfg["experiment"]["cache_provenance"] = first["cache_fingerprint"]
    assert run_provenance(cfg, ds, {})["cache_fingerprint"] == first["cache_fingerprint"]
    data.write_bytes(b"changed")
    assert run_provenance(cfg, ds, {})["cache_fingerprint"] != first["cache_fingerprint"]
    assert first["effective_config"]["experiment"] == {"subjects": ["sub-001"]}
