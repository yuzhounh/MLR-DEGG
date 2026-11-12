"""Failure-injection tests for resumable B101 outer fitting and state files."""
import dataclasses
import importlib.util
from pathlib import Path
import sys

import numpy as np
from mlrdegg import MLRDEGGClassifier
from mlrdegg.graphs import chain_graph
from mlrdegg.solver import SolverState, solve_mm_admm


def test_interrupted_outer_fit_resumes_same_iteration_schedule():
    rng = np.random.default_rng(418)
    X = rng.normal(size=(48, 12))
    y = np.tile(np.arange(3), 16)
    model = MLRDEGGClassifier(6, 2, temporal_graph=chain_graph(2),
        p_s=1, p_g=1, eta=1, lambda_sel=.01, lambda_g=.02,
        lambda_scale="none", residual_diagnostic=False, max_outer=9, max_inner=20,
        tol_obj=0, tol_x=0)
    saved = {}

    def exercise(Xa, Y, H, branches, options, initial):
        baseline = solve_mm_admm(Xa, Y, H, branches, options, initial)
        def interrupt(theta, row):
            if row["outer"] != 2:
                return
            frame = sys._getframe(1).f_locals
            saved.update(state=SolverState(theta.copy(), [u.copy() for u in frame["U"]],
                         [z.copy() for z in frame["Z"]], frame["rho"]),
                         history=list(frame["history"]), rho=frame["n_rho_updates"],
                         kkt=frame["n_kkt_violations"])
            raise InterruptedError("planned interruption")
        options.monitor = interrupt
        try:
            solve_mm_admm(Xa, Y, H, branches, options, initial)
        except InterruptedError:
            pass
        resumed_options = dataclasses.replace(options, resume_history=saved["history"],
                         resume_rho_updates=saved["rho"], resume_kkt_violations=saved["kkt"])
        resumed = solve_mm_admm(Xa, Y, H, branches, resumed_options, saved["state"])
        np.testing.assert_allclose(resumed.Theta, baseline.Theta, rtol=1e-12, atol=1e-12)
        assert resumed.n_inner_total == baseline.n_inner_total
        assert [h["outer"] for h in resumed.history] == list(range(9))
        finished_options = dataclasses.replace(options, resume_history=baseline.history,
                                              resume_converged=True)
        finished = solve_mm_admm(Xa, Y, H, branches, finished_options,
                    SolverState(baseline.Theta, baseline.U, baseline.Z, baseline.rho))
        np.testing.assert_array_equal(finished.Theta, baseline.Theta)
        assert finished.n_outer == baseline.n_outer
        return resumed
    model._solve = exercise
    model.fit(X, y)


def test_atomic_state_can_fall_back_to_previous(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "26_refine_hcp181_primary.py"
    spec = importlib.util.spec_from_file_location("primary_refinement", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "checkpoint.pkl"
    module.save_state(path, {"accepted": 1})
    module.save_state(path, {"accepted": 2})
    path.write_bytes(b"planned corrupt latest checkpoint")
    assert module.load_state(path) == {"accepted": 1}


def test_scaled_worker_fit_keeps_raw_input_scaler():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    script = Path(__file__).parents[1] / "scripts" / "26_refine_hcp181_primary.py"
    spec = importlib.util.spec_from_file_location("primary_scaled_fit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(53)
    X = rng.normal(size=(60, 5)) * 7 + 19
    y = np.tile([0, 1, 2], 20)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
    scaled = pipe.named_steps["scale"].fit_transform(X)
    original_mean = pipe.named_steps["scale"].mean_.copy()
    module.fit_scaled_training(pipe, scaled, y)
    np.testing.assert_array_equal(pipe.named_steps["scale"].mean_, original_mean)
    np.testing.assert_allclose(module.predict_scaled_test(pipe, scaled), pipe.predict_proba(X))
