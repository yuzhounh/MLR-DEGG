"""EEG trial selection, ocular artifact detectors and ERP CORE exclusion rules."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mne")
from mlrdegg.datasets import erpcore  # noqa: E402

TIMES = np.arange(-0.2, 0.8, 1 / 1000)


def test_select_trials_rt_window_and_delay():
    events = pd.DataFrame({
        "onset": [1.0, 1.5, 3.0, 3.1, 5.0, 6.3, 8.0, 8.6],
        "trial_type": ["stimulus", "response"] * 4,
        "value": [5, 201, 50, 201, 120, 201, 150, 202],
    })
    t = erpcore.select_trials(events, True, (0.2, 1.0), 0.026)
    assert list(t["label"]) == ["face", "car", "scrambled_face", "scrambled_car"]
    np.testing.assert_allclose(t["onset"], [1.026, 3.026, 5.026, 8.026])
    # RT 474 ms ok; 74 ms too fast; 1274 ms too slow; incorrect response
    assert list(t["valid"]) == [True, False, False, False]


def test_blink_detector_window():
    x = np.zeros((2, TIMES.size))
    x[0, (TIMES > 0.1) & (TIMES < 0.2)] = 160.0      # blink during the stimulus window
    x[1, (TIMES > 0.5) & (TIMES < 0.6)] = 160.0      # blink after the detection window
    p2p = erpcore.moving_window_p2p(x, TIMES, -0.025, 0.325, 0.175, 0.01)
    assert p2p[0] == pytest.approx(160.0) and p2p[1] == 0.0


def test_step_detector():
    x = np.zeros((2, TIMES.size))
    x[0, TIMES >= 0.1] = 40.0                          # step of 40 uV (saccade)
    x[1] = 40.0 * np.sin(2 * np.pi * 40 * TIMES)       # oscillation averages out
    step = erpcore.step_like(x, TIMES, -0.05, 0.35, 0.1, 0.01)
    assert step[0] == pytest.approx(40.0, rel=0.05)
    assert step[1] < 5.0


def test_quality_rules():
    labels = ["face"] * 10 + ["car"] * 10
    trials = pd.DataFrame({"label": labels, "valid": [True] * 20})
    q = erpcore.quality(trials, np.array(["face"] * 9 + ["car"] * 9), {})
    assert not q["excluded"] and q["artifact_fraction"] == pytest.approx(0.1)
    q = erpcore.quality(trials, np.array(["face"] * 2 + ["car"] * 10), {})
    assert q["excluded"] and set(q["reasons"].split(";")) == {"artifacts", "condition"}
