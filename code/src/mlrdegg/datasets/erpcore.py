"""ERP CORE N170 (Kappenman et al., 2021): four-class visual category decoding from EEG.

Classes follow the ERP CORE N170 event scheme: stimulus codes 1-40 faces, 41-80 cars,
101-140 scrambled faces, 141-180 scrambled cars; response code 201 is correct, 202 incorrect.
Preprocessing follows the ERP CORE N170 pipeline (26 ms monitor-delay shift, correct responses
within 200-1000 ms, average reference, -200..800 ms epochs with pre-stimulus baseline, rejection
of blinks and horizontal eye movements during stimulus presentation with the published
per-subject thresholds, participant exclusion rules) with two simplifications: EOG regression
replaces ICA, and a fixed peak-to-peak threshold replaces the individually tuned artifact
thresholds on the scalp channels.  The signal is then averaged in short time bins, giving one
``V x T`` matrix per trial.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .common import save_subject

log = logging.getLogger(__name__)

CLASS_CODES = {
    "face": range(1, 41),
    "car": range(41, 81),
    "scrambled_face": range(101, 141),
    "scrambled_car": range(141, 181),
}
CORRECT, INCORRECT = 201, 202
EOG_CHANNELS = ("HEOG_left", "HEOG_right", "VEOG_lower")
BIPOLAR = {"VEOG_bipolar": ("VEOG_lower", "FP2"), "HEOG_bipolar": ("HEOG_left", "HEOG_right")}
# ERP CORE defaults for the stimulus-period ocular artifact detectors (used when a subject has no
# row in the per-subject parameter file): threshold [uV], window start/end, width, step [ms].
DEFAULT_AR = {"blink": (150, -25, 325, 175, 10), "heog": (32, -50, 350, 100, 10)}
AR_FIELDS = ("threshold_uv", "tmin_ms", "tmax_ms", "window_ms", "step_ms")


def code_to_class(code: int) -> str | None:
    for name, codes in CLASS_CODES.items():
        if code in codes:
            return name
    return None


def subject_files(raw_root: Path, subject: int) -> dict[str, Path]:
    base = raw_root / f"sub-{subject:03d}" / "eeg" / f"sub-{subject:03d}_task-N170"
    return {
        "set": base.with_name(base.name + "_eeg.set"),
        "events": base.with_name(base.name + "_events.tsv"),
        "electrodes": base.with_name(base.name + "_electrodes.tsv"),
    }


def read_electrodes(path: Path, ch_names: list[str]) -> np.ndarray:
    el = pd.read_csv(path, sep="\t").set_index("name")
    return el.loc[ch_names, ["x", "y", "z"]].to_numpy(float)


def detect_bad_channels(raw: mne.io.BaseRaw, eeg_names: list[str], mad_ratio: float,
                        sd_ratio: float) -> list[str]:
    """Bad channels of the filtered, unreferenced recording.

    A channel is bad when its robust scale (1.4826 x MAD) departs from the median scale by more
    than ``mad_ratio`` (persistently noisy or flat), or when its standard deviation exceeds
    ``sd_ratio`` times the median SD (intermittent gross artifacts).  Blinks inflate the SD of the
    frontal-pole channels by roughly an order of magnitude, so ``sd_ratio`` must stay well above 10.
    """
    x = raw.get_data(picks=eeg_names)
    sd = x.std(axis=1)
    mad = 1.4826 * np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
    r_mad, r_sd = mad / np.median(mad), sd / np.median(sd)
    bad = (r_mad > mad_ratio) | (r_mad < 1.0 / mad_ratio) | (r_sd > sd_ratio)
    return [ch for ch, b in zip(eeg_names, bad) if b]


def select_trials(events: pd.DataFrame, require_correct: bool, rt_window: tuple[float, float] | None,
                  stim_delay: float) -> pd.DataFrame:
    """All labelled stimulus events; ``valid`` marks the trials that enter the analysis.

    Stimulus onsets are shifted by ``stim_delay`` (monitor delay).  As in the ERP CORE bin
    descriptor ``{t<200-1000>201}``, a trial is valid when the first response after the stimulus
    is correct and its latency from the shifted onset lies in ``rt_window``.
    """
    ev = events.sort_values("onset").reset_index(drop=True)
    rows = []
    stim_idx = np.flatnonzero(ev["trial_type"].to_numpy() == "stimulus")
    for k, i in enumerate(stim_idx):
        code = int(ev.at[i, "value"])
        label = code_to_class(code)
        if label is None:
            continue
        onset = float(ev.at[i, "onset"]) + stim_delay
        stop = stim_idx[k + 1] if k + 1 < len(stim_idx) else len(ev)
        resp = ev.iloc[i + 1 : stop]
        resp = resp[resp["trial_type"] == "response"]
        response = int(resp["value"].iloc[0]) if len(resp) else -1
        rt = float(resp["onset"].iloc[0]) - onset if len(resp) else np.nan
        valid = not require_correct or response == CORRECT
        if rt_window is not None:
            valid = valid and rt_window[0] <= rt <= rt_window[1]
        rows.append({"onset": onset, "code": code, "label": label, "response": response,
                     "rt": rt, "valid": valid})
    return pd.DataFrame(rows)


def _windows(times: np.ndarray, tmin: float, tmax: float, width: float, step: float):
    """Sample masks of the moving windows of ERPLAB's detectors, all inside ``[tmin, tmax]``."""
    for a in np.arange(tmin, tmax - width + 1e-9, step):
        yield a, (times >= a - 1e-9) & (times <= a + width + 1e-9)


def moving_window_p2p(x: np.ndarray, times: np.ndarray, tmin, tmax, width, step) -> np.ndarray:
    """Largest peak-to-peak amplitude over moving windows (ERPLAB ``pop_artmwppth``); ``(n, S) -> (n,)``."""
    out = np.zeros(len(x))
    for _, m in _windows(times, tmin, tmax, width, step):
        seg = x[:, m]
        out = np.maximum(out, seg.max(axis=1) - seg.min(axis=1))
    return out


def step_like(x: np.ndarray, times: np.ndarray, tmin, tmax, width, step) -> np.ndarray:
    """Largest |mean(second half) - mean(first half)| over moving windows (ERPLAB ``pop_artstep``)."""
    out = np.zeros(len(x))
    for a, m in _windows(times, tmin, tmax, width, step):
        first = m & (times < a + width / 2 - 1e-9)
        second = m & ~first
        out = np.maximum(out, np.abs(x[:, second].mean(axis=1) - x[:, first].mean(axis=1)))
    return out


def ar_parameters(path: str | Path | None, subject: int) -> dict[str, tuple]:
    """Per-subject blink / HEOG detector parameters (ERP CORE values), falling back to defaults."""
    params = dict(DEFAULT_AR)
    if path:
        table = pd.read_csv(path).set_index("subject")
        if subject in table.index:
            row = table.loc[subject]
            params = {k: tuple(float(row[f"{k}_{f}"]) for f in AR_FIELDS) for k in DEFAULT_AR}
    return params


def detect_artifacts(eeg_uv: np.ndarray, bip_uv: np.ndarray, times: np.ndarray, reject_eeg_uv: float,
                     ar: dict[str, tuple]) -> dict[str, np.ndarray]:
    """Boolean artifact flags per epoch: scalp peak-to-peak, blink and HEOG step in the stimulus window.

    ``bip_uv`` holds the uncorrected bipolar VEOG and HEOG, ``(n, 2, S)`` in microvolts.
    """
    flags = {"eeg_p2p": (eeg_uv.max(axis=2) - eeg_uv.min(axis=2)).max(axis=1) > reject_eeg_uv}
    for key, detector, ch in (("blink", moving_window_p2p, 0), ("heog", step_like, 1)):
        thr, tmin, tmax, width, step = ar[key]
        flags[key] = detector(bip_uv[:, ch], times, tmin / 1e3, tmax / 1e3, width / 1e3,
                              step / 1e3) > thr
    return flags


def quality(trials: pd.DataFrame, kept_labels: np.ndarray, rules: dict) -> dict:
    """ERP CORE participant exclusion criteria.

    ``accuracy``: valid trials (correct, RT in window) over all trials; ``artifact_fraction``:
    rejected over valid trials; ``min_condition_fraction``: smallest per-class ratio of analysed
    trials to presented trials.
    """
    presented = trials["label"].value_counts()
    kept = pd.Series(kept_labels).value_counts().reindex(presented.index, fill_value=0)
    n_valid = int(trials["valid"].sum())
    q = {"n_presented": len(trials), "n_valid": n_valid, "n_kept": len(kept_labels),
         "accuracy": n_valid / len(trials),
         "artifact_fraction": 1.0 - len(kept_labels) / max(n_valid, 1),
         "min_condition_fraction": float((kept / presented).min())}
    reasons = []
    if q["artifact_fraction"] >= rules.get("max_artifact_fraction", 0.25):
        reasons.append("artifacts")
    if q["accuracy"] < rules.get("min_accuracy", 0.75):
        reasons.append("accuracy")
    if q["min_condition_fraction"] < rules.get("min_condition_fraction", 0.5):
        reasons.append("condition")
    q["excluded"] = bool(reasons)
    q["reasons"] = ";".join(reasons)
    return q


def bin_average(data: np.ndarray, times: np.ndarray, tmin: float, tmax: float, width: float):
    """Average ``(n, V, S)`` epochs in consecutive bins of ``width`` seconds on ``[tmin, tmax)``."""
    edges = np.arange(tmin, tmax + 1e-9, width)
    edges = edges[edges <= tmax + 1e-9]
    out, centers = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (times >= a - 1e-9) & (times < b - 1e-9)
        out.append(data[:, :, m].mean(axis=2))
        centers.append(0.5 * (a + b))
    return np.stack(out, axis=2), np.asarray(centers)


def preprocess_subject(subject: int, raw_root: Path, out_dir: Path, cfg: dict) -> dict:
    files = subject_files(raw_root, subject)
    raw = mne.io.read_raw_eeglab(files["set"], preload=True, verbose="error")
    raw.set_channel_types({ch: "eog" for ch in EOG_CHANNELS if ch in raw.ch_names})
    eeg_names = [ch for ch in raw.ch_names if ch not in EOG_CHANNELS]
    coords, montage = template_coords(eeg_names)
    raw.set_montage(mne.channels.make_dig_montage(dict(zip(eeg_names, coords / 1000.0)),
                                                  coord_frame="head"), on_missing="ignore")
    raw.filter(cfg["l_freq"], cfg["h_freq"], picks=["eeg", "eog"], verbose="error")
    bads = detect_bad_channels(raw, eeg_names, cfg.get("bad_mad_ratio", 5.0),
                               cfg.get("bad_sd_ratio", 50.0))
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True, verbose="error")
    # Uncorrected bipolar EOG for the stimulus-period ocular artifact detectors (reference-free).
    bip = np.vstack([raw.get_data(picks=[a]) - raw.get_data(picks=[b]) for a, b in BIPOLAR.values()])
    raw.add_channels([mne.io.RawArray(bip, mne.create_info(list(BIPOLAR), raw.info["sfreq"], "misc"),
                                      first_samp=raw.first_samp, verbose="error")],
                     force_update_info=True)
    raw.set_eeg_reference("average", projection=False, verbose="error")
    if cfg.get("eog_regression", True):
        # Ocular-artifact correction by least-squares regression on the EOG channels
        # (Gratton-type; deterministic alternative to the ICA step of the ERP CORE pipeline).
        from mne.preprocessing import EOGRegression

        EOGRegression(picks="eeg", picks_artifact="eog").fit(raw).apply(raw, copy=False)

    events = pd.read_csv(files["events"], sep="\t")
    rt_window = cfg.get("rt_window_s")
    all_trials = select_trials(events, cfg.get("require_correct", True),
                               tuple(rt_window) if rt_window else None, cfg.get("stim_delay_s", 0.0))
    trials = all_trials[all_trials["valid"]].reset_index(drop=True)
    samples = raw.first_samp + np.round(trials["onset"].to_numpy() * raw.info["sfreq"]).astype(int)
    mne_events = np.column_stack([samples, np.zeros_like(samples), trials["code"].to_numpy()])

    epochs = mne.Epochs(raw, mne_events, tmin=cfg["epoch_tmin"], tmax=cfg["epoch_tmax"],
                        baseline=(cfg["epoch_tmin"], 0.0), picks=eeg_names + list(BIPOLAR),
                        reject=None, preload=True, event_repeated="drop", verbose="error")
    data = epochs.get_data(picks=eeg_names) * 1e6  # microvolts, (n, V, S)
    flags = detect_artifacts(data, epochs.get_data(picks=list(BIPOLAR)) * 1e6, epochs.times,
                             cfg["reject_eeg_uv"], ar_parameters(cfg.get("ar_params"), subject))
    good = ~np.logical_or.reduce(list(flags.values()))
    kept = epochs.selection[good]
    X, centers = bin_average(data[good], epochs.times, cfg["feature_tmin"], cfg["feature_tmax"],
                             cfg["bin_width"])
    y = trials["label"].to_numpy()[kept]
    save_subject(out_dir / f"sub-{subject:03d}.npz", X, y,
                 code=trials["code"].to_numpy()[kept], trial=kept.astype(int),
                 rt=trials["rt"].to_numpy()[kept])
    counts = pd.Series(y).value_counts().to_dict()
    q = quality(all_trials, y, cfg.get("exclusion", {}))
    q.update({f"n_flag_{k}": int(v.sum()) for k, v in flags.items()})
    return {"subject": subject, "n_candidates": len(trials), "n_kept": len(kept), "bads": bads,
            "counts": counts, "quality": q, "ch_names": eeg_names, "times": centers,
            "coords": read_electrodes(files["electrodes"], eeg_names)}


def template_coords(ch_names: list[str]) -> tuple[np.ndarray, str]:
    """Subject-independent electrode positions (mm) from a standard 10-05 montage."""
    name = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"
    pos = mne.channels.make_standard_montage(name).get_positions()["ch_pos"]
    lut = {k.upper(): v for k, v in pos.items()}
    return np.array([lut[ch.upper()] for ch in ch_names]) * 1000.0, name


def write_space(out_dir: Path, infos: list[dict]) -> None:
    """``coords`` is the template montage used for the graph; digitized positions are kept too."""
    ref = infos[0]
    coords, montage = template_coords(ref["ch_names"])
    np.savez(out_dir / "space.npz", ch_names=np.asarray(ref["ch_names"]), coords=coords,
             montage=np.asarray(montage),
             coords_digitized_mean=np.mean([i["coords"] for i in infos], axis=0),
             times=ref["times"])
