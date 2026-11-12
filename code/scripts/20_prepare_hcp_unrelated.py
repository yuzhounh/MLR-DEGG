"""Prepare the family-filtered HCP cohort using the frozen surface/segment pipeline.

Only input preparation: no model fits or manuscript edits. Existing subject NPZs
are validated and copied; missing subjects are processed with the original hcp
functions. Subject-level atomic files support resuming an interrupted run.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import struct
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import _bootstrap
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

from mlrdegg.datasets import hcp
from mlrdegg.paths import resolve

OUT = resolve("data/derived/hcp_wm_surf_unrelated")
FAMILY = resolve("data/HCP_YA_2025/metadata/restricted/hcp_ya_family_20261013.csv")
SOURCES = [resolve("data/derived/hcp_wm_surf_dev"), resolve("data/derived/hcp_wm_surf_final")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_npz(path):
    with np.load(path, allow_pickle=False) as z:
        assert z["X"].shape == (64, 4801, 4), f"invalid sample dimensions: {path}"
        assert np.isfinite(z["X"]).all(), f"nonfinite features: {path}"
        assert dict(zip(*np.unique(z["y"], return_counts=True))) == {c: 16 for c in hcp.CATEGORIES}
        assert dict(zip(*np.unique(z["run"], return_counts=True))) == {"LR": 32, "RL": 32}
        assert all(z[k].shape == (64,) for k in ("y", "run", "block", "segment", "load"))
        for run in hcp.RUNS:
            mask = z["run"] == run
            assert dict(zip(*np.unique(z["block"][mask], return_counts=True))) == {b: 4 for b in range(8)}
            assert dict(zip(*np.unique(z["load"][mask], return_counts=True))) == {"0bk": 16, "2bk": 16}


def inspect_zip(raw_root, subject, cfg):
    summary = {"subject": subject, "runs": {}}
    with zipfile.ZipFile(hcp.zip_path(raw_root, subject)) as z:
        for run in hcp.RUNS:
            prefix = f"{subject}/MNINonLinear/Results/tfMRI_WM_{run}/"
            cfile = prefix + f"tfMRI_WM_{run}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii"
            with z.open(cfile) as stream:
                head = nib.Nifti2Header.from_fileobj(stream)
                stream.seek(int(head["sizeof_hdr"]))
                stream.read(4)
                size, code = struct.unpack(head.endianness + "ii", stream.read(8))
                assert code == 32 and size > 8
                xml = ET.fromstring(stream.read(size - 8).rstrip(b"\x00"))
            series = next(a.attrib for a in xml.iter("MatrixIndicesMap")
                          if a.attrib.get("IndicesMapToDataType") == "CIFTI_INDEX_TYPE_SERIES")
            with z.open(prefix + f"tfMRI_WM_{run}_hp0_clean_rclean_tclean.nii.gz") as stream:
                with gzip.GzipFile(fileobj=stream) as g:
                    volume = nib.Nifti1Header.from_fileobj(g)
            frames = int(series["NumberOfSeriesPoints"])
            tr = float(volume.get_zooms()[3])
            assert volume.get_xyzt_units()[1] == "sec" and np.isclose(tr, hcp.TR, atol=1e-6)
            assert frames == volume.get_data_shape()[3] == 405
            blocks = hcp.read_blocks(z, subject, run)
            assert len(blocks) == 8
            assert {(b["load"], b["category"]) for b in blocks} == {(l, c) for l in hcp.LOADS for c in hcp.CATEGORIES}
            ends = []
            for b in blocks:
                start = int(np.ceil((b["onset"] + cfg["hrf_delay_s"]) / hcp.TR))
                stop = int(np.floor((b["onset"] + b["duration"] + cfg["hrf_delay_s"]) / hcp.TR))
                n = min((stop - start) // cfg["segment_trs"], cfg["max_segments_per_block"])
                assert n == 4 and 0 <= start and start + n * cfg["segment_trs"] <= frames
                ends.append((start, start + n * cfg["segment_trs"]))
            assert all(a[1] <= b[0] for a, b in zip(ends, ends[1:]))
            summary["runs"][run] = {"frames": frames, "volume_tr": tr,
                "cifti_series_step": float(series["SeriesStep"]) * 10 ** int(series["SeriesExponent"]),
                "blocks": len(blocks), "segments": 32, "cifti_crc": z.getinfo(cfile).CRC}
    return summary


def prepare():
    cfg = _bootstrap.load_config(resolve("code/configs/hcp_wm_surf_final.yaml"))["preprocess"]
    dev = _bootstrap.load_config(resolve("code/configs/hcp_wm_surf_dev.yaml"))["preprocess"]
    assert {k: v for k, v in cfg.items() if k not in ("subjects", "out_dir", "n_jobs")} == {
        k: v for k, v in dev.items() if k not in ("subjects", "out_dir", "n_jobs")}
    raw = resolve(cfg["raw_root"])
    downloaded = sorted(f.name.split("_")[0] for f in raw.glob("*_Task3TRecommended.zip"))
    candidates, incomplete = [], []
    for subject in downloaded:
        with zipfile.ZipFile(hcp.zip_path(raw, subject)) as z:
            names = set(z.namelist())
            required = []
            for run in hcp.RUNS:
                prefix = f"{subject}/MNINonLinear/Results/tfMRI_WM_{run}/"
                required += [prefix + f"tfMRI_WM_{run}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii",
                             prefix + f"tfMRI_WM_{run}_hp0_clean_rclean_tclean.nii.gz"]
                required += [prefix + f"EVs/{load}_{cat}.txt" for load in hcp.LOADS for cat in hcp.CATEGORIES]
            missing_files = [name for name in required if name not in names]
        if missing_files:
            incomplete.append({"subject": subject, "missing_files": missing_files})
        else:
            candidates.append(subject)
    with FAMILY.open(newline="", encoding="utf-8-sig") as f:
        family = {r["Subject"]: r["Family_ID"] for r in csv.DictReader(f)}
    groups = defaultdict(list)
    for subject in candidates:
        assert subject in family and family[subject]
        groups[family[subject]].append(subject)
    rng = np.random.default_rng(2026)
    selected = sorted(str(rng.choice(sorted(groups[k]))) for k in sorted(groups))
    assert len({family[s] for s in selected}) == len(selected)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": 2026, "downloaded_count": len(downloaded), "incomplete_inputs": incomplete,
        "candidate_count": len(candidates), "selected_count": len(selected),
        "subjects": selected, "family_table_sha256": sha(FAMILY),
        "preprocess": {k: v for k, v in cfg.items() if k not in ("subjects", "out_dir", "n_jobs")},
        "geometry_sha256": sha(SOURCES[0] / "space.npz")}
    mfile = OUT / "cohort_manifest.json"
    if mfile.exists():
        assert json.loads(mfile.read_text(encoding="utf-8")) == manifest, "cohort/input settings changed"
    else:
        _bootstrap.write_status(OUT / "selection_receipt.json", **manifest)
        mfile.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assert sha(SOURCES[1] / "space.npz") == manifest["geometry_sha256"]
    audit = [inspect_zip(raw, s, cfg) for s in selected]
    _bootstrap.write_status(OUT / "input_audit.json", subjects=len(audit), runs=len(audit) * 2,
        tr=hcp.TR, cifti_header_step_mismatches=sum(not np.isclose(r["cifti_series_step"], hcp.TR)
        for a in audit for r in a["runs"].values()), records=audit)
    existing = []
    for subject in selected:
        dst = OUT / f"sub-{subject}.npz"
        if dst.exists():
            inspect_npz(dst)
            existing.append(subject)
            continue
        src = next((d / dst.name for d in SOURCES if (d / dst.name).exists()), None)
        if src is not None:
            inspect_npz(src)
            shutil.copy2(src, dst)
            assert sha(src) == sha(dst)
            existing.append(subject)
    cfg["subjects"], cfg["out_dir"] = selected, str(OUT)
    return cfg, selected, existing


def process(subject, space, cfg, work):
    try:
        segments = [seg for run in hcp.RUNS for seg in hcp.preprocess_run(
            space, resolve(cfg["raw_root"]), subject, run, cfg, work)]
        hcp.save_hcp_subject(OUT, subject, segments, cfg["label"])
        inspect_npz(OUT / f"sub-{subject}.npz")
        return {"subject": subject, "state": "completed"}
    except Exception:
        return {"subject": subject, "state": "failed", "error": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = _bootstrap.setup_logging(OUT / "preprocess.log")
    started = time.monotonic()
    try:
        cfg, selected, existing = prepare()
        missing = [s for s in selected if s not in existing]
        log.info("selected %d subjects; %d reusable; %d missing", len(selected), len(existing), len(missing))
        if args.prepare_only:
            print(json.dumps({"selected": len(selected), "reused": len(existing), "missing": len(missing)}), flush=True)
            return 0
        work = OUT / "work"
        space = hcp.make_space(cfg, resolve)
        geometry = space.prepare(resolve(cfg["raw_root"]), selected, work, args.n_jobs)
        with np.load(SOURCES[0] / "space.npz", allow_pickle=False) as old:
            assert all(np.array_equal(old[k], v) for k, v in geometry.items())
            assert np.array_equal(old["times"], hcp.sample_times(cfg))
        shutil.copy2(SOURCES[0] / "space.npz", OUT / "space.npz")
        _bootstrap.write_status(OUT / "completion.json", state="running", target=len(selected), done=len(existing))
        failures = []
        done = len(existing)
        results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
            delayed(process)(s, space, cfg, work) for s in missing)
        for result in results:
            if result["state"] == "completed":
                done += 1
            else:
                failures.append(result)
                log.error("subject failed: %s", result["error"])
            _bootstrap.write_status(OUT / "progress.json", done=done, target=len(selected), failures=failures)
            log.info("prepared %d/%d; failures %d", done, len(selected), len(failures))
        if failures:
            raise RuntimeError(f"{len(failures)} subjects failed; see progress.json")
        for s in selected:
            inspect_npz(OUT / f"sub-{s}.npz")
        _bootstrap.write_status(OUT / "completion.json", state="completed", subjects=done,
            samples=done * 64, n_space=4801, n_time=4, reused=len(existing), newly_processed=len(missing),
            elapsed_s=round(time.monotonic() - started, 2), model_fits_started=False,
            geometry_sha256=sha(OUT / "space.npz"))
        log.info("COMPLETED: %d subjects, %.1f seconds", done, time.monotonic() - started)
        return 0
    except Exception:
        _bootstrap.write_status(OUT / "completion.json", state="failed",
            elapsed_s=round(time.monotonic() - started, 2), error=traceback.format_exc())
        log.exception("preprocessing failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
