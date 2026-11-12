"""Preprocess HCP-YA working-memory task fMRI into per-subject ``(n, V, T)`` segment arrays.

``preprocess.space`` selects the spatial representation: ``volume`` (MNI grid, default) or
``surface`` (cortex on an fsaverage mesh, needs Connectome Workbench).

Usage:  python code/scripts/02_preprocess_hcp.py --config code/configs/hcp_wm.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import _bootstrap  # noqa: F401
import numpy as np
from joblib import Parallel, delayed

from mlrdegg.datasets import hcp
from mlrdegg.paths import resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = _bootstrap.load_config(args.config)["preprocess"]
    raw_root, out_dir = resolve(cfg["raw_root"]), resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)
    log = _bootstrap.setup_logging(out_dir / "preprocess.log")
    subjects = [str(s) for s in cfg["subjects"]]
    n_jobs = cfg.get("n_jobs", 1)
    t0 = time.time()

    space = hcp.make_space(cfg, resolve)
    space_arrays = space.prepare(raw_root, subjects, work_dir, n_jobs)
    log.info("space %s, V = %d, extra smoothing %.1f mm FWHM", cfg.get("space", "volume"),
             len(space_arrays["coords"]), space.smooth_fwhm_mm)

    def run(s, r):
        try:
            return (s, r, hcp.preprocess_run(space, raw_root, s, r, cfg, work_dir))
        except Exception:
            return (s, r, traceback.format_exc())

    results = Parallel(n_jobs=n_jobs)(delayed(run)(s, r) for s in subjects for r in hcp.RUNS)
    failed = [(s, r, e) for s, r, e in results if isinstance(e, str)]
    for s, r, e in failed:
        log.error("sub-%s run %s failed:\n%s", s, r, e)
    if failed:
        return 1
    for s in subjects:
        segs = [seg for s2, _, segs in results if s2 == s for seg in segs]
        info = hcp.save_hcp_subject(out_dir, s, segs, cfg["label"])
        log.info("sub-%s: %d samples %s", s, info["n"], info["counts"])

    np.savez(out_dir / "space.npz", times=hcp.sample_times(cfg), **space_arrays)
    _bootstrap.write_status(out_dir / "preprocess_status.json", n_subjects=len(subjects),
                            n_voxels=len(space_arrays["coords"]), space=cfg.get("space", "volume"),
                            elapsed_s=round(time.time() - t0, 1))
    log.info("done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
