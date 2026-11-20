"""Preprocess ERP CORE N170 into per-subject ``(n, V, T)`` trial arrays.

Usage:  python code/scripts/01_preprocess_erpcore.py --config code/configs/erpcore_n170.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mlrdegg.datasets import erpcore
from mlrdegg.paths import resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--subjects", nargs="*", type=int, help="override the configured subjects")
    args = ap.parse_args()
    cfg = _bootstrap.load_config(args.config)["preprocess"]
    raw_root, out_dir = resolve(cfg["raw_root"]), resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _bootstrap.setup_logging(out_dir / "preprocess.log")
    subjects = args.subjects or (list(range(1, 41)) if cfg["subjects"] == "all" else cfg["subjects"])
    if cfg.get("ar_params"):
        cfg["ar_params"] = str(resolve(cfg["ar_params"]))
    log.info("ERP CORE N170: %d subjects -> %s", len(subjects), out_dir)

    def run(s):
        try:
            return erpcore.preprocess_subject(s, raw_root, out_dir, cfg)
        except Exception:  # recorded and reported below
            return {"subject": s, "error": traceback.format_exc()}

    results = Parallel(n_jobs=cfg.get("n_jobs", 1))(delayed(run)(s) for s in subjects)
    failed = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    for r in ok:
        q = r["quality"]
        log.info("sub-%03d kept %d/%d %s bad channels %s | accuracy %.2f, artifacts %.1f %% "
                 "(p2p %d, blink %d, heog %d)%s", r["subject"], r["n_kept"], r["n_candidates"],
                 r["counts"], r["bads"], q["accuracy"], 100 * q["artifact_fraction"],
                 q["n_flag_eeg_p2p"], q["n_flag_blink"], q["n_flag_heog"],
                 f" -> EXCLUDED ({q['reasons']})" if q["excluded"] else "")
    for r in failed:
        log.error("sub-%03d failed:\n%s", r["subject"], r["error"])
    if ok:
        ref = ok[0]
        for r in ok[1:]:
            if list(r["ch_names"]) != list(ref["ch_names"]) or not np.allclose(r["times"], ref["times"]):
                log.error("inconsistent channel layout or bins for sub-%03d", r["subject"])
                return 1
        erpcore.write_space(out_dir, ok)
        # One row per subject with the ERP CORE exclusion criteria, for checking the subject list.
        pd.DataFrame([{"subject": f"sub-{r['subject']:03d}", **r["quality"]} for r in ok]) \
            .round(4).to_csv(out_dir / "quality.csv", index=False)
    excluded = [r["subject"] for r in ok if r["quality"]["excluded"]]
    log.info("excluded by the ERP CORE criteria: %s", excluded)
    _bootstrap.write_status(out_dir / "preprocess_status.json", n_ok=len(ok),
                            failed=[r["subject"] for r in failed], excluded=excluded)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
