"""Run the remaining supplementary simulation studies one after another, then write the report.

Usage (from the project root):
    python code/scripts/12_run_simulation_studies.py [--n-jobs 30] [--dry-run]

Each batch is one call of ``10_simulation_studies.py``; a batch whose ``status.json`` says
"completed" is skipped, so after an interruption the same command resumes with the first
unfinished batch (that batch restarts from scratch).  Progress: the console and
``results/sim_studies/driver.log`` / ``driver_status.json`` show the batch being run, the
batches done, the elapsed time and an ETA from the estimated CPU minutes of the batches; each
batch also has its own progress bar and ``results/sim_studies/<batch>/status.json``.  At the end
``11_simulation_studies_report.py`` writes ``results/sim_studies/report/report.md``.  The exit
code is nonzero if any batch failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta

import _bootstrap  # noqa: F401

from mlrdegg.paths import PROJECT_ROOT, RESULTS_ROOT

CONFIG = "code/configs/simulation_studies.yaml"
# (study, seeds, conditions or None, estimated CPU minutes at one busy core per job)
BATCHES = [
    ("seeds", [2031, 2032], None, 160),
    ("seeds", [2033, 2034, 2035], None, 240),
    ("snr", [2026, 2027, 2028], ["amp0.5", "amp1.0"], 250),
    ("ntrain", [2026, 2027, 2028], ["n80"], 80),
    ("ntrain", [2026, 2027, 2028], ["n320"], 250),
    ("graph", [2026, 2027, 2028], ["drop50", "wnoise"], 230),
    ("graph", [2026, 2027, 2028], ["spur20", "wide"], 230),
]
SIM3D = ["sim_exponents", "sim_exponents_s2031"]


def batch_name(study, seeds, conds) -> str:
    return f"{study}_{seeds[0]}-{seeds[-1]}" + (f"_{'_'.join(conds)}" if conds else "")


def fmt(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def done(name: str) -> bool:
    p = RESULTS_ROOT / "sim_studies" / name / "status.json"
    return p.exists() and json.loads(p.read_text(encoding="utf-8")).get("state") == "completed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = RESULTS_ROOT / "sim_studies"
    out.mkdir(parents=True, exist_ok=True)
    log = _bootstrap.setup_logging(out / "driver.log")

    todo = [(b, batch_name(*b[:3])) for b in BATCHES]
    todo = [(b, n) for b, n in todo if not done(n)]
    total_cpu = sum(b[3] for b, _ in todo)
    log.info("%d of %d batches to run, estimated %.0f CPU minutes = about %s with %d jobs",
             len(todo), len(BATCHES), total_cpu, fmt(60 * total_cpu / args.n_jobs), args.n_jobs)
    for b, n in todo:
        log.info("  %s (%.0f CPU min)", n, b[3])
    if args.dry_run:
        return 0

    t0, cpu_done, failed = time.time(), 0.0, []
    for i, ((study, seeds, conds, cpu), name) in enumerate(todo):
        elapsed = time.time() - t0
        rate = elapsed / cpu_done if cpu_done else 60.0 / args.n_jobs  # s per CPU minute
        eta = rate * (total_cpu - cpu_done)
        status = {"batch": name, "batches_done": i, "batches_total": len(todo),
                  "percent": round(100 * cpu_done / total_cpu, 1), "elapsed": fmt(elapsed), "eta": fmt(eta),
                  "expected_finish": (datetime.now() + timedelta(seconds=eta)).isoformat(timespec="seconds"),
                  "failed": failed, "state": "running"}
        _bootstrap.write_status(out / "driver_status.json", **status)
        log.info("batch %d/%d %s: elapsed %s, ETA %s (%s)", i + 1, len(todo), name, status["elapsed"],
                 status["eta"], status["expected_finish"])
        cmd = [sys.executable, "code/scripts/10_simulation_studies.py", "--config", CONFIG, "--study", study,
               "--seeds", *map(str, seeds), "--name", name, "--n-jobs", str(args.n_jobs),
               "--time-limit-min", "600"]
        if conds:
            cmd += ["--conditions", *conds]
        rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
        if rc != 0:
            failed.append(name)
            log.error("batch %s failed with exit code %d (see results/sim_studies/%s/run.log)", name, rc, name)
        cpu_done += cpu

    sim3d = [r for r in SIM3D if (RESULTS_ROOT / r / "summary.csv").exists()]
    rc = subprocess.call([sys.executable, "code/scripts/11_simulation_studies_report.py", "--config", CONFIG,
                          "--sim3d", *sim3d], cwd=PROJECT_ROOT)
    if rc != 0:
        failed.append("report")
    _bootstrap.write_status(out / "driver_status.json", batches_done=len(todo), batches_total=len(todo),
                            percent=100.0, elapsed=fmt(time.time() - t0), eta=fmt(0), failed=failed,
                            state="completed" if not failed else "completed_with_errors")
    log.info("finished in %s; failed: %s", fmt(time.time() - t0), failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
