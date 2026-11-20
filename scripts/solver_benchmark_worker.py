"""Runs one B048 benchmark job in its own process (started by ``22_solver_benchmark.py``).

Usage: python code/scripts/solver_benchmark_worker.py <run_dir>/job.json

The ``mlrdegg`` package is registered without executing its ``__init__`` (which imports
scikit-learn), so each timed process loads only NumPy/SciPy and the solver modules.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if "mlrdegg" not in sys.modules:
    package = types.ModuleType("mlrdegg")
    package.__path__ = [str(SRC / "mlrdegg")]
    sys.modules["mlrdegg"] = package

from mlrdegg.solver_benchmark.core import read_json  # noqa: E402
from mlrdegg.solver_benchmark.methods import run_job  # noqa: E402


def main() -> int:
    job_path = Path(sys.argv[1])
    run_job(read_json(job_path), job_path.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
