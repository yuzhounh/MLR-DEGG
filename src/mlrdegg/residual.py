"""Post-fit common residual of the original objective (diagnostic only; never a stopping rule).

After a fit, the fixed S + P decomposition of the original objective gives the mapping residual

    r = ||Theta - prox_{alpha P}(Theta - alpha grad S(Theta))||_F / alpha / max(1, ||grad S(Theta_null)||_F)

with ``alpha = 1 / (2 L_S)`` (the common evaluator of ``review/solver_benchmark_plan.md``).  The
joint prox is solved within a time budget and an error bound is attached; a band ``b`` is
``pass`` when the estimate is at most ``b`` and the bound at most ``b/100``, ``fail`` when the
estimate minus the bound exceeds ``b``, else ``unresolved``.  The solver's own stopping rules and
the fitted coefficients are unchanged.  A small residual measures first-order accuracy; it does
not certify local or global optimality.
"""

from __future__ import annotations

import time

import numpy as np

from .solver import Branch

BANDS = (1e-3, 1e-5)
FIELDS = ("common_residual", "common_residual_error_bound", "common_residual_bound_type",
          "common_residual_1e-03", "common_residual_1e-05", "common_residual_status",
          "common_residual_seconds")


def common_residual(Xa, Y, H, branches: list[Branch], Theta, Theta_null, epsilon: float,
                    budget: float = 2.0) -> dict:
    """Common residual of a fitted ``Theta`` as flat, report-ready fields."""
    t0 = time.perf_counter()
    try:
        from .solver_benchmark.core import Problem
        from .solver_benchmark.evaluate import Evaluator

        frozen = [Branch(br.name, br.op, br.mu, br.p, br.offset(epsilon)) for br in branches]
        problem = Problem({"Xa": Xa, "Y": Y, "H": H, "branches": frozen, "theta0": Theta,
                           "theta_null": Theta_null, "meta": {}})
        info, _ = Evaluator(problem).residual(np.asarray(Theta, dtype=float), budget, BANDS)
        return {"common_residual": info["r_orig_estimate"],
                "common_residual_error_bound": info["mapping_error_bound"],
                "common_residual_bound_type": info["bound_type"],
                "common_residual_1e-03": info["residual_1e-03"],
                "common_residual_1e-05": info["residual_1e-05"],
                "common_residual_status": info["residual_status"],
                "common_residual_seconds": time.perf_counter() - t0}
    except Exception as err:  # a diagnostic must never break a fit
        return {"common_residual": np.nan, "common_residual_error_bound": np.nan,
                "common_residual_bound_type": f"error: {type(err).__name__}: {err}"[:200],
                "common_residual_1e-03": "unresolved", "common_residual_1e-05": "unresolved",
                "common_residual_status": "error",
                "common_residual_seconds": time.perf_counter() - t0}


def summarize(rows) -> str:
    """One-line count of residual bands over fit rows (for run logs)."""
    rows = [r for r in rows if isinstance(r, dict) and "common_residual_1e-03" in r]
    if not rows:
        return "common residual: not recorded"
    out = []
    for b in ("1e-03", "1e-05"):
        vals = [r.get(f"common_residual_{b}") for r in rows]
        out.append(f"{b}: pass {vals.count('pass')}, fail {vals.count('fail')}, "
                   f"unresolved {len(vals) - vals.count('pass') - vals.count('fail')}")
    return f"common residual over {len(rows)} fits ({'; '.join(out)})"
