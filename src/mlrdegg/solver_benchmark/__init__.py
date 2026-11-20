"""B048 small-simulation solver benchmark (``review/solver_benchmark_plan.md``).

Separate from production model fitting and ``solver="auto"``.  Modules:

* ``core``: frozen problem and original objective, convex row functions, joint proximal maps,
  operation counters and the per-run algorithm clock (light imports for the timed worker).
* ``instances``: small simulated instances, frozen calibration and stored initial values.
* ``methods``: route registry and the M01-M24 adapters.
* ``evaluate``: the common original-objective evaluator (fixed S + P decomposition,
  certified mapping residual), convex dual lower bounds and offline test metrics.
* ``analysis``: run/trace/reference tables, accuracy-time figures and the result summary.

The package ``__init__`` deliberately imports nothing, so the timed worker processes load only
NumPy/SciPy and the solver modules.
"""
