"""Report a decoding run with several selection rules (experiment.selection_rules).

For every rule and method: accuracy, log loss, split-half reproducibility R of the weight maps,
cross-fold importance-map stability, row sparsity and the share above the graph grid minimum;
paired Wilcoxon tests against group_lasso under the same rule (Holm-corrected over methods).
R is taken from ``fresh_repro_seed<seed>.csv`` (``08_fresh_reproducibility.py``: selected points
refitted on a new partition of the same training data, reducing split-specific optimism) when it
exists, otherwise from the selection itself (marked "selection R").

The report is written for all subjects and, when ``experiment.development_subjects`` is set,
also for the subjects outside the development subset (the confirmation sample).

Usage:
    python code/scripts/09_selection_rules_report.py --config code/configs/erpcore_n170.yaml [--fresh-seed 5003]

Output: ``results/<name>/report/`` with ``report.md``, ``table_<subset>.csv`` and
``tests_<subset>.csv``.
"""

from __future__ import annotations

import argparse
import itertools
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mlrdegg.paths import RESULTS_ROOT, resolve
from mlrdegg.experiment.reporting import prediction_support_from_maps

REFERENCE = "group_lasso"


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def map_stability(out) -> pd.DataFrame:
    """Mean pairwise correlation of the outer-fold importance maps, per unit and method."""
    rows = []
    for f in sorted((out / "units").glob("*_maps.npz")):
        unit = f.name.removesuffix("_maps.npz")
        with np.load(f) as z:
            for m in z.files:
                M = z[m].reshape(len(z[m]), -1)
                cs = [np.corrcoef(M[a], M[b])[0, 1] for a, b in itertools.combinations(range(len(M)), 2)
                      if M[a].std() > 0 and M[b].std() > 0]
                rows.append({"unit": unit, "method": m, "stability": np.mean(cs) if cs else np.nan})
    return pd.DataFrame(rows)


def base_and_rule(method: str, first_rule: str) -> tuple[str, str]:
    base, _, rule = method.partition("@")
    return base, rule or first_rule


def report(df: pd.DataFrame, rules: list[str], order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit = df.groupby(["unit", "method"]).agg(
        accuracy=("accuracy", "mean"), log_loss=("log_loss", "mean"), R=("R", "mean"),
        stability=("stability", "first"), zeros=("sel_zero_fraction", "mean"),
        graph_above_min=("graph_above_min", "mean")).reset_index()
    table, tests = [], []
    for rule in rules:
        for base in order:
            name = base if rule == rules[0] else f"{base}@{rule}"
            g = unit[unit.method == name]
            if g.empty:
                continue
            row = {"rule": rule, "method": base, "n": len(g)}
            for c in ("accuracy", "log_loss", "R", "stability", "zeros", "graph_above_min"):
                row[c], row[f"{c}_sd"] = g[c].mean(), g[c].std(ddof=1)
            table.append(row)
        ref = REFERENCE if rule == rules[0] else f"{REFERENCE}@{rule}"
        r = unit[unit.method == ref].set_index("unit")
        block = []
        for base in order:
            name = base if rule == rules[0] else f"{base}@{rule}"
            if base == REFERENCE or name not in set(unit.method):
                continue
            x = unit[unit.method == name].set_index("unit")
            for c in ("accuracy", "R", "stability"):
                d = (x[c] - r[c]).dropna()
                p = wilcoxon(d).pvalue if len(d) and np.any(d != 0) else 1.0
                block.append({"rule": rule, "method": base, "metric": c, "diff": d.mean(),
                              "wins": int((d > 0).sum()), "n": len(d), "p": p})
        block = pd.DataFrame(block)
        if not block.empty:
            for c in ("accuracy", "R", "stability"):
                sel = block.metric == c
                block.loc[sel, "p_holm"] = holm(block.loc[sel, "p"].to_numpy())
            tests.append(block)
    return pd.DataFrame(table), (pd.concat(tests) if tests else pd.DataFrame())


def markdown(table: pd.DataFrame, tests: pd.DataFrame, title: str, r_label: str) -> list[str]:
    lines = [f"## {title}", "",
             f"| rule | method | n | accuracy | log-loss | {r_label} | stability | row zeros | graph above grid minimum |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.rule} | {r.method} | {r.n} | {r.accuracy:.3f} ± {r.accuracy_sd:.3f} | "
                     f"{r.log_loss:.3f} | {r.R:.3f} | {r.stability:.3f} | {r.zeros:.2f} | "
                     + ("—" if np.isnan(r.graph_above_min) else f"{r.graph_above_min:.0%}") + " |")
    lines += ["", f"Paired differences to {REFERENCE} under the same rule (Wilcoxon, Holm over methods):", "",
              "| rule | method | Δ accuracy (wins, p_holm) | Δ R (wins, p_holm) | Δ stability (wins, p_holm) |",
              "|---|---|---|---|---|"]
    if not tests.empty:
        for (rule, method), g in tests.groupby(["rule", "method"], sort=False):
            cell = {r.metric: f"{r.diff:+.3f} ({r.wins}/{r.n}, {r.p_holm:.2g})" for r in g.itertuples()}
            lines.append(f"| {rule} | {method} | {cell.get('accuracy', '')} | {cell.get('R', '')} | "
                         f"{cell.get('stability', '')} |")
    return lines + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name")
    ap.add_argument("--fresh-seed", type=int, default=5003)
    args = ap.parse_args()
    cfg = _bootstrap.load_config(args.config)
    exp = cfg["experiment"]
    out = RESULTS_ROOT / (args.name or exp["name"])
    rdir = out / "report"
    rdir.mkdir(exist_ok=True)
    rules = list(exp.get("selection_rules", ["min"]))
    order = [m["name"] for m in exp["methods"]]
    smallest_g = {m["name"]: min(m["grid"]["lambda_g"]) for m in exp["methods"]
                  if "lambda_g" in (m.get("grid") or {})}

    df = pd.read_csv(out / "outer_results.csv")
    df = df[df["error"].isna() | (df["error"].astype(str) == "")].copy()
    support = prediction_support_from_maps(out, df)
    df = df.drop(columns=["sel_zero_fraction", "sel_aux_zero_fraction"], errors="ignore").merge(
        support, on=["unit", "outer_fold", "method"], validate="one_to_one")
    base = df.method.map(lambda m: base_and_rule(m, rules[0])[0])
    df["graph_above_min"] = [float(lg > smallest_g[b]) if b in smallest_g else np.nan
                      for b, lg in zip(base, df.get("lambda_g", pd.Series(np.nan, index=df.index)))]
    fresh = out / f"fresh_repro_seed{args.fresh_seed}.csv"
    if fresh.exists():
        df = df.merge(pd.read_csv(fresh), on=["unit", "outer_fold", "method"], how="left")
        df["R"], r_label = df["fresh_repro"], "R (fresh halves)"
    else:
        df["R"], r_label = df.get("repro", np.nan), "selection R"
    df = df.merge(map_stability(out), on=["unit", "method"], how="left")

    def norm(s) -> str:
        return f"sub-{int(s):03d}" if str(s).isdigit() else str(s)

    dev = {norm(s) for s in exp.get("development_subjects", [])}
    subsets = [("all", "All subjects", df)]
    if dev:
        subsets.append(("confirmation", "Subjects outside the development subset",
                        df[~df.unit.isin(dev)]))
    lines = [f"# Selection-rule report: {out.name}", "",
             f"Rules: {', '.join(rules)}.  Reproducibility: {r_label}.  Stability: mean correlation "
             "of the outer-fold importance maps.  Graph above grid minimum: share of folds with "
             "lambda_g above its smallest grid value; the graph remains active at a positive minimum.", ""]
    for key, title, sub in subsets:
        table, tests = report(sub, rules, order)
        table.to_csv(rdir / f"table_{key}.csv", index=False)
        tests.to_csv(rdir / f"tests_{key}.csv", index=False)
        lines += markdown(table, tests, f"{title} (n = {sub.unit.nunique()})", r_label)
    (rdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
