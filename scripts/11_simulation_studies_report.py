"""Report of the supplementary simulation studies (``10_simulation_studies.py``).

Usage:
    python code/scripts/11_simulation_studies_report.py --config code/configs/simulation_studies.yaml [--sim3d sim_exponents sim_exponents_s2031]

Collects ``results/sim_studies/*/fits.csv``, selects the grid point of every method, condition and
seed with the rules of ``05_simulation_benchmark.py`` (``min``: smallest validation log loss;
``1se_repro``: most reproducible point within one standard error of it; ``npairs``: closest to
perfect validation accuracy and split-half reproducibility), and compares methods (paired over
seeds) and conditions (each sweep condition against the base setting, reselected on the sweep
grid).  With ``--sim3d`` it also pools the selected fits (``summary.csv``) of 3D blob runs of
``05_simulation_benchmark.py`` over their seeds.

Outputs in ``results/sim_studies/report/``: ``report.md`` (all tables), ``selected.csv`` (every
selected fit) and ``paired.csv`` (every paired comparison).
"""

from __future__ import annotations

import argparse
import importlib
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats

from mlrdegg.experiment.models import grid_points
from mlrdegg.paths import RESULTS_ROOT

bench = importlib.import_module("05_simulation_benchmark")
studies_mod = importlib.import_module("10_simulation_studies")

RULES = ("min", "1se_repro", "npairs")
GRAPH_METHODS = ("graph_tv", "graph_tv_tm", "graph_p15", "graphnet", "superquadratic", "nonconvex")
REFERENCES = ("l2_logreg", "group_lasso")


def load_fits() -> pd.DataFrame:
    runs = sorted(p for p in (RESULTS_ROOT / "sim_studies").glob("*/fits.csv"))
    if not runs:
        raise SystemExit("no results/sim_studies/*/fits.csv found")
    fits = pd.concat([pd.read_csv(p).assign(run=p.parent.name) for p in runs], ignore_index=True)
    fits = fits[(fits.label == "grid") & (fits.error.fillna("") == "")]
    # A grid point can appear in two runs when a run hit its time limit before the split-half fits
    # and was repeated; keep the copy with the reproducibility.
    fits = fits.sort_values("repro", na_position="last", key=lambda r: r.notna(), ascending=False)
    dup = fits.duplicated(["condition", "seed", "method", "lambda_sel", "lambda_g", "C"])
    if dup.any():
        print(f"note: {int(dup.sum())} repeated fits (same condition, seed and grid point) dropped")
        fits = fits[~dup]
    return fits


def on_grid(fits: pd.DataFrame, method: dict) -> pd.DataFrame:
    """Fits of ``method`` at the points of its grid."""
    f = fits[fits.method == method["name"]]
    keep = np.zeros(len(f), bool)
    for p in grid_points(method):
        hit = np.ones(len(f), bool)
        for k, v in p.items():
            hit &= np.isclose(f[k].to_numpy(float), v)
        keep |= hit
    return f[keep]


def select_all(fits: pd.DataFrame, methods: list[dict], conditions: list[str], study: str) -> pd.DataFrame:
    rows = []
    for cond in conditions:
        fc = fits[fits.condition == cond]
        for m in methods:
            fm = on_grid(fc, m)
            n_grid = len(grid_points(m))
            for sd, ok in fm.groupby("seed"):
                if len(ok) < n_grid:
                    print(f"warning: {study}/{cond}/{m['name']}/seed {sd}: {len(ok)} of {n_grid} grid points")
                for rule in RULES:
                    r = bench.select(ok, m, rule).copy()
                    r["rule"], r["study"] = rule, study
                    for k, vals in (m.get("grid") or {}).items():
                        r[f"{k}_at_edge"] = bool(np.isclose(r[k], [min(vals), max(vals)]).any()) if len(vals) > 1 else False
                    rows.append(r)
    return pd.DataFrame(rows)


def mean_sd(x: pd.Series, digits: int = 3) -> str:
    x = x.dropna()
    if x.empty:
        return "–"
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}" if len(x) > 1 else f"{x.mean():.{digits}f}"


def paired(sel: pd.DataFrame, a: str, b: str, metric: str, by: str = "method") -> dict | None:
    """``a - b`` over seeds (``by`` = "method": two methods in one condition; "condition": one
    method in two conditions)."""
    if metric not in sel or sel[metric].isna().all():
        return None
    p = sel.pivot_table(index="seed", columns=by, values=metric)
    if a not in p or b not in p:
        return None
    d = (p[a] - p[b]).dropna()
    n = len(d)
    if n == 0:
        return None
    out = {"a": a, "b": b, "metric": metric, "n": n, "mean_diff": d.mean(), "wins": int((d > 0).sum()),
           "losses": int((d < 0).sum()), "ci95": np.nan, "p_wilcoxon": np.nan}
    if n > 2 and d.std(ddof=1) > 0:
        out["ci95"] = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    if n >= 6 and np.any(d != 0):
        out["p_wilcoxon"] = stats.wilcoxon(d, zero_method="zsplit").pvalue
    return out


def fmt_diff(r: dict | None) -> str:
    if r is None:
        return "–"
    s = f"{r['mean_diff']:+.3f}"
    if not np.isnan(r["ci95"]):
        s += f" ± {r['ci95']:.3f}"
    s += f" ({r['wins']}/{r['n']}"
    if not np.isnan(r["p_wilcoxon"]):
        s += f", p={r['p_wilcoxon']:.3f}"
    return s + ")"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def order(methods) -> list[str]:
    ref = ["l2_logreg", "group_lasso", "graph_tv", "graph_tv_tm", "graph_p15", "graphnet", "superquadratic",
           "nonconvex"]
    return sorted(set(methods), key=lambda m: ref.index(m) if m in ref else len(ref))


def seeds_section(sel: pd.DataFrame, slugs: list[str], pairs_out: list, title: str, metrics: list[str],
                  per_shape: bool, rules=RULES) -> list[str]:
    out = [f"## {title}", ""]
    n_seeds = sel.seed.nunique()
    out.append(f"{n_seeds} seeds ({sel.seed.min()}–{sel.seed.max()}); mean ± SD over seeds.")
    for rule in [r for r in rules if r in set(sel.rule)]:
        s = sel[sel.rule == rule]
        out += ["", f"### Rule `{rule}`", ""]
        tab = pd.DataFrame([{"method": m, **{k: mean_sd(s[s.method == m][k]) for k in metrics if k in s}}
                            for m in order(s.method)])
        out.append(md_table(tab))
        # Paired comparisons: every graph member against group lasso and L2, and graph members
        # against GraphNet.
        rows = []
        for m in order(s.method):
            for ref in REFERENCES + ("graphnet",):
                if m == ref or (ref == "graphnet" and m not in GRAPH_METHODS):
                    continue
                if ref in REFERENCES and m in REFERENCES and REFERENCES.index(m) < REFERENCES.index(ref):
                    continue
                cells = {"comparison": f"{m} − {ref}"}
                for k in ("test_accuracy", "map_corr", "f1"):
                    r = paired(s, m, ref, k)
                    if r:
                        pairs_out.append({"section": title, "rule": rule, **r})
                    cells[k] = fmt_diff(r)
                rows.append(cells)
        out += ["", "Paired differences over seeds: mean ± 95 % CI (wins/seeds, Wilcoxon p):", "",
                md_table(pd.DataFrame(rows))]
        if per_shape:
            shp = pd.DataFrame([{"method": m, **{sl: mean_sd(s[s.method == m][f"corr_{sl}"], 2)
                                                 for sl in slugs},
                                 **{f"bg {sl}": mean_sd(s[s.method == m][f"bg_{sl}"], 2) for sl in slugs}}
                                for m in order(s.method)])
            out += ["", "Per-shape map correlation and background energy fraction:", "", md_table(shp)]
    return out + [""]


def sweep_section(sel: pd.DataFrame, conds: list[str], base_label: str, pairs_out: list, title: str) -> list[str]:
    """Rows: methods; columns: conditions (base in its natural position); plus condition effects."""
    out = [f"## {title}", ""]
    seeds = sorted(sel.seed.unique())
    out.append(f"Seeds {seeds}; base = {base_label}, reselected on the sweep grid. Mean ± SD over seeds.")
    for rule in RULES:
        s = sel[sel.rule == rule]
        out += ["", f"### Rule `{rule}`", ""]
        for metric in ("test_accuracy", "map_corr", "f1"):
            tab = pd.DataFrame([{"method": m, **{c: mean_sd(s[(s.method == m) & (s.condition == c)][metric])
                                                 for c in conds}} for m in order(s.method)])
            out += [f"`{metric}`:", "", md_table(tab), ""]
        # Gains of the graph members over group lasso within each condition (sweep studies with
        # the reference methods), or the change against the base graph (graph study).
        rows = []
        for m in order(s.method):
            if m not in GRAPH_METHODS:
                continue
            for c in conds:
                if c == "base":
                    continue
                cells = {"method": m, "condition": c}
                for k in ("test_accuracy", "map_corr"):
                    r = paired(s[s.method == m], c, "base", k, by="condition")
                    if r:
                        pairs_out.append({"section": title, "rule": rule, "method": m, **r})
                    cells[f"{k}: {c} − base"] = fmt_diff(r)
                    if "group_lasso" in set(s.method):
                        g = paired(s[s.condition == c], m, "group_lasso", k)
                        if g:
                            pairs_out.append({"section": title, "rule": rule, "condition": c, **g})
                        cells[f"{k}: − group_lasso"] = fmt_diff(g)
                rows.append(cells)
        if rows:
            out += ["Paired differences over seeds, mean ± 95 % CI (wins/seeds):", "", md_table(pd.DataFrame(rows)), ""]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sim3d", nargs="*", default=[], help="runs of 05_simulation_benchmark.py (3D blobs)")
    args = ap.parse_args()
    cfg = _bootstrap.load_config(args.config)
    out_dir = RESULTS_ROOT / "sim_studies" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    fits = load_fits()
    base_cfg, base_methods = studies_mod.study_setup(cfg, cfg["studies"]["seeds"])
    slugs = studies_mod.shape_slugs(base_cfg["simulation"])
    lines = ["# Supplementary simulation studies", "",
             f"Runs: {', '.join(sorted(fits.run.unique()))}.", ""]
    selected, pairs_out = [], []
    metrics = ["test_accuracy", "test_log_loss", "map_corr", "repro", "support_auc", "f1",
               "n_selected", "fit_time"]

    # 1. Base setting over all seeds.
    sel = select_all(fits, base_methods, ["base"], "seeds")
    selected.append(sel)
    lines += seeds_section(sel, slugs, pairs_out, "2D shapes, base setting", metrics, per_shape=True)
    edge = sel.groupby(["rule", "method"])[[c for c in sel if c.endswith("_at_edge")]].mean().round(2)
    lines += ["Fraction of seeds whose selected point lies on a grid edge:", "",
              md_table(edge.reset_index()), ""]

    # 2. Sweeps against the base setting on the sweep grid.
    for name, title in (("snr", "Signal amplitude (base 0.7)"), ("ntrain", "Training-set size (base 160)"),
                        ("graph", "Misspecified spatial graph (base: true lattice)")):
        study = cfg["studies"][name]
        conds = [c["name"] for c in study["conditions"]]
        if not set(conds) & set(fits.condition):
            lines += [f"## {title}", "", "not run yet", ""]
            continue
        _, methods = studies_mod.study_setup(cfg, study)
        seeds = sorted(fits[fits.condition.isin(conds)].seed.unique())
        sel = select_all(fits[fits.seed.isin(seeds)], methods, ["base"] + conds, name)
        if name == "graph":  # graph-free references on the same seeds, for "is a wrong graph still useful?"
            _, refs = studies_mod.study_setup(cfg, {**study, "methods": list(REFERENCES)})
            sel = pd.concat([sel, select_all(fits[fits.seed.isin(seeds)], refs, ["base"], name)])
        selected.append(sel)
        order_c = {"snr": ["amp0.5", "base", "amp1.0"], "ntrain": ["n80", "base", "n320"]}.get(name, ["base"] + conds)
        lines += sweep_section(sel, order_c, {"snr": "amplitude 0.7", "ntrain": "n_train 160"}.get(name, "true lattice"),
                               pairs_out, title)
        if name == "graph":
            refs_tab = sel[(sel.method.isin(REFERENCES)) & (sel.condition == "base")]
            lines += ["Graph-free references on the same seeds (base data):", "",
                      md_table(pd.DataFrame([{"rule": r, "method": m,
                                              "test_accuracy": mean_sd(g.test_accuracy),
                                              "map_corr": mean_sd(g.map_corr), "f1": mean_sd(g.f1)}
                                             for (r, m), g in refs_tab.groupby(["rule", "method"])])), ""]
            gr = pd.concat([pd.read_csv(p) for p in (RESULTS_ROOT / "sim_studies").glob("graph_*/graphs.csv")])
            lines += ["Graphs (mean over seeds):", "",
                      md_table(gr.groupby("condition")[["n_edges", "mean_degree", "n_components", "weight_sum"]]
                               .mean().round(1).reset_index()), ""]

    # 3. 3D blobs over pooled seeds.
    if args.sim3d:
        s3 = pd.concat([pd.read_csv(RESULTS_ROOT / r / "summary.csv").assign(run=r) for r in args.sim3d])
        s3["rule"] = s3.rule.replace({"1se": "1se_sparsest"})  # the older runs' name of this rule
        s3 = s3[s3.rule.isin(["min", "1se_sparsest"])].drop_duplicates(["seed", "rule", "method"])
        lines += [f"## 3D blobs ({', '.join(args.sim3d)})", "",
                  "Rules: `min` and `1se_sparsest` (fewest selected voxels within one standard error of the "
                  "smallest validation log loss; called `1se` in older runs).  No split-half fits.", ""]
        sec = seeds_section(s3.assign(study="sim3d"), [], pairs_out, "3D blobs",
                            ["test_accuracy", "test_log_loss", "support_auc", "f1", "precision", "recall",
                             "n_selected", "fit_time"], per_shape=False, rules=("min", "1se_sparsest"))
        lines += sec[1:]
        selected.append(s3.assign(study="sim3d", condition="base"))

    pd.concat(selected, ignore_index=True).to_csv(out_dir / "selected.csv", index=False)
    pd.DataFrame(pairs_out).to_csv(out_dir / "paired.csv", index=False)
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
