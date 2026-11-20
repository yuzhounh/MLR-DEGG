"""Reference values, accuracy-time tables, figures and the result summary (plan Sections 6-11).

References.  Convex instances: the best-known value is the lowest original objective among all
evaluated states; it is certified when the best numerically feasible dual lower bound is within
the target relative gap (otherwise ``reference_unresolved``).  A conic optimal status or two close
objective values alone never certify it.  Nonconvex instances: the best-known stationary point
is the lowest objective among states whose common residual passes 1e-5; it is a best-known
stationary point, not a global optimum.

Time to accuracy is the algorithm time of the first stored checkpoint (geometric, last five or
final) that meets the band, so it is an upper bound at checkpoint resolution.  Test data are
read only here, offline, at stored checkpoints.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Problem, load_instance, plain, read_json, write_json
from .methods import METHODS

OBJ_BANDS = (1e-3, 1e-5, 1e-7)
RES_BANDS = (1e-3, 1e-5)
# Pre-specified conventions for the automatic indicators of Section 8 (interpretation aids only).
LOGLOSS_UNCHANGED = 1e-3
CLOSE_OBJECTIVE = 1e-6
CLOSE_PROBABILITY = 1e-3


def _band(b: float) -> str:
    return f"{b:.0e}"


def load_runs(out: Path) -> list[dict]:
    runs = []
    for d in sorted((Path(out) / "runs").iterdir()) if (Path(out) / "runs").exists() else []:
        if not (d / "job.json").exists() or not (d / "result.json").exists():
            continue
        run = {"dir": d, "job": read_json(d / "job.json"), "result": read_json(d / "result.json")}
        run["eval"] = read_json(d / "eval.json") if (d / "eval.json").exists() else {"rows": []}
        runs.append(run)
    return runs


class Instances:
    def __init__(self, out: Path):
        self.dir = Path(out) / "instances"
        self.meta = {m["id"]: m for m in read_json(self.dir / "instances.json")}
        self._problems = {}

    def problem(self, iid: str) -> Problem:
        if iid not in self._problems:
            self._problems[iid] = Problem(load_instance(self.dir / f"{iid}.npz"))
        return self._problems[iid]


def _state_rows(run: dict) -> list[dict]:
    return [r for r in run["eval"].get("rows", []) if r and r.get("finite", True) and "objective" in r]


def compute_references(runs: list[dict], insts: Instances, target: float) -> dict:
    by_inst: dict[str, list] = {}
    for run in runs:
        by_inst.setdefault(run["job"]["instance"], []).append(run)
    refs = {}
    for iid, rs in by_inst.items():
        meta = insts.meta[iid]
        states = [(run, row) for run in rs for row in _state_rows(run)]
        ref = {"instance": iid, "scale": meta["scale"], "seed": meta["seed"], "point": meta["point"],
               "strength": meta["strength"], "nonconvex": meta["nonconvex"], "n_states": len(states)}
        if not states:
            ref["reference_status"] = "no_evaluated_state"
            refs[iid] = ref
            continue
        if not meta["nonconvex"]:
            run, row = min(states, key=lambda s: s[1]["objective"])
            lbs = [(r, w) for r, w in states if w.get("dual_lower_bound") is not None]
            ref["best_known_objective"] = row["objective"]
            ref["best_known_source"] = run["job"]["run_id"]
            ref["best_known_file"] = str(run["dir"] / row["file"])
            if lbs:
                rl, wl = max(lbs, key=lambda s: s[1]["dual_lower_bound"])
                ref["dual_lower_bound"] = wl["dual_lower_bound"]
                ref["dual_lower_bound_source"] = rl["job"]["run_id"]
                gap = (row["objective"] - wl["dual_lower_bound"]) / max(1.0, abs(row["objective"]))
                ref["certified_relative_gap"] = gap
                ref["reference_status"] = "certified" if gap <= target else "reference_unresolved"
            else:
                ref["reference_status"] = "reference_unresolved"
            conic = [r for r in rs if r["job"]["method"] == "M17"]
            for r in conic:
                fin = [w for w in _state_rows(r) if w.get("kind") == "final"]
                ref["conic_exit"] = r["result"].get("exit_status")
                ref["conic_native_status"] = r["result"].get("native_status")
                ref["conic_representation_abs_difference"] = r["result"].get("representation_abs_difference")
                if fin:
                    ref["conic_objective"] = fin[0]["objective"]
                    if fin[0].get("dual_lower_bound") is not None:
                        ref["conic_own_relative_gap"] = ((fin[0]["objective"] - fin[0]["dual_lower_bound"])
                                                         / max(1.0, abs(fin[0]["objective"])))
        else:
            ok5 = [(r, w) for r, w in states if w.get("residual_1e-05") == "pass"]
            ok3 = [(r, w) for r, w in states if w.get("residual_1e-03") == "pass"]
            pool, level = (ok5, "best_known_stationary_residual_1e-05") if ok5 else (
                (ok3, "best_known_residual_1e-03_only") if ok3 else ([], "reference_unresolved"))
            ref["reference_status"] = level
            if pool:
                run, row = min(pool, key=lambda s: s[1]["objective"])
                ref["best_known_objective"] = row["objective"]
                ref["best_known_source"] = run["job"]["run_id"]
                ref["best_known_file"] = str(run["dir"] / row["file"])
                ref["best_known_residual"] = row.get("r_orig_estimate")
            low = min(states, key=lambda s: s[1]["objective"])
            ref["lowest_objective_any_state"] = low[1]["objective"]
            ref["lowest_objective_any_state_source"] = low[0]["job"]["run_id"]
        refs[iid] = ref
    return refs


def _theta(path) -> np.ndarray:
    return np.load(path)


def trace_table(runs: list[dict], refs: dict, insts: Instances) -> pd.DataFrame:
    rows = []
    ref_cache = {}
    for run in runs:
        job, res = run["job"], run["result"]
        iid = job["instance"]
        ref = refs.get(iid, {})
        problem = insts.problem(iid)
        if iid not in ref_cache:
            th_ref = _theta(ref["best_known_file"]) if ref.get("best_known_file") else None
            P_ref = problem.probabilities(problem.inst["Xte"], th_ref) if th_ref is not None else None
            ref_cache[iid] = (th_ref, P_ref)
        th_ref, P_ref = ref_cache[iid]
        F_ref = ref.get("best_known_objective")
        LB = ref.get("dual_lower_bound")
        for row in _state_rows(run):
            out = {"run_id": job["run_id"], "phase": job["phase"], "instance": iid,
                   "method": job["method"], "variant": job.get("variant", ""),
                   "kind": row.get("kind"), "last_five": row.get("last_five"),
                   "solver_seconds": row.get("solver_seconds"), "iteration": row.get("iteration"),
                   "stage": row.get("stage"), "objective": row["objective"],
                   "native_objective": row.get("native_objective"),
                   "r_orig": row.get("r_orig_estimate"), "residual_error_bound": row.get("mapping_error_bound"),
                   "residual_bound_type": row.get("bound_type"),
                   "residual_status": row.get("residual_status"),
                   "test_log_loss": row.get("test_log_loss"), "test_accuracy": row.get("test_accuracy"),
                   "evaluation_seconds": row.get("evaluation_seconds")}
            for b in RES_BANDS:
                out[f"residual_{_band(b)}"] = row.get(f"residual_{_band(b)}")
            if F_ref is not None:
                out["relative_objective_gap"] = (row["objective"] - F_ref) / max(1.0, abs(F_ref))
            if LB is not None:
                out["optimality_upper_bound"] = (row["objective"] - LB) / max(1.0, abs(row["objective"]))
            if th_ref is not None:
                th = _theta(run["dir"] / row["file"])
                out["coefficient_relative_difference"] = float(
                    np.linalg.norm(th[:-1] - th_ref[:-1]) / max(np.linalg.norm(th_ref[:-1]), 1e-12))
                P = problem.probabilities(problem.inst["Xte"], th)
                out["probability_tv_difference"] = float(0.5 * np.mean(np.sum(np.abs(P - P_ref), axis=1)))
            for key, value in (row.get("operations") or {}).items():
                out[f"op_{key}"] = value
            rows.append(out)
    return pd.DataFrame(rows)


def time_to_bands(tr: pd.DataFrame, nonconvex: bool) -> dict:
    tr = tr.sort_values("solver_seconds")
    out = {}

    def first(mask):
        hit = tr[mask]
        return float(hit["solver_seconds"].iloc[0]) if len(hit) else None

    gap = tr.get("relative_objective_gap")
    for b in RES_BANDS:
        res_ok = tr[f"residual_{_band(b)}"] == "pass"
        out[f"t_res_{_band(b)}"] = first(res_ok)
        if gap is not None and not nonconvex:
            out[f"t_obj_res_{_band(b)}"] = first(res_ok & (gap <= b))
    if gap is not None:
        out["t_obj_1e-07"] = first(gap <= 1e-7)
    return out


def run_table(runs: list[dict], traces: pd.DataFrame, refs: dict, insts: Instances) -> pd.DataFrame:
    rows = []
    groups = dict(tuple(traces.groupby("run_id"))) if len(traces) else {}
    for run in runs:
        job, res = run["job"], run["result"]
        meta = insts.meta[job["instance"]]
        row = {"run_id": job["run_id"], "phase": job["phase"], "instance": job["instance"],
               "scale": meta["scale"], "seed": meta["seed"], "point": meta["point"],
               "strength": meta["strength"], "nonconvex": meta["nonconvex"],
               "method": job["method"], "method_name": METHODS[job["method"]][0],
               "objective_kind": METHODS[job["method"]][1], "variant": job.get("variant", ""),
               "init": job.get("init", "common"), "cap_seconds": job["cap"],
               "exit_status": res.get("exit_status"), "native_status": res.get("native_status"),
               "reason": res.get("reason"), "solver_seconds": res.get("solver_seconds"),
               "process_seconds": res.get("process_seconds"),
               "monitoring_seconds": res.get("monitoring_seconds"),
               "peak_memory_mb": res.get("peak_memory_mb"),
               "n_kkt_violations": res.get("n_kkt_violations"),
               "uniform_bias_bound": res.get("uniform_bias_bound"),
               "fine_ratio": job.get("fine_ratio"),
               "reference_status": refs.get(job["instance"], {}).get("reference_status")}
        mon = res.get("objective_monitor") or {}
        row.update({"objective_increases": mon.get("increases"),
                    "max_relative_objective_increase": mon.get("max_relative_increase"),
                    "objective_monitor_skipped": mon.get("skipped")})
        warm = meta.get("warm_start") or {}
        row["instance_preparation_seconds"] = meta.get("calibration_seconds", 0.0) + (warm.get("seconds") or 0.0)
        for key, value in (res.get("operations") or {}).items():
            row[f"op_{key}"] = value
        tr = groups.get(job["run_id"])
        if tr is not None and len(tr):
            row["iterations"] = pd.to_numeric(tr["iteration"], errors="coerce").max()
            row["evaluation_seconds_total"] = pd.to_numeric(tr["evaluation_seconds"], errors="coerce").sum()
            if row["solver_seconds"] is not None and not pd.isna(row["solver_seconds"]):
                row["end_to_end_seconds"] = (row["solver_seconds"] + row["instance_preparation_seconds"]
                                             + row["evaluation_seconds_total"])
            final = tr[tr["kind"] == "final"]
            if len(final):
                f = final.iloc[-1]
                for key in ("objective", "relative_objective_gap", "optimality_upper_bound", "r_orig",
                            "residual_error_bound", "residual_bound_type", "residual_status",
                            "residual_1e-03", "residual_1e-05", "test_log_loss", "test_accuracy",
                            "coefficient_relative_difference", "probability_tv_difference",
                            "native_objective"):
                    if key in f:
                        row[f"final_{key}"] = f[key]
            row.update(time_to_bands(tr, meta["nonconvex"]))
        rows.append(row)
    return pd.DataFrame(rows)


def tv_pairing(rt: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b = cfg["phases"]["B"]
    sub = rt[(rt["phase"] == "B") & (rt["point"] == "C1")]
    rows = []
    n_inst = sub["instance"].nunique()
    for base in ("M01", "M02"):
        for method in b["tv_methods"]:
            if method == base:
                continue
            for band in RES_BANDS:
                col = f"t_obj_res_{_band(band)}"
                if col not in sub:
                    continue
                m = sub[sub["method"] == method].set_index("instance")[col]
                bb = sub[sub["method"] == base].set_index("instance")[col]
                both = [i for i in m.index if i in bb.index and pd.notna(m[i]) and pd.notna(bb[i])]
                ratios = [m[i] / bb[i] for i in both if bb[i] > 0]
                med = float(np.median(ratios)) if ratios else None
                rows.append({"method": method, "baseline": base, "band": band,
                             "instances": n_inst, "method_reached": int(m.notna().sum()),
                             "baseline_reached": int(bb.notna().sum()), "both_reached": len(both),
                             "median_time_ratio": med,
                             "min_ratio": float(min(ratios)) if ratios else None,
                             "max_ratio": float(max(ratios)) if ratios else None,
                             "speedup_signal": bool(len(both) == n_inst and n_inst > 0 and med is not None
                                                    and med <= 0.5)})
    return pd.DataFrame(rows)


def nonconvex_attribution(rt: pd.DataFrame, refs: dict) -> pd.DataFrame:
    rows = []
    sub = rt[(rt["point"] == "N3") & rt["phase"].isin(["B", "B-cont"])]
    for iid, g in sub.groupby("instance"):
        meta = g.iloc[0]
        d = g[(g["method"] == "M01") & (g["phase"] == "B")]
        c = g[g["phase"] == "B-cont"]
        s = g[(g["method"] == "M02") & (g["phase"] == "B")]
        if d.empty:
            continue
        d = d.iloc[0]
        row = {"instance": iid, "seed": meta["seed"], "strength": meta["strength"],
               "reference_status": refs.get(iid, {}).get("reference_status"),
               "best_known_objective": refs.get(iid, {}).get("best_known_objective"),
               "best_known_source": refs.get(iid, {}).get("best_known_source")}
        for tag, r in (("default", d), ("continuation", c.iloc[0] if len(c) else None),
                       ("strict_common_init", s.iloc[0] if len(s) else None)):
            if r is None:
                continue
            for key in ("exit_status", "native_status", "solver_seconds", "final_objective",
                        "final_relative_objective_gap", "final_r_orig", "final_residual_1e-03",
                        "final_residual_1e-05", "final_test_log_loss", "final_test_accuracy",
                        "final_probability_tv_difference"):
                row[f"{tag}_{key}"] = r.get(key)
        if len(c):
            row["continuation_cumulative_seconds"] = float(d["solver_seconds"] + c.iloc[0]["solver_seconds"])
        convex = rt[(rt["phase"] == "B") & (rt["point"] == "C1") & (rt["seed"] == meta["seed"])
                    & (rt["strength"] == meta["strength"])]
        cref = refs.get(convex["instance"].iloc[0], {}) if len(convex) else {}
        if cref.get("best_known_source"):
            cc = convex[convex["run_id"] == cref["best_known_source"]]
            if len(cc):
                row["convex_control_test_log_loss"] = cc.iloc[0].get("final_test_log_loss")
                row["convex_control_test_accuracy"] = cc.iloc[0].get("final_test_accuracy")
        row.update(_attribution_flags(row))
        rows.append(row)
    return pd.DataFrame(rows)


def _attribution_flags(row: dict) -> dict:
    """Automatic indicators for the combinations of plan Section 8 (not conclusions)."""
    f = {}
    unresolved = row.get("reference_status") != "best_known_stationary_residual_1e-05" or \
        row.get("continuation_final_residual_1e-05") in (None, "unresolved")
    f["indicator_unresolved"] = bool(unresolved)
    dr, cr = row.get("default_final_r_orig"), row.get("continuation_final_r_orig")
    dl, cl = row.get("default_final_test_log_loss"), row.get("continuation_final_test_log_loss")
    dF, cF = row.get("default_final_objective"), row.get("continuation_final_objective")
    if None not in (dr, cr, dl, cl, dF, cF) and not any(isinstance(v, float) and math.isnan(v)
                                                       for v in (dr, cr, dl, cl, dF, cF)):
        improved_numerics = cr < dr and cF < dF
        f["indicator_numerical_shortfall_contributes"] = bool(improved_numerics and dl - cl > LOGLOSS_UNCHANGED)
        f["indicator_accuracy_without_task_gain"] = bool(improved_numerics and dl - cl <= LOGLOSS_UNCHANGED)
    ctl = row.get("convex_control_test_log_loss")
    if ctl is not None and cl is not None and not (isinstance(ctl, float) and math.isnan(ctl)):
        f["indicator_refined_worse_than_convex_control"] = bool(cl - ctl > LOGLOSS_UNCHANGED)
    return f


def initial_value_table(rt: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    iv = cfg["phases"]["initial_values"]
    sub = rt[(rt["scale"] == "A") & (rt["point"] == iv["point"]) & (rt["strength"] == "S1")
             & rt["method"].isin(list(iv["methods"]) + ["M19"]) & (rt["phase"] == "A")]
    cols = ["method", "init", "exit_status", "native_status", "solver_seconds", "final_objective",
            "final_relative_objective_gap", "final_r_orig", "final_residual_1e-05",
            "final_test_log_loss", "final_test_accuracy", "final_coefficient_relative_difference", "reason"]
    return sub[[c for c in cols if c in sub]].sort_values(["method", "init"])


# ----------------------------------------------------------------------------------------------
# Figures (matplotlib, vector PDF with Arial)
# ----------------------------------------------------------------------------------------------


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 8,
                         "axes.titlesize": 9, "legend.fontsize": 6.5})
    return plt


def _label(run_row) -> str:
    v = run_row.get("variant") or ""
    return run_row["method"] + (f" {v}" if v else "")


def figures(out: Path, traces: pd.DataFrame, rt: pd.DataFrame, refs: dict, pairing: pd.DataFrame,
            attribution: pd.DataFrame) -> list[str]:
    plt = _setup_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path, png_path = Path(out) / "accuracy_time.pdf", Path(out) / "accuracy_time.png"
    cmap = plt.get_cmap("tab20")
    methods = sorted(rt["method"].unique()) if len(rt) else []
    colors = {m: cmap(i % 20) for i, m in enumerate(methods)}
    floor = 1e-16
    with PdfPages(pdf_path) as pdf:
        summary_fig = _summary_figure(plt, rt, pairing, attribution)
        if summary_fig is not None:
            pdf.savefig(summary_fig)
            summary_fig.savefig(png_path, dpi=200)
            plt.close(summary_fig)
        for iid in sorted(traces["instance"].unique()) if len(traces) else []:
            tr = traces[traces["instance"] == iid]
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
            plotted = [False, False]
            for run_id, g in tr.groupby("run_id"):
                g = g.sort_values("solver_seconds")
                rr = rt[rt["run_id"] == run_id].iloc[0]
                t = np.maximum(pd.to_numeric(g["solver_seconds"], errors="coerce").to_numpy(dtype=float), 1e-4)
                lab = _label(rr) + ("" if rr["phase"] in ("A", "B", "C") else f" [{rr['phase']}]")
                series = []
                if "relative_objective_gap" in g:
                    series.append((0, np.abs(pd.to_numeric(g["relative_objective_gap"], errors="coerce").to_numpy(dtype=float))))
                series.append((1, pd.to_numeric(g["r_orig"], errors="coerce").to_numpy(dtype=float)))
                for k, y in series:
                    ok = np.isfinite(y) & np.isfinite(t)
                    if not np.any(ok):
                        continue
                    axes[k].plot(t[ok], np.maximum(y[ok], floor), "-", marker=".", ms=2.5, lw=0.8,
                                 color=colors[rr["method"]], label=lab)
                    plotted[k] = True
            titles = ("|relative objective gap| to reference", "common residual r_orig")
            for k, (ax, title) in enumerate(zip(axes, titles)):
                ax.set_title(title)
                ax.set_xlabel("algorithm time (s)")
                ax.grid(alpha=0.3, lw=0.4)
                if plotted[k]:
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                else:
                    ax.text(0.5, 0.5, "no reference / no finite value", ha="center", va="center",
                            transform=ax.transAxes)
            if plotted[1]:
                for b in RES_BANDS:
                    axes[1].axhline(b, color="0.5", lw=0.5, ls="--")
                axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
            ref = refs.get(iid, {})
            fig.suptitle(f"{iid}  reference: {ref.get('reference_status')}", fontsize=9)
            try:
                fig.tight_layout()
            except ValueError:
                pass
            pdf.savefig(fig)
            plt.close(fig)
    return [str(pdf_path), str(png_path)]


def _summary_figure(plt, rt, pairing, attribution):
    sub = rt[(rt["phase"] == "B") & (rt["point"] == "C1")]
    if sub.empty and attribution.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    col = "t_obj_res_1e-05"
    if not sub.empty:
        ms = sorted(sub["method"].unique())
        for i, m in enumerate(ms):
            g = sub[sub["method"] == m]
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float) if col in g \
                else np.full(len(g), np.nan)
            reached = vals[np.isfinite(vals)]
            ax.scatter(np.full(reached.size, i), np.maximum(reached, 1e-4), s=10, color="C0")
            miss = int((~np.isfinite(vals)).sum())
            if miss:
                ax.scatter([i] * miss, g["cap_seconds"].iloc[:miss], s=14, marker="x", color="C3")
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels(ms, rotation=45)
        ax.set_yscale("log")
        ax.set_ylabel("time to objective & residual 1e-5 (s)")
        ax.set_title("TV (scale B); x: not reached (plotted at cap)")
    ax = axes[1]
    plotted = False
    for _, r in attribution.iterrows():
        xs = pd.to_numeric(pd.Series([r.get("default_final_r_orig"), r.get("continuation_final_r_orig")]),
                           errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(pd.Series([r.get("default_final_test_log_loss"),
                                      r.get("continuation_final_test_log_loss")]),
                           errors="coerce").to_numpy(dtype=float)
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)) and np.all(xs > 0)):
            continue
        ax.plot(xs, ys, "-o", ms=3, lw=0.8)
        plotted = True
        ctl = pd.to_numeric(pd.Series([r.get("convex_control_test_log_loss")]), errors="coerce").iloc[0]
        if np.isfinite(ctl):
            ax.axhline(ctl, color="0.7", lw=0.4)
    if plotted:
        ax.set_xscale("log")
    ax.set_xlabel("common residual (default -> refined)")
    ax.set_ylabel("test log-loss")
    ax.set_title("N3 default -> M02 refinement (scale B); grey: convex TV")
    try:
        fig.tight_layout()
    except ValueError:
        pass
    return fig


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------


def _md_table(df: pd.DataFrame, cols: list[str], floatfmt=".3g") -> str:
    cols = [c for c in cols if c in df]
    if df.empty or not cols:
        return "_none_\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if math.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def summary_markdown(out: Path, cfg: dict, rt: pd.DataFrame, refs: dict, pairing: pd.DataFrame,
                     attribution: pd.DataFrame, init_tab: pd.DataFrame, manifest: pd.DataFrame | None,
                     status: dict) -> str:
    lines = ["# B048 benchmark result summary (automatically generated)", "",
             "Generated from stored runs; automatic indicators use pre-specified thresholds and are "
             "interpretation aids, not conclusions. Time to accuracy is at checkpoint resolution.", ""]
    lines += ["## Counts", ""]
    if manifest is not None and len(manifest):
        lines.append(f"Planned jobs in manifest: {len(manifest)}.")
    if len(rt):
        counts = rt.groupby(["phase", "exit_status"]).size().unstack(fill_value=0).reset_index()
        lines.append(_md_table(counts, list(counts.columns), floatfmt=".0f"))
    lines += ["", f"Run status: {status}", ""]
    lines += ["## References", ""]
    rdf = pd.DataFrame(list(refs.values()))
    lines.append(_md_table(rdf, ["instance", "reference_status", "best_known_objective",
                                 "certified_relative_gap", "best_known_source", "conic_native_status",
                                 "conic_own_relative_gap"]))
    lines += ["## Q1: TV speed at equal original-objective accuracy (scale B)", ""]
    lines.append(_md_table(pairing, ["method", "baseline", "band", "method_reached", "baseline_reached",
                                     "both_reached", "median_time_ratio", "min_ratio", "max_ratio",
                                     "speedup_signal"]))
    lines += ["## Q2: nonconvex default vs refinement (scale B, N3)", ""]
    lines.append(_md_table(attribution, [c for c in attribution.columns if c.startswith(("instance", "default_final", "continuation_final", "convex_control", "indicator", "continuation_cumulative"))]))
    lines += ["## Initial-value diagnostics (scale A, N3/S1)", ""]
    lines.append(_md_table(init_tab, list(init_tab.columns)))
    lines += ["## Scale A coverage (exit status by route)", ""]
    a = rt[rt["phase"] == "A"]
    if len(a):
        cov = a.groupby(["method", "exit_status"]).size().unstack(fill_value=0).reset_index()
        lines.append(_md_table(cov, list(cov.columns), floatfmt=".0f"))
    lines += ["", "## Scale C", ""]
    lines.append(_md_table(rt[rt["phase"] == "C"], ["instance", "method", "exit_status", "solver_seconds",
                                                     "final_r_orig", "final_relative_objective_gap",
                                                     "t_obj_res_1e-05", "t_res_1e-05"]))
    text = "\n".join(lines) + "\n"
    (Path(out) / "results_summary.md").write_text(text, encoding="utf-8")
    return text


def analyze(out: Path, cfg: dict, status: dict | None = None) -> dict:
    out = Path(out)
    insts = Instances(out)
    runs = load_runs(out)
    target = cfg["references"]["target_relative_gap"]
    refs = compute_references(runs, insts, target)
    traces = trace_table(runs, refs, insts)
    rt = run_table(runs, traces, refs, insts)
    traces.to_csv(out / "traces.csv", index=False)
    rt.to_csv(out / "runs.csv", index=False)
    pd.DataFrame(list(refs.values())).to_csv(out / "references.csv", index=False)
    pairing = tv_pairing(rt, cfg) if len(rt) else pd.DataFrame()
    attribution = nonconvex_attribution(rt, refs) if len(rt) else pd.DataFrame()
    init_tab = initial_value_table(rt, cfg) if len(rt) else pd.DataFrame()
    pairing.to_csv(out / "tv_pairing.csv", index=False)
    attribution.to_csv(out / "nonconvex_attribution.csv", index=False)
    init_tab.to_csv(out / "initial_values.csv", index=False)
    files = figures(out, traces, rt, refs, pairing, attribution) if len(traces) else []
    manifest = pd.read_csv(out / "manifest.csv") if (out / "manifest.csv").exists() else None
    summary_markdown(out, cfg, rt, refs, pairing, attribution, init_tab, manifest, status or {})
    summary = {"runs": len(rt), "states": len(traces), "references": refs, "figures": files,
               "speedup_signals": pairing[pairing.get("speedup_signal", False) == True].to_dict("records")  # noqa: E712
               if len(pairing) else []}
    write_json(out / "analysis.json", plain(summary))
    return summary
