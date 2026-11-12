"""B048 small-simulation solver benchmark (review/solver_benchmark_plan.md).

Usage (from the project root):
    python code/scripts/22_solver_benchmark.py plan     --config code/configs/solver_benchmark.yaml
    python code/scripts/22_solver_benchmark.py run      --config code/configs/solver_benchmark.yaml
    python code/scripts/22_solver_benchmark.py evaluate --config code/configs/solver_benchmark.yaml
    python code/scripts/22_solver_benchmark.py analyze  --config code/configs/solver_benchmark.yaml

``plan`` builds (or reuses) the frozen instances and writes ``methods.csv`` and ``manifest.csv``
with run counts and the maximal algorithm time; it runs no solver.  ``run`` executes, under one
hard time budget: phase A (all routes, scale A) -> phase B (TV and nonconvex comparison, scale B)
-> M02 refinement from the M01 state -> references (A, B) -> evaluation -> phase C (scale C,
candidates chosen by B accuracy-time only) -> references and evaluation of C -> timing repeats
-> analysis.  Every solve runs in its own process with one BLAS/OpenMP thread, strictly one at
a time; routes are shuffled with a fixed seed within each instance.  Jobs whose cap no longer fits
the remaining budget are recorded as ``not_run_budget``; no computation is added automatically.

The program writes ``status.json`` while running and ``COMPLETED.json`` (exit 0) or
``ATTENTION.json`` (exit 2) at the end.  Outputs: ``tmp/solver_benchmark/<name>/``.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from mlrdegg.paths import PROJECT_ROOT, resolve
from mlrdegg.solver_benchmark import analysis
from mlrdegg.solver_benchmark import instances as instmod
from mlrdegg.solver_benchmark.core import Problem, load_instance, plain, read_json, write_json
from mlrdegg.solver_benchmark.evaluate import evaluate_run
from mlrdegg.solver_benchmark.methods import METHODS, fine_ratio

SMOOTHING_METHODS = ("M20", "M21", "M22")
WORKER = Path(__file__).resolve().parent / "solver_benchmark_worker.py"
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


class Benchmark:
    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args
        self.name = args.name or cfg["name"]
        self.out = resolve(cfg["output_root"]) / self.name
        self.inst_dir = self.out / "instances"
        self.runs_dir = self.out / "runs"
        self.out.mkdir(parents=True, exist_ok=True)
        b = cfg["budget"]
        self.t0 = time.time()
        self.total = 60.0 * (args.budget_minutes or b["total_minutes"])
        self.reserve = 60.0 * b["reserve_evaluation_minutes"]
        self.cap_override = getattr(args, "cap_override", None)
        self._problems: dict[str, Problem] = {}
        self.metas: dict[str, dict] = {}
        self.stage = "init"
        self.done, self.planned = 0, 0

    # --- bookkeeping ------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{self.elapsed() / 60:6.2f} min] {msg}"
        print(line, flush=True)
        with open(self.out / "run.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def elapsed(self) -> float:
        return time.time() - self.t0

    def write_status(self, state: str = "running", **extra) -> None:
        write_json(self.out / "status.json", {
            "state": state, "stage": self.stage, "elapsed_minutes": self.elapsed() / 60,
            "budget_minutes": self.total / 60, "jobs_done_this_invocation": self.done,
            "pid": os.getpid(), "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cap_override": self.cap_override, **extra})

    def problem(self, iid: str) -> Problem:
        if iid not in self._problems:
            self._problems[iid] = Problem(load_instance(self.inst_dir / f"{iid}.npz"))
        return self._problems[iid]

    # --- instances and jobs -----------------------------------------------------------------
    def prepare(self) -> None:
        self.stage = "prepare"
        t = time.perf_counter()
        metas = instmod.prepare_instances(self.cfg, self.inst_dir, PROJECT_ROOT, log=self.log)
        self.metas = {m["id"]: m for m in metas}
        write_json(self.out / "instances.json", metas)  # plan Section 11 deliverable
        self.log(f"{len(metas)} frozen instances ready ({time.perf_counter() - t:.1f} s)")

    def job(self, phase: str, iid: str, method: str, cap: float, variant: str = "", **kw) -> dict:
        run_id = "__".join(x for x in (phase, iid, method, variant) if x)
        job = {"run_id": run_id, "phase": phase, "instance": iid,
               "instance_path": str(self.inst_dir / f"{iid}.npz"), "method": method,
               "variant": variant, "cap": float(self.cap_override or cap),
               "cvxpy_site": str(resolve(self.cfg["cvxpy_site"])) if self.cfg.get("cvxpy_site") else None,
               **kw}
        if method in SMOOTHING_METHODS:
            job.setdefault("smoothing", "continuation")
            ratio, bound = fine_ratio(self.problem(iid), method)
            job["fine_ratio"], job["fine_bias_bound"] = ratio, bound
        return job

    def _shuffled(self, blocks: list[list[dict]]) -> list[dict]:
        rng = random.Random(self.cfg["budget"]["method_order_seed"])
        out = []
        for block in blocks:
            block = list(block)
            rng.shuffle(block)
            out += block
        return out

    def phase_a_jobs(self) -> list[dict]:
        a = self.cfg["phases"]["A"]
        iids = [instmod.instance_id("A", a["seed"], pt, a["strength"]) for pt in instmod.POINTS]
        iids += [instmod.instance_id("A", a["seed"], pt, "S2") for pt in a["extra_strength_points"]]
        blocks = []
        for iid in iids:
            point = self.metas[iid]["point"]
            block = []
            for method, (_, _, points, _, _) in METHODS.items():
                if method == "M17" or point not in points:
                    continue  # M17 runs as the separately listed reference (Section 7)
                if method in SMOOTHING_METHODS:
                    block += [self.job("A", iid, method, a["cap_seconds"], v, smoothing=v)
                              for v in a["smoothing_variants"]]
                else:
                    block.append(self.job("A", iid, method, a["cap_seconds"]))
            blocks.append(block)
        iv = self.cfg["phases"]["initial_values"]
        iid = instmod.instance_id("A", a["seed"], iv["point"], a["strength"])
        extra = [self.job("A", iid, m, a["cap_seconds"], f"init_{init}", init=init,
                          perturbation_seed=iv["perturbation_seed"])
                 for m in iv["methods"] for init in iv["inits"] if init != "common"]
        # M19 reuses the perturbed initial value as a conditional check (no new data).
        extra.append(self.job("A", iid, "M19", a["cap_seconds"], "init_perturbed", init="perturbed",
                              perturbation_seed=iv["perturbation_seed"]))
        blocks.append(extra)
        return self._shuffled(blocks)

    def phase_b_jobs(self) -> list[dict]:
        b = self.cfg["phases"]["B"]
        blocks = []
        for seed in b["seeds"]:
            for strength in b["strengths"]:
                for point in b["points"]:
                    iid = instmod.instance_id("B", seed, point, strength)
                    methods = b["tv_methods"] if point == "C1" else b["nonconvex_methods"]
                    blocks.append([self.job("B", iid, m, b["cap_seconds"]) for m in methods])
        for point in b["apg_overlap_points"]:
            iid = instmod.instance_id("B", b["apg_overlap_seed"], point, b["apg_overlap_strength"])
            blocks.append([self.job("B", iid, m, b["cap_seconds"]) for m in b["apg_overlap_methods"]])
        return self._shuffled(blocks)

    def continuation_jobs(self) -> list[dict]:
        b = self.cfg["phases"]["B"]
        jobs = []
        for seed in b["seeds"]:
            for strength in b["strengths"]:
                iid = instmod.instance_id("B", seed, "N3", strength)
                src = self.runs_dir / f"B__{iid}__M01" / "final.npz"
                jobs.append(self.job("B-cont", iid, "M02", self.cfg["references"]["continuation_cap_seconds"],
                                     "from_M01", initial_state=str(src)))
        return jobs

    def reference_jobs(self, scale: str) -> list[dict]:
        r = self.cfg["references"]
        jobs = []
        for sc, seed, point, strength in instmod.planned_instances(self.cfg):
            if sc != scale:
                continue
            iid = instmod.instance_id(sc, seed, point, strength)
            if self.metas[iid]["nonconvex"]:
                jobs += [self.job("R", iid, m, r["nonconvex_cap_seconds"][scale], "reference",
                                  reference=True) for m in r["nonconvex_methods"]]
            else:
                jobs.append(self.job("R", iid, "M17", r["conic_cap_seconds"][scale]))
        return jobs

    def phase_c_jobs(self, chosen: dict) -> list[dict]:
        c = self.cfg["phases"]["C"]
        jobs = []
        for point in c["points"]:
            iid = instmod.instance_id("C", c["seed"], point, c["strength"])
            jobs += [self.job("C", iid, m, c["cap_seconds"]) for m in ["M01"] + chosen[point]]
        return jobs

    # --- execution ----------------------------------------------------------------------------
    def execute(self, job: dict, reserve: float) -> dict:
        run_dir = self.runs_dir / job["run_id"]
        result_path = run_dir / "result.json"
        if result_path.exists() and not self.args.rerun:
            prev = read_json(result_path)
            if prev.get("exit_status") != "not_run_budget":
                return prev
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "job.json", job)
        if job.get("initial_state") and not Path(job["initial_state"]).exists():
            res = {"run_id": job["run_id"], "method": job["method"], "instance": job["instance"],
                   "phase": job["phase"], "exit_status": "not_run_missing_state",
                   "solver_seconds": None, "reason": f"missing {job['initial_state']}"}
            write_json(result_path, res)
            self.log(f"not started (missing state): {job['run_id']}")
            return res
        factor = self.cfg["budget"]["hard_kill_factor"]
        if self.elapsed() + job["cap"] * factor + 20 > self.total - reserve:
            res = {"run_id": job["run_id"], "method": job["method"], "instance": job["instance"],
                   "phase": job["phase"], "exit_status": "not_run_budget", "solver_seconds": None}
            write_json(result_path, res)
            self.log(f"not started (budget): {job['run_id']}")
            return res
        if result_path.exists():
            result_path.unlink()
        env = {**os.environ, **THREAD_ENV, "PYTHONHASHSEED": "0"}
        t = time.perf_counter()
        with open(run_dir / "worker.log", "w", encoding="utf-8") as fh:
            try:
                proc = subprocess.run([sys.executable, str(WORKER), str(run_dir / "job.json")],
                                      env=env, stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=job["cap"] * factor + 20, cwd=str(PROJECT_ROOT))
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = "hard_killed"
        seconds = time.perf_counter() - t
        if result_path.exists() and code == 0:
            res = read_json(result_path)
        else:
            res = {"run_id": job["run_id"], "method": job["method"], "instance": job["instance"],
                   "phase": job["phase"], "solver_seconds": None,
                   "exit_status": "hard_killed" if code == "hard_killed" else "worker_failed",
                   "reason": f"worker return code {code}; see worker.log"}
            if not (run_dir / "trace.json").exists() and (run_dir / "trace.jsonl").exists():
                write_json(run_dir / "trace.json", analysis_records(run_dir))
        res["process_seconds"] = seconds
        write_json(result_path, res)
        self.done += 1
        return res

    def run_jobs(self, jobs: list[dict], stage: str, reserve: float) -> list[dict]:
        self.stage = stage
        self.log(f"stage {stage}: {len(jobs)} jobs")
        out = []
        for i, job in enumerate(jobs):
            res = self.execute(job, reserve)
            out.append(res)
            self.write_status(stage_progress=f"{i + 1}/{len(jobs)}", last=job["run_id"])
            if res.get("exit_status") in ("error", "worker_failed", "hard_killed"):
                self.log(f"ATTENTION {res['exit_status']}: {job['run_id']} {res.get('reason', '')[:200]}")
        counts = pd.Series([r.get("exit_status") for r in out]).value_counts().to_dict()
        self.log(f"stage {stage} finished: {counts}")
        return out

    def evaluate(self, stage: str, deadline: float, phases: set[str] | None = None) -> None:
        """Evaluate stored states in parallel (not timed as algorithm work)."""
        from joblib import Parallel, delayed

        self.stage = stage
        ev = self.cfg["evaluation"]
        todo = []
        for d in sorted(self.runs_dir.iterdir()):
            if not (d / "job.json").exists() or not (d / "result.json").exists():
                continue
            job, res = read_json(d / "job.json"), read_json(d / "result.json")
            if res.get("exit_status") == "not_run_budget":
                continue
            if phases is not None and job["phase"] not in phases:
                continue
            if (d / "eval.json").exists() and not self.args.re_evaluate:
                old = read_json(d / "eval.json")
                if not any(r and r.get("evaluation_note") == "evaluation budget"
                           for r in old.get("rows", [])):
                    continue
            scale = self.metas[job["instance"]]["scale"]
            todo.append((d, job["instance_path"], ev["residual_seconds_final"][scale],
                         ev["residual_seconds_point"][scale]))
        self.log(f"evaluation {stage}: {len(todo)} runs, {ev['jobs']} parallel evaluators")
        if not todo:
            return
        os.environ.update(THREAD_ENV)
        wall_deadline = self.t0 + deadline  # wall clock, comparable across processes
        Parallel(n_jobs=ev["jobs"], backend="loky")(
            delayed(_evaluate_one)(d, p, bf, bp, wall_deadline) for d, p, bf, bp in todo)
        self.log(f"evaluation {stage} finished")

    def choose_c_candidates(self) -> tuple[dict, dict]:
        """Rank phase-B routes by median time to the selection band (all six instances reached)."""
        c, b = self.cfg["phases"]["C"], self.cfg["phases"]["B"]
        analysis.analyze(self.out, self.cfg, {"stage": "C candidate selection"})
        rt = pd.read_csv(self.out / "runs.csv")
        band = f"{c['selection_band']:.0e}"
        chosen, notes = {}, {}
        for point, methods, col in (("C1", b["tv_methods"], f"t_obj_res_{band}"),
                                    ("N3", b["nonconvex_methods"], f"t_res_{band}")):
            sub = rt[(rt["phase"] == "B") & (rt["point"] == point)]
            n_inst = sub["instance"].nunique()
            ranking = []
            for m in methods:
                if m == "M01" or col not in sub:
                    continue
                vals = sub[sub["method"] == m][col]
                if len(vals) == n_inst and vals.notna().all() and n_inst > 0:
                    ranking.append((float(vals.median()), m))
            ranking.sort()
            picks = [m for _, m in ranking[: c["max_candidates"]]]
            fallback = [m for m in c["default_candidates"][point] if m not in picks]
            while len(picks) < c["max_candidates"] and fallback:
                picks.append(fallback.pop(0))
            chosen[point] = picks
            notes[point] = {"ranking_median_seconds": ranking, "chosen": picks,
                            "filled_from_defaults": len(ranking) < c["max_candidates"]}
        write_json(self.out / "phase_c_selection.json", notes)
        self.log(f"phase C candidates: {chosen}")
        return chosen, notes

    def timing_jobs(self) -> list[dict]:
        rt = pd.read_csv(self.out / "runs.csv")
        ctl = rt[(rt["phase"] == "A") & rt["method"].isin(["M01", "M03"]) & (rt["exit_status"] == "native_stop")]
        if ctl.empty:
            return []
        best = ctl.sort_values("solver_seconds").iloc[0]
        job = read_json(self.runs_dir / best["run_id"] / "job.json")
        out = []
        for k in range(self.cfg["budget"]["timing_repeats"]):
            j = dict(job)
            j["phase"], j["variant"] = "T", f"repeat{k + 1}"
            j["run_id"] = f"T__{job['instance']}__{job['method']}__repeat{k + 1}"
            j["repeat_of"] = job["run_id"]
            out.append(j)
        return out

    # --- manifest -----------------------------------------------------------------------------
    def manifest(self) -> pd.DataFrame:
        rows = []
        for stage, jobs in (("A", self.phase_a_jobs()), ("B", self.phase_b_jobs()),
                            ("B-cont", self.continuation_jobs()),
                            ("R", self.reference_jobs("A") + self.reference_jobs("B") + self.reference_jobs("C"))):
            for j in jobs:
                rows.append({"stage": stage, "run_id": j["run_id"], "instance": j["instance"],
                             "method": j["method"], "variant": j.get("variant", ""),
                             "cap_seconds": j["cap"], "fine_ratio": j.get("fine_ratio"),
                             "fine_bias_bound": j.get("fine_bias_bound")})
        c = self.cfg["phases"]["C"]
        for point in c["points"]:
            iid = instmod.instance_id("C", c["seed"], point, c["strength"])
            for k in range(1 + c["max_candidates"]):
                rows.append({"stage": "C", "run_id": f"C__{iid}__(selected after B)#{k}",
                             "instance": iid, "method": "M01" if k == 0 else "selected",
                             "variant": "", "cap_seconds": c["cap_seconds"]})
        return pd.DataFrame(rows)

    def methods_table(self, manifest: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for m, (name, kind, points, conditional, notes) in METHODS.items():
            sub = manifest[manifest["method"] == m]
            rows.append({"method": m, "name": name, "objective": kind, "points": " ".join(points),
                         "conditional": conditional, "implementation": "existing code" if m in ("M01", "M02", "M03")
                         else "B048 adapter", "conditions_and_settings": notes,
                         "planned_A": int((sub["stage"] == "A").sum()), "planned_B": int((sub["stage"] == "B").sum()),
                         "planned_reference": int((sub["stage"] == "R").sum())})
        return pd.DataFrame(rows)


def analysis_records(run_dir: Path):
    from mlrdegg.solver_benchmark.evaluate import load_records
    return load_records(run_dir)


def _evaluate_one(run_dir, instance_path, budget_final, budget_point, wall_deadline):
    try:
        evaluate_run(Path(run_dir), Path(instance_path), budget_final, budget_point,
                     wall_deadline=wall_deadline)
        if (Path(run_dir) / "eval_error.json").exists():
            (Path(run_dir) / "eval_error.json").unlink()
    except Exception as err:  # recorded for the attention report
        import traceback
        write_json(Path(run_dir) / "eval_error.json", {"error": f"{type(err).__name__}: {err}",
                                                       "traceback": traceback.format_exc()})


def cmd_plan(bench: Benchmark) -> int:
    bench.prepare()
    man = bench.manifest()
    man.to_csv(bench.out / "manifest.csv", index=False)
    bench.methods_table(man).to_csv(bench.out / "methods.csv", index=False)
    counts = man.groupby("stage").agg(jobs=("run_id", "size"), max_seconds=("cap_seconds", "sum"))
    bench.log("planned jobs and maximal algorithm time:\n" + counts.to_string())
    a = counts.loc["A", "jobs"]
    limit = bench.cfg["phases"]["A"]["max_runs"]
    bench.log(f"phase A jobs {a} (limit {limit}); total maximal algorithm minutes "
              f"{counts['max_seconds'].sum() / 60:.1f}; hard budget {bench.total / 60:.0f} min")
    return 0 if a <= limit else 2


def cmd_run(bench: Benchmark) -> int:
    write_json(bench.out / "invocation.json", {"argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                "budget_minutes": bench.total / 60,
                                                "cap_override": bench.cap_override})
    for name in ("COMPLETED.json", "ATTENTION.json"):
        if (bench.out / name).exists():
            (bench.out / name).unlink()
    bench.write_status()
    bench.prepare()
    man = bench.manifest()
    man.to_csv(bench.out / "manifest.csv", index=False)
    bench.methods_table(man).to_csv(bench.out / "methods.csv", index=False)
    phases = set(bench.args.phases.split(",")) if bench.args.phases else {"A", "B", "C"}
    reserve = bench.reserve
    if "A" in phases:
        bench.run_jobs(bench.phase_a_jobs(), "phase A", reserve)
    if "B" in phases:
        bench.run_jobs(bench.phase_b_jobs(), "phase B", reserve)
        bench.run_jobs(bench.continuation_jobs(), "M02 refinement from M01", reserve)
    refs = []
    for sc in ("A", "B"):
        if sc in phases:
            refs += bench.reference_jobs(sc)
    bench.run_jobs(refs, "references A/B", reserve)
    c_time = 0.0
    if "C" in phases:
        c = bench.cfg["phases"]["C"]
        r = bench.cfg["references"]
        c_time = len(c["points"]) * (1 + c["max_candidates"]) * c["cap_seconds"] + \
            r["conic_cap_seconds"]["C"] + 2 * r["nonconvex_cap_seconds"]["C"] + 120
    bench.evaluate("evaluation A/B", bench.total - c_time - 90)
    if "C" in phases:
        chosen, _ = bench.choose_c_candidates()
        late_reserve = 120.0
        bench.run_jobs(bench.phase_c_jobs(chosen), "phase C", late_reserve)
        bench.run_jobs(bench.reference_jobs("C"), "references C", late_reserve)
        bench.evaluate("evaluation C", bench.total - 60, phases={"C", "R"})
    try:
        analysis.analyze(bench.out, bench.cfg, {"stage": "timing"})
        tj = bench.timing_jobs()
        bench.run_jobs(tj, "timing repeats", 45.0)
        if tj:
            bench.evaluate("evaluation timing", bench.total - 30, phases={"T"})
    except Exception as err:
        bench.log(f"timing repeats skipped: {err}")
    bench.stage = "analysis"
    summary = analysis.analyze(bench.out, bench.cfg, {"elapsed_minutes": bench.elapsed() / 60})
    rt = pd.read_csv(bench.out / "runs.csv")
    attention = {
        "errors": rt[rt["exit_status"].isin(["error", "worker_failed", "hard_killed"])]["run_id"].tolist(),
        "not_run_budget": rt[rt["exit_status"] == "not_run_budget"]["run_id"].tolist(),
        "evaluation_errors": [d.name for d in bench.runs_dir.iterdir() if (d / "eval_error.json").exists()],
        "references_unresolved": [k for k, v in summary["references"].items()
                                  if v.get("reference_status") in ("reference_unresolved", "no_evaluated_state")],
        "cap_override": bench.cap_override,
    }
    final = {"elapsed_minutes": bench.elapsed() / 60, "runs": int(len(rt)),
             "exit_status_counts": rt["exit_status"].value_counts().to_dict(),
             "outputs": [str(bench.out / f) for f in ("runs.csv", "traces.csv", "references.csv",
                                                       "methods.csv", "instances/instances.json",
                                                       "accuracy_time.pdf", "accuracy_time.png",
                                                       "results_summary.md")],
             "attention": attention}
    needs = any(attention[k] for k in ("errors", "not_run_budget", "evaluation_errors")) or bench.cap_override
    write_json(bench.out / ("ATTENTION.json" if needs else "COMPLETED.json"), plain(final))
    bench.write_status("attention" if needs else "completed")
    bench.log(f"finished: {final['exit_status_counts']}; attention: "
              f"{ {k: len(v) if isinstance(v, list) else v for k, v in attention.items()} }")
    return 2 if needs else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "run", "evaluate", "analyze"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", help="result directory name (default: config name)")
    ap.add_argument("--budget-minutes", type=float, help="override the hard total budget")
    ap.add_argument("--phases", help="subset, e.g. A or A,B (default: A,B,C)")
    ap.add_argument("--rerun", action="store_true", help="recompute runs that already have results")
    ap.add_argument("--re-evaluate", action="store_true", help="recompute stored evaluations")
    ap.add_argument("--cap-override", type=float,
                    help="smoke test only: replace every cap (results are not plan results)")
    args = ap.parse_args()
    cfg = _bootstrap.load_config(args.config)
    bench = Benchmark(cfg, args)
    if args.command == "plan":
        return cmd_plan(bench)
    if args.command == "run":
        try:
            return cmd_run(bench)
        except Exception as err:
            import traceback
            write_json(bench.out / "ATTENTION.json", {"fatal": f"{type(err).__name__}: {err}",
                                                      "traceback": traceback.format_exc()})
            bench.write_status("failed", error=str(err))
            raise
    if args.command == "evaluate":
        bench.prepare()
        bench.evaluate("evaluation", bench.total)
        return 0
    if args.command == "analyze":
        bench.prepare()
        analysis.analyze(bench.out, cfg, {})
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
