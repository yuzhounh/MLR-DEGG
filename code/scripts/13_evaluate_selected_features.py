"""Evaluate the features (channels / voxels, times) retained by the fitted models.

Usage:
    python code/scripts/13_evaluate_selected_features.py --mode sim --results results/sim_shapes_repro
    python code/scripts/13_evaluate_selected_features.py --mode erp --results results/erpcore_n170_dev_repro

``--mode sim`` reads ``selected_maps.npz`` and ``simulated_data_seed*.npz`` of a run of
``05_simulation_benchmark.py``; the reference is the true support.  ``--mode erp`` reads the
per-fold importance maps ``units/*_maps.npz`` of a run of ``03_run_decoding.py`` (needs
``save_maps: true``); the reference is a task-effect map of the *other* subjects (one-way ANOVA
across the classes in every subject, combined over subjects by Stouffer's z; the strongest
``--ref-top-frac`` of bins (and channels) give the binary reference, or Benjamini-Hochberg at ``--fdr``), so it never uses the evaluated subject.

Metrics (see ``mlrdegg/experiment/support_eval.py``): precision / recall / Dice / enrichment /
hypergeometric p against the binary reference, AUC / Spearman / top-k overlap against the
continuous reference, stability of the retained set (mean pairwise Jaccard and Nogueira index
over outer folds, or over seeds in simulations), and decoding with only the retained features or
without them against random sets of the same size.  Outputs in ``<results>/feature_eval/``:
``rows.csv`` and ``summary.csv`` (plus ``reference.csv`` for ``--mode erp``) and ``run.log``.
"""

from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import f as f_dist, norm
from threadpoolctl import threadpool_limits

from mlrdegg.datasets import load_derived
from mlrdegg.experiment.cv import outer_splits
from mlrdegg.experiment.support_eval import (anova_f_map, auc_against_reference, benjamini_hochberg,
                                             nogueira_stability, overlap_metrics, pairwise_jaccard,
                                             rank_metrics, retained_mask, sufficiency_necessity)
from mlrdegg.graphs import vectorize
from mlrdegg.paths import resolve


def top_fraction(score, frac):
    """Mask of the ``ceil(frac * size)`` highest scores."""
    k = int(np.ceil(frac * score.size))
    return np.argsort(-score.ravel(), kind="stable").argsort().reshape(score.shape) < k


def evaluate(imp_vt, R_vt, score_vt, X, y, tr, te, n_random, seed, C, R_ch=None):
    """Metrics of one retained set; ``imp_vt`` is the ``(V, T)`` importance map, ``R_vt`` the
    binary reference (``R_ch`` for channels, default: channels with any reference bin)."""
    S = retained_mask(imp_vt)
    dense = bool(S.all())
    rows = {}
    for level, (s, r, imp, sc) in {
            "channel_time": (S, R_vt, imp_vt, score_vt),
            "channel": (S.any(1), R_vt.any(1) if R_ch is None else R_ch, imp_vt.max(1), score_vt.max(1))}.items():
        row = {"auc": auc_against_reference(imp, r), **rank_metrics(s, imp, sc)}
        row.update({} if dense else overlap_metrics(s, r))
        row["n_selected"] = int(s.sum())
        rows[level] = row
    if not dense:
        sn = sufficiency_necessity(X, y, tr, te, vectorize(S[None])[0], n_random=n_random, C=C, seed=seed)
        sn.update(suff_sig=float(sn["p_suff"] <= 0.05), nec_sig=float(sn["p_nec"] <= 0.05))
        rows["channel_time"].update(sn)
    return rows, S


def sim_task(key, W, data, n_random, C):
    seed, rule, method = key.split("__")
    imp = np.linalg.norm(W - W.mean(2, keepdims=True), axis=2)  # (V, T)
    truth_vt = (np.abs(data["patterns"]) > 0).any(0)
    X = np.vstack([data["Xtr"], data["Xte"]])
    y = np.concatenate([data["ytr"], data["yte"]])
    tr, te = np.arange(len(data["ytr"])), len(data["ytr"]) + np.arange(len(data["yte"]))
    with threadpool_limits(limits=1):
        rows, S = evaluate(imp, truth_vt, truth_vt.astype(float), X, y, tr, te, n_random, 0, C)
    return [{"seed": int(seed.removeprefix("seed")), "rule": rule, "method": method, "level": lv, **r}
            for lv, r in rows.items()], S


def run_sim(args, out, log):
    res = resolve(args.results)
    maps = np.load(res / "selected_maps.npz")
    keys = [k for k in maps.files if k.split("__")[1] == args.rule]
    data = {}
    for k in keys:
        sd = k.split("__")[0]
        if sd not in data:
            data[sd] = dict(np.load(res / f"simulated_data_{sd}.npz"))
    log.info("%d selected fits (rule %s)", len(keys), args.rule)
    out_rows = Parallel(n_jobs=args.n_jobs)(
        delayed(sim_task)(k, maps[k], data[k.split("__")[0]], args.n_random, args.C) for k in keys)
    rows = [r for rr, _ in out_rows for r in rr]
    sets = {}
    for k, (_, S) in zip(keys, out_rows):
        sets.setdefault(k.split("__")[2], []).append(S)
    rows_df = pd.DataFrame(rows)
    stab = pd.DataFrame([{"method": m, "level": lv, "stability_jaccard": pairwise_jaccard(
        [s if lv == "channel_time" else s.any(1) for s in ss]), "stability_nogueira": nogueira_stability(
        [s if lv == "channel_time" else s.any(1) for s in ss])} for m, ss in sets.items()
        for lv in ("channel_time", "channel")])
    return rows_df, stab, None


def reference_maps(ds):
    """Stouffer z of the class effect of every subject: returns ``(sum of z, per-subject z)``."""
    zs = []
    for s in ds.subjects:
        F, d1, d2 = anova_f_map(s.X, s.y)
        zs.append(norm.isf(np.clip(f_dist.sf(F, d1, d2), 1e-300, 1.0)))
    zs = np.stack(zs)
    return zs.sum(0), zs


def erp_task(sub, imaps, ref_vt, X, y, groups, exp_cfg, args):
    """All outer folds of one subject; ``imaps[method]`` has shape ``(n_outer, V, T)``."""
    folds = outer_splits(exp_cfg, y, groups)
    rows, sets = [], {}
    with threadpool_limits(limits=1):
        for method, arr in imaps.items():
            for f, imp in enumerate(arr):
                tr, te = folds[f]
                r, S = evaluate(imp, ref_vt["R"], ref_vt["Z"], X, y, tr, te, args.n_random,
                                1000 * f + 1, args.C, R_ch=ref_vt["R_ch"])
                sets.setdefault(method, []).append(S)
                rows += [{"subject": sub, "outer_fold": f, "method": method, "level": lv, **v}
                         for lv, v in r.items()]
    stab = [{"subject": sub, "method": m, "level": lv,
             "stability_jaccard": pairwise_jaccard([s if lv == "channel_time" else s.any(1) for s in ss]),
             "stability_nogueira": nogueira_stability([s if lv == "channel_time" else s.any(1) for s in ss])}
            for m, ss in sets.items() for lv in ("channel_time", "channel")]
    return rows, stab


def run_erp(args, out, log):
    res = resolve(args.results)
    cfg = _bootstrap.load_config(res / "config.yaml")
    exp = cfg["experiment"]
    ds = load_derived(resolve(cfg["preprocess"]["out_dir"]), None, exclude=exp.get("exclude_subjects"))
    zsum, zs = reference_maps(ds)
    n_sub = len(ds.subjects)
    names = [s.subject for s in ds.subjects]
    log.info("reference from %d subjects (leave-one-subject-out), V=%d, T=%d", n_sub, ds.n_space, ds.n_time)
    ref_rows, tasks = [], []
    wanted = args.subjects or (None if exp.get("subjects", "all") == "all" else exp["subjects"])
    for i, s in enumerate(ds.subjects):
        f = res / "units" / f"{s.subject}_maps.npz"
        if not f.exists() or (wanted and int(s.subject.removeprefix("sub-")) not in {int(w) for w in wanted}):
            continue
        z = (zsum - zs[i]) / np.sqrt(n_sub - 1)
        if args.ref_top_frac > 0:  # strongest bins / channels of the group effect map
            R, R_ch = top_fraction(z, args.ref_top_frac), top_fraction(z.max(1), args.ref_top_frac)
        else:
            R = benjamini_hochberg(norm.sf(z), args.fdr).reshape(z.shape)
            R_ch = R.any(1)
        ref_rows.append({"subject": s.subject, "n_reference": int(R.sum()), "reference_fraction": R.mean(),
                         "channels_with_reference": int(R.any(1).sum())})
        maps = np.load(f)
        keep = [k for k in maps.files if (args.methods and k in args.methods) or
                (not args.methods and k.endswith("@" + args.rule))]
        groups = {k: np.asarray(v) for k, v in s.meta.items()}
        tasks.append(delayed(erp_task)(s.subject, {k: maps[k] for k in keep}, {"R": R, "R_ch": R_ch, "Z": z},
                                       vectorize(s.X).astype(float), s.y, groups, exp, args))
    log.info("%d subjects, methods %s, %d random sets per test", len(tasks), args.methods or f"*@{args.rule}",
             args.n_random)
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs)(tasks)
    log.info("evaluated in %.0f s", time.time() - t0)
    rows = pd.DataFrame([r for rr, _ in results for r in rr])
    stab = pd.DataFrame([r for _, ss in results for r in ss])
    return rows, stab, pd.DataFrame(ref_rows)


def summarize(rows: pd.DataFrame, stab: pd.DataFrame) -> pd.DataFrame:
    """Mean over folds within unit (subject or seed), then mean and SD over units."""
    unit = "subject" if "subject" in rows else "seed"
    num = [c for c in rows.columns if c not in ("subject", "seed", "outer_fold", "method", "level", "rule")]
    per_unit = rows.groupby([unit, "method", "level"])[num].mean().reset_index()
    agg = per_unit.groupby(["method", "level"])[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    if unit == "subject":
        s = stab.groupby(["method", "level"])[["stability_jaccard", "stability_nogueira"]].agg(["mean", "std"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
    else:
        s = stab.set_index(["method", "level"])
    return agg.join(s).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "erp"], required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--rule", default="1se_repro", help="selection rule whose maps are evaluated")
    ap.add_argument("--methods", nargs="*", help="erp: map names in units/*_maps.npz (default *@rule)")
    ap.add_argument("--subjects", nargs="*", help="erp: restrict to these subjects")
    ap.add_argument("--ref-top-frac", type=float, default=0.1,
                    help="erp: the binary reference is this fraction of the strongest bins (and "
                         "channels) of the group effect map; 0 uses the BH-FDR mask instead")
    ap.add_argument("--fdr", type=float, default=0.05, help="erp: FDR level when --ref-top-frac is 0")
    ap.add_argument("--n-random", type=int, default=50, help="random sets per sufficiency/necessity test")
    ap.add_argument("--C", type=float, default=0.01, help="L2 logistic regression strength for the decoding tests")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--out", default="feature_eval")
    args = ap.parse_args()

    out = resolve(args.results) / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = _bootstrap.setup_logging(out / "run.log")
    rows, stab, ref = (run_sim if args.mode == "sim" else run_erp)(args, out, log)
    rows.to_csv(out / "rows.csv", index=False)
    summary = summarize(rows, stab)
    summary.to_csv(out / "summary.csv", index=False)
    if ref is not None:
        ref.to_csv(out / "reference.csv", index=False)
        log.info("reference size (mean over subjects): %.1f%% of channel-time bins, %.1f channels",
                 100 * ref.reference_fraction.mean(), ref.channels_with_reference.mean())
    show = [c for c in ["method", "level", "n_selected_mean", "precision_mean", "recall_mean", "dice_mean",
                        "enrichment_mean", "auc_mean", "spearman_mean", "topk_overlap_mean",
                        "topk_chance_mean", "stability_jaccard_mean", "stability_nogueira_mean",
                        "acc_all_mean", "acc_S_mean", "acc_random_mean", "acc_without_S_mean",
                        "acc_without_random_mean", "suff_sig_mean", "nec_sig_mean"] if c in summary]
    log.info("summary:\n%s", summary[show].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
