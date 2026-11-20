"""Refit, for one seed, each method's grid point with the highest map correlation (oracle).

Usage:
    python code/scripts/07_oracle_maps.py --results results/sim_shapes --seed 2026
    python code/scripts/06_plot_simulation_maps.py --results results/sim_shapes --seed 2026         --rule oracle_map --maps oracle_maps.npz --summary oracle_summary.csv

The oracle uses the true maps, so it shows what each penalty geometry can recover, not a
legitimate model-selection rule.
"""
import argparse
import importlib.util
import time
from pathlib import Path

import _bootstrap
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
spec = importlib.util.spec_from_file_location("sb", Path(__file__).with_name("05_simulation_benchmark.py"))
sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
from mlrdegg.experiment.models import Context, build_graphs
from mlrdegg.paths import resolve

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--seed", type=int, required=True)
args = ap.parse_args()
res = str(resolve(args.results)) + "/"
seed = args.seed
cfg = _bootstrap.load_config(res + "config.yaml")
data = sb.simulate({**cfg["simulation"], "seed": seed})
sg, tg = build_graphs(cfg["graph"], {"coords": data["coords"]}, cfg["simulation"]["n_time"])
ctx = Context(len(data["coords"]), cfg["simulation"]["n_time"], sg, tg)
fits = pd.read_csv(res + "fits.csv"); fits = fits[(fits.seed == seed) & (fits.error.fillna("") == "")]
tasks, meta = [], []
for m in cfg["experiment"]["methods"]:
    best = fits[fits.method == m["name"]].sort_values("map_corr").iloc[-1]
    params = {k: float(best[k]) for k in m["grid"]}
    tasks.append(delayed(sb.fit_one)(m, params, data, ctx, cfg["experiment"]["solver"], {}, time.time() + 600, "oracle_map", seed))
rows = Parallel(n_jobs=len(tasks))(tasks)
np.savez_compressed(res + "oracle_maps.npz", **{f"seed{seed}__oracle_map__{r['method']}": r.pop("_coef") for r in rows})
df = pd.DataFrame(rows); df["rule"] = "oracle_map"; df.to_csv(res + "oracle_summary.csv", index=False)
print(df[["method"] + [c for c in ("lambda_sel", "lambda_g", "C") if c in df] + ["test_accuracy", "map_corr", "n_selected", "f1"]].to_string(index=False))
