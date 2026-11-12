"""Class maps of the temporal-TV and nonconvex members for one simulated data set (1se-repro rule).

The members study (results/sim_studies/members_2026-2035) fitted these two members on the base setting
but did not keep their maps.  The grid point chosen there on the validation set with the 1se-repro rule
is refitted here for one seed on the same data, so that Figure "simulation maps" can show all members.

Usage:
    python code/scripts/17_simulation_member_maps.py [--seed 2026] [--rule 1se_repro]

Output: results/sim_shapes_repro_wide/selected_maps_members.npz with keys seed<seed>__<rule>__<method>.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import _bootstrap
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("sb", Path(__file__).with_name("05_simulation_benchmark.py"))
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)
from mlrdegg.experiment.models import Context, build_graphs
from mlrdegg.paths import RESULTS_ROOT

METHODS = ("graph_tv_tm", "nonconvex")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--rule", default="1se_repro")
ap.add_argument("--study", default="members_2026-2035")
ap.add_argument("--out", default="sim_shapes_repro_wide")
args = ap.parse_args()

study = RESULTS_ROOT / "sim_studies" / args.study
cfg = _bootstrap.load_config(study / "base_config.yaml")
data = sb.simulate({**cfg["simulation"], "seed": args.seed})
ref = np.load(RESULTS_ROOT / args.out / f"simulated_data_seed{args.seed}.npz")
assert np.allclose(ref["patterns"], data["patterns"]) and np.allclose(ref["Xtr"], data["Xtr"]), \
    "simulated data differ from the saved data of the figure"
sg, tg = build_graphs(cfg["graph"], {"coords": data["coords"]}, cfg["simulation"]["n_time"])
ctx = Context(len(data["coords"]), cfg["simulation"]["n_time"], sg, tg)

fits = pd.read_csv(study / "fits.csv")
fits = fits[(fits.seed == args.seed) & (fits.error.fillna("") == "")]
out = {}
for m in cfg["experiment"]["methods"]:
    if m["name"] not in METHODS:
        continue
    ok = fits[fits.method == m["name"]]
    row = sb.select(ok, m, args.rule)
    params = {k: float(row[k]) for k in m["grid"]}
    print(m["name"], params, "val log-loss", round(float(row.val_log_loss), 4))
    r = sb.fit_one(m, params, data, ctx, cfg["experiment"]["solver"], {}, time.time() + 1800, args.rule, args.seed)
    assert not r.get("error"), r
    print("  refit test accuracy %.3f (members study: %.3f)" % (r["test_accuracy"], row.test_accuracy))
    out[f"seed{args.seed}__{args.rule}__{m['name']}"] = r.pop("_coef")
assert set(k.split("__")[-1] for k in out) == set(METHODS), "methods missing in base_config.yaml"
np.savez_compressed(RESULTS_ROOT / args.out / "selected_maps_members.npz", **out)
print("wrote", RESULTS_ROOT / args.out / "selected_maps_members.npz")
