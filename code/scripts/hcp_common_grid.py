"""Replay the four-member HCP common-grid report from completed original fits."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import _bootstrap
import numpy as np
import pandas as pd
import yaml

report = importlib.import_module("paper_report")
rules = importlib.import_module("09_selection_rules_report")
MEMBERS = ("group_lasso", "graphnet", "graph_tv", "superquadratic")
RULES = ("min", "1se_repro")
MODE = "original_stopping_results"
MERGED_FILES = ("config.yaml", "inner_scores.csv", "outer_results.csv",
                "units/pooled_predictions.csv", "units/pooled_maps.npz")


def check_original_metrics(rows, source):
    markers = {"reported_solution_source", "precision_accepted", "precision_source"}
    if markers.intersection(rows):
        raise ValueError(f"Refinement markers in {source}; supply the original stopping results")


def require_original_primary(directory):
    receipt = directory / "source_receipt.json"
    if not receipt.is_file():
        raise ValueError(f"Original common-grid source receipt required in {directory}")
    source = json.loads(receipt.read_text(encoding="utf-8"))
    if source.get("mode") != MODE:
        raise ValueError(f"Original common-grid results required in {directory}")
    for name in MERGED_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if "sha256" in source and source["sha256"].get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Common-grid replay input changed: {path}")
    columns = pd.read_csv(directory / "outer_results.csv", nrows=0).columns
    check_original_metrics(columns, directory)


def paired_accuracy_tests(accuracy):
    tests = report.tests(accuracy, ["accuracy"], "wilcox")
    participants = accuracy.unit.nunique()
    resamples = np.random.default_rng(2036).integers(0, participants, (10000, participants))
    for i, row in tests.iterrows():
        pivot = accuracy[accuracy.rule.eq(row.rule)].pivot(index="unit", columns="method", values="accuracy")
        delta = (pivot[row.method] - pivot.group_lasso).to_numpy()
        tests.loc[i, ["ci_low", "ci_high"]] = np.percentile(delta[resamples].mean(axis=1), [2.5, 97.5])
    return tests


def merge_config(sources):
    apg, tv = [yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8")) for source in sources]
    methods = []
    for config in (apg, tv):
        experiment = config["experiment"]
        methods.extend(experiment.pop("methods"))
        for key in ("name", "n_jobs", "cache_provenance"):
            experiment.pop(key, None)
    if apg != tv or {method["name"] for method in methods} != set(MEMBERS) or len(methods) != len(MEMBERS):
        raise ValueError("APG and TV runs must contain the four paper members and matching scientific settings")
    apg["experiment"].update(name="hcp181_primary", n_jobs=1, methods=sorted(methods, key=lambda m: MEMBERS.index(m["name"])))
    return apg


def replay_primary(sources, dest, *, replace=False):
    if not replace and dest.exists() and any(dest.iterdir()):
        raise ValueError(f"Output directory must be empty: {dest}")
    config = merge_config(sources)
    frames, predictions, inner_frames, maps = [], [], [], {}
    for source in sources:
        frame = pd.read_csv(source / "outer_results.csv")
        check_original_metrics(frame.columns, source)
        if frame.error.fillna("").ne("").any():
            raise ValueError(f"Original outer fit errors in {source}")
        frames.append(frame)
        inner_frames.append(pd.read_csv(source / "inner_scores.csv"))
        predictions.append(pd.read_csv(source / "units/pooled_predictions.csv"))
        with np.load(source / "units/pooled_maps.npz") as arrays:
            if maps.keys() & set(arrays.files):
                raise ValueError(f"Repeated method maps in {source}")
            maps.update({name: arrays[name] for name in arrays.files})

    outer = pd.concat(frames, ignore_index=True)
    pred = pd.concat(predictions, ignore_index=True)
    names = set(MEMBERS) | {f"{name}@1se_repro" for name in MEMBERS}
    assert set(outer.method) == set(pred.method) == maps.keys() == names
    assert set(outer.outer_fold) == set(range(5))
    assert len(outer) == 40 and not outer.duplicated(["method", "outer_fold"]).any()
    assert pred.subject.nunique() == 181 and not pred.duplicated(["sample", "method"]).any()
    assert pred.groupby(["subject", "method"]).size().eq(64).all() and len(pred) == 181 * 64 * len(names)
    base = pred[pred.method.eq("group_lasso")].sort_values("sample")
    for _, frame in pred.groupby("method"):
        assert np.array_equal(
            frame.sort_values("sample")[["sample", "subject", "y_true", "outer_fold"]],
            base[["sample", "subject", "y_true", "outer_fold"]],
        )
    assert pred.groupby("subject").outer_fold.nunique().eq(1).all()
    assert all(m.shape == (5, 4801, 4) and np.isfinite(m).all() for m in maps.values())
    observed = pred.assign(correct=pred.y_true.eq(pred.y_pred)).groupby(["outer_fold", "method"]).correct.mean()
    expected = outer.set_index(["outer_fold", "method"]).accuracy
    assert observed.sort_index().index.equals(expected.sort_index().index)
    assert np.allclose(observed.sort_index(), expected.sort_index(), atol=1e-12, rtol=0)
    outer["sel_zero_fraction"] = [float(np.mean(maps[row.method][row.outer_fold] == 0)) for row in outer.itertuples()]

    units = dest / "units"
    summary_dir = dest / "report"
    units.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(exist_ok=True)
    outer.to_csv(dest / "outer_results.csv", index=False)
    pd.concat(inner_frames, ignore_index=True).to_csv(dest / "inner_scores.csv", index=False)
    (dest / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pred.to_csv(units / "pooled_predictions.csv", index=False)
    np.savez_compressed(units / "pooled_maps.npz", **maps)
    acc = pred.assign(correct=pred.y_true.eq(pred.y_pred)).groupby(["subject", "method"]).correct.mean()
    accuracy = report.split_names(acc.rename("accuracy").reset_index().rename(columns={"subject": "unit"}))
    folds = report.split_names(outer.merge(rules.map_stability(dest), on=["unit", "method"], validate="many_to_one")
                               .rename(columns={"sel_zero_fraction": "zeros"}))
    tests = paired_accuracy_tests(accuracy)
    sums = report.summarize(folds, ["log_loss", "stability", "zeros"]).drop(columns="n").merge(
        report.summarize(accuracy, ["accuracy"]), on=["rule", "method"], validate="one_to_one"
    )
    for data, name in [(accuracy, "accuracy"), (folds, "folds"), (tests, "tests"), (sums, "summary")]:
        data.to_csv(summary_dir / f"primary_{name}.csv", index=False)
    receipt = {"mode": MODE, "sources": [str(path.resolve()) for path in sources],
               "participants": 181, "fits_replayed": len(outer), "new_fit_calls": 0,
               "sha256": {name: hashlib.sha256((dest / name).read_bytes()).hexdigest() for name in MERGED_FILES}}
    (dest / "source_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apg-run", type=Path, required=True)
    parser.add_argument("--tv-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    replay_primary([args.apg_run, args.tv_run], args.output_dir)
    importlib.import_module("update_hcp181_report").render_primary(
        args.output_dir, args.output_dir / "report", prefix=""
    )


if __name__ == "__main__":
    main()
