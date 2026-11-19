"""Replay ERP zero-graph selection, paired results and table rows from saved fits."""

import argparse
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "scripts"))

import _bootstrap
import pandas as pd
from erp_zero_reporting import analyze, read_sources, select_with_zero

tables = importlib.import_module("48_format_erp_zero_tables")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-run", type=Path, required=True)
    parser.add_argument("--zero-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manuscript-reference", type=Path, help="Optional saved ERP primary-table CSV")
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error("The output directory must be empty")

    config, original, inner, fresh = read_sources(args.positive_run)
    choices, _ = select_with_zero(config, original, inner)
    columns = ["unit", "outer_fold", "method", "rule", "canonical_method", "target_id", "lambda_sel", "lambda_g"]
    expected = pd.read_csv(args.zero_run / "selection.csv", usecols=columns)
    replayed = pd.DataFrame(choices, columns=columns)
    order = ["unit", "outer_fold", "method"]
    pd.testing.assert_frame_equal(replayed.sort_values(order).reset_index(drop=True),
                                  expected[columns].sort_values(order).reset_index(drop=True))
    output.mkdir(parents=True, exist_ok=True)
    replayed.to_csv(output / "selection.csv", index=False)
    analyze(args.positive_run, args.zero_run, output, original, fresh, args.manuscript_reference)
    shutil.copyfile(args.zero_run / "validation.json", output / "validation.json")
    tables.render_tables(output, output / "erp_zero_tables.tex", output / "table widths")


if __name__ == "__main__":
    main()
