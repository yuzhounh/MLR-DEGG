"""Read-only reconstruction of reported quantities from saved final outputs."""
from pathlib import Path

import numpy as np
import pandas as pd


def prediction_support_from_maps(out: Path, outer: pd.DataFrame) -> pd.DataFrame:
    """Audit historical auxiliary zeros against saved prediction importance maps.

    Historical CSVs without sel_aux_zero_fraction used U for sel_zero_fraction.
    Saved maps contain norms of B/W, so exact zero support can be reconstructed.
    They cannot recover U masks, signed DB or feasibility norms.
    """
    rows = []
    for unit, group in outer.groupby("unit", sort=False):
        with np.load(Path(out) / "units" / f"{unit}_maps.npz") as maps:
            for _, record in group.iterrows():
                importance = maps[record["method"]][int(record["outer_fold"])]
                if not np.isfinite(importance).all() or (importance < 0).any():
                    raise ValueError(f"Invalid importance map: {unit}, {record['method']}")
                actual = float(np.mean(importance == 0))
                auxiliary = record.get("sel_aux_zero_fraction", record.get("sel_zero_fraction", np.nan))
                rows.append({
                    "unit": unit, "outer_fold": record["outer_fold"], "method": record["method"],
                    "sel_zero_fraction": actual, "sel_aux_zero_fraction": auxiliary,
                    "zero_fraction_difference": actual - auxiliary,
                    "n_prediction_rows": importance.size,
                    "support_source": "saved_prediction_importance",
                })
    return pd.DataFrame(rows)
