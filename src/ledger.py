"""The immutable per-patient prediction + operation ledger.

Why this exists
----------------
Architecture 4R's own design requires that "recomputing all metrics from the
immutable per-patient ledger must reproduce the aggregate file exactly" as
one of its mandatory integrity tests. This module is the ledger: one row per
(patient, arm, stage) with the prediction and the directly-measured operation
counts that produced it, plus a `recompute_aggregate` that takes only those
raw rows and a metric function -- never the driver's own in-memory arrays --
so "the aggregate is reproducible from the ledger" is actually exercised, not
just asserted.

Stored as CSV rather than parquet: `requirements.txt` does not pin a parquet
engine (pyarrow/fastparquet), and CSV via pandas round-trips float64 exactly
under Python's `repr`-based formatting, which is what this module relies on
for the "exact reproduction" test rather than an approximate one.
"""

from pathlib import Path

import numpy as np
import pandas as pd

_COLUMNS = (
    "patient_id", "split", "stage", "p_hat", "y", "route_flag",
    "edges_touched", "nodes_visited", "queue_pops", "wall_clock_s",
)


class PredictionLedger:
    """Append-only per-patient rows for one (seed, architecture) run."""

    def __init__(self, seed: int, arch: str):
        self.seed = int(seed)
        self.arch = str(arch)
        self._rows = []

    def add_row(self, patient_id, split, stage, p_hat, y, route_flag,
                edges_touched=0, nodes_visited=0, queue_pops=0,
                wall_clock_s=0.0) -> None:
        self._rows.append({
            "patient_id": patient_id,
            "split": split,
            "stage": int(stage),
            "p_hat": float(p_hat),
            "y": int(y),
            "route_flag": route_flag,
            "edges_touched": int(edges_touched),
            "nodes_visited": int(nodes_visited),
            "queue_pops": int(queue_pops),
            "wall_clock_s": float(wall_clock_s),
        })

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame(self._rows, columns=_COLUMNS)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        df.insert(0, "arch", self.arch)
        df.insert(0, "seed", self.seed)
        df.to_csv(path, index=False)

    @staticmethod
    def load(path: str | Path) -> "PredictionLedger":
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"ledger at {path} has no rows")
        seeds = df["seed"].unique()
        archs = df["arch"].unique()
        if len(seeds) != 1 or len(archs) != 1:
            raise ValueError(
                f"ledger at {path} must hold exactly one (seed, arch); "
                f"found seeds={seeds}, archs={archs}"
            )
        ledger = PredictionLedger(seed=int(seeds[0]), arch=str(archs[0]))
        ledger._rows = df[list(_COLUMNS)].to_dict("records")
        return ledger


def recompute_aggregate(ledger_df: pd.DataFrame, metric_fn) -> dict:
    """Recompute aggregate metrics purely from raw per-patient ledger rows.

    `metric_fn(y, p_hat) -> dict` is any of `src.evaluate`'s metric
    functions. This function does no more than pull `y`/`p_hat` back out of
    the stored rows and hand them to `metric_fn` -- there is no shortcut back
    to the driver's own in-memory arrays, which is what makes "the ledger
    reproduces the aggregate exactly" a real test rather than a tautology.
    """
    y = ledger_df["y"].to_numpy(dtype=np.int64)
    p_hat = ledger_df["p_hat"].to_numpy(dtype=np.float64)
    return metric_fn(y, p_hat)
