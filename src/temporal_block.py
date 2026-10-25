"""Time-binned trajectory features for labs and vitals (notebook 15).

`mimic_iv._aggregate_events` summarises each concept over the whole
observation window (first/last/min/max/mean/count). YAIB and DuETT
(RESEARCH.md prompt 1) both find the *trajectory* inside the window carries
signal those whole-window summaries lose. This adds, per concept:

    <prefix>_<concept>_b{0..n_bins-1}   mean inside each equal-width time bin
    <prefix>_<concept>_slope           least-squares slope, value per hour
    <prefix>_<concept>_hrs_since_last  window_end minus the last measurement, hours

Same window discipline as `mimic_iv` (events strictly inside
[intime, window_end)), same per-concept plausibility ranges
(`mimic_concepts` items: (concept, unit, lo, hi)), multi-itemid concepts
pooled at the concept level. Needs DuckDB.
"""

import numpy as np
import pandas as pd

from src import mimic_iv


def temporal_block(root, df, rel, id_col, time_col, value_col, items, prefix,
                   window_h=24.0, n_bins=4, verbose=True):
    """-> DataFrame [len(df), n_concepts * (n_bins + 2)], rows aligned with df."""
    concepts = sorted({v[0] for v in items.values()})
    cols = [f"{prefix}_{c}_{s}" for c in concepts
            for s in [f"b{b}" for b in range(n_bins)] + ["slope", "hrs_since_last"]]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols, dtype=np.float32)
    path = mimic_iv.table_path(root, rel)
    if path is None:
        if verbose:
            print(f"  [{prefix}_t] {rel} absent -- block skipped")
        return out
    con = mimic_iv._duckdb()
    assert con is not None, "temporal_block needs duckdb"

    win = df[[id_col, "intime", "window_end"]].copy()
    win["row_id"] = np.arange(len(df))
    rng = pd.DataFrame([(int(i), v[0], float(v[2]), float(v[3])) for i, v in items.items()],
                       columns=["itemid", "concept", "lo", "hi"])
    con.register("win", win)
    con.register("rng", rng)
    bw = window_h / n_bins
    bins = ",\n".join(
        f"avg(CASE WHEN hr >= {b * bw} AND hr < {(b + 1) * bw} THEN v END) AS b{b}"
        for b in range(n_bins))
    sql = f"""
        WITH ev AS (
            SELECT w.row_id, r.concept, CAST(e.{value_col} AS DOUBLE) AS v,
                   epoch(CAST(e.{time_col} AS TIMESTAMP) - CAST(w.intime AS TIMESTAMP)) / 3600.0 AS hr
            FROM read_csv_auto('{path}', ignore_errors=true) e
            JOIN win w ON e.{id_col} = w.{id_col}
            JOIN rng r ON e.itemid = r.itemid
            WHERE e.{value_col} IS NOT NULL
              AND e.{time_col} >= w.intime AND e.{time_col} < w.window_end
              AND e.{value_col} BETWEEN r.lo AND r.hi
        )
        SELECT row_id, concept, {bins},
               regr_slope(v, hr) AS slope,
               {window_h} - max(hr) AS hrs_since_last
        FROM ev GROUP BY row_id, concept
    """
    if verbose:
        print(f"  [{prefix}_t] duckdb scan of {rel} ...", flush=True)
    agg = con.execute(sql).fetch_df()
    con.close()
    for concept, blk in agg.groupby("concept"):
        rows = blk["row_id"].to_numpy()
        for s in [f"b{b}" for b in range(n_bins)] + ["slope", "hrs_since_last"]:
            out.iloc[rows, out.columns.get_loc(f"{prefix}_{concept}_{s}")] = \
                blk[s].to_numpy(dtype=np.float64).astype(np.float32)
    if verbose:
        print(f"  [{prefix}_t] {len(cols)} cols, {len(agg):,} (stay, concept) groups", flush=True)
    return out.astype(np.float32)
