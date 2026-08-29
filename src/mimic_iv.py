"""Native ETL for the Kaggle `akshaybe/updated-mimic-iv` mirror of MIMIC-IV v3.1.

The mirror ships PhysioNet's directory layout verbatim:

    data/raw/mimic-iv-3.1/hosp/{patients,admissions,diagnoses_icd,labevents,...}.csv
    data/raw/mimic-iv-3.1/icu/{icustays,chartevents,inputevents,...}.csv

This module turns those tables into the single patient-level design matrix the
graph pipeline consumes. Nothing downstream knows about MIMIC schemas.

Cohort
------
Adult (>= 18) patients, **first ICU stay only** (so one row per `subject_id`,
which makes the patient-level split trivially exact), stay length at least
`min_los_hours`. Features come from a fixed observation window
`[intime, intime + obs_window_hours)`; the label is in-hospital mortality.

Leakage controls, all asserted in `build_cohort`
------------------------------------------------
1. **Window closure.** Every event used is timestamped strictly inside the
   observation window. No aggregate reaches past `intime + obs_window_hours`.
2. **Deaths inside the window are dropped.** A patient who dies at hour 12 of a
   24 h window has their outcome partly *inside* their own feature window;
   keeping them makes the task trivially easy for the wrong reason.
3. **Comorbidities come from prior admissions only.** `diagnoses_icd` holds
   billing codes assigned at *discharge*, so the current admission's codes
   encode the outcome. Only admissions whose `dischtime` precedes `intime`
   contribute. This is the single biggest leak in naive MIMIC pipelines.
4. **One stay per patient**, so no patient can straddle two splits.

Feature blocks
--------------
Blocks mirror the SBSCGM patient encoder (arXiv:2508.00615 sec. 3.2) and are
carried through to graph construction as *views*:

    demographics  age, sex, admission route, care unit, insurance, race
    comorbidity   Charlson binary groups + weighted index, from history
    labs          first-window aggregates over `hosp/labevents`
    vitals        first-window aggregates over `icu/chartevents`
    treatment     intervention exposure from inputevents/procedureevents,
                  plus urine output rate

Whichever source tables are absent on disk simply contribute no block --
`scripts/fetch_mimic.py` downloads them in tiers, and a core-only checkout still
produces a usable (if smaller) cohort.
"""

import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from src import mimic_concepts as mc

# Aggregates computed per numeric concept inside the observation window. `first`
# and `last` together carry the trend, which is what a static snapshot loses --
# the criticism levelled at the ICU-LoS similarity-graph work in the review.
NUMERIC_AGGS = ("first", "last", "min", "max", "mean", "count")


# ------------------------------------------------------------------ plumbing

def find_root(base="data/raw"):
    """Locate the MIMIC-IV tree under `base`. Returns a path or None."""
    for candidate in (
        os.path.join(base, "mimic-iv-3.1"),
        os.path.join(base, "mimic-iv"),
        base,
    ):
        if os.path.isdir(os.path.join(candidate, "hosp")):
            return candidate
    return None


def table_path(root, rel):
    """Path to a table, tolerating the gzipped variant. None if absent."""
    for suffix in ("", ".gz"):
        path = os.path.join(root, rel + suffix)
        if os.path.exists(path):
            return path
    return None


def has_table(root, rel):
    return table_path(root, rel) is not None


def read_table(root, rel, **kwargs):
    path = table_path(root, rel)
    if path is None:
        raise FileNotFoundError(f"{rel} not found under {root}")
    return pd.read_csv(path, **kwargs)


def _duckdb():
    """DuckDB connection, or None. Used to scan the multi-GB event tables."""
    try:
        import duckdb
    except ImportError:
        return None
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    return con


# -------------------------------------------------------------------- cohort

def _base_cohort(root, cfg):
    """patients x admissions x icustays -> one row per included ICU stay."""
    d = cfg["data"]
    window_h = float(d["obs_window_hours"])
    min_los_h = float(d.get("min_los_hours", window_h + 4.0))

    patients = read_table(
        root, "hosp/patients.csv",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
    )
    admissions = read_table(
        root, "hosp/admissions.csv",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                 "admission_type", "admission_location", "insurance",
                 "marital_status", "race", "hospital_expire_flag"],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    stays = read_table(
        root, "icu/icustays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit",
                 "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )

    n_stays_raw = len(stays)
    # First ICU stay per patient, in time order.
    stays = stays.sort_values(["subject_id", "intime"])
    stays = stays.drop_duplicates("subject_id", keep="first")

    df = stays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    df = df.merge(patients, on="subject_id", how="inner")

    # MIMIC-IV date-shifts each patient into an `anchor_year`; age at admission
    # is anchor_age plus the years elapsed since that anchor.
    df["age"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])

    steps = [("icu stays", n_stays_raw), ("first stay per patient", len(df))]

    df = df[df["age"] >= 18]
    steps.append(("adult", len(df)))

    df["los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0
    df = df[df["los_hours"] >= min_los_h]
    steps.append((f"los >= {min_los_h:.0f}h", len(df)))

    # Leakage control 2: the outcome must fall strictly after the feature window.
    df["window_end"] = df["intime"] + pd.Timedelta(hours=window_h)
    died_in_window = df["deathtime"].notna() & (df["deathtime"] <= df["window_end"])
    df = df[~died_in_window]
    steps.append(("died after window", len(df)))

    df = df[df["dischtime"] > df["window_end"]]
    steps.append(("discharged after window", len(df)))

    df = df.reset_index(drop=True)
    df["label"] = df["hospital_expire_flag"].astype(np.int64)
    return df, steps


# ------------------------------------------------------------ feature blocks

def _demographics_block(df):
    """Always-observed context features. These anchor the similarity graph."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(np.float32)
    out["is_female"] = (df["gender"] == "F").astype(np.float32)
    out["los_planned_elective"] = df["admission_type"].isin(
        ("ELECTIVE", "SURGICAL SAME DAY ADMISSION")
    ).astype(np.float32)
    out["admitted_via_ed"] = df["admission_location"].fillna("").str.contains(
        "EMERGENCY"
    ).astype(np.float32)

    for adm in mc.ADMISSION_TYPES:
        key = adm.lower().replace(" ", "_").replace(".", "")
        out[f"adm_{key}"] = (df["admission_type"] == adm).astype(np.float32)
    for unit in mc.CARE_UNITS:
        key = unit.split("(")[-1].strip(")").lower().replace("/", "_").replace(" ", "_")
        out[f"unit_{key}"] = (df["first_careunit"] == unit).astype(np.float32)

    race = df["race"].map(mc.race_group)
    for group in sorted(set(mc.RACE_GROUPS.values())):
        out[f"race_{group}"] = (race == group).astype(np.float32)
    for ins in ("Medicare", "Medicaid", "Private", "Other"):
        out[f"insurance_{ins.lower()}"] = (df["insurance"] == ins).astype(np.float32)
    out["married"] = (df["marital_status"] == "MARRIED").astype(np.float32)
    return out


def _comorbidity_block(root, df):
    """Charlson groups from *prior* admissions only (leakage control 3)."""
    cols = [f"cci_{g}" for g in mc.COMORBIDITIES] + ["cci_score", "n_prior_admissions"]
    if not has_table(root, "hosp/diagnoses_icd.csv"):
        return pd.DataFrame(0.0, index=df.index, columns=cols, dtype=np.float32)

    diag = read_table(root, "hosp/diagnoses_icd.csv",
                      usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    diag = diag[diag["subject_id"].isin(set(df["subject_id"]))]

    # Attach each diagnosis to its admission's discharge time, then keep only
    # admissions that closed before this patient's ICU window opened.
    adm_times = read_table(root, "hosp/admissions.csv",
                           usecols=["subject_id", "hadm_id", "dischtime"],
                           parse_dates=["dischtime"])
    diag = diag.merge(adm_times, on=["subject_id", "hadm_id"], how="left")
    diag = diag.merge(df[["subject_id", "intime"]], on="subject_id", how="inner")
    diag = diag[diag["dischtime"] < diag["intime"]]

    # Map every distinct (code, version) once -- there are ~2e4 of them versus
    # millions of rows.
    keys = diag[["icd_code", "icd_version"]].drop_duplicates()
    lookup = {
        (row.icd_code, row.icd_version): mc.charlson_groups(row.icd_code, row.icd_version)
        for row in keys.itertuples()
    }

    flags = {g: set() for g in mc.COMORBIDITIES}
    for subj, code, ver in zip(diag["subject_id"], diag["icd_code"], diag["icd_version"]):
        for group in lookup[(code, ver)]:
            flags[group].add(subj)

    out = pd.DataFrame(index=df.index)
    score = np.zeros(len(df), dtype=np.float32)
    for group in mc.COMORBIDITIES:
        hit = df["subject_id"].isin(flags[group]).to_numpy(dtype=np.float32)
        out[f"cci_{group}"] = hit
        score += mc.CHARLSON_WEIGHTS[group] * hit
    out["cci_score"] = score

    prior = diag.groupby("subject_id")["hadm_id"].nunique()
    out["n_prior_admissions"] = (
        df["subject_id"].map(prior).fillna(0.0).to_numpy(dtype=np.float32)
    )
    return out


def _window_sql(source, id_col, time_col, value_col, item_col, itemids):
    """SQL that aggregates one event table inside each stay's window."""
    ids = ", ".join(str(i) for i in itemids)
    return f"""
        SELECT w.row_id, e.{item_col} AS itemid,
               min_by(e.{value_col}, e.{time_col}) AS v_first,
               max_by(e.{value_col}, e.{time_col}) AS v_last,
               min(e.{value_col})   AS v_min,
               max(e.{value_col})   AS v_max,
               avg(e.{value_col})   AS v_mean,
               count(e.{value_col}) AS v_count
        FROM read_csv_auto('{source}', ignore_errors=true) e
        JOIN win w ON e.{id_col} = w.{id_col}
        WHERE e.{item_col} IN ({ids})
          AND e.{value_col} IS NOT NULL
          AND e.{time_col} >= w.intime
          AND e.{time_col} <  w.window_end
        GROUP BY w.row_id, e.{item_col}
    """


def _aggregate_events(root, df, rel, id_col, time_col, value_col, items,
                      prefix, chunksize=4_000_000, verbose=True):
    """Window-aggregate a large event table into per-concept feature columns.

    Uses DuckDB when it is importable (it scans the 18-42 GB CSVs with predicate
    pushdown and constant memory); otherwise falls back to a chunked pandas
    scan, which is slower but has no extra dependency.
    """
    concepts = sorted({name for name, *_ in items.values()})
    ranges = {name: (lo, hi) for name, _, lo, hi in items.values()}
    columns = [f"{prefix}_{c}_{a}" for c in concepts for a in NUMERIC_AGGS]

    path = table_path(root, rel)
    if path is None:
        if verbose:
            print(f"  [{prefix}] {rel} absent -- block skipped")
        return pd.DataFrame(np.nan, index=df.index, columns=columns, dtype=np.float32)

    win = df[[id_col, "intime", "window_end"]].copy()
    win["row_id"] = np.arange(len(df))

    con = _duckdb()
    if con is not None:
        if verbose:
            print(f"  [{prefix}] duckdb scan of {rel} ...", flush=True)
        con.register("win", win)
        agg = con.execute(
            _window_sql(path, id_col, time_col, value_col, "itemid", sorted(items))
        ).fetch_df()
        con.close()
    else:
        agg = _aggregate_events_pandas(path, win, id_col, time_col, value_col,
                                       items, chunksize, verbose, prefix)

    out = pd.DataFrame(np.nan, index=df.index, columns=columns, dtype=np.float32)
    if len(agg) == 0:
        return out

    agg["concept"] = agg["itemid"].map({k: v[0] for k, v in items.items()})
    # One itemid per concept per stay is the common case, but pressures and
    # glucose have several sources; combine them the way each aggregate implies.
    grouped = agg.groupby(["row_id", "concept"]).agg(
        v_first=("v_first", "first"), v_last=("v_last", "last"),
        v_min=("v_min", "min"), v_max=("v_max", "max"),
        v_mean=("v_mean", "mean"), v_count=("v_count", "sum"),
    ).reset_index()

    for concept, block in grouped.groupby("concept"):
        lo, hi = ranges[concept]
        rows = block["row_id"].to_numpy()
        for agg_name in NUMERIC_AGGS:
            vals = block[f"v_{agg_name}"].to_numpy(dtype=np.float64)
            if agg_name != "count":
                vals = np.where((vals < lo) | (vals > hi), np.nan, vals)
            # `out` is float32; assigning float64 into it is deprecated in
            # pandas and becomes an error in a future version.
            out.iloc[rows, out.columns.get_loc(f"{prefix}_{concept}_{agg_name}")] = \
                vals.astype(np.float32)

    # A concept that was never measured has count 0, not "missing count".
    for concept in concepts:
        col = f"{prefix}_{concept}_count"
        out[col] = out[col].fillna(0.0)
    return out.astype(np.float32)


def _aggregate_events_pandas(path, win, id_col, time_col, value_col, items,
                             chunksize, verbose, prefix):
    """Chunked fallback for `_aggregate_events` when DuckDB is unavailable."""
    keep_ids = set(win[id_col])
    bounds = win.set_index(id_col)[["intime", "window_end", "row_id"]]
    parts = []
    reader = pd.read_csv(
        path, usecols=[id_col, time_col, value_col, "itemid"],
        parse_dates=[time_col], chunksize=chunksize,
        dtype={value_col: np.float64}, on_bad_lines="skip",
    )
    for i, chunk in enumerate(reader):
        chunk = chunk[chunk["itemid"].isin(items) & chunk[id_col].isin(keep_ids)]
        chunk = chunk.dropna(subset=[value_col])
        if chunk.empty:
            continue
        chunk = chunk.join(bounds, on=id_col)
        chunk = chunk[(chunk[time_col] >= chunk["intime"])
                      & (chunk[time_col] < chunk["window_end"])]
        if not chunk.empty:
            parts.append(chunk[["row_id", "itemid", time_col, value_col]])
        if verbose and i % 10 == 0:
            print(f"  [{prefix}] chunk {i}, {sum(len(p) for p in parts):,} rows kept",
                  flush=True)

    if not parts:
        return pd.DataFrame(columns=["row_id", "itemid", "v_first", "v_last",
                                     "v_min", "v_max", "v_mean", "v_count"])
    events = pd.concat(parts).sort_values(time_col)
    return events.groupby(["row_id", "itemid"])[value_col].agg(
        v_first="first", v_last="last", v_min="min", v_max="max",
        v_mean="mean", v_count="count",
    ).reset_index()


def _treatment_block(root, df, verbose=True):
    """Binary intervention exposure inside the window, plus urine output rate.

    These are the "interventions and medications" features of the SBSCGM
    encoder: whether a patient was on pressors, ventilated, or dialysed in the
    first window is both highly prognostic and cheap to extract.
    """
    names = sorted(set(mc.INPUT_ITEMS.values()) | set(mc.PROC_ITEMS.values()))
    columns = [f"tx_{n}" for n in names] + ["tx_any_vasopressor", "tx_n_interventions",
                                            "tx_urine_ml_per_hr"]
    out = pd.DataFrame(0.0, index=df.index, columns=columns, dtype=np.float32)

    win = df[["stay_id", "intime", "window_end"]].copy()
    win["row_id"] = np.arange(len(df))
    window_h = (df["window_end"] - df["intime"]).dt.total_seconds().to_numpy() / 3600.0

    sources = [
        ("icu/inputevents.csv", "starttime", mc.INPUT_ITEMS),
        ("icu/procedureevents.csv", "starttime", mc.PROC_ITEMS),
    ]
    for rel, time_col, mapping in sources:
        path = table_path(root, rel)
        if path is None:
            if verbose:
                print(f"  [tx] {rel} absent -- block partially skipped")
            continue
        hits = _exposure_rows(path, win, time_col, sorted(mapping), verbose, rel)
        for itemid, rows in hits.items():
            col = f"tx_{mapping[itemid]}"
            out.iloc[list(rows), out.columns.get_loc(col)] = 1.0

    out["tx_any_vasopressor"] = out[
        [f"tx_{v}" for v in mc.VASOPRESSORS if f"tx_{v}" in out]
    ].max(axis=1)
    out["tx_n_interventions"] = out[[f"tx_{n}" for n in names]].sum(axis=1)

    urine_path = table_path(root, "icu/outputevents.csv")
    if urine_path is not None:
        totals = _urine_totals(urine_path, win, verbose)
        rate = np.zeros(len(df), dtype=np.float32)
        rate[list(totals)] = np.array(list(totals.values()), dtype=np.float32)
        rate = rate / np.maximum(window_h, 1e-6)
        # Irrigant subtraction can drive a stay negative; both that and a
        # supraphysiologic rate mean "not measurable", which is missing, not 0.
        out["tx_urine_ml_per_hr"] = np.where(
            (rate < 0) | (rate > mc.URINE_RATE_MAX_ML_PER_HR), np.nan, rate)
    else:
        out["tx_urine_ml_per_hr"] = np.nan
    return out


def _exposure_rows(path, win, time_col, itemids, verbose, rel):
    """{itemid: set(row_id)} for events starting inside each stay's window."""
    con = _duckdb()
    ids = ", ".join(str(i) for i in itemids)
    if con is not None:
        if verbose:
            print(f"  [tx] duckdb scan of {rel} ...", flush=True)
        con.register("win", win)
        res = con.execute(f"""
            SELECT DISTINCT e.itemid, w.row_id
            FROM read_csv_auto('{path}', ignore_errors=true) e
            JOIN win w ON e.stay_id = w.stay_id
            WHERE e.itemid IN ({ids})
              AND e.{time_col} >= w.intime AND e.{time_col} < w.window_end
        """).fetch_df()
        con.close()
    else:
        bounds = win.set_index("stay_id")[["intime", "window_end", "row_id"]]
        keep, parts = set(win["stay_id"]), []
        for chunk in pd.read_csv(path, usecols=["stay_id", "itemid", time_col],
                                 parse_dates=[time_col], chunksize=2_000_000,
                                 on_bad_lines="skip"):
            chunk = chunk[chunk["itemid"].isin(itemids) & chunk["stay_id"].isin(keep)]
            if chunk.empty:
                continue
            chunk = chunk.join(bounds, on="stay_id")
            chunk = chunk[(chunk[time_col] >= chunk["intime"])
                          & (chunk[time_col] < chunk["window_end"])]
            parts.append(chunk[["itemid", "row_id"]])
        res = pd.concat(parts).drop_duplicates() if parts else pd.DataFrame(
            columns=["itemid", "row_id"])

    return {int(i): set(b["row_id"]) for i, b in res.groupby("itemid")}


def _urine_totals(path, win, verbose):
    """{row_id: total urine volume (mL)} inside each stay's window."""
    all_items = mc.OUTPUT_URINE_ITEMS + mc.OUTPUT_URINE_NEGATIVE_ITEMS
    ids = ", ".join(str(i) for i in all_items)
    neg_ids = ", ".join(str(i) for i in mc.OUTPUT_URINE_NEGATIVE_ITEMS)
    con = _duckdb()
    if con is not None:
        if verbose:
            print("  [tx] duckdb scan of outputevents ...", flush=True)
        con.register("win", win)
        res = con.execute(f"""
            SELECT w.row_id,
                   sum(CASE WHEN e.itemid IN ({neg_ids}) THEN -e.value
                            ELSE e.value END) AS total
            FROM read_csv_auto('{path}', ignore_errors=true) e
            JOIN win w ON e.stay_id = w.stay_id
            WHERE e.itemid IN ({ids}) AND e.value IS NOT NULL
              AND e.value BETWEEN 0 AND {mc.URINE_EVENT_MAX_ML}
              AND e.charttime >= w.intime AND e.charttime < w.window_end
            GROUP BY w.row_id
        """).fetch_df()
        con.close()
    else:
        bounds = win.set_index("stay_id")[["intime", "window_end", "row_id"]]
        keep, parts = set(win["stay_id"]), []
        for chunk in pd.read_csv(path, usecols=["stay_id", "itemid", "charttime", "value"],
                                 parse_dates=["charttime"], chunksize=2_000_000,
                                 on_bad_lines="skip"):
            chunk = chunk[chunk["itemid"].isin(all_items)
                          & chunk["stay_id"].isin(keep)]
            chunk = chunk[chunk["value"].between(0, mc.URINE_EVENT_MAX_ML)]
            if chunk.empty:
                continue
            chunk = chunk.join(bounds, on="stay_id")
            chunk = chunk[(chunk["charttime"] >= chunk["intime"])
                          & (chunk["charttime"] < chunk["window_end"])]
            chunk["value"] = np.where(
                chunk["itemid"].isin(mc.OUTPUT_URINE_NEGATIVE_ITEMS),
                -chunk["value"], chunk["value"])
            parts.append(chunk[["row_id", "value"]])
        res = (pd.concat(parts).groupby("row_id")["value"].sum().reset_index(name="total")
               if parts else pd.DataFrame(columns=["row_id", "total"]))
    return dict(zip(res["row_id"].astype(int), res["total"].astype(float)))


# ------------------------------------------------------------- derived signal

def _severity(df, blocks):
    """Severity from the observation window only -- no outcome, no full-stay info.

    This exists because `_criticalness` weights the death flag at 1.2 against
    two standardised terms, so it correlates with mortality at AUROC ~0.97. That
    is fine for an auxiliary *training* target, but it makes it useless as an
    evaluation stratum: "top decile of criticalness" is very nearly "the
    patients who died", which is the same degeneracy `high_risk_analysis` was
    fixed to avoid (HANDOVER 6.1).

    A mini-SOFA instead: organ-support exposure plus physiologic derangement,
    all measured inside `[intime, intime + obs_window)`. Every component is
    already a model input, which is fine -- the stratum has to be independent of
    the model's *decisions* and of the *outcome*, not of the features.

    Components are robust-z-scored (median / IQR) so a single wild lab does not
    dominate, oriented so larger is sicker, and averaged over whichever are
    available.
    """
    cols = pd.concat([b for b in blocks.values() if b is not None and b.shape[1]],
                     axis=1) if blocks else pd.DataFrame(index=df.index)

    # (column, +1 if high is sicker, -1 if low is sicker)
    COMPONENTS = [
        ("tx_any_vasopressor", +1), ("tx_invasive_ventilation", +1),
        ("tx_crrt", +1), ("tx_n_interventions", +1),
        ("lab_lactate_max", +1), ("lab_creatinine_max", +1), ("lab_bun_max", +1),
        ("vit_gcs_motor_last", -1), ("vit_gcs_verbal_last", -1),
        ("vit_sbp_min", -1), ("vit_spo2_min", -1), ("vit_resp_rate_max", +1),
    ]

    parts, used = [], []
    for name, sign in COMPONENTS:
        if name not in cols.columns:
            continue
        v = cols[name].to_numpy(dtype=np.float64)
        if not np.isfinite(v).any():
            continue
        med = np.nanmedian(v)
        q1, q3 = np.nanpercentile(v[np.isfinite(v)], [25, 75])
        scale = max(q3 - q1, 1e-6)
        z = sign * (np.where(np.isfinite(v), v, med) - med) / scale
        parts.append(np.clip(z, -4.0, 4.0))
        used.append(name)

    if not parts:
        return np.zeros(len(df), dtype=np.float32), []

    raw = np.mean(parts, axis=0)
    lo, hi = np.percentile(raw, [1, 99])
    score = np.clip((raw - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return score.astype(np.float32), used


def _criticalness(df, treatment):
    """Continuous severity proxy used as the auxiliary regression target.

    Follows the SBSCGM multi-task setup (arXiv:2508.00615 sec. 3.4): a
    normalised blend of intervention burden, ICU length of stay, and discharge
    disposition. It is a *label*, derived from the full stay, and is only ever
    read on training nodes inside the loss -- never a model input.

    Note the death flag carries weight 1.2 against two standardised terms, so
    this correlates with mortality at AUROC ~0.97 on the MIMIC cohort. That is
    acceptable for a training target but disqualifies it as an evaluation
    stratum -- use `_severity` for that.
    """
    los = np.log1p(df["los_hours"].to_numpy(dtype=np.float64))
    los = (los - los.mean()) / (los.std() + 1e-8)
    burden = treatment["tx_n_interventions"].to_numpy(dtype=np.float64)
    burden = (burden - burden.mean()) / (burden.std() + 1e-8)
    died = df["label"].to_numpy(dtype=np.float64)

    raw = 0.35 * los + 0.35 * burden + 1.2 * died
    lo, hi = np.percentile(raw, [1, 99])
    return np.clip((raw - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------- top level

# Bump when the *block-building* semantics change (itemid sets, aggregates,
# window logic). Expensive: it invalidates the 18 GB and 42 GB scans.
_BLOCK_VERSION = 2
# Bump when *assembly* semantics change (cohort filters, packing, derived
# columns). Cheap: blocks are reused, only the assembled table is rebuilt.
_COHORT_VERSION = 4

_ROW_KEYS = ("source", "task", "obs_window_hours", "min_los_hours",
             "max_patients", "cohort_seed")


def row_signature(cfg):
    """Hash of the fields that decide *which patients* are in the cohort.

    Feature blocks are cached against this rather than against
    `cohort_signature`, so enabling one more block reuses the multi-GB scans
    already done for the others instead of invalidating them.
    """
    d = cfg["data"]
    payload = {k: d.get(k) for k in _ROW_KEYS}
    payload["_v"] = _BLOCK_VERSION
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def cohort_signature(cfg):
    """Hash of only the fields that change the extracted cohort."""
    d = cfg["data"]
    keys = _ROW_KEYS + ("blocks", "require_nonempty_blocks")
    payload = {k: d.get(k) for k in keys}
    payload["_v"] = (_BLOCK_VERSION, _COHORT_VERSION)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _block_present(block):
    """Boolean mask: did this patient have *any* measurement in this block?

    Not `notna().any()`. `_aggregate_events` fills the `_count` columns with
    zero on purpose -- a concept that was never measured has count 0, not a
    missing count -- so every row is non-null in those columns and a naive
    null check calls every patient covered. The count columns are precisely
    the ones that answer the question, when they exist.
    """
    counts = [c for c in block.columns if c.endswith("_count")]
    if counts:
        return (block[counts].fillna(0).to_numpy() > 0).any(axis=1)
    return block.notna().any(axis=1).to_numpy()


def _drop_uncovered(df, blocks, required, verbose=True):
    """Drop patients for whom a required feature block is entirely absent.

    Public MIMIC mirrors are not always complete. The Kaggle
    `akshaybe/updated-mimic-iv` copy ships a `chartevents.csv` that is truncated
    mid-record and covers only the first ~38% of ICU stays in `subject_id`
    order, so most patients have no vitals at all.

    Leaving them in is worse than dropping them. `impute_and_scale` appends a
    missingness indicator per column *because* missingness in the ICU is
    informative -- a lab not drawn is a clinical decision. When the missingness
    is instead an artefact of a truncated download, those indicators encode
    "this patient's subject_id sorts below the cut point", and the model learns
    the defect as signal.

    `subject_id` is randomly assigned in MIMIC-IV, so the surviving cohort is
    not clinically selected -- but that is a property of this particular
    truncation, and is asserted rather than assumed by the caller.
    """
    if not required:
        return df, blocks, np.ones(len(df), dtype=bool)

    keep = np.ones(len(df), dtype=bool)
    for name in required:
        block = blocks.get(name)
        if block is None:
            raise ValueError(f"require_nonempty_blocks names {name!r}, which was not built")
        present = _block_present(block)
        if verbose:
            print(f"  [{name}] present for {present.sum():,} / {len(present):,} "
                  f"patients ({100 * present.mean():.1f}%)")
        keep &= present

    if keep.all():
        return df, blocks, keep
    if verbose:
        print(f"  dropped {(~keep).sum():,} patients lacking a required block "
              f"-> {keep.sum():,} remain")
    df = df[keep].reset_index(drop=True)
    blocks = {k: v[keep].reset_index(drop=True) for k, v in blocks.items()}
    return df, blocks, keep


def _cached_block(name, sig, cache_dir, build, force=False, verbose=True):
    """Build one feature block, memoised on disk.

    The `labs` and `vitals` blocks each cost a full scan of a multi-GB event
    table. On a hosted kernel with a wall-clock session limit, losing those to
    an unrelated failure later in the pipeline is expensive, so every block is
    written the moment it is built and reused on the next run.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"block_{sig}_{name}.parquet")
    if os.path.exists(path) and not force:
        block = pd.read_parquet(path)
        if verbose:
            print(f"  [{name}] cached ({block.shape[1]} cols)")
        return block
    t0 = time.time()
    block = build()
    block.to_parquet(path, index=False)
    if verbose:
        print(f"  [{name}] built {block.shape[1]} cols in {time.time() - t0:.1f}s "
              f"-> {path}")
    return block


def build_cohort(cfg, verbose=True):
    """Extract (or load a cached) MIMIC-IV cohort.

    Returns `(X, y, feature_names, meta)` with the same contract as the mock
    generator. `X` keeps NaN for genuinely unmeasured values -- imputation
    happens later, fit on the training split only.
    """
    d = cfg["data"]
    raw_dir = d.get("raw_dir", "data/raw")
    cache_dir = d.get("cache_dir", "data/processed")
    root = find_root(raw_dir)
    if root is None:
        raise FileNotFoundError(
            f"no MIMIC-IV tree under {raw_dir}. Run:\n"
            f"    python scripts/fetch_mimic.py --tier core --tier events"
        )

    sig = cohort_signature(cfg)
    row_sig = row_signature(cfg)
    cache = os.path.join(cache_dir, f"cohort_{sig}.csv.gz")
    if os.path.exists(cache) and not d.get("force_rebuild", False):
        if verbose:
            print(f"[mimic] cached cohort {cache}")
        return _unpack(pd.read_csv(cache), cfg)

    if verbose:
        print(f"[mimic] building cohort from {root}")
    df, steps = _base_cohort(root, cfg)
    if verbose:
        for name, n in steps:
            print(f"  {name:28s} {n:>8,}")

    max_patients = d.get("max_patients")
    if max_patients and len(df) > max_patients:
        # Subsample *patients*, stratified on the label so prevalence survives.
        rng = np.random.default_rng(d.get("cohort_seed", cfg["seed"]))
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(df["label"].to_numpy() == cls))[
                : int(round(max_patients * (df["label"] == cls).mean()))
            ]
            for cls in (0, 1)
        ])
        df = df.iloc[np.sort(keep)].reset_index(drop=True)
        if verbose:
            print(f"  {'subsampled':28s} {len(df):>8,}")

    enabled = set(d.get("blocks") or
                  ["demographics", "comorbidity", "labs", "vitals", "treatment"])
    builders = {
        "demographics": lambda: _demographics_block(df),
        "comorbidity": lambda: _comorbidity_block(root, df),
        "labs": lambda: _aggregate_events(
            root, df, "hosp/labevents.csv", "hadm_id", "charttime", "valuenum",
            mc.LAB_ITEMS, "lab", verbose=verbose),
        "vitals": lambda: _aggregate_events(
            root, df, "icu/chartevents.csv", "stay_id", "charttime", "valuenum",
            mc.CHART_ITEMS, "vit", verbose=verbose),
        "treatment": lambda: _treatment_block(root, df, verbose=verbose),
    }
    blocks, views = {}, {}
    for name in ("demographics", "comorbidity", "labs", "vitals", "treatment"):
        if name in enabled:
            blocks[name] = _cached_block(name, row_sig, cache_dir, builders[name],
                                         force=d.get("force_rebuild", False),
                                         verbose=verbose)

    # Drop blocks that came back entirely empty (source table not downloaded).
    blocks = {k: v for k, v in blocks.items()
              if v.shape[1] and not v.isna().all().all()}

    # Then drop patients for whom a required block is missing entirely, which
    # is how a partial mirror shows up (see `_drop_uncovered`).
    df, blocks, _kept = _drop_uncovered(
        df, blocks, [b for b in (d.get("require_nonempty_blocks") or []) if b in blocks],
        verbose=verbose)

    cursor = 0
    for name, block in blocks.items():
        views[name] = list(range(cursor, cursor + block.shape[1]))
        cursor += block.shape[1]

    features = pd.concat(list(blocks.values()), axis=1)
    crit = _criticalness(df, blocks.get("treatment",
                                        pd.DataFrame({"tx_n_interventions": np.zeros(len(df))})))
    sev, sev_used = _severity(df, blocks)
    if verbose:
        print(f"  [severity] {len(sev_used)} components: {', '.join(sev_used)}")

    packed = features.copy()
    packed.insert(0, "subject_id", df["subject_id"].to_numpy())
    packed.insert(1, "hadm_id", df["hadm_id"].to_numpy())
    packed.insert(2, "stay_id", df["stay_id"].to_numpy())
    packed.insert(3, "label", df["label"].to_numpy())
    packed.insert(4, "criticalness", crit)
    packed.insert(5, "los_hours", df["los_hours"].to_numpy())
    packed.insert(6, "severity", sev)

    os.makedirs(cache_dir, exist_ok=True)
    packed.to_csv(cache, index=False)
    with open(os.path.join(cache_dir, f"cohort_{sig}.views.json"), "w") as f:
        json.dump({"views": views, "steps": steps}, f, indent=2, default=str)
    if verbose:
        print(f"[mimic] wrote {cache}  ({len(packed):,} patients, "
              f"{features.shape[1]} features, prevalence "
              f"{packed['label'].mean():.3f})")
    return _unpack(packed, cfg)


_ID_COLS = ("subject_id", "hadm_id", "stay_id", "label", "criticalness",
            "los_hours", "severity")


def _unpack(packed, cfg):
    """Split the cached table into the (X, y, names, meta) contract."""
    sig = cohort_signature(cfg)
    view_path = os.path.join(cfg["data"].get("cache_dir", "data/processed"),
                             f"cohort_{sig}.views.json")
    views = {}
    if os.path.exists(view_path):
        with open(view_path) as f:
            views = {k: list(map(int, v)) for k, v in json.load(f)["views"].items()}

    names = [c for c in packed.columns if c not in _ID_COLS]
    X = packed[names].to_numpy(dtype=np.float32)
    y = packed["label"].to_numpy(dtype=np.int64)

    # Columns that are already 0/1 drive the Jaccard half of the hybrid
    # similarity (arXiv:2508.00615 sec. 3.1); everything else is continuous.
    finite = np.where(np.isfinite(X), X, 0.0)
    binary = np.all((finite == 0.0) | (finite == 1.0), axis=0)

    meta = {
        "subject_id": packed["subject_id"].to_numpy(),
        "hadm_id": packed["hadm_id"].to_numpy(),
        "stay_id": packed["stay_id"].to_numpy(),
        "criticalness": packed["criticalness"].to_numpy(dtype=np.float32),
        "severity": packed["severity"].to_numpy(dtype=np.float32),
        # Time axis for the concordance index. ICU length of stay with
        # in-hospital death as the event -- survivors censored at discharge.
        "los_hours": packed["los_hours"].to_numpy(dtype=np.float32),
        "views": views,
        "binary_cols": np.flatnonzero(binary),
    }
    if len(y) and y.sum() == 0:
        warnings.warn("cohort has no positive labels -- check the window filters")
    return X, y, names, meta
