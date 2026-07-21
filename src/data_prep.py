"""Cohort loading, imputation, scaling and splits.

Framing is node-level classification on one patient-similarity graph: every
patient is a node, training is transductive over boolean masks.

Leakage rules enforced here:
  * the feature window strictly precedes the label event (MIMIC extraction),
  * imputation and scaling statistics are fit on the train split only,
  * splits are disjoint (asserted).
"""

import hashlib
import json

import numpy as np

from src.mock_data import make_mock_cohort


def load_cohort(cfg):
    """Dispatch on cfg["data"]["source"]. Returns (X, y, feature_names, meta)."""
    source = cfg["data"]["source"]
    if source == "mock":
        return make_mock_cohort(
            n_patients=cfg["data"]["n_patients"],
            missing_rate=cfg["data"]["missing_rate"],
            seed=cfg["seed"],
        )
    if source == "mimic":
        return _load_mimic(cfg)
    raise ValueError(f"unknown data.source: {source}")


def _load_mimic(cfg):
    """Load a pre-extracted MIMIC-IV cohort table.

    Extraction SQL lives in data/ and is run offline; this only reads the
    resulting CSV so the loader stays dumb. Expected columns: subject_id,
    label, and one column per feature.
    """
    import pandas as pd

    path = cfg["data"].get("csv_path", "data/raw/cohort.csv")
    df = pd.read_csv(path)
    label_col = "label"
    id_cols = [c for c in ("subject_id", "hadm_id", "stay_id") if c in df.columns]
    feature_names = [c for c in df.columns if c not in id_cols + [label_col]]
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int64)
    return X, y, feature_names, {}


def make_splits(y, seed, fracs=(0.6, 0.2, 0.2)):
    """Stratified 60/20/20 split. Returns boolean masks (train, val, test)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_tr = int(round(fracs[0] * len(idx)))
        n_va = int(round(fracs[1] * len(idx)))
        train[idx[:n_tr]] = True
        val[idx[n_tr:n_tr + n_va]] = True
        test[idx[n_tr + n_va:]] = True

    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    assert (train | val | test).all(), "every patient must land in exactly one split"
    return train, val, test


def impute_and_scale(X, train_mask, add_missing_indicators=True):
    """Median-impute + standardise, fitting statistics on the train split only.

    Missingness in the ICU is informative (a lab not drawn is a decision), so
    indicator columns are appended rather than discarded.
    """
    X = np.asarray(X, dtype=np.float32)
    missing = np.isnan(X)

    med = np.nanmedian(X[train_mask], axis=0)
    med = np.where(np.isnan(med), 0.0, med)  # column entirely NaN in train
    Xi = np.where(missing, med, X)

    mu = Xi[train_mask].mean(axis=0)
    sd = Xi[train_mask].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xi = (Xi - mu) / sd

    if add_missing_indicators:
        keep = missing[train_mask].any(axis=0)
        if keep.any():
            Xi = np.concatenate([Xi, missing[:, keep].astype(np.float32)], axis=1)
    return Xi.astype(np.float32)


def feature_names_after_scaling(names, X_raw, train_mask, add_missing_indicators=True):
    """Mirror of impute_and_scale's column expansion, for SHAP labelling."""
    if not add_missing_indicators:
        return list(names)
    missing = np.isnan(np.asarray(X_raw, dtype=np.float32))
    keep = missing[train_mask].any(axis=0)
    return list(names) + [f"{names[j]}_missing" for j in np.flatnonzero(keep)]


def config_hash(cfg):
    """Stable short hash of a config dict -- used for caching and result logs."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
