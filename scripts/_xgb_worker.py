#!/usr/bin/env python
"""Fit XGBoost (and optionally SHAP) in a torch-free process.

Exists purely to keep OpenMP runtimes apart -- see src/_omp_guard.py. Nothing
here may import torch, directly or transitively.

    python scripts/_xgb_worker.py <input.npz> <output.npz>

Two modes, selected by the optional `mode` field in the input npz (absent ->
"plain", for backward compatibility with every existing caller):

  "plain"  -- one XGBoost over every column. Original behaviour, untouched.
  "staged" -- `src.tabular.fit_staged_boosting` over per-stage column sets,
              for rerunning the 3-stage cascade baseline as a paired
              comparator alongside torch-based Architecture 4R code that
              cannot import xgboost in-process.
"""

import sys

import numpy as np


def _main_plain(d, out_path):
    X, y = d["X"], d["y"]
    train, val, test = d["train"], d["val"], d["test"]
    seed = int(d["seed"])
    want_shap = bool(d["want_shap"])

    from xgboost import XGBClassifier

    pos = max(int((y[train] == 1).sum()), 1)
    clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=float((y[train] == 0).sum() / pos),
        eval_metric="logloss", random_state=seed, n_jobs=4,
    )
    clf.fit(X[train], y[train])

    out = {
        "p_val": clf.predict_proba(X[val])[:, 1],
        "p_test": clf.predict_proba(X[test])[:, 1],
    }
    if want_shap:
        import shap
        out["shap_values"] = shap.TreeExplainer(clf).shap_values(X[test])

    np.savez(out_path, **out)


def _main_staged(d, out_path):
    # Local import: keeps `src.tabular` (torch-free by contract) out of the
    # "plain" path's import graph too, not just out of the caller's process.
    sys.path.insert(0, __import__("os").path.dirname(
        __import__("os").path.dirname(__import__("os").path.abspath(__file__))))
    from src.tabular import fit_staged_boosting

    X, y = d["X"], d["y"]
    train = d["train"]
    seed = int(d["seed"])
    n_folds = int(d["n_folds"])
    device = str(d["device"])
    cols_per_stage = [np.asarray(c) for c in d["cols_per_stage"]]

    result = fit_staged_boosting(
        X, y, train, cols_per_stage, seed=seed, n_folds=n_folds,
        device=device, verbose=False,
    )
    np.savez(
        out_path,
        margins=result["margins"],
        cost_us=result["cost_us"],
        n_cols=result["n_cols"],
        oof_folds=result["oof_folds"],
    )


def main(in_path, out_path):
    # allow_pickle=True: this file is written moments earlier by the same
    # trusted parent process (src._omp_guard) into its own tempdir, never by
    # an external/untrusted source -- needed only to round-trip the ragged
    # cols_per_stage list for "staged" mode.
    d = np.load(in_path, allow_pickle=True)
    mode = str(d["mode"]) if "mode" in d.files else "plain"
    if mode == "plain":
        _main_plain(d, out_path)
    elif mode == "staged":
        _main_staged(d, out_path)
    else:
        raise ValueError(f"unknown xgb worker mode: {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
