#!/usr/bin/env python
"""Fit XGBoost (and optionally SHAP) in a torch-free process.

Exists purely to keep OpenMP runtimes apart -- see src/_omp_guard.py. Nothing
here may import torch, directly or transitively.

    python scripts/_xgb_worker.py <input.npz> <output.npz>
"""

import sys

import numpy as np


def main(in_path, out_path):
    d = np.load(in_path, allow_pickle=True)
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


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
