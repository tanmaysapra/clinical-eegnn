"""The tabular half of the staged hybrid: one boosted score, grown in stages.

Torch-free on purpose -- nothing here may import torch, directly or
transitively (see `src/_omp_guard`).

The idea
--------
Everything measured on this cohort says the same thing (HANDOVER 5f-5j):
gradient boosting over the feature matrix is the strongest component, and
message passing over a graph *derived from that matrix* adds nothing on top of
it. A depth cascade therefore had nothing to choose between -- layer 1 was as
accurate as layer 10, so "early exit" could only ever be a way of trimming an
oversized budget.

So stop cascading over depth and cascade over **information**. Clinical
information does not arrive all at once or at equal cost:

    stage 0  bedside     demographics + vitals      free, continuous, minute 0
    stage 1  chart       + treatment + comorbidity  free, needs record linkage
    stage 2  labs        + laboratory panel         costs money and turnaround
    stage 3  cohort      + patient-graph residual   needs the whole cohort

Each stage is a full model over its own nested feature set. Unlike depth, the
stages are not redundant -- each is fitted on information the one before it
could not see -- so the exits differ in **accuracy** as well as in cost, which
is exactly what a depth cascade never had and what gives a gate something real
to decide.

Why the stages are independent models rather than boosting continuations
------------------------------------------------------------------------
The tidier design is one additive score, each stage handed its predecessor's
margin as `base_margin` and learning only the increment. That was implemented
and measured first, because "an additive score is natively truncatable" is a
much prettier argument for early exit. It does not work here: chaining the
margin degrades the final stage rather than improving it (mock cohort, test
AUROC 0.7954 / 0.8087 / 0.7939 across the three stages, against 0.7977 / 0.8286
/ 0.8318 for independent nested models, and 0.8285 for the reference XGBoost).
Two effects compound -- the base margin on training rows is out-of-fold while
the one at inference comes from a full-train model, so each continuation is
calibrated against a weaker base than it is applied to; and every stage piles
more trees onto an already-fitted score. Using the fold ensemble at inference
narrowed the gap but did not close it.

So the stages are independent, and the additive structure survives only where
it earns its place: the graph stage in `src/hybrid`, which genuinely is a
correction term on top of the final margin.

Out-of-fold discipline
----------------------
Training-row margins are still out-of-fold at every stage, and that is not
decoration: the graph stage propagates *residuals*, and in-sample residuals
collapse to ~0, leaving nothing to propagate. Fold models supply the training
margins; a full-train model supplies val/test.

Tree budgets rise with the stage (200/300/400). A stage with less information
has less to exploit, so this is the right capacity ordering and it also makes
prediction cost rise monotonically -- XGBoost's cost is O(trees x depth) and is
otherwise independent of how many features the stage may read.
"""

import time

import numpy as np

# Feature blocks in the order they become available, and the stage each one
# joins at. Cumulative: stage s sees every block from stages 0..s.
STAGE_BLOCKS = [
    ("bedside", ["demographics", "vitals"]),
    ("chart",   ["treatment", "comorbidity"]),
    ("labs",    ["labs"]),
]

XGB_PARAMS = dict(
    max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
)
# Capacity rises with information; see the module docstring.
STAGE_TREES = [200, 300, 400]


# --------------------------------------------------------------- stage columns

def stage_columns(views, n_raw, n_scaled, indicator_source=None):
    """Column indices of `X_scaled` belonging to each cumulative stage.

    `views` maps block name -> raw column indices (from `mimic_iv._unpack`).
    `impute_and_scale` appends one missingness indicator per raw column that
    had any NaN in train, so indicator columns inherit the block of the raw
    column they were built from -- `indicator_source[i]` is that raw column for
    the i-th appended indicator.

    Returns (list of index arrays, list of stage names).
    """
    block_of = np.full(n_raw, -1, dtype=int)
    names, sets = [], []
    for s, (name, blocks) in enumerate(STAGE_BLOCKS):
        names.append(name)
        for b in blocks:
            for j in views.get(b, []):
                if 0 <= j < n_raw:
                    block_of[j] = s
        sets.append(s)

    # Raw columns belonging to no declared block (shouldn't happen, but a
    # mirror missing a table can produce one) go to the last stage rather than
    # being silently dropped.
    block_of[block_of < 0] = len(STAGE_BLOCKS) - 1

    stage_of = np.empty(n_scaled, dtype=int)
    stage_of[:n_raw] = block_of
    if n_scaled > n_raw:
        src = np.asarray(indicator_source, dtype=int)
        assert len(src) == n_scaled - n_raw, (
            f"{len(src)} indicator sources for {n_scaled - n_raw} indicator columns"
        )
        stage_of[n_raw:] = block_of[src]

    cols = [np.flatnonzero(stage_of <= s) for s in range(len(STAGE_BLOCKS))]
    assert all(len(c) for c in cols), "a stage ended up with no columns"
    for a, b in zip(cols, cols[1:]):
        assert set(a.tolist()) <= set(b.tolist()), "stages must be nested"
    return cols, names


def indicator_sources(X_raw, train_mask):
    """Raw column index behind each appended missingness indicator.

    Mirrors `data_prep.impute_and_scale`'s expansion exactly; kept next to the
    stage logic because the two have to agree or features leak across stages.
    """
    missing = np.isnan(np.asarray(X_raw, dtype=np.float32))
    return np.flatnonzero(missing[train_mask].any(axis=0))


# ------------------------------------------------------------------ the stages

def _fit(X, y, cols, n_trees, seed, device):
    from xgboost import XGBClassifier

    clf = XGBClassifier(n_estimators=n_trees, random_state=seed, n_jobs=4,
                        device=device, tree_method="hist", **XGB_PARAMS)
    clf.fit(X[:, cols], y)
    return clf


def _margin(clf, X, cols):
    return clf.predict(X[:, cols], output_margin=True)


def fit_staged_boosting(X, y, train_mask, cols_per_stage, seed=42, n_folds=5,
                        device="cpu", verbose=True):
    """Fit one model per cumulative stage. -> dict of arrays.

    `margins[s]` is stage s's score for *every* node, honest on each split:
    out-of-fold on train, full-train-model on val/test. The out-of-fold half is
    what makes the graph stage's residuals meaningful.

    `cost_us[s]` is stage s's own measured prediction cost in microseconds per
    patient; reaching stage s costs the cumulative sum, since a patient has to
    run every earlier stage to be routed there. It is only half the story --
    the other half is that a patient
    resolved before the lab stage never needed the lab panel at all, which is
    the cost that matters clinically and does not show up in microseconds.
    """
    n = len(y)
    n_stages = len(cols_per_stage)
    tr = np.flatnonzero(train_mask)
    rest = np.flatnonzero(~train_mask)
    rng = np.random.default_rng(seed)

    # Label-stratified folds, so every fold model sees positives.
    fold = np.full(n, -1, dtype=int)
    for cls in (0, 1):
        idx = tr[y[tr] == cls]
        fold[rng.permutation(idx)] = np.arange(len(idx)) % n_folds

    margins = np.zeros((n_stages, n), dtype=np.float64)
    models, cost_us, n_cols = [], [], []

    for s in range(n_stages):
        cols = cols_per_stage[s]
        t0 = time.time()

        oof = np.zeros(n, dtype=np.float64)
        for f in range(n_folds):
            fit_idx = tr[fold[tr] != f]
            hold_idx = tr[fold[tr] == f]
            m = _fit(X[fit_idx], y[fit_idx], cols, STAGE_TREES[s], seed + f, device)
            oof[hold_idx] = _margin(m, X[hold_idx], cols)

        # One full-train model carries val/test, and is the model that ships.
        full = _fit(X[tr], y[tr], cols, STAGE_TREES[s], seed, device)
        t_fit = time.time() - t0

        t1 = time.perf_counter()
        m_rest = _margin(full, X[rest], cols)
        own_us = 1e6 * (time.perf_counter() - t1) / max(len(rest), 1)

        oof[rest] = m_rest
        margins[s] = oof
        models.append(full)
        # The stages are independent models, and a patient who exits at stage s
        # has still had to run every stage before it to get there -- so this is
        # stage s's own cost, and reaching stage s costs the cumulative sum.
        cost_us.append(own_us)
        n_cols.append(int(len(cols)))
        if verbose:
            print(f"  [stage {s}] {len(cols):4d} cols  {STAGE_TREES[s]} trees  "
                  f"fit {t_fit:5.1f}s  predict {cost_us[-1]:6.1f} us/patient")

    return {"margins": margins, "models": models, "cost_us": np.array(cost_us),
            "n_cols": np.array(n_cols), "oof_folds": fold}


def leaf_indices(model, X, cols):
    """Per-tree leaf assignment, [n, n_trees]."""
    import xgboost as xgb
    return model.get_booster().predict(
        xgb.DMatrix(X[:, cols]), pred_leaf=True).astype(np.int32)


def leaf_onehot(leaves):
    """One-hot the leaf codes into a [n, sum_trees(max_leaf)] indicator matrix.

    Cosine similarity over these rows is the share of trees in which two
    patients land in the same leaf -- similarity in the geometry the booster
    actually learned, rather than in raw feature space.

    That distinction is the whole reason to bother. A kNN graph over `X` is
    redundant with `X`, which is HANDOVER 5f's diagnosis of why every graph
    model so far has failed to beat gradient boosting. A graph over leaf
    co-occurrence is built from the *fitted model*, so "these patients are
    treated alike by the booster" is the relation, and asking whether the
    booster is systematically wrong on such a group is a question the booster
    cannot ask itself.
    """
    leaves = np.asarray(leaves, dtype=np.int64)
    n, t = leaves.shape
    width = leaves.max(axis=0) + 1
    offset = np.concatenate([[0], np.cumsum(width)[:-1]])
    cols = (leaves + offset[None, :]).reshape(-1)
    out = np.zeros((n, int(width.sum())), dtype=np.float32)
    out[np.repeat(np.arange(n), t), cols] = 1.0
    return out


def reference_xgboost(X, y, train_mask, seed=42, device="cpu"):
    """The incumbent to beat: one XGBoost on everything, as in `baselines`."""
    from xgboost import XGBClassifier

    tr = np.flatnonzero(train_mask)
    pos = max(int((y[tr] == 1).sum()), 1)
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=float((y[tr] == 0).sum() / pos),
                        eval_metric="logloss", random_state=seed, n_jobs=4,
                        device=device, tree_method="hist")
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X)[:, 1]
