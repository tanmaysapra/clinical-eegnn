"""Unit tests for the staged hybrid cascade.

The failures these guard against are the silent kind: a stage that can see a
later stage's features, a residual that lets a node read its own label back
through the graph, and a safety certificate that certifies nothing.
"""

import numpy as np
import pytest
import torch

from src import cascade, hybrid, tabular
from src.data_prep import impute_and_scale, make_splits
from src.mock_data import make_mock_cohort


@pytest.fixture(scope="module")
def cohort():
    X_raw, y, names, meta = make_mock_cohort(600, missing_rate=0.2, seed=0)
    train, val, test = make_splits(y, seed=0)
    X = impute_and_scale(X_raw, train)
    cut = np.array_split(np.arange(X_raw.shape[1]), 5)
    views = dict(zip(["demographics", "vitals", "treatment", "comorbidity", "labs"],
                     [c.tolist() for c in cut]))
    cols, names_ = tabular.stage_columns(
        views, X_raw.shape[1], X.shape[1], tabular.indicator_sources(X_raw, train))
    return dict(X_raw=X_raw, X=X, y=y, train=train, val=val, test=test,
                cols=cols, stage_names=names_, views=views)


# ---------------------------------------------------------------- stage columns

def test_stages_are_nested_and_cover_every_column(cohort):
    cols = cohort["cols"]
    for a, b in zip(cols, cols[1:]):
        assert set(a.tolist()) < set(b.tolist()), "each stage must strictly grow"
    assert set(cols[-1].tolist()) == set(range(cohort["X"].shape[1])), \
        "the last stage must see every column"


def test_missingness_indicators_inherit_their_block(cohort):
    """An indicator must not smuggle a later block's information into stage 0."""
    n_raw = cohort["X_raw"].shape[1]
    src = tabular.indicator_sources(cohort["X_raw"], cohort["train"])
    labs = set(cohort["views"]["labs"])
    stage0 = set(cohort["cols"][0].tolist())
    for i, raw_col in enumerate(src):
        if raw_col in labs:
            assert (n_raw + i) not in stage0, \
                "a lab column's missingness indicator leaked into the bedside stage"


# --------------------------------------------------------------- the graph stage

def test_correct_and_smooth_never_reads_a_held_out_label(cohort):
    """Perturbing a val/test label must not move any prediction at all."""
    n = len(cohort["y"])
    ei = np.stack([np.arange(n), (np.arange(n) + 1) % n])
    base = np.full(n, 0.3)
    y2 = cohort["y"].copy()
    flip = np.flatnonzero(cohort["test"])[:20]
    y2[flip] = 1 - y2[flip]
    kw = dict(edge_index=ei, edge_weight=None, base_prob=base,
              train_mask=cohort["train"], n=n, scale=1.0, hops=2)
    p1 = hybrid.correct_and_smooth(y=cohort["y"], **kw)
    p2 = hybrid.correct_and_smooth(y=y2, **kw)
    assert np.allclose(p1, p2), "a held-out label changed the prediction"


def test_a_node_cannot_read_its_own_residual_at_one_hop(cohort):
    n = 50
    ei = np.stack([np.arange(n), (np.arange(n) + 1) % n])
    A = hybrid.normalised_adjacency(ei, None, n)
    assert float(torch.sparse.mm(A, torch.eye(n)).diagonal().abs().max()) == 0.0


def test_zero_scale_leaves_the_boosted_score_untouched(cohort):
    n = len(cohort["y"])
    ei = np.stack([np.arange(n), (np.arange(n) + 1) % n])
    base = np.clip(np.random.default_rng(0).random(n), 0.01, 0.99)
    p = hybrid.correct_and_smooth(ei, None, base, cohort["y"], cohort["train"],
                                  n, scale=0.0, hops=2)
    assert np.allclose(p, base, atol=1e-6), "the graph stage is not a no-op at scale 0"


# ------------------------------------------------------------ safety certificate

def test_clopper_pearson_brackets_the_point_estimate():
    for k, m in [(0, 100), (1, 100), (5, 100), (0, 1000)]:
        ub = cascade.clopper_pearson_upper(k, m, gamma=0.05)
        assert k / m <= ub <= 1.0
    # More clean observations must buy a tighter certificate.
    assert (cascade.clopper_pearson_upper(0, 1000)
            < cascade.clopper_pearson_upper(0, 100))
    assert cascade.clopper_pearson_upper(10, 10) == 1.0


def test_conformal_gate_is_never_looser_than_the_empirical_one():
    rng = np.random.default_rng(0)
    n = 4000
    p = np.sort(rng.random((3, n)), axis=1)
    y = (rng.random(n) < p[2]).astype(int)
    sel = np.zeros(n, bool); sel[: n // 2] = True
    g_conf = cascade.fit_gate(p, y, sel, alpha=0.05, beta=0.5, conformal=True)
    g_emp = cascade.fit_gate(p, y, sel, alpha=0.05, beta=0.5, conformal=False)
    assert (g_conf["lo"] <= g_emp["lo"] + 1e-12).all(), \
        "the certified threshold must be at most the empirical one"


def test_routing_respects_the_thresholds_and_costs_are_monotone():
    p = np.array([[0.01, 0.5, 0.99], [0.2, 0.5, 0.6], [0.3, 0.4, 0.5]])
    gate = {"lo": np.array([0.05, 0.0, 0.0]), "hi": np.array([0.95, 1.0, 1.0])}
    stage, risk, side = cascade.route(p, gate)
    assert stage.tolist() == [0, 2, 0]
    assert side.tolist() == [-1, 0, +1]
    assert risk[0] == pytest.approx(0.01) and risk[2] == pytest.approx(0.99)


def test_gate_refuses_a_degenerate_fit():
    """tau_lo above tau_hi would gate everyone out; the gate must decline."""
    rng = np.random.default_rng(1)
    n = 2000
    p = np.tile(rng.random(n), (2, 1))
    y = np.zeros(n, dtype=int)          # no deaths at all -> low side unbounded
    sel = np.ones(n, bool)
    g = cascade.fit_gate(p, y, sel, alpha=0.5, beta=0.99, conformal=False)
    assert (g["lo"] < g["hi"]).all()


# ------------------------------------------------------------------ calibration

def test_stratified_calibrator_falls_back_when_a_stratum_is_too_small():
    rng = np.random.default_rng(2)
    m = rng.normal(size=400)
    y = (rng.random(400) < 0.3).astype(int)
    strata = rng.random(400)
    c = cascade.Calibrator(n_strata=8, min_per_stratum=300).fit(m, y, strata)
    assert all(mdl is c.global_ for mdl in c.models_)


def test_split_validation_3way_is_disjoint_and_covers_val(cohort):
    val = cohort["val"]
    y = cohort["y"]
    parts = cascade.split_validation_3way(val, y, seed=0)
    a, b, c = parts["model_val"], parts["gate_selection"], parts["gate_certification"]
    assert not (a & b).any() and not (b & c).any() and not (a & c).any()
    assert (a | b | c).sum() == val.sum()
    assert not (a & ~val).any() and not (b & ~val).any() and not (c & ~val).any()


def test_split_validation_3way_respects_fracs_per_class():
    n = 2000
    val = np.ones(n, dtype=bool)
    y = np.zeros(n, dtype=int); y[: n // 5] = 1  # imbalanced, like the real cohort
    parts = cascade.split_validation_3way(val, y, seed=1, fracs=(0.5, 0.25, 0.25))
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        n_cls = len(idx)
        assert parts["model_val"][idx].sum() == pytest.approx(0.5 * n_cls, abs=1)
        assert parts["gate_selection"][idx].sum() == pytest.approx(0.25 * n_cls, abs=1)


def test_calibration_uses_only_the_calibration_split(cohort):
    """Changing a test-split label must not move a calibrated probability."""
    n = len(cohort["y"])
    margins = np.stack([np.linspace(-3, 3, n), np.linspace(-2, 2, n)])
    cal = cohort["val"]
    y2 = cohort["y"].copy()
    flip = np.flatnonzero(cohort["test"])[:30]
    y2[flip] = 1 - y2[flip]
    p1, _ = cascade.calibrate_stages(margins, cohort["y"], cal)
    p2, _ = cascade.calibrate_stages(margins, y2, cal)
    assert np.allclose(p1, p2)
