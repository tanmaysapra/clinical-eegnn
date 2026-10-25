"""Unit tests -- run these throughout, not at the end.

Covers the things that silently corrupt results: leakage in the splits, a graph
whose degree does not match k, a loss that does not actually decrease, and
Gumbel-Softmax gradients failing to reach the halt branch.
"""

import numpy as np
import pytest
import torch
import yaml

from src.data_prep import impute_and_scale, make_splits
from src.graph_build import build_graph, build_knn_graph
from src.layers import SASGNNLayer
from src.losses import multi_exit_loss
from src.mock_data import make_mock_cohort
from src.model import EEGNN


@pytest.fixture(scope="module")
def cfg():
    with open("configs/default.yaml") as f:
        c = yaml.safe_load(f)
    c["data"]["n_patients"] = 300
    c["model"]["max_layers"] = 4
    c["model"]["hidden_dim"] = 16
    return c


@pytest.fixture(scope="module")
def graph(cfg):
    X_raw, y, names, meta = make_mock_cohort(300, missing_rate=0.2, seed=0)
    masks = make_splits(y, seed=0)
    X = impute_and_scale(X_raw, masks[0])
    return build_graph(X, y, masks, cfg, meta=meta,
                       cache_dir="/tmp/eegnn_test")


# ------------------------------------------------------------------ mock data

def test_mock_shapes_and_prevalence():
    X, y, names, meta = make_mock_cohort(1000, n_features=40, missing_rate=0.2, seed=0)
    assert X.shape == (1000, 40) and len(names) == 40
    assert set(np.unique(y)) <= {0, 1}
    assert 0.05 < y.mean() < 0.25, "prevalence should be realistically imbalanced"
    assert 0.1 < np.isnan(X).mean() < 0.3, "missingness must be injected"
    assert meta["true_risk"].shape == (1000,)


# --------------------------------------------------------------------- splits

def test_splits_are_disjoint_and_stratified():
    _, y, _, _m = make_mock_cohort(1000, seed=0)
    tr, va, te = make_splits(y, seed=0)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all(), "no patient may be dropped"
    base = y.mean()
    for m in (tr, va, te):
        assert abs(y[m].mean() - base) < 0.05, "label prevalence must survive the split"


def test_imputation_does_not_see_test_statistics():
    """Perturbing test rows must not change the imputed train rows."""
    X, y, _, _m = make_mock_cohort(400, seed=0)
    tr, va, te = make_splits(y, seed=0)
    a = impute_and_scale(X.copy(), tr)
    X2 = X.copy()
    X2[te] = np.nan_to_num(X2[te]) + 100.0
    b = impute_and_scale(X2, tr)
    np.testing.assert_allclose(a[tr], b[tr], rtol=1e-5)


# ---------------------------------------------------------------------- graph

def test_knn_graph_degree_and_symmetry():
    X, _, _, _m = make_mock_cohort(200, seed=0)
    X = np.nan_to_num(X)
    ei, ew = build_knn_graph(X, k=10)
    assert (ei[0] != ei[1]).all(), "self-loops must be dropped"
    assert ei.shape[1] / 200 >= 10, "symmetrising can only add edges"
    edges = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert all((d, s) in edges for s, d in list(edges)[:200]), "graph must be undirected"
    assert (ew >= 0).all(), "message passing needs non-negative weights"


def test_graph_preserves_prevalence(graph):
    base = graph.y.float().mean()
    for split in ("train", "val", "test"):
        m = getattr(graph, f"{split}_mask")
        assert abs(float(graph.y[m].float().mean()) - float(base)) < 0.05


# ---------------------------------------------------------------------- model

def test_sasgnn_weight_is_shared_and_decomposes():
    layer = SASGNNLayer(8, epsilon=0.1, gamma=0.1)
    n_params = sum(p.numel() for p in layer.parameters())
    assert n_params == 8 * 8 + 8, "one shared W plus bias -- depth must be free"
    W_eff = layer.effective_weight()
    W_anti = 0.5 * (layer.W - layer.W.t())
    W_sym = 0.5 * (layer.W + layer.W.t())
    torch.testing.assert_close(W_eff, W_anti - 0.1 * W_sym - 0.1 * torch.eye(8))


def test_depth_costs_no_parameters(cfg, graph):
    deep = dict(cfg)
    deep["model"] = {**cfg["model"], "max_layers": 40, "exit_every": 40}
    shallow = dict(cfg)
    shallow["model"] = {**cfg["model"], "max_layers": 2, "exit_every": 2}
    a = EEGNN(graph.x.shape[1], deep)
    b = EEGNN(graph.x.shape[1], shallow)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_forward_shapes_and_exit_mass(cfg, graph):
    model = EEGNN(graph.x.shape[1], cfg)
    out = model(graph.x, graph.edge_index, graph.edge_weight, tau=1.0)
    n = graph.num_nodes
    assert out["final_logits"].shape == (n, 2)
    assert out["exit_probs"].shape[0] == n
    torch.testing.assert_close(out["exit_probs"].sum(1), torch.ones(n), atol=1e-4, rtol=0)
    assert out["exit_layer"].min() >= 1
    assert out["exit_layer"].max() <= cfg["model"]["max_layers"]


def test_gumbel_gradients_reach_halt_branch(cfg, graph):
    model = EEGNN(graph.x.shape[1], cfg)
    model.train()
    out = model(graph.x, graph.edge_index, graph.edge_weight, tau=1.0)
    loss, _ = multi_exit_loss(out, graph.y, graph.train_mask, cfg)
    loss.backward()
    halt_grads = [e.halt.weight.grad for e in model.exits[:-1]]
    assert any(g is not None and g.abs().sum() > 0 for g in halt_grads), \
        "the halt policy must be learnable end to end"


def test_loss_decreases_on_overfit(cfg, graph):
    """Sanity: the model can drive training loss down on its own training set."""
    torch.manual_seed(0)
    model = EEGNN(graph.x.shape[1], cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    first, last = None, None
    for i in range(60):
        model.train()
        opt.zero_grad()
        out = model(graph.x, graph.edge_index, graph.edge_weight, tau=1.0)
        loss, parts = multi_exit_loss(out, graph.y, graph.train_mask, cfg)
        loss.backward()
        opt.step()
        if i == 0:
            first = parts["task"]
        last = parts["task"]
    assert last < first * 0.9, f"task loss did not decrease: {first:.4f} -> {last:.4f}"


def test_high_risk_metric_is_not_degenerate():
    """The safety stratum must respond to the model, not to the threshold rule.

    Defining "high risk" as `y == 1` and then thresholding at a fixed operating
    sensitivity pins the miss rate to `1 - sensitivity` for every model, which
    is what made the old metric constant at 0.104 across a whole sweep
    (HANDOVER 6.1). Stratifying on an independent severity proxy must restore
    the metric's ability to distinguish a good model from a bad one.
    """
    import numpy as np
    from src.evaluate import high_risk_analysis

    rng = np.random.default_rng(0)
    n = 4000
    risk = rng.uniform(size=n)
    y = (rng.uniform(size=n) < 0.05 + 0.4 * risk).astype(int)

    good = np.clip(risk + rng.normal(0, 0.05, n), 0, 1)   # tracks true severity
    bad = rng.uniform(size=n)                             # ignores it
    depth = np.clip(1 + 9 * risk, 1, 10)

    label_based, severity_based = [], []
    for p in (good, bad):
        label_based.append(high_risk_analysis(y, p, depth, 10, 0.90)["high_risk_miss_rate"])
        severity_based.append(
            high_risk_analysis(y, p, depth, 10, 0.90, risk=risk)["high_risk_miss_rate"])

    # Old behaviour: reported as nan now rather than as a misleading constant.
    assert all(np.isnan(v) for v in label_based)

    # New behaviour: a model that tracks severity misses fewer high-risk deaths.
    assert all(np.isfinite(v) for v in severity_based)
    assert severity_based[0] < severity_based[1], (
        f"stratified miss rate failed to separate models: {severity_based}")

    # And the safety ordering is reported explicitly.
    res = high_risk_analysis(y, good, depth, 10, 0.90, risk=risk)
    assert res["safety_ordering_ok"] and res["depth_gap"] > 0
    assert res["high_risk_n"] == int(round(0.10 * n))


def test_safety_ordering_tie_is_not_reported_as_a_violation():
    """A strict `>` on `depth_gap == 0` reports a legitimate tie as an
    ordering inversion. Equal depth is not a safety violation -- only
    hi_depth < lo_depth (the sick cut short) is."""
    import numpy as np
    from src.evaluate import high_risk_analysis

    n = 200
    risk = np.arange(n, dtype=np.float64)          # well-defined top 10%
    y = np.zeros(n, dtype=int)
    p = np.full(n, 0.01)
    exit_layer = np.full(n, 5.0)                    # identical depth everywhere

    res = high_risk_analysis(y, p, exit_layer, max_layers=10,
                             operating_sensitivity=0.90, risk=risk, top_frac=0.10)
    assert res["depth_gap"] == 0.0
    assert res["safety_ordering_ok"] is True


def test_coverage_filter_keeps_blocks_aligned():
    """Dropping uncovered patients must drop the same rows from every block.

    A filter applied to the cohort table but not to each feature block would
    silently pair one patient's features with another's label -- catastrophic
    and invisible in aggregate metrics. Needed because the Kaggle mirror ships
    a truncated `chartevents.csv` covering only ~38% of ICU stays.
    """
    import numpy as np
    import pandas as pd
    from src.mimic_iv import _drop_uncovered

    df = pd.DataFrame({"subject_id": [10, 11, 12, 13, 14, 15],
                       "label": [0, 1, 0, 1, 0, 1]})
    vitals = pd.DataFrame({"hr": [1.0, np.nan, 3.0, np.nan, 5.0, np.nan],
                           "bp": [np.nan, np.nan, 2.0, np.nan, 4.0, np.nan]})
    treatment = pd.DataFrame({"vent": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]})

    out_df, out_blocks, keep = _drop_uncovered(
        df, {"vitals": vitals, "treatment": treatment}, ["vitals"], verbose=False)

    assert keep.tolist() == [True, False, True, False, True, False]
    assert list(out_df["subject_id"]) == [10, 12, 14]
    # every block filtered identically, and index reset so concat aligns
    for name, block in out_blocks.items():
        assert len(block) == len(out_df), name
        assert list(block.index) == list(range(len(out_df))), name
    # the surviving treatment values are the ones belonging to those patients
    assert out_blocks["treatment"]["vent"].tolist() == [0.0, 0.0, 0.0]

    # no requirement -> untouched
    same_df, same_blocks, same_keep = _drop_uncovered(
        df, {"vitals": vitals}, [], verbose=False)
    assert same_keep.all() and len(same_df) == 6


def test_coverage_filter_ignores_zero_filled_count_columns():
    """`_count` columns are zero-filled by design and must not imply coverage.

    `_aggregate_events` sets a never-measured concept's count to 0 rather than
    NaN, so a plain `notna().any(axis=1)` marks every patient as covered and
    the filter silently does nothing -- which is exactly what happened on the
    first run against the truncated mirror.
    """
    import numpy as np
    import pandas as pd
    from src.mimic_iv import _drop_uncovered

    df = pd.DataFrame({"subject_id": [1, 2, 3, 4], "label": [0, 1, 0, 1]})
    # patients 0 and 2 were measured; 1 and 3 never were, but still carry count 0
    vitals = pd.DataFrame({
        "vit_hr_mean": [80.0, np.nan, 95.0, np.nan],
        "vit_hr_count": [24.0, 0.0, 18.0, 0.0],
        "vit_sbp_count": [22.0, 0.0, 17.0, 0.0],
    })
    out_df, out_blocks, keep = _drop_uncovered(
        df, {"vitals": vitals}, ["vitals"], verbose=False)

    assert keep.tolist() == [True, False, True, False]
    assert list(out_df["subject_id"]) == [1, 3]
