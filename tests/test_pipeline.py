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
    X_raw, y, names, risk = make_mock_cohort(300, missing_rate=0.2, seed=0)
    masks = make_splits(y, seed=0)
    X = impute_and_scale(X_raw, masks[0])
    return build_graph(X, y, masks, cfg, meta={"true_risk": risk},
                       cache_dir="/tmp/eegnn_test")


# ------------------------------------------------------------------ mock data

def test_mock_shapes_and_prevalence():
    X, y, names, risk = make_mock_cohort(1000, n_features=40, missing_rate=0.2, seed=0)
    assert X.shape == (1000, 40) and len(names) == 40
    assert set(np.unique(y)) <= {0, 1}
    assert 0.05 < y.mean() < 0.25, "prevalence should be realistically imbalanced"
    assert 0.1 < np.isnan(X).mean() < 0.3, "missingness must be injected"
    assert risk.shape == (1000,)


# --------------------------------------------------------------------- splits

def test_splits_are_disjoint_and_stratified():
    _, y, _, _ = make_mock_cohort(1000, seed=0)
    tr, va, te = make_splits(y, seed=0)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all(), "no patient may be dropped"
    base = y.mean()
    for m in (tr, va, te):
        assert abs(y[m].mean() - base) < 0.05, "label prevalence must survive the split"


def test_imputation_does_not_see_test_statistics():
    """Perturbing test rows must not change the imputed train rows."""
    X, y, _, _ = make_mock_cohort(400, seed=0)
    tr, va, te = make_splits(y, seed=0)
    a = impute_and_scale(X.copy(), tr)
    X2 = X.copy()
    X2[te] = np.nan_to_num(X2[te]) + 100.0
    b = impute_and_scale(X2, tr)
    np.testing.assert_allclose(a[tr], b[tr], rtol=1e-5)


# ---------------------------------------------------------------------- graph

def test_knn_graph_degree_and_symmetry():
    X, _, _, _ = make_mock_cohort(200, seed=0)
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
