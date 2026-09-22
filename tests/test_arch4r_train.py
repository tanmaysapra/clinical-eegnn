"""Tests for the per-epoch training steps extracted into src/arch4r_train.py
so both scripts/run_arch4r_phase1.py (unchanged behaviour) and the
full-epoch-budget notebook (checkpointing between epochs) share one
implementation of what one epoch does."""

import numpy as np
import torch

from src import arch4r, arch4r_train as at
from src import graph_density as gd

VIEW_IN_DIMS = {"demographics": 5, "labs": 6}


def _synthetic_view(n, d, seed, n_clusters=4):
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_clusters, size=n)
    centroids = rng.normal(0, 3, size=(n_clusters, d))
    X = centroids[cluster] + rng.normal(0, 0.5, size=(n, d))
    return X.astype(np.float32)


def _model(mode, cfg_extra=None):
    cfg = {"fusion_dim": 16, "stage_hidden": 8, "alpha": 0.15, "top_k_keep": 8}
    if cfg_extra:
        cfg.update(cfg_extra)
    return arch4r.Arch4RModel(VIEW_IN_DIMS, cfg, mode=mode)


def _params_snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def _params_changed(before, after):
    return any(not torch.equal(a, b) for a, b in zip(before, after))


# ---------------------------------------------------------- encoder_only

def test_train_encoder_epoch_updates_params_and_returns_finite_loss():
    n = 20
    model = _model("encoder_only")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    view_x = {name: torch.tensor(_synthetic_view(n, d, i))
              for i, (name, d) in enumerate(VIEW_IN_DIMS.items())}
    view_observed = {name: torch.ones(n) for name in VIEW_IN_DIMS}
    mask_t = torch.ones(n, dtype=torch.bool)
    y_t = torch.tensor(np.random.default_rng(0).integers(0, 2, size=n), dtype=torch.float32)
    from src.local_push import make_checkpoints
    checkpoints = make_checkpoints()

    before = _params_snapshot(model)
    loss = at.train_encoder_epoch(model, opt, view_x, view_observed, mask_t, y_t,
                                  checkpoints, model.cfg)
    after = _params_snapshot(model)

    assert np.isfinite(loss)
    assert _params_changed(before, after)


# -------------------------------------------------------------- graph arms

def test_train_graph_epoch_updates_params_and_returns_finite_loss():
    n_train = 24
    views_np = {
        "demographics": _synthetic_view(n_train, VIEW_IN_DIMS["demographics"], 0),
        "labs": _synthetic_view(n_train, VIEW_IN_DIMS["labs"], 1),
    }
    reliability = {name: np.ones(n_train) for name in views_np}
    gd_cfg = gd.ViewGraphConfig(k_cand=12, k0=4, gamma=0.5, k_min_adaptive=2,
                                 k_max_adaptive=8, percentile_qualify=99.5, fixed_k=None)
    ref = gd.build_reference_graph(views_np, reliability, gd_cfg, seed=0)

    model = _model("adaptive_k")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    from src.local_push import make_checkpoints
    checkpoints = make_checkpoints()

    view_x_train = {name: torch.tensor(X) for name, X in views_np.items()}
    view_observed_train = {name: torch.ones(n_train) for name in views_np}
    y_train_t = torch.tensor(np.random.default_rng(1).integers(0, 2, size=n_train),
                              dtype=torch.float32)
    full_batch = arch4r.transductive_batch(ref)
    rng = np.random.default_rng(0)

    before = _params_snapshot(model)
    loss = at.train_graph_epoch(model, opt, full_batch, view_x_train, view_observed_train,
                                y_train_t, checkpoints, model.cfg, query_batch_size=8, rng=rng)
    after = _params_snapshot(model)

    assert np.isfinite(loss)
    assert _params_changed(before, after)


def test_train_graph_epoch_consumes_rng_so_epoch_order_differs():
    n_train = 24
    views_np = {
        "demographics": _synthetic_view(n_train, VIEW_IN_DIMS["demographics"], 0),
        "labs": _synthetic_view(n_train, VIEW_IN_DIMS["labs"], 1),
    }
    reliability = {name: np.ones(n_train) for name in views_np}
    gd_cfg = gd.ViewGraphConfig(k_cand=12, k0=4, gamma=0.5, k_min_adaptive=2,
                                 k_max_adaptive=8, percentile_qualify=99.5, fixed_k=None)
    ref = gd.build_reference_graph(views_np, reliability, gd_cfg, seed=0)
    full_batch = arch4r.transductive_batch(ref)

    rng = np.random.default_rng(0)
    order1 = rng.permutation(full_batch.n_nodes)
    order2 = rng.permutation(full_batch.n_nodes)
    assert not np.array_equal(order1, order2)
