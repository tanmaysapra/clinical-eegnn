import numpy as np
import pytest
import torch

from src import arch4r
from src import graph_density as gd
from src.encoder_mv import observed_fraction
from src.local_push import make_checkpoints

VIEW_IN_DIMS = {"demographics": 5, "labs": 6}


def _synthetic_view(n, d, seed, n_clusters=4):
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_clusters, size=n)
    centroids = rng.normal(0, 3, size=(n_clusters, d))
    X = centroids[cluster] + rng.normal(0, 0.5, size=(n, d))
    return X.astype(np.float32)


def _build_ref_and_inputs(n_train=24, seed=0, fixed_k=None):
    views_np = {
        "demographics": _synthetic_view(n_train, VIEW_IN_DIMS["demographics"], seed),
        "labs": _synthetic_view(n_train, VIEW_IN_DIMS["labs"], seed + 1),
    }
    reliability = {name: np.ones(n_train) for name in views_np}
    cfg = gd.ViewGraphConfig(k_cand=12, k0=4, gamma=0.5, k_min_adaptive=2,
                             k_max_adaptive=8, percentile_qualify=99.5, fixed_k=fixed_k)
    ref = gd.build_reference_graph(views_np, reliability, cfg, seed=seed)

    view_x = {name: torch.tensor(X) for name, X in views_np.items()}
    view_observed = {name: torch.ones(n_train) for name in views_np}
    return ref, cfg, view_x, view_observed


def _model(mode="adaptive_k", cfg_extra=None):
    cfg = {"fusion_dim": 16, "stage_hidden": 8, "alpha": 0.15, "top_k_keep": 8}
    if cfg_extra:
        cfg.update(cfg_extra)
    return arch4r.Arch4RModel(VIEW_IN_DIMS, cfg, mode=mode)


# ------------------------------------------------------------- construction

def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        _model(mode="bogus")


def test_all_three_modes_build_identical_graph_dependent_submodules():
    """The plan's own required invariant: fixed_k/adaptive_k (and, for the
    graph-dependent submodules specifically, encoder_only too, since they're
    always built regardless of mode -- see arch4r.py's module docstring)
    must share every component's architecture. Values differ (independent
    random init per instance); names and shapes must not."""
    models = {m: _model(mode=m) for m in arch4r.MODES}
    keysets = {m: set(mdl.state_dict().keys()) for m, mdl in models.items()}
    ref_keys = keysets["encoder_only"]
    for m, keys in keysets.items():
        assert keys == ref_keys, f"mode {m} has different parameter set: {keys ^ ref_keys}"
    ref_shapes = {k: v.shape for k, v in models["fixed_k"].state_dict().items()}
    for m, mdl in models.items():
        for k, v in mdl.state_dict().items():
            assert v.shape == ref_shapes[k], f"mode {m} param {k} shape mismatch"


def test_fixed_k_and_adaptive_k_share_every_component_except_k():
    """Model-side: fixed_k and adaptive_k models are architecturally
    identical (previous test). Graph-side: the only thing that should differ
    between the two arms is `ViewGraphConfig.fixed_k` driving a different
    edge count -- Arch4RModel itself holds no notion of k at all."""
    model_fixed = _model(mode="fixed_k")
    model_adaptive = _model(mode="adaptive_k")
    assert set(model_fixed.state_dict().keys()) == set(model_adaptive.state_dict().keys())

    ref_fixed, _, view_x, view_observed = _build_ref_and_inputs(fixed_k=4)
    ref_adaptive, _, _, _ = _build_ref_and_inputs(fixed_k=None)
    # Same seed/data, only cfg.fixed_k differs -> edge counts should differ
    # (adaptive k is not pinned to exactly 4 everywhere in this synthetic setup).
    e_fixed = ref_fixed.views["labs"].edge_index_ref.shape[1]
    e_adaptive = ref_adaptive.views["labs"].edge_index_ref.shape[1]
    assert e_fixed != e_adaptive

    batch_fixed = arch4r.transductive_batch(ref_fixed)
    batch_adaptive = arch4r.transductive_batch(ref_adaptive)
    ckpts = make_checkpoints()
    out_fixed = model_fixed(view_x, view_observed, batch_fixed, ckpts)
    out_adaptive = model_adaptive(view_x, view_observed, batch_adaptive, ckpts)
    assert len(out_fixed["stage_logits"]) == len(out_adaptive["stage_logits"]) == len(ckpts)


# ------------------------------------------------------------ encoder_only

def test_encoder_only_arm_has_no_graph_dependency():
    """mode='encoder_only' must never touch the graph-dependent submodules,
    even when a fully-populated attached_graph is handed to it -- not just
    'ignores the result', but structurally never runs them, verified by
    every graph-dependent parameter having no gradient at all after
    backward()."""
    model = _model(mode="encoder_only")
    ref, _, view_x, view_observed = _build_ref_and_inputs()
    batch = arch4r.transductive_batch(ref)
    ckpts = make_checkpoints()

    out = model(view_x, view_observed, batch, ckpts)
    loss = sum(l.sum() for l in out["stage_logits"]) + out["error_head_raw"].sum()
    loss.backward()

    for name in model.view_names:
        for p in model.scorers[name].parameters():
            assert p.grad is None, f"scorers[{name}] got a gradient in encoder_only mode"
    for p in model.reliability.parameters():
        assert p.grad is None, "reliability got a gradient in encoder_only mode"
    for p in model.V.parameters():
        assert p.grad is None, "V got a gradient in encoder_only mode"

    # And every checkpoint's logits must be identical (no diffusion happened,
    # so h_bar_q^(s) == h0_q at every stage).
    for s in range(1, len(ckpts)):
        assert torch.equal(out["stage_logits"][0], out["stage_logits"][s])


def test_encoder_only_works_with_attached_graph_none():
    model = _model(mode="encoder_only")
    _, _, view_x, view_observed = _build_ref_and_inputs()
    ckpts = make_checkpoints()
    out = model(view_x, view_observed, None, ckpts)
    n = view_x["labs"].shape[0]
    assert out["stage_logits"][0].shape == (n,)


# ---------------------------------------------------------- graph-available

def test_forward_shapes_with_transductive_batch():
    model = _model(mode="adaptive_k")
    ref, _, view_x, view_observed = _build_ref_and_inputs(n_train=20, seed=7)
    batch = arch4r.transductive_batch(ref)
    ckpts = make_checkpoints()
    out = model(view_x, view_observed, batch, ckpts)

    assert len(out["stage_logits"]) == len(ckpts)
    for logits in out["stage_logits"]:
        assert logits.shape == (20,)
    assert out["h0"].shape == (20, 16)
    assert out["error_head_raw"].shape == (20,)
    assert "edges" in out["ledger"] and "queue_pops" in out["ledger"]


def test_forward_graph_available_query_gets_graph_dependent_gradient():
    model = _model(mode="adaptive_k")
    ref, _, view_x, view_observed = _build_ref_and_inputs(n_train=20, seed=8)
    batch = arch4r.transductive_batch(ref)
    if not batch.graph_available.any():
        pytest.skip("synthetic setup produced no available train nodes at this seed")
    ckpts = make_checkpoints()
    out = model(view_x, view_observed, batch, ckpts)
    out["stage_logits"][-1].sum().backward()
    assert any(p.grad is not None and (p.grad.abs() > 0).any()
              for p in model.V.parameters())


def test_attach_batch_scores_held_out_queries_through_reference_graph():
    """End-to-end held-out path: build a train reference graph, attach a
    fresh (in-distribution) held-out batch via ReferenceGraph.attach(...),
    wire it through attach_batch(...), and confirm forward() runs and scores
    exactly the held-out rows (not the train rows)."""
    n_train = 20
    ref, cfg, view_x_train, view_observed_train = _build_ref_and_inputs(
        n_train=n_train, seed=11)

    rng = np.random.default_rng(12)
    n_q = 4
    query_np = {
        "demographics": ref.views["demographics"].X_train[:n_q]
                       + rng.normal(0, 0.05, size=(n_q, VIEW_IN_DIMS["demographics"])),
        "labs": ref.views["labs"].X_train[:n_q]
               + rng.normal(0, 0.05, size=(n_q, VIEW_IN_DIMS["labs"])),
    }
    query_reliability = {name: np.ones(n_q) for name in query_np}
    attached = ref.attach(query_np, query_reliability)

    batch = arch4r.attach_batch(ref, attached, n_train)
    assert batch.n_nodes == n_train + n_q
    assert np.array_equal(batch.query_idx, np.arange(n_train, n_train + n_q))

    all_view_x = {name: torch.cat([view_x_train[name], torch.tensor(query_np[name], dtype=torch.float32)])
                 for name in view_x_train}
    all_view_observed = {name: torch.cat([view_observed_train[name], torch.ones(n_q)])
                         for name in view_observed_train}

    model = _model(mode="adaptive_k")
    ckpts = make_checkpoints()
    out = model(all_view_x, all_view_observed, batch, ckpts)

    assert len(out["stage_logits"]) == len(ckpts)
    for logits in out["stage_logits"]:
        assert logits.shape == (n_q,)
    assert out["h0"].shape == (n_q, 16)


def test_cold_start_query_matches_encoder_only_fallback():
    """A query with graph_available=False must fall back to the same
    h_bar_q^(s) == h0_q behaviour as encoder_only, at every checkpoint."""
    model = _model(mode="adaptive_k")
    ref, _, view_x, view_observed = _build_ref_and_inputs(n_train=10, seed=9)
    n_train = 10
    per_view_edges = {name: (v.edge_index_ref[0], v.edge_index_ref[1], v.edge_weight_ref)
                      for name, v in ref.views.items()}
    batch = arch4r.Arch4RGraphBatch(per_view_edges=per_view_edges, n_nodes=n_train,
                                    query_idx=np.arange(n_train),
                                    graph_available=np.zeros(n_train, dtype=bool))
    ckpts = make_checkpoints()
    out = model(view_x, view_observed, batch, ckpts)
    for s in range(1, len(ckpts)):
        assert torch.equal(out["stage_logits"][0], out["stage_logits"][s])


# ------------------------------------------------------------------- loss

def test_arch4r_loss_matches_manual_mean_bce():
    torch.manual_seed(0)
    n, n_ckpt = 8, 3
    stage_logits = [torch.randn(n, requires_grad=True) for _ in range(n_ckpt)]
    y = torch.randint(0, 2, (n,))
    mask = torch.ones(n, dtype=torch.bool)
    out = {"stage_logits": stage_logits}
    loss, info = arch4r.arch4r_loss(out, y, mask, cfg={})

    manual = sum(torch.nn.functional.binary_cross_entropy_with_logits(l, y.float())
                for l in stage_logits) / n_ckpt
    assert torch.allclose(loss, manual, atol=1e-6)
    assert "stage0_bce" in info and "total" in info


def test_arch4r_loss_rejects_nonzero_lam_surv():
    out = {"stage_logits": [torch.zeros(4)]}
    y = torch.zeros(4)
    mask = torch.ones(4, dtype=torch.bool)
    with pytest.raises(NotImplementedError):
        arch4r.arch4r_loss(out, y, mask, cfg={}, lam_surv=0.5)


def test_arch4r_loss_respects_mask():
    torch.manual_seed(1)
    n = 10
    logits = [torch.randn(n, requires_grad=True)]
    y = torch.randint(0, 2, (n,))
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:4] = True
    out = {"stage_logits": logits}
    loss, _ = arch4r.arch4r_loss(out, y, mask, cfg={})
    manual = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[0][mask], y[mask].float())
    assert torch.allclose(loss, manual, atol=1e-6)
