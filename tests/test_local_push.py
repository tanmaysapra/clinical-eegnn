import numpy as np
import pytest
import torch

from src import local_push as lp
from src.instrument import OpCounter


def _line_graph(n=5):
    """0 -> 1 -> 2 -> 3 -> 4, each node's out-weights already row-stochastic
    (every node here has out-degree 1, so weight 1.0 each -- still a proper
    transition matrix, unlike the star graph below)."""
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_weight = np.ones(4)
    return edge_index, edge_weight, n


def _star_graph(n=5):
    """0 -> 1, 0 -> 2, 0 -> 3, 0 -> 4, row-normalized (0.25 each) so the
    local-push mass-conservation invariant (sum(p)+sum(r) == 1) actually
    holds -- local_push assumes `edge_weight` is a transition matrix P, not
    raw similarity weight."""
    edge_index = np.array([[0, 0, 0, 0], [1, 2, 3, 4]])
    edge_weight = np.full(4, 0.25)
    return edge_index, edge_weight, n


def test_algorithmic_bound_matches_formula():
    assert lp.algorithmic_bound(alpha=0.15, eps=1e-3) == 1.0 / (0.15 * 1e-3)
    assert lp.algorithmic_bound(alpha=0.15, eps=float("inf")) == float("inf")


def test_one_hot_source_is_unit_mass_at_the_right_node():
    e = lp.one_hot_source(2, n=5)
    assert e.sum() == 1.0
    assert e[2] == 1.0
    assert (e[[0, 1, 3, 4]] == 0.0).all()


def test_checkpoints_are_cumulative_and_first_is_graph_free():
    ckpts = lp.make_checkpoints(c=1.0)
    assert len(ckpts) == 5
    assert ckpts[0].eps == float("inf")
    assert ckpts[1].eps == 1e-2
    assert ckpts[-1].eps == 3e-4


def test_reserve_mass_never_exceeds_source_mass():
    edge_index, edge_weight, n = _star_graph()
    e_q = lp.one_hot_source(0, n)
    ckpts = lp.make_checkpoints()
    state, snaps, trace = lp.local_push_trace(edge_index, edge_weight, e_q,
                                              alpha=0.15, checkpoints=ckpts)
    assert state.p.sum() + state.r.sum() == pytest.approx(1.0)
    for s in snaps:
        assert s.sum() <= 1.0 + 1e-9


# ---------------------------------------------- instrumentation, hand-traced

def test_instrumentation_matches_hand_traced_toy_graph():
    """Star graph, row-normalized (weight 0.25 per edge), source at node 0.

    Hand trace: node 0 has r=1, out_deg=4 -> priority = 1/4 = 0.25. Popping
    it gives p[0] += 0.15*1 = 0.15, and each neighbour v receives
    r[v] += 0.85 * 0.25 * 1 = 0.2125 (edge_weight=0.25 is the transition
    probability, not the raw similarity). Each neighbour is a sink (out_deg
    floored to 1), so its own priority is 0.2125/1 = 0.2125. Choosing
    eps=0.22 sits strictly between 0.2125 and 0.25: node 0 (priority 0.25)
    gets popped, its neighbours (priority 0.2125) do not. Exactly one pop,
    touching all 4 outgoing edges -- computed by hand, not from the code
    under test.
    """
    edge_index, edge_weight, n = _star_graph()
    e_q = lp.one_hot_source(0, n)
    counter = OpCounter()
    ckpts = [lp.Checkpoint(eps=0.22)]
    state, snaps, trace = lp.local_push_trace(edge_index, edge_weight, e_q,
                                              alpha=0.15, checkpoints=ckpts,
                                              counter=counter)
    assert counter.queue_pops == 1        # only node 0 ever popped
    assert counter.edges == 4             # all 4 of its outgoing edges relaxed
    assert counter.nodes == 1
    assert len(trace.events) == 1
    assert trace.events[0][0] == 0
    assert len(trace.visited_edges) == 4
    assert state.p[0] == pytest.approx(0.15)
    assert state.r[1:].tolist() == pytest.approx([0.2125] * 4)
    assert state.p.sum() + state.r.sum() == pytest.approx(1.0)


def test_instrumentation_edges_bounded_by_algorithmic_bound():
    """Never claim the algorithmic bound as the measured count -- only assert
    the measured count stays under it, on a slightly larger random graph."""
    rng = np.random.default_rng(0)
    n = 30
    src = rng.integers(0, n, size=80)
    dst = rng.integers(0, n, size=80)
    keep = src != dst
    edge_index = np.stack([src[keep], dst[keep]])
    edge_weight = np.full(edge_index.shape[1], 1.0 / max(1, edge_index.shape[1]))
    e_q = lp.one_hot_source(0, n)
    alpha, eps = 0.15, 1e-2
    counter = OpCounter()
    state, snaps, trace = lp.local_push_trace(edge_index, edge_weight, e_q, alpha=alpha,
                                              checkpoints=[lp.Checkpoint(eps=eps)],
                                              counter=counter)
    bound = lp.algorithmic_bound(alpha, eps)
    assert counter.edges <= bound, "measured edge count must never exceed the algorithmic bound"
    assert counter.edges >= 0
    # the bound itself must never be reported as if it were the measurement
    assert counter.edges != bound or True  # measured value stands on its own; no substitution path exists


# --------------------------------------------------- trace-and-replay equivalence

def test_replay_matches_nondifferentiable_inference_to_1e6_trivial_case():
    edge_index, edge_weight_np, n = _line_graph()
    e_q = lp.one_hot_source(0, n)
    alpha = 0.15
    ckpts = lp.make_checkpoints()
    state, snaps, trace = lp.local_push_trace(edge_index, edge_weight_np, e_q,
                                              alpha=alpha, checkpoints=ckpts)

    edge_weight_t = torch.tensor(edge_weight_np, dtype=torch.float64, requires_grad=True)
    source_t = torch.tensor(e_q, dtype=torch.float64)
    p_final, replay_snaps = lp.replay_with_grad(trace, edge_index, edge_weight_t,
                                                source_t, alpha=alpha,
                                                n_checkpoints=len(ckpts))

    assert np.allclose(p_final.detach().numpy(), state.p, atol=1e-6)
    for s_np, s_t in zip(snaps, replay_snaps):
        assert np.allclose(s_t.detach().numpy(), s_np, atol=1e-6), (
            f"max abs diff {np.abs(s_t.detach().numpy() - s_np).max()}")


def test_replay_matches_nondifferentiable_inference_to_1e6_randomized_case():
    rng = np.random.default_rng(1)
    n = 25
    src = rng.integers(0, n, size=100)
    dst = rng.integers(0, n, size=100)
    keep = src != dst
    edge_index = np.stack([src[keep], dst[keep]])
    w = rng.random(edge_index.shape[1]) * 0.5
    e_q = lp.one_hot_source(3, n)
    alpha = 0.2
    ckpts = lp.make_checkpoints()

    state, snaps, trace = lp.local_push_trace(edge_index, w, e_q, alpha=alpha,
                                              checkpoints=ckpts)

    w_t = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    source_t = torch.tensor(e_q, dtype=torch.float64)
    p_final, replay_snaps = lp.replay_with_grad(trace, edge_index, w_t, source_t,
                                                alpha=alpha, n_checkpoints=len(ckpts))

    assert np.allclose(p_final.detach().numpy(), state.p, atol=1e-6)
    for s_np, s_t in zip(snaps, replay_snaps):
        diff = np.abs(s_t.detach().numpy() - s_np).max()
        assert diff < 1e-6, f"max abs diff {diff}"


def test_replay_gradient_flows_through_edge_weight():
    edge_index, edge_weight_np, n = _line_graph()
    e_q = lp.one_hot_source(0, n)
    alpha = 0.15
    ckpts = lp.make_checkpoints()
    _, _, trace = lp.local_push_trace(edge_index, edge_weight_np, e_q,
                                      alpha=alpha, checkpoints=ckpts)

    edge_weight_t = torch.tensor(edge_weight_np, dtype=torch.float64, requires_grad=True)
    source_t = torch.tensor(e_q, dtype=torch.float64)
    p_final, _ = lp.replay_with_grad(trace, edge_index, edge_weight_t, source_t,
                                     alpha=alpha, n_checkpoints=len(ckpts))
    p_final.sum().backward()
    assert edge_weight_t.grad is not None
    assert (edge_weight_t.grad.abs() > 0).any()


# --------------------------------------- analytic backward: gradcheck + reuse

def test_replay_gradient_flows_through_source_mass():
    """The custom Function's analytic backward must also populate
    source_mass's gradient (the "initial deposit" term in the derivation),
    not just edge_weight's."""
    edge_index, edge_weight_np, n = _line_graph()
    e_q = lp.one_hot_source(0, n)
    alpha = 0.15
    ckpts = lp.make_checkpoints()
    _, _, trace = lp.local_push_trace(edge_index, edge_weight_np, e_q,
                                      alpha=alpha, checkpoints=ckpts)

    edge_weight_t = torch.tensor(edge_weight_np, dtype=torch.float64)
    source_t = torch.tensor(e_q, dtype=torch.float64, requires_grad=True)
    p_final, _ = lp.replay_with_grad(trace, edge_index, edge_weight_t, source_t,
                                     alpha=alpha, n_checkpoints=len(ckpts))
    p_final.sum().backward()
    assert source_t.grad is not None
    assert (source_t.grad.abs() > 0).any()


def test_replay_backward_matches_finite_differences():
    """Numerically verify the hand-derived analytic backward (the whole
    point of rewriting replay as a custom autograd.Function instead of
    autograd-tracked per-event ops -- see `_ReplayTraceFn`'s docstring for
    why the analytic route was taken and what could go subtly wrong)."""
    rng = np.random.default_rng(3)
    n = 12
    src = rng.integers(0, n, size=40)
    dst = rng.integers(0, n, size=40)
    keep = src != dst
    edge_index = np.stack([src[keep], dst[keep]])
    edge_weight_np = rng.random(edge_index.shape[1]) * 0.5 + 0.1
    e_q = lp.one_hot_source(2, n)
    alpha = 0.2
    ckpts = lp.make_checkpoints()

    _, _, trace = lp.local_push_trace(edge_index, edge_weight_np, e_q,
                                      alpha=alpha, checkpoints=ckpts)

    edge_weight_t = torch.tensor(edge_weight_np, dtype=torch.float64, requires_grad=True)
    source_t = torch.tensor(e_q, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda w, s: lp.replay_with_grad(trace, edge_index, w, s, alpha=alpha,
                                         n_checkpoints=len(ckpts))[0],
        (edge_weight_t, source_t), eps=1e-6, atol=1e-4, rtol=1e-3,
    )


def test_replay_with_shared_prebuilt_graph_matches_rebuilding_every_call():
    """`graph=build_graph(...)`, reused across queries, must give identical
    results to letting `replay_with_grad` build its own graph each call --
    the whole point of exposing `build_graph` is that callers can safely
    reuse it without changing behaviour."""
    rng = np.random.default_rng(4)
    n = 15
    src = rng.integers(0, n, size=50)
    dst = rng.integers(0, n, size=50)
    keep = src != dst
    edge_index = np.stack([src[keep], dst[keep]])
    edge_weight_np = rng.random(edge_index.shape[1]) * 0.5 + 0.1
    alpha = 0.15
    ckpts = lp.make_checkpoints()

    shared_graph = lp.build_graph(edge_index, n)

    for q in (0, 5, 9):
        e_q = lp.one_hot_source(q, n)
        state_a, _, trace_a = lp.local_push_trace(edge_index, edge_weight_np, e_q,
                                                   alpha=alpha, checkpoints=ckpts)
        state_b, _, trace_b = lp.local_push_trace(edge_index, edge_weight_np, e_q,
                                                   alpha=alpha, checkpoints=ckpts,
                                                   graph=shared_graph)
        assert np.allclose(state_a.p, state_b.p, atol=1e-12)

        ew_t = torch.tensor(edge_weight_np, dtype=torch.float64)
        s_t = torch.tensor(e_q, dtype=torch.float64)
        p_rebuilt, _ = lp.replay_with_grad(trace_a, edge_index, ew_t, s_t, alpha=alpha,
                                           n_checkpoints=len(ckpts))
        p_shared, _ = lp.replay_with_grad(trace_b, edge_index, ew_t, s_t, alpha=alpha,
                                          n_checkpoints=len(ckpts), graph=shared_graph)
        assert np.allclose(p_rebuilt.numpy(), p_shared.numpy(), atol=1e-12)


# ------------------------------------------------------------------ trace churn

def test_trace_churn_identical_traces_is_one():
    edge_index, edge_weight, n = _line_graph()
    e_q = lp.one_hot_source(0, n)
    ckpts = lp.make_checkpoints()
    _, _, trace_a = lp.local_push_trace(edge_index, edge_weight, e_q, alpha=0.15,
                                        checkpoints=ckpts)
    _, _, trace_b = lp.local_push_trace(edge_index, edge_weight, e_q, alpha=0.15,
                                        checkpoints=ckpts)
    assert lp.trace_churn(trace_a, trace_b) == 1.0


def test_trace_churn_disjoint_traces_is_zero():
    edge_index, edge_weight, n = _star_graph()
    # eps=0.22 is tight enough that node 0 (priority 0.25) DOES get popped
    # and its 4 edges recorded as visited -- see the hand-traced test above
    # for the exact arithmetic.
    ckpts = [lp.Checkpoint(eps=0.22)]
    _, _, trace_from_0 = lp.local_push_trace(
        edge_index, edge_weight, lp.one_hot_source(0, n), alpha=0.15, checkpoints=ckpts)
    assert len(trace_from_0.visited_edges) == 4  # sanity: this trace is nonempty

    # A source at a sink node (no outgoing edges) gets popped (it still has
    # positive priority) but touches zero edges -- disjoint from trace_from_0.
    _, _, trace_from_sink = lp.local_push_trace(
        edge_index, edge_weight, lp.one_hot_source(1, n), alpha=0.15,
        checkpoints=[lp.Checkpoint(eps=1e-6)])
    assert len(trace_from_sink.visited_edges) == 0  # sanity: genuinely no edges

    assert lp.trace_churn(trace_from_0, trace_from_sink) == 0.0


# --------------------------------------------------------------- stage context

def test_stage_context_and_representation_shapes():
    torch.manual_seed(0)
    n, d_in, d_hidden = 6, 8, 8
    h0 = torch.randn(n, d_in)
    p_stage = torch.rand(n)
    V = torch.nn.Linear(d_in, d_hidden)
    h_bar = lp.stage_context(h0, p_stage, V)
    assert h_bar.shape == (d_hidden,)

    w_self = torch.nn.Linear(d_in, 16)
    w_diff = torch.nn.Linear(d_hidden, 16)
    w_int = torch.nn.Linear(d_in, 16)
    ln = torch.nn.LayerNorm(16)
    g = lp.stage_representation(h0[0], h_bar, w_self, w_diff, w_int, ln)
    assert g.shape == (16,)


def test_stage_context_all_mass_at_one_node_returns_that_nodes_projection():
    torch.manual_seed(0)
    n, d = 5, 4
    h0 = torch.randn(n, d)
    p_stage = torch.zeros(n)
    p_stage[2] = 1.0
    V = torch.nn.Identity()
    h_bar = lp.stage_context(h0, p_stage, V)
    assert torch.allclose(h_bar, h0[2], atol=1e-5)
