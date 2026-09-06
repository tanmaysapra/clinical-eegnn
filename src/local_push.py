"""Checkpointed, priority-queue local-push diffusion, with trace-and-replay
differentiation.

Why this is a new module, not an extension of `src/ppr_push.py`
------------------------------------------------------------------
`ppr_push.py`'s `diffuse` pushes the *signal itself* (`r` starts at the
encoder's `z`, and `p[i]` ends up being node i's own diffused representation
directly) -- an efficient trick for transductive node classification that
avoids ever running personalised PageRank per source. Architecture 4R's own
spec does not use that trick: its stage representation is

    h_bar_q^(s) = sum_u p_q^{(s)}(u) * V h_u^{(0)}  /  sum_u p_q^{(s)}(u)

which weights *every node's own initial representation* `h_u^{(0)}` by how
much of query q's personalised-PageRank mass has reached it -- i.e. `p_q` is
a genuine per-query distribution over graph nodes (`e_q`, one unit of mass at
q's own attach node), not a diffused feature vector. That is a real,
per-query local-push computation, one query at a time; batching several
queries into one shared priority queue (an earlier draft of this module did
that) silently breaks each query's own tolerance guarantee, since one
query's large residual can drag a *different* query's node above the queue's
priority threshold before that query's own mass there would justify it. So
this module processes one query at a time; a caller wanting several queries
loops over them explicitly (`arch4r.py`), and that cost -- not this
module's correctness -- is the thing flagged as an open risk in the Phase
0/1 implementation plan (running every training node as its own query is
`O(n / (alpha*eps))` per epoch).

The differentiation problem
----------------------------
The push *order* (which node the priority queue pops next) is a discrete,
data-dependent decision. There is no way to backpropagate through "which
node was popped when" without either pretending it's continuous (a lie) or
fixing it. This module does the latter: run the queue once with gradients
off to produce a `Trace` (the exact sequence of node pops), then *replay*
that exact arithmetic a second time with gradients on, through the edge
weights and the source vector but never through the queue ordering itself.
`tests/test_local_push.py`'s numerical-equivalence test is what makes this
trustworthy: replay must reproduce the non-differentiable pass exactly, to
1e-6, not approximately.
"""

import heapq
from dataclasses import dataclass, field

import numpy as np
import torch

from src.instrument import OpCounter

CHECKPOINT_RATIOS = (float("inf"), 1e-2, 3e-3, 1e-3, 3e-4)


@dataclass
class Checkpoint:
    eps: float


def make_checkpoints(c: float = 1.0) -> list:
    """The 5 cumulative checkpoints, ratios fixed, one tunable global scale `c`."""
    return [Checkpoint(eps=(r if r == float("inf") else r * c)) for r in CHECKPOINT_RATIOS]


def algorithmic_bound(alpha: float, eps: float) -> float:
    """W_q(eps) < 1/(alpha*eps) directed edges for unit source mass.

    Reported alongside the measured count, never substituted for it -- see
    `test_instrumentation_matches_one_real_batch`.
    """
    if eps <= 0 or eps == float("inf"):
        return float("inf")
    return 1.0 / (alpha * eps)


def one_hot_source(query_node_index: int, n: int) -> np.ndarray:
    """e_q: one unit of mass at a single query's attach node -- length n."""
    e = np.zeros(n, dtype=np.float64)
    e[query_node_index] = 1.0
    return e


# ------------------------------------------------------------------- the graph

@dataclass
class _CSRGraph:
    """Compressed out-edge lists -- topology only (v, edge_k), no weight baked
    in. Deliberately decoupled from `edge_weight`: topology (which edges
    exist) is frozen for many training steps (`graph_density.py`'s "at most
    one rebuild after epoch 10" rule), while `edge_weight` -- the GATv2
    scorer's output over the current `z` -- changes every optimizer step.
    Baking weight values into this structure would force a full O(E) python
    rebuild on every call; keeping it weight-free lets one `build_graph(...)`
    call be reused for the whole topology-frozen period, with fresh
    `edge_weight` arrays passed into each push/replay call at zero rebuild
    cost. (Measured: rebuilding this per query, one query per training node
    per epoch, was the dominant cost in a from-scratch transductive-training
    benchmark -- see the implementation plan/handover for the numbers.)"""
    out_edges: list           # out_edges[u] = list of (v, edge_k)
    n: int
    out_deg: np.ndarray = field(default=None)   # out_deg[u] = max(len(out_edges[u]), 1)

    @staticmethod
    def build(edge_index: np.ndarray, n: int) -> "_CSRGraph":
        out = [[] for _ in range(n)]
        for k in range(edge_index.shape[1]):
            u, v = int(edge_index[0, k]), int(edge_index[1, k])
            out[u].append((v, k))
        out_deg = np.array([max(len(e), 1) for e in out], dtype=np.float64)
        return _CSRGraph(out_edges=out, n=n, out_deg=out_deg)


def build_graph(edge_index: np.ndarray, n: int) -> "_CSRGraph":
    """Build the reusable, weight-free adjacency once; pass the result to
    `local_push_trace(..., graph=...)` and `replay_with_grad(..., graph=...)`
    for every query scored against the same (frozen) topology, instead of
    letting each call rebuild it. See `_CSRGraph`'s docstring for why this
    split (topology here, weights passed fresh per call) matters."""
    return _CSRGraph.build(edge_index, n)


# ------------------------------------------------------------------ the trace

@dataclass
class Trace:
    """The exact, replayable record of one non-differentiable push run."""
    events: list = field(default_factory=list)                 # [(node, checkpoint_idx), ...]
    checkpoint_boundaries: list = field(default_factory=list)   # len == n_checkpoints
    visited_edges: set = field(default_factory=set)             # {(u, v, edge_k), ...}, whole run
    n: int = 0
    alpha: float = 0.15

    def edges_visited_between(self, s0: int, s1: int) -> set:
        lo = self.checkpoint_boundaries[s0 - 1] if s0 > 0 else 0
        hi = self.checkpoint_boundaries[s1]
        return {node for node, _ in self.events[lo:hi]}


def trace_churn(trace_a: Trace, trace_b: Trace) -> float:
    """Jaccard overlap of visited-edge sets between two traces of the same
    graph, taken at different points in training. `local_push.py` computes
    this per query; a caller averaging (or taking the median of) several
    queries' churn is what the training procedure's "median overlap" check
    actually means -- this function itself just does the one-trace-pair math."""
    a, b = trace_a.visited_edges, trace_b.visited_edges
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


# --------------------------------------------------------- the push, plain

class PushState:
    def __init__(self, p: np.ndarray, r: np.ndarray):
        self.p = p
        self.r = r


def local_push_trace(edge_index: np.ndarray, edge_weight: np.ndarray,
                     source_mass: np.ndarray, alpha: float,
                     checkpoints: list, counter: OpCounter | None = None,
                     graph: "_CSRGraph | None" = None):
    """Non-differentiable, priority-queue-driven local push for ONE query.

    `source_mass` is `e_q`, a length-n distribution over graph nodes
    (typically one-hot, from `one_hot_source`). `p[u]` accumulates how much
    of the query's PPR mass has settled at node u; `p.sum()` is always <= 1.
    Queue key is `r(u) / out_degree(u)`, the standard residual-to-degree
    local-push stopping criterion. Returns (final PushState, list of
    per-checkpoint `p` snapshots, Trace).

    `graph`, if given, is a `build_graph(edge_index, n)` result reused across
    many queries against the same frozen topology -- avoids rebuilding the
    O(E) adjacency on every call (the caller is responsible for passing a
    graph whose topology actually matches `edge_index`; not checked here).
    """
    n = source_mass.shape[0]
    counter = counter or OpCounter()
    graph = graph if graph is not None else _CSRGraph.build(edge_index, n)
    out_deg = graph.out_deg

    p = np.zeros(n, dtype=np.float64)
    r = source_mass.astype(np.float64).copy()
    trace = Trace(n=n, alpha=alpha)
    snapshots = []

    def _priority(u):
        return r[u] / out_deg[u]

    heap = [(-_priority(u), u) for u in range(n) if r[u] > 0]
    heapq.heapify(heap)
    in_heap = {u for _, u in heap}

    for ci, ckpt in enumerate(checkpoints):
        eps = ckpt.eps
        if eps == float("inf"):
            snapshots.append(p.copy())
            trace.checkpoint_boundaries.append(len(trace.events))
            continue
        while heap:
            neg_pri, u = heap[0]
            if -neg_pri <= eps:
                break
            heapq.heappop(heap)
            in_heap.discard(u)
            if _priority(u) <= eps:
                continue  # stale queue entry -- r[u] shrank since it was queued

            counter.count_queue_pop()
            counter.count_node_visit()
            trace.events.append((u, ci))

            if graph.out_edges[u]:
                p[u] += alpha * r[u]
                for v, ek in graph.out_edges[u]:
                    w = edge_weight[ek]
                    r[v] += (1.0 - alpha) * w * r[u]
                    counter.count_edge()
                    counter.count_madd()
                    trace.visited_edges.add((u, v, ek))
                    if r[v] > 0 and v not in in_heap:
                        heapq.heappush(heap, (-_priority(v), v))
                        in_heap.add(v)
            else:
                # A genuinely edgeless node (no self-loop either) has nowhere
                # for the (1-alpha) remainder to go; absorb it all into p[u]
                # rather than silently losing mass. In the real pipeline this
                # never triggers -- `graph_density.row_normalize_with_selfloop`
                # guarantees every node has at least a self-loop -- but this
                # module must not leak mass for an arbitrary input graph.
                p[u] += r[u]
            r[u] = 0.0  # fully settled -- priority(u) is now 0, nothing to requeue

        snapshots.append(p.copy())
        trace.checkpoint_boundaries.append(len(trace.events))

    # `p` alone is the final estimate (standard forward-push: sum(p) + sum(r)
    # == 1 exactly at every point). No further "+alpha*r" adjustment here --
    # that belongs to `ppr_push.py`'s different push-the-signal formulation
    # (every node is simultaneously its own source there), not to this
    # single-query personalised-PageRank push.
    return PushState(p=p, r=r), snapshots, trace


# ------------------------------------------------------ replay, differentiable

class _ReplayTraceFn(torch.autograd.Function):
    """Forward replays `trace` in plain numpy (as cheap as `local_push_trace`
    itself); backward is one hand-derived reverse walk over the same trace,
    not autograd tracking each of the ~thousands of per-event pop/push
    micro-ops. That per-event autograd bookkeeping (individual `index_add`
    calls, one torch node each) was measured to cost ~two orders of
    magnitude more than the equivalent plain-numpy push at real cohort
    scale -- see the implementation plan/handover for the benchmark that
    caught this. Collapsing it to one `Function.apply` call is what makes
    every-training-node-as-its-own-query per epoch tractable.

    The backward derivation (why it's correct, not just fast): every op the
    forward pass performs is an additive accumulation --
    `p[u] += alpha*r[u]` (or the sink's `p[u] += r[u]`) and
    `r[v] += (1-alpha)*w_ek*r[u]` -- so gradients distribute by the ordinary
    sum rule. Walking events in *reverse* chronological order and keeping one
    running per-node adjoint `next_r_bar[u]` (meaning: "the gradient that
    should apply to whatever fed r[u] before its next -- in forward time --
    pop") is sufficient: at a pop of `u`, `next_r_bar[u]` (already resolved,
    since anything at a later forward time was visited earlier in this
    reverse walk) plus the checkpoint-snapshot gradients accumulated so far
    (`p_bar`) give this pop's own `grad_r_before`, which both (a) attributes
    to every incoming edge that fed it -- accumulating `edge_weight`'s
    gradient there -- and (b) overwrites `next_r_bar[u]` for the next
    (earlier) increment into `u`. Whatever remains in `next_r_bar` once every
    event has been walked is exactly the gradient into `source_mass`, since
    the initial deposit is just the earliest increment in the same additive
    chain. `p_bar` itself is built by adding each checkpoint's own output
    gradient before entering the segment of events that checkpoint's
    snapshot actually includes (working from the last checkpoint back to the
    first), matching that a snapshot is the sum of every increment at or
    before its own boundary.
    """

    @staticmethod
    def forward(ctx, edge_weight: torch.Tensor, source_mass: torch.Tensor,
               trace: Trace, graph: "_CSRGraph", alpha: float, n_checkpoints: int):
        w_np = edge_weight.detach().cpu().numpy().astype(np.float64)
        s_np = source_mass.detach().cpu().numpy().astype(np.float64)
        n = s_np.shape[0]

        p = np.zeros(n, dtype=np.float64)
        r = s_np.copy()
        n_events = len(trace.events)
        r_before = np.empty(n_events, dtype=np.float64)
        is_sink = np.empty(n_events, dtype=bool)

        snapshots = []
        boundary_iter = iter(trace.checkpoint_boundaries)
        next_boundary = next(boundary_iter, n_events)
        idx = 0
        for _ in range(n_checkpoints):
            while idx < next_boundary:
                u, _ci = trace.events[idx]
                rb = r[u]
                r_before[idx] = rb
                edges_u = graph.out_edges[u]
                if edges_u:
                    is_sink[idx] = False
                    p[u] += alpha * rb
                    for v, ek in edges_u:
                        r[v] += (1.0 - alpha) * w_np[ek] * rb
                else:
                    is_sink[idx] = True
                    p[u] += rb
                r[u] = 0.0
                idx += 1
            snapshots.append(p.copy())
            next_boundary = next(boundary_iter, n_events)

        ctx.trace, ctx.graph, ctx.alpha = trace, graph, alpha
        ctx.r_before, ctx.is_sink, ctx.w_np = r_before, is_sink, w_np
        ctx.ew_dtype, ctx.ew_device = edge_weight.dtype, edge_weight.device
        ctx.sm_dtype, ctx.sm_device = source_mass.dtype, source_mass.device

        y = np.stack(snapshots, axis=0)  # [n_checkpoints, n]
        return torch.tensor(y, dtype=edge_weight.dtype, device=edge_weight.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        trace, graph, alpha = ctx.trace, ctx.graph, ctx.alpha
        r_before, is_sink, w_np = ctx.r_before, ctx.is_sink, ctx.w_np
        events, boundaries = trace.events, trace.checkpoint_boundaries

        go = grad_output.detach().cpu().numpy().astype(np.float64)  # [K, n]
        n = go.shape[1]

        p_bar = np.zeros(n, dtype=np.float64)
        next_r_bar = np.zeros(n, dtype=np.float64)
        grad_w = np.zeros_like(w_np, dtype=np.float64)

        for s in range(go.shape[0] - 1, -1, -1):
            p_bar += go[s]
            seg_start = boundaries[s - 1] if s > 0 else 0
            seg_end = boundaries[s]
            for e_idx in range(seg_end - 1, seg_start - 1, -1):
                u, _ci = events[e_idx]
                rb = r_before[e_idx]
                if not is_sink[e_idx]:
                    grad_rb = alpha * p_bar[u]
                    for v, ek in graph.out_edges[u]:
                        rb_v = next_r_bar[v]
                        if rb_v != 0.0:
                            grad_w[ek] += rb_v * (1.0 - alpha) * rb
                            grad_rb += rb_v * (1.0 - alpha) * w_np[ek]
                else:
                    grad_rb = p_bar[u]
                next_r_bar[u] = grad_rb

        grad_edge_weight = torch.tensor(grad_w, dtype=ctx.ew_dtype, device=ctx.ew_device)
        grad_source_mass = torch.tensor(next_r_bar, dtype=ctx.sm_dtype, device=ctx.sm_device)
        return grad_edge_weight, grad_source_mass, None, None, None, None


def replay_with_grad(trace: Trace, edge_index: np.ndarray, edge_weight: torch.Tensor,
                     source_mass: torch.Tensor, alpha: float, n_checkpoints: int,
                     graph: "_CSRGraph | None" = None):
    """Replay `trace`'s exact node-pop order for ONE query with gradients
    flowing through `edge_weight` and `source_mass`, never through the
    (fixed, detached) pop order.

    Returns (final p [n] tensor with grad, list of per-checkpoint p
    snapshots with grad). Must reproduce `local_push_trace`'s own snapshots
    to 1e-6 when `edge_weight`/`source_mass` hold the same values used to
    build the trace -- that equality is the test that makes this
    trustworthy (`test_replay_matches_nondifferentiable_inference_to_1e6`).

    `graph`, if given, is a `build_graph(edge_index, n)` result reused across
    many queries against the same frozen topology -- see `_CSRGraph`.
    """
    if graph is None:
        graph = _CSRGraph.build(edge_index, trace.n)
    y = _ReplayTraceFn.apply(edge_weight, source_mass, trace, graph, alpha, n_checkpoints)
    snapshots = list(y.unbind(0))
    return snapshots[-1], snapshots


# ------------------------------------------------------------ stage output

def stage_context(h0: torch.Tensor, p_stage: torch.Tensor, V: torch.nn.Module) -> torch.Tensor:
    """h_bar_q^(s) = (sum_u p_q^{(s)}(u) V h_u^{(0)}) / (sum_u p_q^{(s)}(u) + eps).

    `h0` is [n, d_in] (every graph node's own initial representation),
    `p_stage` is [n] (one query's PPR mass at checkpoint s). Returns [d_hidden].
    """
    weighted = (p_stage.unsqueeze(1) * V(h0)).sum(dim=0)
    return weighted / (p_stage.sum() + 1e-8)


def stage_representation(h0_q: torch.Tensor, h_bar_q: torch.Tensor,
                         w_self: torch.nn.Module, w_diff: torch.nn.Module,
                         w_int: torch.nn.Module, ln: torch.nn.Module) -> torch.Tensor:
    """g_q^(s) = LN(W_self h0_q + W_diff h_bar_q + W_int (h0_q * h_bar_q) + b).

    `h0_q` is query q's own initial representation [d_in]; `h_bar_q` is
    `stage_context`'s output for the same query and checkpoint. The
    elementwise product `h0_q * h_bar_q` requires `V`'s output dimension to
    equal `d_in` -- the caller (`arch4r.py`) is responsible for that, e.g.
    by giving `V` the same width as the fusion layer's output.
    """
    combined = w_self(h0_q) + w_diff(h_bar_q) + w_int(h0_q * h_bar_q)
    return ln(combined)
