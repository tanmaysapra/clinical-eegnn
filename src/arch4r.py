"""Architecture 4R model wiring: encoder + density-adaptive graph + local-push
diffusion, in three ablation arms sharing every learned component except the
topology's own `k`.

Why the three arms are one model class, not three
----------------------------------------------------
`encoder_only`, `fixed_k`, and `adaptive_k` must share the encoder, the
GATv2-style edge scorer, and the view-reliability fusion -- the plan's own
required invariant (`test_fixed_k_and_adaptive_k_share_every_component_
except_k`). This module gets that for free by construction: `Arch4RModel`
always builds every graph-dependent submodule regardless of `mode`, and
`mode=="encoder_only"` simply never calls them (no `attached_graph` is
consulted at all in that branch -- not "consulted but ignored", genuinely
untouched, which is what `test_encoder_only_arm_has_no_graph_dependency`
checks via a no-gradient assertion). `fixed_k` vs `adaptive_k` never appear
as a code branch inside this module at all: which `k` a given arm uses is a
property of the `graph_density.ViewGraphConfig` that built the
`Arch4RGraphBatch` fed into `forward`, not of anything this model owns.

Cold start / G_q=0 (no graph) fallback
----------------------------------------
A query with `graph_available[i] == False` (or, in `encoder_only` mode,
every query) gets `h_bar_q^(s) = h0_q` at every checkpoint -- there is no
diffused context to weight by, so the stage representation degenerates to
using the query's own encoded representation at every stage. This makes
`encoder_only`'s output structurally identical to a graph-available query
whose graph happened to return nothing, which is the intended behaviour
(the plan calls `encoder_only` "also...the real code path other arms fall
back into").

Transductive training's per-query cost (see `src/local_push.py`)
--------------------------------------------------------------------
Training is transductive over the training reference graph: every train
node is simultaneously a member of the graph and, one at a time, a query
scored against it. A from-scratch benchmark (see the implementation
handover) found this looked infeasible at real cohort scale purely from two
redundant per-query rebuild costs in `local_push.py`, now fixed there
(`build_graph(...)` built once per `forward()` call and reused for every
query in `attached_graph.query_idx`, not rebuilt per query) -- this module
relies on that fix by building the fused adjacency and its `build_graph(...)`
topology exactly once per `forward()` call.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder_mv import MultiViewEncoder, VIEW_OUT_DIMS
from src.graph_density import (EdgeScorerGATv2, ViewReliability, fuse_views,
                               row_normalize_with_selfloop)
from src.instrument import OpCounter
from src.local_push import (build_graph, local_push_trace, one_hot_source,
                            replay_with_grad, stage_context_from_Vh0,
                            stage_representation)

MODES = ("encoder_only", "fixed_k", "adaptive_k")


@dataclass
class Arch4RGraphBatch:
    """Everything `Arch4RModel.forward` needs to run local-push diffusion for
    a batch of query nodes against a fixed graph-node set.

    `per_view_edges[name]` is `(src, dst, base_weight)`, indices into the
    COMBINED node space `[0, n_nodes)` that `view_x`/`view_observed` (passed
    to `forward` alongside this batch) were encoded over.

    Two constructions, via the helpers below:
      - `transductive_batch(...)`: `n_nodes == n_train`, every train node is
        both a graph member and its own query (`query_idx ==
        arange(n_train)`), edges are the frozen train-train topology.
      - `attach_batch(...)`: `n_nodes == n_train + n_query`, held-out queries
        are appended after the frozen train block, edges are the train-train
        topology PLUS `ReferenceGraph.attach(...)`'s query->train edges (a
        query must be able to diffuse through the whole reference graph, not
        just reach its first ring).
    """
    per_view_edges: dict
    n_nodes: int
    query_idx: np.ndarray
    graph_available: np.ndarray


def transductive_batch(ref_graph, k_min: int | None = None) -> Arch4RGraphBatch:
    """Build the transductive-training batch: every train node is
    simultaneously a graph node and its own query, over the frozen
    train-train edges `graph_density.build_reference_graph` already built."""
    per_view_edges = {
        name: (v.edge_index_ref[0], v.edge_index_ref[1], v.edge_weight_ref)
        for name, v in ref_graph.views.items()
    }
    n_train = next(iter(ref_graph.views.values())).X_train.shape[0]
    available = ref_graph.train_graph_available(k_min=k_min)
    return Arch4RGraphBatch(per_view_edges=per_view_edges, n_nodes=n_train,
                            query_idx=np.arange(n_train), graph_available=available)


def attach_batch(ref_graph, attached, n_train: int) -> Arch4RGraphBatch:
    """Build the held-out-evaluation batch: `attached` (a
    `graph_density.AttachedGraph` from `ReferenceGraph.attach(...)`) is
    appended after the frozen train block so held-out queries can diffuse
    through the whole reference graph."""
    n_query = attached.n_query
    n_nodes = n_train + n_query
    per_view_edges = {}
    for name, v in ref_graph.views.items():
        tt_src, tt_dst, tt_w = v.edge_index_ref[0], v.edge_index_ref[1], v.edge_weight_ref
        if name in attached.per_view_edges:
            q_src, q_dst, q_w = attached.per_view_edges[name]
            q_src = q_src + n_train  # shift query-local indices into combined space
        else:
            q_src = np.zeros(0, dtype=int)
            q_dst = np.zeros(0, dtype=int)
            q_w = np.zeros(0, dtype=tt_w.dtype)
        per_view_edges[name] = (
            np.concatenate([tt_src, q_src]),
            np.concatenate([tt_dst, q_dst]),
            np.concatenate([tt_w, q_w]),
        )
    query_idx = np.arange(n_train, n_nodes)
    return Arch4RGraphBatch(per_view_edges=per_view_edges, n_nodes=n_nodes,
                            query_idx=query_idx, graph_available=attached.graph_available)


class Arch4RModel(nn.Module):
    def __init__(self, view_in_dims: dict[str, int], cfg: dict, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.cfg = dict(cfg)

        fusion_dim = self.cfg.get("fusion_dim", 64)
        stage_hidden = self.cfg.get("stage_hidden", 32)
        out_dims = self.cfg.get("view_out_dims", VIEW_OUT_DIMS)
        self.alpha = self.cfg.get("alpha", 0.15)
        self.top_k_keep = self.cfg.get("top_k_keep", 64)

        self.encoder = MultiViewEncoder(view_in_dims, fusion_dim=fusion_dim,
                                        dropout=self.cfg.get("dropout", 0.15),
                                        out_dims=out_dims)
        self.view_names = list(self.encoder.view_names)

        # Always built, regardless of mode -- see module docstring.
        z_dims = {v: out_dims[v] for v in self.view_names}
        self.scorers = nn.ModuleDict({
            v: EdgeScorerGATv2(z_dims[v], hidden=16) for v in self.view_names
        })
        self.reliability = ViewReliability(z_dims)
        self.V = nn.Linear(fusion_dim, fusion_dim)
        self.w_self = nn.Linear(fusion_dim, stage_hidden)
        self.w_diff = nn.Linear(fusion_dim, stage_hidden)
        self.w_int = nn.Linear(fusion_dim, stage_hidden)
        self.stage_ln = nn.LayerNorm(stage_hidden)
        self.stage_head = nn.Linear(stage_hidden, 1)
        # Graph-independent by construction (operates on h0 alone) so it
        # stays available even in encoder_only / cold-start fallback.
        self.error_head = nn.Linear(fusion_dim, 1)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------- forward

    def forward(self, view_x: dict[str, torch.Tensor], view_observed: dict[str, torch.Tensor],
               attached_graph: Arch4RGraphBatch | None, checkpoints: list) -> dict:
        enc_out = self.encoder(view_x, view_observed)
        h0 = enc_out["h0"]
        n_nodes = h0.shape[0]
        n_checkpoints = len(checkpoints)
        counter = OpCounter()

        if attached_graph is not None:
            query_idx = np.asarray(attached_graph.query_idx)
        else:
            query_idx = np.arange(n_nodes)
        n_query = len(query_idx)

        use_graph = (self.mode != "encoder_only") and (attached_graph is not None)

        query_idx_t = torch.as_tensor(query_idx, dtype=torch.long, device=h0.device)
        h0_q = h0[query_idx_t]
        error_head_raw = self.error_head(h0_q).squeeze(-1)

        if not use_graph:
            g = stage_representation(h0_q, h0_q, self.w_self, self.w_diff, self.w_int,
                                     self.stage_ln)
            logit = self.stage_head(g).squeeze(-1)
            stage_logits = [logit for _ in range(n_checkpoints)]
            return {"h0": h0_q, "stage_logits": stage_logits,
                   "error_head_raw": error_head_raw, "ledger": counter.snapshot()}

        graph_available = np.asarray(attached_graph.graph_available)
        view_z = enc_out["view_z"]
        reliability_pi = self.reliability(view_z, enc_out["reliability_raw"])

        scorer_output = {}
        for v, (src, dst, _base_w) in attached_graph.per_view_edges.items():
            if len(src) == 0:
                scorer_output[v] = torch.zeros(0, device=h0.device)
                continue
            src_t = torch.as_tensor(src, dtype=torch.long, device=h0.device)
            dst_t = torch.as_tensor(dst, dtype=torch.long, device=h0.device)
            scorer_output[v] = self.scorers[v](view_z[v][src_t], view_z[v][dst_t])

        edge_index, edge_weight = fuse_views(attached_graph.per_view_edges, reliability_pi,
                                             scorer_output, n_nodes=n_nodes,
                                             top_k_keep=self.top_k_keep)
        full_edge_index, full_p_weight = row_normalize_with_selfloop(edge_index, edge_weight, n_nodes)

        edge_index_np = full_edge_index.detach().cpu().numpy()
        edge_weight_np = full_p_weight.detach().cpu().numpy()
        # Built once, reused for every query below -- the fix that makes
        # scoring every training node as its own query per epoch tractable.
        shared_graph = build_graph(edge_index_np, n_nodes)
        # V(h0) doesn't depend on the query or the checkpoint -- computing it
        # once here instead of inside the loop below avoids keeping one
        # [n_nodes, d_hidden] activation alive per (query, checkpoint) pair,
        # which is what OOM'd a 16 GB GPU even after batching the query loop.
        Vh0 = self.V(h0)

        stage_logits_rows = [[] for _ in range(n_checkpoints)]
        for local_i, q in enumerate(query_idx):
            if not graph_available[local_i]:
                h_bar = h0[q]
                g = stage_representation(h0[q], h_bar, self.w_self, self.w_diff,
                                         self.w_int, self.stage_ln)
                logit = self.stage_head(g).squeeze(-1)
                for s in range(n_checkpoints):
                    stage_logits_rows[s].append(logit)
                continue

            e_q = one_hot_source(int(q), n_nodes)
            _, _, trace = local_push_trace(edge_index_np, edge_weight_np, e_q,
                                           alpha=self.alpha, checkpoints=checkpoints,
                                           counter=counter, graph=shared_graph)
            source_t = torch.tensor(e_q, dtype=full_p_weight.dtype, device=h0.device)
            _, snapshots = replay_with_grad(trace, edge_index_np, full_p_weight, source_t,
                                            alpha=self.alpha, n_checkpoints=n_checkpoints,
                                            graph=shared_graph)
            for s in range(n_checkpoints):
                h_bar = stage_context_from_Vh0(Vh0, snapshots[s])
                g = stage_representation(h0[q], h_bar, self.w_self, self.w_diff,
                                         self.w_int, self.stage_ln)
                logit = self.stage_head(g).squeeze(-1)
                stage_logits_rows[s].append(logit)

        stage_logits = [torch.stack(rows) for rows in stage_logits_rows]
        return {"h0": h0_q, "stage_logits": stage_logits,
               "error_head_raw": error_head_raw, "ledger": counter.snapshot()}


# ------------------------------------------------------------------- loss

def arch4r_loss(out: dict, y: torch.Tensor, mask: torch.Tensor, cfg: dict,
                lam_surv: float = 0.0):
    """Binary cross-entropy over every checkpoint's stage logits, averaged
    over checkpoints (deep supervision across exits) then over masked rows.

    `lam_surv` is reserved for the Phase 2/3 survival head -- Phase 1 is
    binary-only, so it must be left at 0.0 here; the parameter exists only
    so this signature won't need to break when that head is added.
    """
    if lam_surv != 0.0:
        raise NotImplementedError("survival head is Phase 2/3 scope; lam_surv must be 0.0")
    y_m = y[mask].float()
    total = 0.0
    info = {}
    for s, logits in enumerate(out["stage_logits"]):
        l = F.binary_cross_entropy_with_logits(logits[mask], y_m)
        info[f"stage{s}_bce"] = float(l.detach())
        total = total + l
    total = total / len(out["stage_logits"])
    info["total"] = float(total.detach())
    return total, info
