"""Architecture 3 from the spec: Fixed-Block early exit, corrected.

Distinct per-block weights -- *not* Architecture 1's shared operator -- with one
exit head per block, plus the two fixes the spec's SS3.3 asks for:

  Fix 1, active-subgraph masking. Block l runs only on nodes that have not yet
  exited, and only over the edges pointing into them. Exited nodes still *send*
  messages (their frozen representation is a constant on the right-hand side),
  so the masked computation is numerically identical to running the full graph
  and freezing afterwards -- but the work shrinks with the active set.
  `test_architectures.py` asserts that equivalence, which is what makes the
  measured saving a saving rather than a different model.

  Architecture 1 never implemented this. `model.py` runs `self.backbone` over
  every node and every edge at every step and applies the mask *after*, so its
  reported savings are analytical (exit depth x per-layer cost), never measured
  from a shrinking frontier. That gap is why the spec's central efficiency
  claim has not so far been tested.

  Fix 2, a calibrated certified gate. Exits are decided by the isotonic +
  Clopper-Pearson two-sided gate in `src/cascade.py`, the same machinery the
  staged information cascade uses, rather than by the hand-set entropy
  thresholds of `baselines.fixed_threshold_ee`. Sharing the gate across
  architectures is what makes their efficiency numbers comparable.

Training runs every block on every node (no masking, all heads supervised).
Masking is an inference-time property: it cannot be applied during training
because the gate that decides exits is fitted afterwards, on a held-out
selection split.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import ExitHead, SASGNNLayer, _gcn_norm


class FixedBlockEE(nn.Module):
    """`n_blocks` separately-weighted SAS operators, one exit head each.

    The operator family is identical to Architecture 1's `SASGNNLayer`; the
    only structural differences are that the weights are *not* tied across
    depth and that the exit points are a fixed, discrete set. That is precisely
    the contrast the spec wanted isolated -- "does ACT actually beat discrete
    exit stages" -- so nothing else about the backbone should vary.
    """

    def __init__(self, in_dim, cfg, n_blocks=4, n_classes=2):
        super().__init__()
        m = cfg["model"]
        self.hidden_dim = m["hidden_dim"]
        self.n_blocks = n_blocks
        self.n_classes = n_classes

        self.encoder = nn.Linear(in_dim, self.hidden_dim)
        self.dropout = nn.Dropout(m["dropout"])
        # Distinct weights per block -- the defining property of Architecture 3,
        # and the reason its parameter count is n_blocks x Architecture 1's.
        self.blocks = nn.ModuleList(
            SASGNNLayer(self.hidden_dim, epsilon=m["epsilon"], gamma=m["gamma"])
            for _ in range(n_blocks)
        )
        self.exits = nn.ModuleList(
            ExitHead(self.hidden_dim, n_classes, dropout=m["dropout"])
            for _ in range(n_blocks)
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def cache_norm(edge_index, edge_weight, n, dtype, device):
        """Full-graph symmetric normalisation, computed once.

        The graph never changes, so the degrees never change. Recomputing the
        normalisation from a masked edge subset would silently renormalise by
        the *active* degree, which is a different operator -- and would make
        masking change the answer instead of only the cost.
        """
        norm, ei = _gcn_norm(edge_index, edge_weight, n, dtype, device)
        return ei, norm

    # -------------------------------------------------------------- training
    def forward(self, x, edge_index, edge_weight=None, norm_cache=None):
        """Dense forward: every block on every node. Returns per-block logits."""
        n = x.size(0)
        ei, norm = norm_cache or self.cache_norm(
            edge_index, edge_weight, n, x.dtype, x.device)

        h = self.dropout(F.relu(self.encoder(x)))
        logits = []
        for blk, head in zip(self.blocks, self.exits):
            h = _sas_step(blk, h, ei, norm)
            logits.append(head.task(self.dropout(h)))
        return {"block_logits": logits, "n_blocks": self.n_blocks}

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def forward_masked(self, x, edge_index, edge_weight=None, exit_block=None,
                       norm_cache=None):
        """Active-set inference. `exit_block[v]` is the block v exits after.

        Returns the realised logits plus a work ledger. The ledger is the point
        of the method: `edges_processed` and `nodes_updated` are counted, not
        inferred from a mean exit depth, so "compute saved" is a measurement.
        """
        self.eval()
        n = x.size(0)
        device = x.device
        ei, norm = norm_cache or self.cache_norm(
            edge_index, edge_weight, n, x.dtype, device)
        dst = ei[1]

        if exit_block is None:
            exit_block = torch.full((n,), self.n_blocks - 1, dtype=torch.long,
                                    device=device)

        h = F.relu(self.encoder(x))          # dropout is off in eval
        realised = torch.zeros(n, self.n_classes, device=device)
        active = torch.ones(n, dtype=torch.bool, device=device)

        edges_processed = 0
        nodes_updated = 0
        dense_edges = int(ei.size(1)) * self.n_blocks
        dense_nodes = n * self.n_blocks

        for l, (blk, head) in enumerate(zip(self.blocks, self.exits)):
            idx = torch.nonzero(active, as_tuple=True)[0]
            if idx.numel() == 0:
                break
            keep = active[dst]                          # edges INTO active nodes
            sub_ei, sub_norm = ei[:, keep], norm[keep]
            edges_processed += int(keep.sum())
            nodes_updated += int(idx.numel())

            h = _sas_step(blk, h, sub_ei, sub_norm, update_idx=idx)
            out = head.task(h[idx])
            done = active & (exit_block <= l)
            if done.any():
                # Rows of `out` correspond to `idx`; select the exiting subset.
                sel = done[idx]
                realised[idx[sel]] = out[sel]
            active = active & ~done

        if active.any():                                # never gated out
            idx = torch.nonzero(active, as_tuple=True)[0]
            realised[idx] = self.exits[-1].task(h[idx])

        return {
            "realised_logits": realised,
            "edges_processed": edges_processed,
            "nodes_updated": nodes_updated,
            "dense_edges": dense_edges,
            "dense_nodes": dense_nodes,
            "edge_work_saved_pct": 100.0 * (1.0 - edges_processed / max(dense_edges, 1)),
            "node_work_saved_pct": 100.0 * (1.0 - nodes_updated / max(dense_nodes, 1)),
        }

    @torch.no_grad()
    def block_margins(self, x, edge_index, edge_weight=None, norm_cache=None):
        """Per-block P(death) margins for every node, at full depth.

        This is what the gate is calibrated and certified on: it must see every
        block's opinion of every patient, unconditioned on any exit rule.
        """
        self.eval()
        out = self.forward(x, edge_index, edge_weight, norm_cache=norm_cache)
        return torch.stack([lg[:, 1] - lg[:, 0] for lg in out["block_logits"]])


def _sas_step(blk, h, edge_index, norm, update_idx=None):
    """One SAS-GNN step over a pre-normalised (possibly restricted) edge set.

    `SASGNNLayer.forward` recomputes the normalisation internally, which is
    exactly what must not happen here -- see `cache_norm`. This drives the
    layer's `propagate`/`message` directly with the cached values instead.

    With `update_idx` the dense transform runs on those rows only, so the
    O(n d^2) half of the step shrinks with the active set too, not just the
    message passing.
    """
    m = blk.propagate(edge_index, x=h, norm=norm)
    W = blk.effective_weight()
    if update_idx is None:
        return h + blk.epsilon * blk.activation(m @ W + blk.bias)
    upd = h[update_idx] + blk.epsilon * blk.activation(
        m[update_idx] @ W + blk.bias)
    return h.index_copy(0, update_idx, upd)


def fixed_block_loss(out, y, mask, cfg, beta=0.5):
    """Per-block cost-sensitive focal loss + inter-block consistency.

    The spec's SS3.4: exponential layer weights `w_l = exp(beta l) / sum exp`,
    so deeper blocks -- which the shallow ones are distilled towards -- carry
    more of the objective, plus a consistency term tying each block to the
    final one. `losses.focal_loss` is reused so the class weighting is
    identical to Architecture 1's.
    """
    from src.losses import focal_loss

    t = cfg["train"]
    logits = [lg[mask] for lg in out["block_logits"]]
    y_m = y[mask]
    device = y.device
    class_weight = torch.tensor([1.0, float(t["cost_ratio"])], device=device)

    L = len(logits)
    w = torch.exp(beta * torch.arange(1, L + 1, dtype=torch.float32, device=device))
    w = w / w.sum()

    l_task = torch.zeros((), device=device)
    for i, lg in enumerate(logits):
        l_task = l_task + w[i] * focal_loss(
            lg, y_m, gamma=t["focal_gamma"], class_weight=class_weight)

    # Consistency: shallow blocks track the deepest block's distribution. The
    # spec calls for this explicitly; without it the early heads are free to
    # disagree with the final one, which is what makes an early exit unsafe.
    teacher = F.softmax(logits[-1].detach(), dim=-1)
    l_consist = torch.zeros((), device=device)
    for lg in logits[:-1]:
        l_consist = l_consist + F.kl_div(
            F.log_softmax(lg, dim=-1), teacher, reduction="batchmean")
    if L > 1:
        l_consist = l_consist / (L - 1)

    total = l_task + t.get("consistency_weight", 0.1) * l_consist
    return total, {"task": l_task.detach().item(),
                   "consistency": l_consist.detach().item(),
                   "total": total.detach().item()}
