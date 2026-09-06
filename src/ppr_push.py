"""Architecture 2 from the spec: PPR-Push, decoupled propagation with a
provably local early exit.

Feature learning and propagation are separated. A shallow MLP encodes each
patient once, with no graph involved; personalised-PageRank diffusion then
spreads those codes over the similarity graph. Adaptivity comes from the
*algorithm* rather than from a learned halting unit: a node stops being touched
when its residual mass falls below tolerance, which is a deterministic function
of the local structure around that patient.

Why this is worth having next to Architectures 1 and 3
------------------------------------------------------
Both of the others buy efficiency with a learned or calibrated rule, and their
savings therefore have to be verified empirically at inference time. Here the
work is bounded by the standard local-push analysis -- total pushed mass is
bounded by the source mass over `eps_push`, independent of N -- so the compute
guarantee is structural. It is also the only one of the three whose exit
mechanism cannot collapse: there is no halting head to saturate shut, which is
exactly the failure mode diagnosed in Architecture 1 (SS5o correction 4).

Forward push, not per-source push
---------------------------------
The spec writes the diffusion per source node v. For transductive node
classification the equivalent and far cheaper formulation pushes the *signal*:
start with residual r = Z (the encoder output), and repeatedly move mass from
nodes whose residual is large relative to their degree. The fixed point is the
same PPR operator, alpha (I - (1-alpha) A)^-1 Z, and the per-node work bound
carries over. A node's "exit round" is the round after which it is never
pushed again.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import _gcn_norm


class PPRPush(nn.Module):
    """Shallow encoder + local-push PPR diffusion + linear read-out.

    `alpha` is the teleport probability (higher = more local). `eps_push` is
    the residual tolerance that sets the work/accuracy trade-off, and is the
    knob this architecture exposes in place of a ponder cost.
    """

    def __init__(self, in_dim, cfg, n_classes=2, alpha=0.15, eps_push=1e-4,
                 max_rounds=None):
        super().__init__()
        m = cfg["model"]
        self.hidden_dim = m["hidden_dim"]
        self.alpha = alpha
        self.eps_push = eps_push
        self.max_rounds = max_rounds or m["max_layers"]
        self.n_classes = n_classes

        # No graph anywhere in the encoder: this is the whole point of the
        # decoupling, and it is why SHAP applies to it cleanly (spec SS2.7).
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.readout = nn.Linear(self.hidden_dim, n_classes)

    # ------------------------------------------------------------------ push
    def diffuse(self, z, ei, norm, deg, snapshots=None, eps_push=None,
                differentiable=False):
        """Local-push PPR. Returns (pi, ledger).

        Only nodes whose residual exceeds `eps_push * deg` are pushed in a
        given round, so the frontier shrinks and the work per round with it.
        The ledger records the edges actually touched -- the measured compute,
        which is the number this architecture exists to produce.
        """
        eps = self.eps_push if eps_push is None else eps_push
        n, d = z.shape
        device = z.device
        src, dst = ei[0], ei[1]

        pi = torch.zeros_like(z)
        r = z.clone()
        exit_round = torch.full((n,), self.max_rounds, dtype=torch.long,
                                device=device)
        ever_pushed = torch.zeros(n, dtype=torch.bool, device=device)

        edges_touched = 0
        pushes = 0
        dense_edges = int(ei.size(1)) * self.max_rounds
        snaps = []

        for rnd in range(1, self.max_rounds + 1):
            # Push criterion: residual mass per unit degree, per the standard
            # local-push rule. Degree-scaling is what makes hub patients push
            # sooner and isolated ones settle immediately.
            mass = r.norm(dim=1)
            frontier = mass > (eps * deg.clamp_min(1.0))
            if not frontier.any():
                break

            fidx = torch.nonzero(frontier, as_tuple=True)[0]
            pushes += int(fidx.numel())
            # Settle alpha of the frontier's residual into the estimate.
            contrib = self.alpha * r[fidx]
            pi = pi.index_add(0, fidx, contrib) if differentiable else \
                pi.index_add_(0, fidx, contrib)

            # The remaining (1-alpha) flows to neighbours along normalised edges.
            out_mask = frontier[src]
            edges_touched += int(out_mask.sum())
            e_src, e_dst = src[out_mask], dst[out_mask]
            flow = (1.0 - self.alpha) * norm[out_mask].unsqueeze(1) * r[e_src]

            r_new = r.clone()
            r_new[fidx] = 0.0
            r_new = r_new.index_add(0, e_dst, flow)
            r = r_new

            ever_pushed |= frontier
            exit_round = torch.where(frontier,
                                     torch.full_like(exit_round, rnd),
                                     exit_round)
            if snapshots and rnd in snapshots:
                snaps.append(pi.clone())

        # Anything never pushed keeps its own encoding: with no residual to
        # move, the PPR estimate for that patient is its teleport term alone.
        pi = pi + self.alpha * r
        ledger = {
            "edges_touched": edges_touched,
            "dense_edges": dense_edges,
            "pushes": pushes,
            "rounds_used": rnd,
            "exit_round": exit_round,
            "mean_exit_round": float(exit_round.float().mean()),
            "edge_work_saved_pct": 100.0 * (1.0 - edges_touched / max(dense_edges, 1)),
        }
        return pi, ledger, snaps

    # --------------------------------------------------------------- forward
    def forward(self, x, edge_index, edge_weight=None, snapshots=None,
                eps_push=None):
        n = x.size(0)
        norm, ei = _gcn_norm(edge_index, edge_weight, n, x.dtype, x.device)
        deg = torch.zeros(n, dtype=x.dtype, device=x.device).scatter_add_(
            0, ei[0], torch.ones(ei.size(1), dtype=x.dtype, device=x.device))

        z = self.encoder(x)
        pi, ledger, snaps = self.diffuse(
            z, ei, norm, deg, snapshots=snapshots, eps_push=eps_push,
            differentiable=self.training)
        out = {
            "logits": self.readout(pi),
            "pi": pi,
            "ledger": ledger,
            "encoder_logits": self.readout(z),   # the no-graph ablation, free
        }
        if snaps:
            out["snapshot_logits"] = [self.readout(s) for s in snaps]
        return out

    @torch.no_grad()
    def stage_margins(self, x, edge_index, edge_weight=None, snapshots=(1, 2, 4)):
        """Margins at increasing push budgets, for the shared certified gate.

        Gives Architecture 2 the same shape of input the gate takes from the
        other two: a [n_stages, n] matrix of progressively better opinions.
        """
        self.eval()
        out = self.forward(x, edge_index, edge_weight, snapshots=list(snapshots))
        stages = out.get("snapshot_logits", []) + [out["logits"]]
        return torch.stack([lg[:, 1] - lg[:, 0] for lg in stages]), out["ledger"]


def ppr_loss(out, y, mask, cfg, lam_smooth=0.0, edge_index=None, edge_weight=None):
    """Cost-sensitive focal loss, plus the spec's optional Laplacian term.

    The smoothness penalty compensates the shallow encoder for not being
    jointly trained through message passing: it pushes connected patients'
    codes together so that diffusion has something informative to spread.
    """
    from src.losses import focal_loss

    t = cfg["train"]
    class_weight = torch.tensor([1.0, float(t["cost_ratio"])], device=y.device)
    l_task = focal_loss(out["logits"][mask], y[mask], gamma=t["focal_gamma"],
                        class_weight=class_weight)

    l_smooth = torch.zeros((), device=y.device)
    if lam_smooth > 0 and edge_index is not None:
        z = out["pi"]
        src, dst = edge_index[0], edge_index[1]
        w = (edge_weight if edge_weight is not None
             else torch.ones(src.numel(), device=z.device))
        l_smooth = (w * (z[src] - z[dst]).pow(2).sum(1)).mean()

    total = l_task + lam_smooth * l_smooth
    return total, {"task": l_task.detach().item(),
                   "smooth": float(l_smooth.detach()),
                   "total": total.detach().item()}
