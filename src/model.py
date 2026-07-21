"""EEGNN: SAS-GNN backbone + per-node confidence-aware exit heads.

Halting follows the ACT-style soft formulation. For each node we track the
probability mass that has *not* yet halted; at exit l the mass that halts is
`remainder_l * p_halt_l`, and the final prediction is the mixture of per-exit
predictions weighted by that mass. The deepest exit absorbs whatever remains,
so `max_layers` is a budget cap rather than a fixed depth.

At inference the policy is hard: a node halts the first time its halt branch
wins, its representation freezes, and it stops consuming compute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import ExitHead, SASGNNLayer


class EEGNN(nn.Module):
    def __init__(self, in_dim, cfg, n_classes=2):
        super().__init__()
        m = cfg["model"]
        self.hidden_dim = m["hidden_dim"]
        self.max_layers = m["max_layers"]
        self.exit_every = m["exit_every"]
        self.n_classes = n_classes
        self.use_exits = True

        self.encoder = nn.Linear(in_dim, self.hidden_dim)
        self.backbone = SASGNNLayer(
            self.hidden_dim, epsilon=m["epsilon"], gamma=m["gamma"]
        )
        self.dropout = nn.Dropout(m["dropout"])

        # Exit layer indices are 1-based propagation steps; the final step
        # always carries a head so every node has a fallback.
        self.exit_layers = sorted(
            set(list(range(self.exit_every, self.max_layers + 1, self.exit_every))
                + [self.max_layers])
        )
        self.exits = nn.ModuleList(
            ExitHead(self.hidden_dim, n_classes, dropout=m["dropout"])
            for _ in self.exit_layers
        )

    # -- full-depth mode (P2 backbone baseline) ---------------------------
    def forward_full_depth(self, x, edge_index, edge_weight=None):
        h = self.dropout(F.relu(self.encoder(x)))
        for _ in range(self.max_layers):
            h = self.backbone(h, edge_index, edge_weight)
        return self.exits[-1].task(self.dropout(h))

    # -- multi-exit mode (P3) ---------------------------------------------
    def forward(self, x, edge_index, edge_weight=None, tau=1.0, hard=False):
        """Returns a dict with per-exit logits, exit mass, and final logits."""
        if not self.use_exits:
            logits = self.forward_full_depth(x, edge_index, edge_weight)
            return {
                "exit_logits": [logits],
                "exit_probs": torch.ones(x.size(0), 1, device=x.device),
                "final_logits": logits,
                "expected_depth": torch.full((x.size(0),), float(self.max_layers),
                                             device=x.device),
                "exit_layer": torch.full((x.size(0),), self.max_layers,
                                         device=x.device, dtype=torch.long),
                "exit_layers": [self.max_layers],
            }

        n = x.size(0)
        device = x.device
        h = self.dropout(F.relu(self.encoder(x)))

        remainder = torch.ones(n, device=device)          # unhalted mass
        alive = torch.ones(n, dtype=torch.bool, device=device)  # hard policy
        exit_logits, exit_mass = [], []
        hard_exit = torch.full((n,), self.max_layers, dtype=torch.long, device=device)
        h_frozen = h

        exit_ptr = 0
        for step in range(1, self.max_layers + 1):
            h_new = self.backbone(h, edge_index, edge_weight)
            # Halted nodes keep the representation they exited with. During
            # training every node is propagated (the mixture is soft), but the
            # freeze is what makes the inference-time saving real.
            h = torch.where(alive.unsqueeze(1), h_new, h) if not self.training else h_new
            h_frozen = torch.where(alive.unsqueeze(1), h, h_frozen)

            if step not in self.exit_layers:
                continue

            head = self.exits[exit_ptr]
            exit_ptr += 1
            is_last = step == self.exit_layers[-1]

            task_logits, _, p_halt = head(
                h, tau=tau, hard=hard, training=self.training
            )
            if is_last:
                p_halt = torch.ones_like(p_halt)  # fallback: everyone exits

            mass = remainder * p_halt
            remainder = remainder - mass

            exit_logits.append(task_logits)
            exit_mass.append(mass)

            newly = alive & (p_halt > 0.5)
            hard_exit = torch.where(newly, torch.full_like(hard_exit, step), hard_exit)
            alive = alive & ~newly
            if is_last:
                hard_exit = torch.where(alive, torch.full_like(hard_exit, step), hard_exit)

        probs = torch.stack(exit_mass, dim=1)              # [n, n_exits]
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        stacked = torch.stack(exit_logits, dim=1)          # [n, n_exits, C]
        final_logits = (probs.unsqueeze(-1) * stacked).sum(dim=1)

        depths = torch.tensor([float(l) for l in self.exit_layers], device=device)
        expected_depth = (probs * depths).sum(dim=1)

        return {
            "exit_logits": exit_logits,
            "exit_probs": probs,
            "final_logits": final_logits,
            "expected_depth": expected_depth,
            "exit_layer": hard_exit,
            "exit_layers": self.exit_layers,
        }

    @torch.no_grad()
    def predict(self, x, edge_index, edge_weight=None, tau=1.0):
        """Hard-policy inference. Each node's prediction comes from its exit."""
        self.eval()
        out = self.forward(x, edge_index, edge_weight, tau=tau, hard=True)
        if not self.use_exits:
            return out

        stacked = torch.stack(out["exit_logits"], dim=1)
        idx = torch.tensor(
            [out["exit_layers"].index(int(l)) for l in out["exit_layer"]],
            device=x.device,
        )
        realised = stacked[torch.arange(x.size(0), device=x.device), idx]
        out["realised_logits"] = realised
        return out
