"""Baselines that make the headline claim falsifiable.

  * GCN / GAT       -- is the graph + task learnable at all?
  * XGBoost         -- does the graph add anything over plain tabular features?
                       If not, say so; that is a real finding, not a bug.
  * fixed_threshold_ee -- entropy-thresholded exits bolted onto a *standard*
                       GCN, with no learned halting. Isolates how much of the
                       win comes from the stable SAS-GNN backbone versus from
                       learning the exit policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from src.evaluate import auroc


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2, n_classes=2, dropout=0.3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(n_layers))
        self.out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        hidden = []
        for conv in self.convs:
            x = self.dropout(F.relu(conv(x, edge_index, edge_weight)))
            hidden.append(x)
        logits = self.out(x)
        return (logits, hidden) if return_hidden else logits


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2, heads=4, n_classes=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.convs.append(GATConv(d, hidden_dim // heads, heads=heads, dropout=dropout))
            d = hidden_dim
        self.out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x, att = conv(x, edge_index, return_attention_weights=True)
            self.last_attention = att
            x = self.dropout(F.elu(x))
        return self.out(x)


def train_plain_gnn(model, data, cfg, epochs=None, verbose=False):
    """Shared training loop for the GCN/GAT baselines (class-weighted CE)."""
    epochs = epochs if epochs is not None else cfg["train"]["epochs"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    w = torch.tensor([1.0, float(cfg["train"]["cost_ratio"])], device=data.x.device)

    best_auc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_weight)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=w)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(data.x, data.edge_index, data.edge_weight), -1)[:, 1]
            va = auroc(data.y[data.val_mask].cpu().numpy(),
                       p[data.val_mask].cpu().numpy())
        if va > best_auc:
            best_auc, best_epoch = va, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch > cfg["train"]["patience"]:
            break
        if verbose and epoch % 50 == 0:
            print(f"  epoch {epoch:4d} loss {float(loss):.4f} val_auroc {va:.4f}")

    model.load_state_dict(best_state)
    return model, best_auc


def xgboost_baseline(X, y, masks, seed=42, want_shap=False):
    """Non-graph tabular reference point. If the GNNs cannot beat this, the
    patient-similarity graph is not adding signal -- report that honestly.

    Runs out-of-process (see src/_omp_guard) because xgboost and torch cannot
    share one process on macOS/arm64.
    """
    from src._omp_guard import run_xgb_worker

    _, val_mask, test_mask = masks
    out = run_xgb_worker(X, y, masks, seed=seed, want_shap=want_shap)
    metrics = {
        "val_auroc": auroc(y[val_mask], out["p_val"]),
        "test_auroc": auroc(y[test_mask], out["p_test"]),
    }
    return out, metrics, out["p_test"]


def fixed_threshold_ee(data, cfg, entropy_threshold=0.35, n_layers=None):
    """Naive early exit: a standard GCN with softmax-entropy exits at hand-set
    thresholds. No learned halting, no stability guarantee from the backbone."""
    n_layers = n_layers or cfg["model"]["max_layers"]
    device = data.x.device
    model = GCN(data.x.shape[1], cfg["model"]["hidden_dim"], n_layers=n_layers,
                dropout=cfg["model"]["dropout"]).to(device)
    heads = nn.ModuleList(
        nn.Linear(cfg["model"]["hidden_dim"], 2) for _ in range(n_layers)
    ).to(device)

    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    w = torch.tensor([1.0, float(cfg["train"]["cost_ratio"])], device=device)

    for _ in range(cfg["train"]["epochs"] // 2):
        model.train()
        opt.zero_grad()
        _, hidden = model(data.x, data.edge_index, data.edge_weight, return_hidden=True)
        loss = sum(
            F.cross_entropy(head(h)[data.train_mask], data.y[data.train_mask], weight=w)
            for head, h in zip(heads, hidden)
        ) / n_layers
        loss.backward()
        opt.step()

    model.eval()
    heads.eval()
    with torch.no_grad():
        _, hidden = model(data.x, data.edge_index, data.edge_weight, return_hidden=True)
        n = data.x.shape[0]
        probs = torch.zeros(n, device=device)
        exit_layer = torch.full((n,), n_layers, dtype=torch.long, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        for i, (head, h) in enumerate(zip(heads, hidden), start=1):
            p = torch.softmax(head(h), dim=-1)
            ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
            confident = alive & ((ent < entropy_threshold) | (i == n_layers))
            probs[confident] = p[confident, 1]
            exit_layer[confident] = i
            alive = alive & ~confident

    return {
        "probs": probs.cpu().numpy(),
        "exit_layer": exit_layer.cpu().numpy(),
        "avg_exit_depth": float(exit_layer.float().mean()),
    }
