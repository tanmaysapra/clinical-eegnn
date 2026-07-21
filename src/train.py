"""Transductive training loop with Gumbel temperature annealing.

Early stopping is on validation AUROC, not loss -- with an imbalanced label and
a multi-term objective, loss is a poor proxy for the number we care about.
"""

import copy
import random

import numpy as np
import torch

from src.evaluate import auroc
from src.losses import multi_exit_loss
from src.model import EEGNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tau_at(epoch, epochs, cfg):
    """Anneal Gumbel temperature high -> low: exploratory, then near-discrete."""
    t0, t1 = cfg["train"]["gumbel_tau_start"], cfg["train"]["gumbel_tau_end"]
    frac = epoch / max(epochs - 1, 1)
    return t0 * (t1 / t0) ** frac


def train_model(data, cfg, use_exits=True, epochs=None, verbose=True, log_every=25):
    """Train an EEGNN (or its full-depth backbone) on one patient graph."""
    device = get_device()
    data = data.to(device)
    epochs = epochs if epochs is not None else cfg["train"]["epochs"]

    model = EEGNN(data.x.shape[1], cfg).to(device)
    model.use_exits = use_exits
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    best = {"auroc": -1.0, "epoch": -1, "state": copy.deepcopy(model.state_dict())}
    history = []
    patience = cfg["train"]["patience"]

    for epoch in range(epochs):
        tau = _tau_at(epoch, epochs, cfg)
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight, tau=tau)
        loss, parts = multi_exit_loss(out, data.y, data.train_mask, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            ev = model(data.x, data.edge_index, data.edge_weight, tau=max(tau, 0.5))
            probs = torch.softmax(ev["final_logits"], dim=-1)[:, 1]
            va = auroc(data.y[data.val_mask].cpu().numpy(),
                       probs[data.val_mask].cpu().numpy())
            depth = float(ev["expected_depth"][data.val_mask].mean())

        history.append({"epoch": epoch, "val_auroc": va, "avg_depth": depth, **parts})
        if va > best["auroc"]:
            best = {"auroc": va, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  loss {parts['total']:.4f}  "
                  f"val_auroc {va:.4f}  avg_depth {depth:.2f}  tau {tau:.2f}")
        if epoch - best["epoch"] > patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best {best['epoch']})")
            break

    model.load_state_dict(best["state"])
    return model, history, best
