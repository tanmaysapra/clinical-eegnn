"""Stage 3: what the patient graph can add *on top of* the boosted score.

This is the join between the two halves of the model, and it is deliberately
posed as a residual question rather than a rival one.

Why residual, and why that matters here
---------------------------------------
HANDOVER 5f diagnosed the project's central negative result: the kNN graph is
built from the feature matrix, so it cannot carry information that is not
already in `X`, and gradient boosting extracts more from those features
directly than message passing recovers by smoothing them. Every graph model
tried so far competed with XGBoost *over the same information* and lost.

A residual stage does not compete. It is handed the boosted margin and asked a
question the booster structurally cannot answer, because a tree ensemble scores
one row at a time:

    are patients like this one systematically under- or over-predicted?

The answer lives in the neighbours' **residuals**, and a residual is not a
function of `X` -- it is a function of `X` *and the labels*. So the graph now
transports something genuinely outside the feature matrix, which is the root
cause HANDOVER 7 said had to be attacked. It is the cheap version of "edges
from a source outside X": the edges are still built from `X`, but the signal
riding along them no longer is.

This is Correct & Smooth (Huang et al., ICLR 2021) used as a cascade stage, and
it depends on the out-of-fold discipline in `src/tabular`: with in-sample base
margins the training residuals collapse to ~0 and there is nothing to
propagate. That is very likely part of why earlier attempts saw the graph add
nothing.

Leakage
-------
Only *training* residuals ever seed the propagation. Validation and test
residuals are identically zero at initialisation, and the aggregation excludes
self-loops so no node can read its own residual back. Both are asserted, and
covered by a regression test.
"""

import numpy as np
import torch


# ------------------------------------------------------------- sparse operator

def normalised_adjacency(edge_index, edge_weight, n, device="cpu", self_loops=False):
    """Row-normalised sparse adjacency, self-loops off by default.

    Self-loops stay off because the whole point is to read the *neighbours'*
    residuals; a self-loop would let a training node recover its own label.
    """
    ei = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    w = (torch.ones(ei.shape[1], device=device) if edge_weight is None
         else torch.as_tensor(edge_weight, dtype=torch.float32, device=device))
    if self_loops:
        loop = torch.arange(n, device=device)
        ei = torch.cat([ei, torch.stack([loop, loop])], dim=1)
        w = torch.cat([w, torch.ones(n, device=device)])
    deg = torch.zeros(n, device=device).index_add_(0, ei[0], w).clamp_min(1e-12)
    val = w / deg[ei[0]]
    return torch.sparse_coo_tensor(ei, val, (n, n)).coalesce()


def propagate(A, x, hops=1, alpha=1.0):
    """`hops` rounds of x <- (1-alpha) x0 + alpha A x."""
    x0 = x
    for _ in range(hops):
        x = (1.0 - alpha) * x0 + alpha * torch.sparse.mm(A, x)
    return x


# --------------------------------------------------------------- correct & smooth

def correct_and_smooth(edge_index, edge_weight, base_prob, y, train_mask,
                       n, scale=1.0, hops=2, alpha=0.9, smooth_hops=0,
                       smooth_alpha=0.5, device="cpu"):
    """Propagate training residuals and correct the boosted probability.

    `scale` multiplies the propagated correction; it is the one free parameter
    and is fitted on the calibration split (see `fit_cs_scale`), never on test.
    Returns corrected probabilities, clipped into (0, 1).
    """
    A = normalised_adjacency(edge_index, edge_weight, n, device)
    resid = torch.zeros(n, 1, device=device)
    tr = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    pb = torch.as_tensor(base_prob, dtype=torch.float32, device=device)
    resid[tr, 0] = yt[tr] - pb[tr]
    assert float(resid[~tr].abs().max()) == 0.0, "non-training residual leaked in"

    corr = propagate(A, resid, hops=hops, alpha=alpha)[:, 0]
    out = (pb + scale * corr).clamp(1e-6, 1 - 1e-6)

    if smooth_hops:
        # Smooth the *corrected* prediction, anchoring training nodes at their
        # own labels -- the second half of Huang et al.
        z = out.clone().unsqueeze(1)
        z[tr, 0] = yt[tr]
        z = propagate(A, z, hops=smooth_hops, alpha=smooth_alpha)[:, 0]
        out = z.clamp(1e-6, 1 - 1e-6)
    return out.cpu().numpy().astype(np.float64)


def fit_cs_scale(edge_index, edge_weight, base_prob, y, train_mask, fit_mask, n,
                 grid=None, **kw):
    """Pick the correction scale on a held-out split, by AUROC."""
    from sklearn.metrics import roc_auc_score
    grid = np.linspace(0.0, 2.0, 21) if grid is None else grid
    best, best_s = -1.0, 0.0
    for s in grid:
        p = correct_and_smooth(edge_index, edge_weight, base_prob, y, train_mask,
                               n, scale=float(s), **kw)
        a = roc_auc_score(y[fit_mask], p[fit_mask])
        if a > best:
            best, best_s = a, float(s)
    return best_s, best


# ------------------------------------------------------------- learned residual

class ResidualGNN(torch.nn.Module):
    """Learns a margin correction from the neighbourhood. -> delta margin.

    Inputs are the scaled features, the running boosted margin, and the
    self-excluded neighbour-mean of the training residuals. The last one is the
    term the booster cannot compute, and giving it to the network explicitly --
    rather than hoping message passing rediscovers it -- is the difference
    between this and the GCN/GAT baselines that lost to XGBoost.

    The head is initialised at zero so the stage starts as an exact no-op: at
    step 0 the cascade is precisely the tabular model, and any accuracy the
    graph stage reports is accuracy it actually earned.
    """

    def __init__(self, in_dim, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.mix = torch.nn.ModuleList(
            torch.nn.Linear(2 * hidden, hidden) for _ in range(layers)
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x, A):
        h = self.enc(x)
        for lin in self.mix:
            h = torch.relu(lin(torch.cat([h, torch.sparse.mm(A, h)], dim=1)))
            h = self.dropout(h)
        return self.head(h).squeeze(-1)


def train_residual_gnn(x, base_margin, y, edge_index, edge_weight, train_mask,
                       val_mask, n, epochs=200, lr=5e-3, weight_decay=5e-4,
                       hidden=64, layers=2, patience=40, device="cpu",
                       seed=42, verbose=False):
    """Fit the correction on train, early-stopping on validation AUROC."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    A = normalised_adjacency(edge_index, edge_weight, n, device)
    X = torch.as_tensor(x, dtype=torch.float32, device=device)
    m = torch.as_tensor(base_margin, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    tr = torch.as_tensor(train_mask, dtype=torch.bool, device=device)

    # Neighbour residual: self-excluded by construction (no self-loops in A).
    resid = torch.zeros(n, 1, device=device)
    resid[tr, 0] = yt[tr] - torch.sigmoid(m[tr])
    nbr = torch.sparse.mm(A, resid)
    seen = torch.sparse.mm(A, tr.float().unsqueeze(1))     # share of train nbrs

    feats = torch.cat([X, m.unsqueeze(1), nbr, seen], dim=1)
    model = ResidualGNN(feats.shape[1], hidden, layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss()

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = m + model(feats, A)
        loss = lossf(out[tr], yt[tr])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(m + model(feats, A)).cpu().numpy()
        a = roc_auc_score(y[val_mask], p[val_mask])
        if a > best:
            best, bad = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f"    ep {ep:3d} loss {loss.item():.4f} val auroc {a:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        delta = model(feats, A)
    return (m + delta).cpu().numpy().astype(np.float64), float(best), model
