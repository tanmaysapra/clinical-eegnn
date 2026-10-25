"""Graph v2: label-aware patient-graph models (notebooks 12-13).

Shared by the Kaggle notebooks via `scripts/nbmake.py`'s `#%% embed` marker.
Everything here is torch/numpy only (no xgboost), device-agnostic.

Pieces
------
- `prep_views`       per-view preprocessing (iqr / pct / quantile) + optional
                     per-feature missingness indicators, fit on train rows only.
- `Encoder`, `EncoderOnly`
                     multi-view MLP encoder -> h; graph-free exit.
- `Retrieval`        TabR-style label-aware retrieval (G1): attends over the m
                     nearest pool patients (self excluded) in a learned key space.
- `fit_minibatch`, `predict_rows`
                     generic mini-batch trainer (AdamW, early stopping on a
                     caller-supplied validation AUROC) + batched inference.
- `crossfit`         K-fold cross-fitting: out-of-fold logits for every training
                     row, fold-averaged logits for held-out rows.
- `cs_features`      leak-free Correct & Smooth features over a kNN graph whose
                     nodes are training patients: 1-hop / 2-hop neighbour OOF
                     residuals and labels, with the query's own label excluded
                     on every path (self-edges removed, 2-hop self-returns
                     subtracted).

Leakage invariants (tested in tests/test_graph_v2.py):
  * a training row's OOF logit never comes from a model that saw its label;
  * a training row's C&S features never include its own residual/label;
  * held-out rows only ever read training rows' labels/residuals.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer


def auroc(y, p):
    return float(roc_auc_score(y, p))


# ------------------------------------------------------------ preprocessing

def prep_views(X, views, tr, kind="iqr", indicators=True, return_cont=False):
    """-> (Xp [n, D] float32, slices {view: (start, end)}). Stats fit on `tr` rows.
    With return_cont=True also returns {view: bool mask over that view's
    columns} marking continuous (non-binary, non-indicator) columns -- the
    ones `Encoder(plr=...)` embeds."""
    blocks, slices, start, conts = [], {}, 0, {}
    for v, idx in views.items():
        x = X[:, idx].astype(np.float64)
        miss = np.isnan(x)
        allmiss = miss[tr].all(0)
        x, miss = x[:, ~allmiss], miss[:, ~allmiss]
        obs_tr = np.where(miss[tr], np.nan, x[tr])
        binary = np.array([np.isin(c[~np.isnan(c)], (0.0, 1.0)).all() for c in obs_tr.T])
        med = np.nan_to_num(np.nanmedian(obs_tr, 0))
        z = np.zeros_like(x)
        cont = ~binary
        if kind == "quantile":
            qt = QuantileTransformer(n_quantiles=256, output_distribution="normal",
                                     subsample=10**9, random_state=0)
            qt.fit(obs_tr[:, cont])
            z[:, cont] = np.nan_to_num(qt.transform(x[:, cont]), nan=0.0)
        else:
            q1, q3 = np.nanpercentile(obs_tr, 25, 0), np.nanpercentile(obs_tr, 75, 0)
            iqr = np.where(np.isnan(q3 - q1) | (q3 - q1 < 1e-6), 1.0, q3 - q1)
            if kind == "iqr":
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:  # pct
                lo, hi = np.nanpercentile(obs_tr, 0.5, 0), np.nanpercentile(obs_tr, 99.5, 0)
            xc = np.clip(np.where(miss, med, x), lo, hi)
            z[:, cont] = ((xc - med) / iqr)[:, cont]
        xb = np.where(miss, med, x)[:, binary]
        z[:, binary] = xb - xb[tr].mean(0)
        parts = [z]
        if indicators:
            keep = miss[tr].any(0)
            parts += [miss[:, keep].astype(np.float64), 1.0 - miss.mean(1, keepdims=True)]
        zz = np.concatenate(parts, 1).astype(np.float32)
        conts[v] = np.concatenate([cont, np.zeros(zz.shape[1] - len(cont), dtype=bool)])
        slices[v] = (start, start + zz.shape[1])
        start += zz.shape[1]
        blocks.append(zz)
    if return_cont:
        return np.concatenate(blocks, 1), slices, conts
    return np.concatenate(blocks, 1), slices


# ------------------------------------------------------------------ models

class PLR(nn.Module):
    """Periodic -> Linear -> ReLU numeric embedding (Gorishniy et al. 2022),
    "lite" variant: per-feature learnable frequencies, one Linear shared by
    all features. [B, n] -> [B, n * e]."""

    def __init__(s, n_feat, n_freq=16, sigma=0.1, e=8):
        super().__init__()
        s.c = nn.Parameter(torch.randn(n_feat, n_freq) * sigma)
        s.lin = nn.Linear(2 * n_freq, e)

    def forward(s, x):
        v = 2 * math.pi * s.c[None] * x[..., None]
        return F.relu(s.lin(torch.cat([torch.cos(v), torch.sin(v)], -1))).flatten(1)


class ViewEnc(nn.Module):
    def __init__(s, i, o, hid, p):
        super().__init__()
        s.fc1, s.fc2 = nn.Linear(i, hid), nn.Linear(hid, o)
        s.drop, s.ln = nn.Dropout(p), nn.LayerNorm(o)

    def forward(s, x):
        return s.ln(s.fc2(s.drop(F.gelu(s.fc1(x)))))


class Encoder(nn.Module):
    """Multi-view encoder: one MLP per view, residual fusion -> h [n, d]."""

    def __init__(s, slices, d=128, hid=128, vout=32, p=0.2, cont=None, plr=None):
        super().__init__()
        s.slices = dict(slices)
        s.plr = nn.ModuleDict()
        s.cont_idx, s.rest_idx = {}, {}
        in_dims = {}
        for v, (a, b) in s.slices.items():
            if plr and cont is not None and cont[v].any():
                ci = torch.as_tensor(np.flatnonzero(cont[v]) + a)
                ri = torch.as_tensor(np.flatnonzero(~cont[v]) + a)
                s.register_buffer(f"ci_{v}", ci); s.register_buffer(f"ri_{v}", ri)
                s.plr[v] = PLR(len(ci), **plr)
                in_dims[v] = len(ci) * plr.get("e", 8) + len(ri)
            else:
                in_dims[v] = b - a
        s.enc = nn.ModuleDict({v: ViewEnc(in_dims[v], vout, hid, p) for v in s.slices})
        cat = vout * len(s.slices)
        s.skip = nn.Linear(cat, d)
        s.fuse = nn.Sequential(nn.Linear(cat, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))
        s.ln = nn.LayerNorm(d)

    def _view_in(s, x, v, a, b):
        if v not in s.plr:
            return x[:, a:b]
        ci, ri = getattr(s, f"ci_{v}"), getattr(s, f"ri_{v}")
        return torch.cat([s.plr[v](x[:, ci]), x[:, ri]], 1)

    def forward(s, x):
        z = torch.cat([s.enc[v](s._view_in(x, v, a, b)) for v, (a, b) in s.slices.items()], 1)
        return s.ln(s.skip(z) + s.fuse(z))


def head(i, p):
    return nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Dropout(p), nn.Linear(i, 1))


class EncoderOnly(nn.Module):
    def __init__(s, slices, d=128, p=0.2, hid=128, cont=None, plr=None):
        super().__init__()
        s.enc, s.h0 = Encoder(slices, d=d, hid=hid, p=p, cont=cont, plr=plr), head(d, p)

    def forward(s, x):
        return [s.h0(s.enc(x)).squeeze(-1)]


class Retrieval(nn.Module):
    """G1. For a query batch: keys of the whole pool are computed without
    gradient, the m nearest (L2) are picked -- the query itself excluded when
    it is a pool member (`q_pool_pos`) -- and only those are re-encoded with
    gradient. Neighbour value = LabelEmb(y_j) + T(k_q - k_j), softmax(-dist)
    weighted. Exit 0 = graph-free head on h, exit 1 = after retrieval."""

    def __init__(s, slices, d=128, m=32, p=0.2, ctx_drop=0.2, hid=128, cont=None, plr=None):
        super().__init__()
        s.enc, s.m = Encoder(slices, d=d, hid=hid, p=p, cont=cont, plr=plr), m
        s.K = nn.Linear(d, d)
        s.lab = nn.Embedding(2, d)
        s.T = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d, bias=False))
        s.block = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(p),
                                nn.Linear(2 * d, d))
        s.ctx_drop = nn.Dropout(ctx_drop)
        s.h0, s.h1 = head(d, p), head(d, p)

    def forward(s, xq, q_pool_pos, x_pool, y_pool):
        hq = s.enc(xq)
        kq = s.K(hq)
        with torch.no_grad():
            kp = torch.cat([s.K(s.enc(x_pool[i:i + 4096])) for i in range(0, len(x_pool), 4096)])
            dist = torch.cdist(kq.detach(), kp) ** 2
            if q_pool_pos is not None:
                own = q_pool_pos >= 0
                rows = torch.arange(len(xq), device=xq.device)[own]
                dist[rows, q_pool_pos[own]] = float("inf")
            nbr = dist.topk(min(s.m, len(x_pool) - 1), dim=1, largest=False).indices
        kn = s.K(s.enc(x_pool[nbr.reshape(-1)])).reshape(len(xq), nbr.shape[1], -1)
        w = s.ctx_drop(torch.softmax(-((kq[:, None, :] - kn) ** 2).sum(-1), dim=1))
        v = s.lab(y_pool[nbr]) + s.T(kq[:, None, :] - kn)
        h1 = hq + (w[..., None] * v).sum(1)
        h1 = h1 + s.block(h1)
        return [s.h0(hq).squeeze(-1), s.h1(h1).squeeze(-1)]


# ---------------------------------------------------------------- training

def fit_minibatch(model, forward_fn, tr_idx, y_t, val_fn, lr=1e-3, wd=1e-4, bs=512,
                  max_ep=300, patience=20, seed=0, restore_best=True):
    """`forward_fn(idx, train=True)` -> list of exit logits for rows idx;
    loss = mean BCE over exits. `val_fn()` -> validation AUROC of the LAST
    exit; best epoch restored (unless restore_best=False)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_ep, t0 = -1.0, None, 0, 0, time.time()
    ep = -1
    for ep in range(max_ep):
        model.train()
        perm = rng.permutation(tr_idx)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            if len(b) < 2:
                continue
            outs = forward_fn(b, train=True)
            loss = sum(F.binary_cross_entropy_with_logits(o, y_t[b]) for o in outs) / len(outs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        v = val_fn()
        if v > best:
            best, bad, best_ep = v, 0, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return {"val_auroc": best, "best_epoch": best_ep, "epochs_run": ep + 1,
            "train_s": time.time() - t0}


@torch.no_grad()
def predict_rows(forward_fn, model, idx, bs=2048):
    model.eval()
    outs = None
    for s in range(0, len(idx), bs):
        o = [t.float().cpu().numpy() for t in forward_fn(idx[s:s + bs], train=False)]
        outs = o if outs is None else [np.concatenate([a, b]) for a, b in zip(outs, o)]
    return outs


def stratified_folds(y_tr, n_folds, seed):
    """Fold id per training row (positions into tr_idx), class-stratified."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y_tr), dtype=int)
    for c in np.unique(y_tr):
        idx = rng.permutation(np.flatnonzero(y_tr == c))
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def crossfit(kind, Xt, y_np, tr_idx, eval_idx, slices, cfg, n_folds=5, seed=0, device="cpu",
             score_mask=None):
    """K-fold cross-fit one model family.

    kind: "encoder" (EncoderOnly) or "retrieval" (Retrieval; the pool is the
    fold's inner-train rows, so fold rows are scored exactly like held-out
    patients). Early stopping on the held-out fold. Returns dict with
      oof   [n_exits, len(tr_idx)]  out-of-fold logits for training rows,
      held  [n_exits, len(eval_idx)] fold-averaged logits for eval rows,
      folds, info (per-fold fit info), oof_auroc (last exit).
    `score_mask` (bool over all rows, optional): only these rows are used for
    early stopping and `oof_auroc` -- e.g. train on core + auxiliary patients
    but select on core patients, whose distribution matches the test set.
    """
    if score_mask is None:
        score_mask = np.ones(len(y_np), dtype=bool)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
    fold = stratified_folds(y_np[tr_idx], n_folds, seed)
    oof, held, infos = None, None, []
    for f in range(n_folds):
        torch.manual_seed(seed * 100 + f)
        inner, out = tr_idx[fold != f], tr_idx[fold == f]
        mk = dict(d=cfg.get("d", 128), p=cfg.get("p", 0.2), hid=cfg.get("hid", 128),
                  cont=cfg.get("cont"), plr=cfg.get("plr"))
        if kind == "encoder":
            model = EncoderOnly(slices, **mk).to(device)
            if cfg.get("init_enc") is not None:
                model.enc.load_state_dict(cfg["init_enc"])
            fwd = lambda b, train=True: model(Xt[b])
            bs = cfg.get("bs", 512)
        else:
            model = Retrieval(slices, m=cfg.get("m", 32), **mk).to(device)
            if cfg.get("init_enc") is not None:
                model.enc.load_state_dict(cfg["init_enc"])
            pool_pos = torch.full((len(y_np),), -1, dtype=torch.long, device=device)
            pool_pos[torch.as_tensor(inner, device=device)] = torch.arange(len(inner), device=device)
            x_pool, y_pool = Xt[inner], y_t[inner].long()
            fwd = (lambda b, train=True:
                   model(Xt[b], pool_pos[torch.as_tensor(b, device=device)] if train else None,
                         x_pool, y_pool))
            bs = cfg.get("bs", 256)
        out_s = out[score_mask[out]]
        vf = lambda: auroc(y_np[out_s], predict_rows(fwd, model, out_s, bs=1024)[-1])
        info = fit_minibatch(model, fwd, inner, y_t, vf, lr=cfg.get("lr", 1e-3),
                             wd=cfg.get("wd", 1e-4), bs=bs, patience=cfg.get("patience", 12),
                             max_ep=cfg.get("max_ep", 200), seed=seed * 100 + f)
        po = np.stack(predict_rows(fwd, model, out, bs=1024))
        ph = np.stack(predict_rows(fwd, model, eval_idx, bs=1024))
        if oof is None:
            oof = np.zeros((po.shape[0], len(tr_idx)))
            held = np.zeros((ph.shape[0], len(eval_idx)))
        oof[:, fold == f] = po
        held += ph / n_folds
        infos.append(info)
        del model
    return {"oof": oof, "held": held, "folds": fold, "info": infos,
            "oof_auroc": auroc(y_np[tr_idx][score_mask[tr_idx]], oof[-1][score_mask[tr_idx]])}


# ------------------------------------------- self-supervised pretraining

def pretrain_encoder(slices, Xt, rows, cfg, epochs=40, mask_p=0.3, bs=512, lr=1e-3, seed=0,
                     device="cpu"):
    """Masked-feature reconstruction (DuETT-style value+presence masking,
    tabular form): a random `mask_p` of every row's columns -- values AND
    missingness indicators -- is zeroed, and a linear decoder on h must
    reconstruct the original row (MSE on masked entries). Uses FEATURES of
    `rows` only, never labels. Returns the Encoder state_dict."""
    torch.manual_seed(seed)
    mk = dict(d=cfg.get("d", 128), p=cfg.get("p", 0.2), hid=cfg.get("hid", 128),
              cont=cfg.get("cont"), plr=cfg.get("plr"))
    enc = Encoder(slices, **mk).to(device)
    dec = nn.Linear(mk["d"], Xt.shape[1]).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for ep in range(epochs):
        enc.train(); tot, nb = 0.0, 0
        perm = rng.permutation(rows)
        for s in range(0, len(perm), bs):
            x = Xt[perm[s:s + bs]]
            m = (torch.rand(x.shape, generator=g, device=device) < mask_p).float()
            rec = dec(enc(x * (1 - m)))
            loss = (((rec - x) ** 2) * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        losses.append(tot / max(nb, 1))
    return {k: v.detach().clone() for k, v in enc.state_dict().items()}, losses


# ------------------------------------------------------ Correct & Smooth

def knn_train_graph(Z, tr_idx, q_idx, k, device="cpu", chunk=4096):
    """Cosine kNN: every query row -> its k nearest TRAINING rows. When a
    query is itself a training row, it is excluded from its own neighbours.
    Returns (nbr [len(q_idx), k] as positions into tr_idx, sim [.., k])."""
    Zt = F.normalize(torch.as_tensor(Z, device=device, dtype=torch.float32), dim=1)
    pool = Zt[torch.as_tensor(tr_idx, device=device)]
    pos = {int(t): i for i, t in enumerate(tr_idx)}
    nbrs, sims = [], []
    for s in range(0, len(q_idx), chunk):
        qi = q_idx[s:s + chunk]
        sim = Zt[torch.as_tensor(qi, device=device)] @ pool.T
        own = [(r, pos[int(q)]) for r, q in enumerate(qi) if int(q) in pos]
        if own:
            r, c = zip(*own)
            sim[torch.as_tensor(r, device=device), torch.as_tensor(c, device=device)] = -2.0
        v, i = sim.topk(k, dim=1)
        nbrs.append(i.cpu().numpy())
        sims.append(v.cpu().numpy())
    return np.concatenate(nbrs), np.concatenate(sims)


def cs_features(nbr_q, sim_q, nbr_tr, sim_tr, q_pos_in_tr, resid_tr, y_tr, tau=0.05):
    """Leak-free C&S features for query rows.

    nbr_q/sim_q:   the queries' k nearest training rows (self already excluded).
    nbr_tr/sim_tr: every training row's k nearest training rows (self excluded).
    q_pos_in_tr:   for each query, its own position in tr_idx, or -1 if held-out.
    resid_tr:      OOF residual y - p for every training row; y_tr: labels.

    Features: [r1, y1, r2, y2] = softmax(sim/tau)-weighted 1-hop and 2-hop
    means of neighbour residuals / labels. 2-hop paths q -> j -> q are removed
    (and the remaining weight renormalised), so a training query never reads
    its own residual or label through a cycle.
    """
    def w_of(sim):
        w = np.exp((sim - sim.max(1, keepdims=True)) / tau)
        return w / w.sum(1, keepdims=True)

    wq, wt = w_of(sim_q), w_of(sim_tr)
    r1 = (wq * resid_tr[nbr_q]).sum(1)
    y1 = (wq * y_tr[nbr_q]).sum(1)
    # 2-hop: weights W2[q, j, l] = wq[q, j] * wt[j, l] over l = nbr_tr[j]
    nb2 = nbr_tr[nbr_q]                                   # [nq, k, k]
    w2 = wq[:, :, None] * wt[nbr_q]                       # [nq, k, k]
    self_ret = (nb2 == q_pos_in_tr[:, None, None]) & (q_pos_in_tr[:, None, None] >= 0)
    w2 = np.where(self_ret, 0.0, w2)
    w2 = w2 / np.clip(w2.sum((1, 2), keepdims=True), 1e-12, None)
    r2 = (w2 * resid_tr[nb2]).sum((1, 2))
    y2 = (w2 * y_tr[nb2]).sum((1, 2))
    return np.stack([r1, y1, r2, y2], 1)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
