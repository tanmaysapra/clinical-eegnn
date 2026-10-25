#%% md
# 12 · Graph v2 experiments — label-aware patient graphs vs. XGBoost

Self-contained: every repo module this needs is written out by the
`%%writefile` cells below (built from the repo by `scripts/nbmake.py`'s
`#%% embed` marker), and the MIMIC-IV mirror is fetched with `kagglehub`, so
this runs with **no secrets and no manual dataset attach**.

What one run does, for every seed in `SEEDS` (same 60/20/20 split and same
model_val/gate_selection/gate_certification cut as notebooks 10/11):

1. **XGBoost references** on the same split -- the phase-1 plain config, and
   a tuned native-NaN config early-stopped on `model_val`.
2. **Corrected Arch4R `encoder_only` baseline** -- the phase-1 arm trained
   one full-batch step per epoch (5 steps total at 5 epochs); here it is
   mini-batched, i.e. what the "graph signal" criterion should have been
   compared against.
3. **Encoder preprocessing ablation** -- the Arch4R 1.5-IQR winsorization vs.
   0.5/99.5-percentile clipping vs. a train-fit quantile transform, each with
   and without per-feature missingness indicators.
4. **Label-homophily diagnostics** of a feature kNN patient graph.
5. **G1 -- TabR-style retrieval** over a learned metric: each patient
   attends over its m nearest *training* patients (self excluded) and reads
   their labels + key differences. Exit 0 = graph-free, exit 1 = retrieved.
6. **G2 -- masked-label PPR diffusion** (Arch4R lineage): a learned kNN graph
   over train patients, APPNP-style personalized-PageRank diffusion of both
   embeddings and *masked* training labels (label trick: a scored patient's
   own label is never an input), one exit per diffusion checkpoint.

Model selection uses `model_val` only; test is read once per config, for
reporting. Results -> `/kaggle/working/results_v2/`.

#%% code
import os, sys, json, time, math, glob, subprocess
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/src", exist_ok=True)
open(f"{WORK}/src/__init__.py", "w").close()
sys.path.insert(0, WORK)
OUT = f"{WORK}/results_v2"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42]
print(f"[v2] device={DEV} torch={torch.__version__}", flush=True)

#%% embed src/mimic_concepts.py
#%% embed src/mimic_iv.py
#%% embed src/instrument.py
#%% embed src/encoder_mv.py
#%% embed src/graph_density.py
#%% embed src/local_push.py
#%% embed src/arch4r.py

#%% md
## Splits -- verbatim copies of `data_prep.make_splits` and `cascade.split_validation_3way`

#%% code
def make_splits(y, seed, fracs=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    n = len(y)
    train, val, test = (np.zeros(n, dtype=bool) for _ in range(3))
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_tr = int(round(fracs[0] * len(idx))); n_va = int(round(fracs[1] * len(idx)))
        train[idx[:n_tr]] = True; val[idx[n_tr:n_tr + n_va]] = True; test[idx[n_tr + n_va:]] = True
    return train, val, test


def split_validation_3way(val_mask, y, seed, fracs=(0.4, 0.3, 0.3)):
    rng = np.random.default_rng(seed + 992)
    parts = [np.zeros_like(val_mask) for _ in range(3)]
    idx = np.flatnonzero(val_mask)
    for cls in (0, 1):
        c = rng.permutation(idx[y[idx] == cls])
        cut1 = int(round(fracs[0] * len(c))); cut2 = cut1 + int(round(fracs[1] * len(c)))
        parts[0][c[:cut1]] = True; parts[1][c[cut1:cut2]] = True; parts[2][c[cut2:]] = True
    return parts


def auroc(y, p):
    return float(roc_auc_score(y, p))


def auprc(y, p):
    return float(average_precision_score(y, p))


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

#%% md
## Preprocessing variants

`iqr` is exactly Arch4R's `encoder_mv.fit_view_preprocessors` (median impute,
clip to Q1-1.5·IQR / Q3+1.5·IQR, scale by IQR) -- which clips away the lab/
vital extremes (e.g. `lab_lactate_max`) that carry most of the mortality
signal. `pct` clips at the 0.5/99.5 train percentiles instead; `quantile`
maps each continuous column through a train-fit quantile transform to a
standard normal. Binary columns are left as 0/1 in every variant. With
`indicators=True`, one 0/1 missingness column is appended per raw column that
is ever missing in train, plus each view's observed fraction.

#%% code
from sklearn.preprocessing import QuantileTransformer


def prep_views(X, views, tr, kind, indicators):
    blocks, slices, start = [], {}, 0
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
            zc = qt.transform(x[:, cont])
            z[:, cont] = np.nan_to_num(zc, nan=0.0)
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
        slices[v] = (start, start + zz.shape[1]); start += zz.shape[1]
        blocks.append(zz)
    return np.concatenate(blocks, 1), slices

#%% md
## Shared model pieces + training loop (AdamW, early stopping on model_val AUROC)

#%% code
class ViewEnc(nn.Module):
    def __init__(s, i, o, hid, p):
        super().__init__()
        s.fc1, s.fc2, s.drop, s.ln = nn.Linear(i, hid), nn.Linear(hid, o), nn.Dropout(p), nn.LayerNorm(o)

    def forward(s, x):
        return s.ln(s.fc2(s.drop(F.gelu(s.fc1(x)))))


class Encoder(nn.Module):
    """Multi-view encoder: one MLP per view, residual fusion -> h [n, d]."""

    def __init__(s, slices, d=128, hid=128, vout=32, p=0.2):
        super().__init__()
        s.slices = slices
        s.enc = nn.ModuleDict({v: ViewEnc(b - a, vout, hid, p) for v, (a, b) in slices.items()})
        cat = vout * len(slices)
        s.skip = nn.Linear(cat, d)
        s.fuse = nn.Sequential(nn.Linear(cat, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))
        s.ln = nn.LayerNorm(d)

    def forward(s, x):
        z = torch.cat([s.enc[v](x[:, a:b]) for v, (a, b) in s.slices.items()], 1)
        return s.ln(s.skip(z) + s.fuse(z))


def head(i, p):
    return nn.Sequential(nn.Linear(i, i), nn.GELU(), nn.Dropout(p), nn.Linear(i, 1))


class EncoderOnly(nn.Module):
    def __init__(s, slices, d=128, p=0.2):
        super().__init__()
        s.enc, s.h0 = Encoder(slices, d=d, p=p), head(d, p)

    def forward(s, x):
        return [s.h0(s.enc(x)).squeeze(-1)]


def fit_minibatch(model, forward_fn, tr_idx, y_t, val_fn, lr=1e-3, wd=1e-4, bs=512,
                  max_ep=300, patience=20, seed=0, restore_best=True):
    """Generic mini-batch loop. `forward_fn(idx, train=True)` -> list of exit
    logits for rows idx; loss = mean BCE over exits (deep supervision, as in
    arch4r_loss). `val_fn()` -> list of per-exit model_val AUROCs; the best
    epoch by the LAST exit's val AUROC is restored."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_ep, t0 = -1.0, None, 0, 0, time.time()
    for ep in range(max_ep):
        model.train()
        perm = rng.permutation(tr_idx)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            outs = forward_fn(b, train=True)
            loss = sum(F.binary_cross_entropy_with_logits(o, y_t[b]) for o in outs) / len(outs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        v = val_fn()[-1]
        if v > best:
            best, bad, best_ep = v, 0, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if restore_best:
        model.load_state_dict(best_state)
    return {"val_auroc": best, "best_epoch": best_ep, "epochs_run": ep + 1,
            "restored_best": restore_best, "train_s": time.time() - t0}


@torch.no_grad()
def predict_rows(forward_fn, model, idx, bs=2048):
    model.eval()
    outs = None
    for s in range(0, len(idx), bs):
        o = [t.float().cpu().numpy() for t in forward_fn(idx[s:s + bs], train=False)]
        outs = o if outs is None else [np.concatenate([a, b]) for a, b in zip(outs, o)]
    return outs

#%% md
## G1 — TabR-style label-aware retrieval over a learned metric

For a query batch, keys of ALL training patients are computed without
gradient, the m nearest (L2 in key space) are selected -- **the query itself
is excluded** when it is a training patient -- and only the selected
neighbours are re-encoded with gradient. Each neighbour contributes
`LabelEmb(y_j) + T(k_q - k_j)` weighted by softmax(-distance); held-out
queries retrieve from the training pool only, so no validation/test label is
ever an input.

#%% code
class Retrieval(nn.Module):
    def __init__(s, slices, d=128, m=96, p=0.2, ctx_drop=0.2):
        super().__init__()
        s.enc, s.m = Encoder(slices, d=d, p=p), m
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
            if q_pool_pos is not None:  # training queries: never retrieve yourself
                dist[torch.arange(len(xq), device=xq.device), q_pool_pos] = float("inf")
            nbr = dist.topk(s.m, dim=1, largest=False).indices          # [B, m]
        kn = s.K(s.enc(x_pool[nbr.reshape(-1)])).reshape(len(xq), s.m, -1)
        w = s.ctx_drop(torch.softmax(-((kq[:, None, :] - kn) ** 2).sum(-1), dim=1))
        v = s.lab(y_pool[nbr]) + s.T(kq[:, None, :] - kn)
        h1 = hq + (w[..., None] * v).sum(1)
        h1 = h1 + s.block(h1)
        return [s.h0(hq).squeeze(-1), s.h1(h1).squeeze(-1)]

#%% md
## G2 — masked-label PPR diffusion on a learned kNN graph (Arch4R lineage)

Every node's out-edges go to its `kk` nearest *training* nodes (self
excluded) in a learned cosine key space, weighted by softmax(sim/τ) -- so
held-out nodes read from the training graph but no training node ever reads
a held-out node (inductive, no held-out-to-held-out edges, exactly Arch4R's
attach discipline). Each node carries `[V h, ỹ, vis]`, where `vis`=1 marks a
visible training label and `ỹ = y·vis`. APPNP diffusion
`Z ← (1-α)·A·Z + α·Z0`; at checkpoints t ∈ `steps`, exit s reads
`[h, Z_emb, Z_ỹ/Z_vis, Z_vis]`. Training uses the label trick: each step a
random `keep` fraction of training labels is visible and the loss is taken
ONLY on the hidden ones -- a patient's own label never enters its own score.
At evaluation every training label is visible; held-out labels never are.

#%% code
class LabelPPR(nn.Module):
    def __init__(s, slices, d=128, kk=16, alpha=0.15, steps=(1, 2, 4, 8, 16), p=0.2, kdim=32):
        super().__init__()
        s.enc, s.kk, s.alpha, s.steps = Encoder(slices, d=d, p=p), kk, alpha, tuple(steps)
        s.K, s.V = nn.Linear(d, kdim), nn.Linear(d, d)
        s.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))
        s.h0 = head(d, p)
        s.hs = nn.ModuleList([head(2 * d + 2, p) for _ in steps])

    def forward(s, x, n_pool, y_vis, vis):
        """x: [N, D] with rows [0, n_pool) = training pool. y_vis/vis: [N]
        (zero outside visible training rows). Returns exit logits for all N."""
        h = s.enc(x)
        k = F.normalize(s.K(h), dim=-1)
        N = len(x)
        with torch.no_grad():
            nbr = torch.empty(N, s.kk, dtype=torch.long, device=x.device)
            kp = k[:n_pool].detach()
            for i in range(0, N, 4096):
                sim = k[i:i + 4096].detach() @ kp.T
                r = torch.arange(i, min(i + 4096, N), device=x.device)
                own = r < n_pool
                sim[own.nonzero().squeeze(1), r[own]] = -float("inf")
                nbr[i:i + 4096] = sim.topk(s.kk, dim=1).indices
        A = torch.softmax((k[:, None, :] * k[nbr]).sum(-1) / s.log_tau.exp(), dim=1)  # [N, kk]
        Z0 = torch.cat([s.V(h), y_vis[:, None], vis[:, None]], 1)
        Z, outs, si = Z0, [s.h0(h).squeeze(-1)], 0
        for t in range(1, max(s.steps) + 1):
            Z = (1 - s.alpha) * (A[..., None] * Z[nbr]).sum(1) + s.alpha * Z0
            if t in s.steps:
                rate = Z[:, -2:-1] / (Z[:, -1:] + 1e-6)
                outs.append(s.hs[si](torch.cat([h, Z[:, :-2], rate, Z[:, -1:]], 1)).squeeze(-1))
                si += 1
        return outs


def fit_labelppr(model, Xt, y_t, tr_idx, mv_idx, keep=0.5, lr=1e-3, wd=1e-4,
                 max_steps=3000, eval_every=20, patience=25, seed=0):
    """Full-batch over the training graph; label-trick masking resampled every step."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device=Xt.device).manual_seed(seed)
    x_tr, y_tr = Xt[tr_idx], y_t[tr_idx]
    n = len(tr_idx)
    best, best_state, bad, best_step, t0 = -1.0, None, 0, 0, time.time()
    for step in range(1, max_steps + 1):
        model.train()
        vis = (torch.rand(n, generator=g, device=Xt.device) < keep).float()
        outs = model(x_tr, n, y_tr * vis, vis)
        hid = vis == 0
        loss = sum(F.binary_cross_entropy_with_logits(o[hid], y_tr[hid]) for o in outs) / len(outs)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % eval_every == 0:
            v = labelppr_predict(model, Xt, y_t, tr_idx, mv_idx)
            a = auroc(Y_NP[mv_idx], v[-1])
            if a > best:
                best, bad, best_step = a, 0, step
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best_state)
    return {"val_auroc": best, "best_step": best_step, "steps_run": step, "train_s": time.time() - t0}


@torch.no_grad()
def labelppr_predict(model, Xt, y_t, tr_idx, q_idx):
    model.eval()
    n = len(tr_idx)
    x = torch.cat([Xt[tr_idx], Xt[q_idx]])
    vis = torch.cat([torch.ones(n, device=Xt.device), torch.zeros(len(q_idx), device=Xt.device)])
    y_vis = torch.cat([y_t[tr_idx], torch.zeros(len(q_idx), device=Xt.device)])
    outs = model(x, n, y_vis, vis)
    return [o[n:].float().cpu().numpy() for o in outs]

#%% md
## Smoke test on synthetic data (fails fast, before the ~7-min cohort build)

#%% code
_sl = {"a": (0, 7), "b": (7, 12)}
_x = torch.randn(300, 12, device=DEV)
_y = (torch.rand(300, device=DEV) < 0.2).float()
_g1 = Retrieval(_sl, m=8).to(DEV)
_o = _g1(_x[:32], torch.arange(32, device=DEV), _x[:200], _y[:200].long())
sum(o.mean() for o in _o).backward()
_g2 = LabelPPR(_sl, kk=4).to(DEV)
_vis = (torch.rand(300, device=DEV) < 0.5).float()
_o = _g2(_x, 200, _y * _vis, _vis)
sum(o.mean() for o in _o).backward()
assert len(_o) == 6 and _o[-1].shape == (300,)
_Xs, _ss = prep_views(np.where(np.random.rand(300, 12) < 0.3, np.nan, np.random.randn(300, 12)),
                      {"a": np.arange(7), "b": np.arange(7, 12)}, np.arange(300) < 200, "quantile", True)
assert np.isfinite(_Xs).all()
print("[v2] smoke test OK", flush=True)

#%% md
## Cohort (same definition/signature as notebooks 09-11: expect `cohort_97de93de81`)

#%% code
def _find_mimic_root(start, max_depth=5):
    for dirpath, dirnames, _files in os.walk(start):
        if "hosp" in dirnames:
            base = os.path.basename(dirpath)
            return os.path.dirname(dirpath) if base in ("mimic-iv-3.1", "mimic-iv") else dirpath
        if dirpath[len(start):].count(os.sep) >= max_depth:
            dirnames[:] = []
    return None

RAW = _find_mimic_root("/kaggle/input") if os.path.isdir("/kaggle/input") else None
if RAW is None:
    import kagglehub
    _dl = kagglehub.dataset_download("akshaybe/updated-mimic-iv")
    RAW = _find_mimic_root(_dl, max_depth=3) or _dl
print("[v2] MIMIC root:", RAW, flush=True)

try:
    import duckdb  # noqa: F401  (mimic_iv falls back to a much slower pandas scan without it)
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb"], check=True)

from src import mimic_iv
CFG = {"seed": 42, "data": {
    "source": "mimic", "task": "mortality", "raw_dir": RAW,
    "cache_dir": f"{WORK}/processed", "obs_window_hours": 24, "min_los_hours": 28,
    "max_patients": None, "cohort_seed": 42, "force_rebuild": False,
    "blocks": ["demographics", "comorbidity", "labs", "vitals", "treatment"],
    "require_nonempty_blocks": ["vitals"]}}
print("[v2] cohort signature:", mimic_iv.cohort_signature(CFG), flush=True)
t0 = time.time()
X_RAW, Y, NAMES, META = mimic_iv.build_cohort(CFG)
VIEWS = {k: np.asarray(v) for k, v in META["views"].items()}
print(f"[v2] cohort n={len(Y)} features={X_RAW.shape[1]} prevalence={Y.mean():.4f} "
      f"views={ {k: len(v) for k, v in VIEWS.items()} } ({time.time()-t0:.0f}s)", flush=True)

#%% md
## Run everything, per seed

#%% code
from xgboost import XGBClassifier
from src import arch4r
from src.encoder_mv import fit_view_preprocessors, observed_fraction
from src.local_push import make_checkpoints

Y_NP = Y.astype(np.int64)
ALL = {}


def rep(name, row):
    print(f"[v2][seed {SEED}] {name:42s} " + " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    ALL.setdefault(str(SEED), {})[name] = row
    with open(f"{OUT}/results_v2.json", "w") as f:
        json.dump(ALL, f, indent=1)


def exit_report(outs, mv, te, extra=None):
    """Per-exit test AUROC; `best_exit` chosen on model_val, never on test."""
    va = [auroc(Y_NP[mv], o[0]) for o in outs]
    ta = [auroc(Y_NP[te], o[1]) for o in outs]
    b = int(np.argmax(va))
    row = {"test_auroc_last": ta[-1], "test_auroc_bestval": ta[b], "best_exit": b,
           "val_auroc_last": va[-1], "test_auprc_last": auprc(Y_NP[te], outs[-1][1]),
           "test_auroc_per_exit": [round(t, 4) for t in ta]}
    row.update(extra or {})
    return row


for SEED in SEEDS:
    set_seed(SEED)
    train, val, test = make_splits(Y_NP, SEED)
    mv, gsel, gcert = split_validation_3way(val, Y_NP, SEED)
    tr_idx, mv_idx, te_idx = np.flatnonzero(train), np.flatnonzero(mv), np.flatnonzero(test)
    print(f"[v2][seed {SEED}] train={len(tr_idx)} model_val={len(mv_idx)} test={len(te_idx)}", flush=True)
    y_t = torch.tensor(Y_NP, dtype=torch.float32, device=DEV)

    # ---- 1. XGBoost references ------------------------------------------------
    Xi = X_RAW.astype(np.float32).copy()
    miss = np.isnan(Xi)
    med = np.nan_to_num(np.nanmedian(Xi[train], 0))
    Xi = np.where(miss, med, Xi)
    mu, sd = Xi[train].mean(0), Xi[train].std(0)
    Xi = (Xi - mu) / np.where(sd < 1e-6, 1.0, sd)
    Xi = np.concatenate([Xi, miss[:, miss[train].any(0)].astype(np.float32)], 1)
    pos = (Y_NP[train] == 1).sum()
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=float((Y_NP[train] == 0).sum() / pos),
                        eval_metric="logloss", random_state=SEED, n_jobs=4, tree_method="hist",
                        device="cuda" if DEV.type == "cuda" else "cpu")
    clf.fit(Xi[train], Y_NP[train])
    rep("xgb_plain_phase1_config", {"test_auroc": auroc(Y_NP[test], clf.predict_proba(Xi[test])[:, 1]),
                                    "test_auprc": auprc(Y_NP[test], clf.predict_proba(Xi[test])[:, 1])})
    clf = XGBClassifier(n_estimators=4000, max_depth=5, learning_rate=0.02, subsample=0.8,
                        colsample_bytree=0.5, min_child_weight=2, reg_lambda=2.0,
                        eval_metric="auc", early_stopping_rounds=300, random_state=SEED, n_jobs=4,
                        tree_method="hist", device="cuda" if DEV.type == "cuda" else "cpu")
    clf.fit(X_RAW[train], Y_NP[train], eval_set=[(X_RAW[mv], Y_NP[mv])], verbose=False)
    p = clf.predict_proba(X_RAW[test])[:, 1]
    rep("xgb_tuned_nativeNaN_earlystop", {"test_auroc": auroc(Y_NP[test], p), "test_auprc": auprc(Y_NP[test], p),
                                          "best_iteration": int(clf.best_iteration)})

    # ---- 2. corrected Arch4R encoder_only baseline (original model + preprocessing)
    raw_views = {v: X_RAW[:, idx] for v, idx in VIEWS.items()}
    pre = fit_view_preprocessors(raw_views, train)
    vx = pre.transform(raw_views)
    vo = observed_fraction(raw_views)
    cfg_a = {"fusion_dim": 64, "stage_hidden": 32}
    for proto in ("phase1_fullbatch_5ep", "minibatch_30ep", "minibatch_earlystop"):
        set_seed(SEED)
        m4 = arch4r.Arch4RModel({v: a.shape[1] for v, a in vx.items()}, cfg_a, "encoder_only").to(DEV)
        ck = make_checkpoints()
        VX = {v: torch.tensor(a, device=DEV) for v, a in vx.items()}
        VO = {v: torch.tensor(a, dtype=torch.float32, device=DEV) for v, a in vo.items()}
        fwd4 = lambda b, train=True: m4({v: t[b] for v, t in VX.items()},
                                        {v: t[b] for v, t in VO.items()}, None, ck)["stage_logits"][-1:]
        val4 = lambda: [auroc(Y_NP[mv_idx], predict_rows(fwd4, m4, mv_idx)[-1])]
        if proto == "phase1_fullbatch_5ep":   # exactly what phase 1 did: 1 Adam step/epoch
            opt = torch.optim.Adam(m4.parameters(), lr=1e-3)
            for _ in range(5):
                m4.train(); out = fwd4(tr_idx)
                loss = F.binary_cross_entropy_with_logits(out[0], y_t[tr_idx])
                opt.zero_grad(); loss.backward(); opt.step()
            info = {}
        elif proto == "minibatch_30ep":
            # phase-1's own optimiser (Adam, no weight decay) and fixed 30-epoch
            # budget, LAST-epoch weights -- only the batching differs.
            m4_opt_wd = 0.0
            info = fit_minibatch(m4, fwd4, tr_idx, y_t, val4, wd=m4_opt_wd, max_ep=30,
                                 patience=10**9, seed=SEED, restore_best=False)
        else:
            info = fit_minibatch(m4, fwd4, tr_idx, y_t, val4, seed=SEED)
        pm = predict_rows(fwd4, m4, mv_idx)[-1]; pt = predict_rows(fwd4, m4, te_idx)[-1]
        rep(f"arch4r_encoder_only[{proto}]", {"test_auroc": auroc(Y_NP[te_idx], pt),
                                              "val_auroc": auroc(Y_NP[mv_idx], pm), **info})

    # ---- 3. encoder preprocessing ablation --------------------------------------
    abl = {}
    for kind in ("iqr", "pct", "quantile"):
        for ind in (False, True):
            set_seed(SEED)
            Xp, sl = prep_views(X_RAW, VIEWS, train, kind, ind)
            Xt = torch.tensor(Xp, device=DEV)
            m0 = EncoderOnly(sl).to(DEV)
            fwd = lambda b, train=True: m0(Xt[b])
            vf = lambda: [auroc(Y_NP[mv_idx], predict_rows(fwd, m0, mv_idx)[-1])]
            info = fit_minibatch(m0, fwd, tr_idx, y_t, vf, seed=SEED)
            outs = list(zip(predict_rows(fwd, m0, mv_idx), predict_rows(fwd, m0, te_idx)))
            abl[(kind, ind)] = info["val_auroc"]
            rep(f"encoder[{kind},ind={int(ind)}]", exit_report(outs, mv_idx, te_idx, info))
    BEST_PREP = max(abl, key=abl.get)
    print(f"[v2][seed {SEED}] best preprocessing on model_val: {BEST_PREP}", flush=True)
    Xp, SL = prep_views(X_RAW, VIEWS, train, *BEST_PREP)
    Xt = torch.tensor(Xp, device=DEV)

    # ---- 4. label-homophily diagnostics (train-train edges only) -----------------
    with torch.no_grad():
        Z = F.normalize(Xt, dim=1)
        ytr = y_t[tr_idx]
        prev = float(ytr.mean())
        for kk in (16, 64):
            nb = []
            for i in range(0, len(tr_idx), 4096):
                sim = Z[tr_idx[i:i + 4096]] @ Z[tr_idx].T
                sim[torch.arange(sim.shape[0], device=DEV), torch.arange(i, i + sim.shape[0], device=DEV)] = -2
                nb.append(sim.topk(kk, 1).indices)
            nb = torch.cat(nb)
            rate = ytr[nb].mean(1)
            same = (ytr[:, None] == ytr[nb]).float().mean()
            h0 = prev ** 2 + (1 - prev) ** 2
            m11 = float(ytr[nb][ytr == 1].mean()); m00 = float(1 - ytr[nb][ytr == 0].mean())
            sim_mv = Z[mv_idx] @ Z[tr_idx].T
            rate_mv = ytr[sim_mv.topk(kk, 1).indices].mean(1).cpu().numpy()
            rep(f"homophily[feature_cosine_knn,k={kk}]", {
                "edge_h": float(same), "adj_h": float((same - h0) / (1 - h0)), "M11": m11, "M00": m00,
                "loo_label_rate_auroc_train": auroc(Y_NP[tr_idx], rate.cpu().numpy()),
                "label_rate_auroc_model_val": auroc(Y_NP[mv_idx], rate_mv)})

    # ---- 5. G1 retrieval ---------------------------------------------------------
    pool_pos = torch.full((len(Y_NP),), -1, dtype=torch.long, device=DEV)
    pool_pos[torch.as_tensor(tr_idx, device=DEV)] = torch.arange(len(tr_idx), device=DEV)
    x_pool, y_pool = Xt[tr_idx], y_t[tr_idx].long()
    for m in (32, 96):
        set_seed(SEED)
        g1 = Retrieval(SL, m=m).to(DEV)
        fwd = lambda b, train=True: g1(Xt[b], pool_pos[torch.as_tensor(b, device=DEV)] if train else None,
                                       x_pool, y_pool)
        vf = lambda: [auroc(Y_NP[mv_idx], predict_rows(fwd, g1, mv_idx, bs=1024)[-1])]
        info = fit_minibatch(g1, fwd, tr_idx, y_t, vf, lr=3e-4, wd=1e-5, bs=256, patience=16, seed=SEED)
        outs = list(zip(predict_rows(fwd, g1, mv_idx, bs=1024), predict_rows(fwd, g1, te_idx, bs=1024)))
        rep(f"G1_retrieval[m={m}]", exit_report(outs, mv_idx, te_idx, info))

    # ---- 6. G2 masked-label PPR ---------------------------------------------------
    for keep in (0.5, 0.75):
        set_seed(SEED)
        g2 = LabelPPR(SL).to(DEV)
        info = fit_labelppr(g2, Xt, y_t, tr_idx, mv_idx, keep=keep, seed=SEED)
        outs = list(zip(labelppr_predict(g2, Xt, y_t, tr_idx, mv_idx),
                        labelppr_predict(g2, Xt, y_t, tr_idx, te_idx)))
        rep(f"G2_labelPPR[keep={keep}]", exit_report(outs, mv_idx, te_idx,
                                                     info | {"tau": float(g2.log_tau.exp())}))

print("[v2] DONE", flush=True)
print(json.dumps(ALL, indent=1))
