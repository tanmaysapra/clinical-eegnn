#%% md
# 13 · Cross-fitted encoder / retrieval ensembles + leak-free Correct & Smooth

Round 2 after notebook 12 (seed 42): every neural model peaked at epoch 2-3
and was selected on only 1,413 `model_val` patients (~152 deaths, AUROC SE
~±0.012), and the end-to-end graph exits added ~0 over the encoder. This
notebook attacks both problems:

1. **5-fold cross-fitting on the training split** -- early stopping and
   configuration choice use ~10.6k out-of-fold (OOF) rows instead of 1.4k,
   and the 5 fold models (x bags) form an ensemble for held-out patients.
2. **Correct & Smooth, leak-free** (RESEARCH.md prompt 3): over a cosine kNN
   graph of training patients, each patient reads its neighbours' OOF
   residuals and labels (1-hop and 2-hop). Its own label/residual is never on
   any path (self-edges dropped, 2-hop self-returns removed -- asserted by
   `tests/test_graph_v2.py`, run below). A small logistic layer fit on
   training OOF rows combines base logit + graph features.
   Exit 0 = base model, exit 1 = + graph correction.

Selection is on training OOF AUROC only; `model_val` and test are reported,
never used to choose. Seeds 42-44, same splits as notebooks 10-12. No XGBoost
is used in any Arch model -- the tuned XGBoost row is the reference to beat.

#%% code
import os, sys, json, time, glob, shutil, subprocess
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

WORK = "/kaggle/working"
for d in ("src", "tests", "processed", "results_v2"):
    os.makedirs(f"{WORK}/{d}", exist_ok=True)
open(f"{WORK}/src/__init__.py", "w").close()
sys.path.insert(0, WORK)
OUT = f"{WORK}/results_v2"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44]
print(f"[v2] device={DEV} torch={torch.__version__}", flush=True)

#%% embed src/mimic_concepts.py
#%% embed src/mimic_iv.py
#%% embed src/graph_v2.py
#%% embed tests/test_graph_v2.py

#%% code
r = subprocess.run([sys.executable, "-m", "pytest", "-q", f"{WORK}/tests/test_graph_v2.py"],
                   cwd=WORK, capture_output=True, text=True,
                   env={**os.environ, "PYTHONPATH": WORK})
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0, "leakage tests failed -- refusing to run"
print("[v2] leakage tests passed", flush=True)

#%% md
## Cohort: reuse notebook 12's output if attached, else rebuild (~7 min)

#%% code
hits = glob.glob("/kaggle/input/**/cohort_97de93de81.csv.gz", recursive=True)
if hits:
    src_dir = os.path.dirname(hits[0])
    for f in ("cohort_97de93de81.csv.gz", "cohort_97de93de81.views.json"):
        shutil.copy(os.path.join(src_dir, f), f"{WORK}/processed/{f}")
    print("[v2] reusing cohort from", src_dir, flush=True)
    RAW = "/kaggle/input"
else:
    def _find(start, max_depth=5):
        for dp, dn, _ in os.walk(start):
            if "hosp" in dn:
                b = os.path.basename(dp)
                return os.path.dirname(dp) if b in ("mimic-iv-3.1", "mimic-iv") else dp
            if dp[len(start):].count(os.sep) >= max_depth:
                dn[:] = []
        return None
    RAW = _find("/kaggle/input")
    if RAW is None:
        import kagglehub
        _dl = kagglehub.dataset_download("akshaybe/updated-mimic-iv")
        RAW = _find(_dl, 3) or _dl

from src import mimic_iv
CFG = {"seed": 42, "data": {
    "source": "mimic", "task": "mortality", "raw_dir": RAW, "cache_dir": f"{WORK}/processed",
    "obs_window_hours": 24, "min_los_hours": 28, "max_patients": None, "cohort_seed": 42,
    "force_rebuild": False, "blocks": ["demographics", "comorbidity", "labs", "vitals", "treatment"],
    "require_nonempty_blocks": ["vitals"]}}
assert mimic_iv.cohort_signature(CFG) == "97de93de81"
if hits:  # cache hit path: build_cohort only needs find_root to succeed
    mimic_iv.find_root = lambda base="data/raw": base
X_RAW, Y, NAMES, META = mimic_iv.build_cohort(CFG)
VIEWS = {k: np.asarray(v) for k, v in META["views"].items()}
Y = Y.astype(np.int64)
print(f"[v2] cohort n={len(Y)} features={X_RAW.shape[1]} prevalence={Y.mean():.4f}", flush=True)

#%% md
## Splits (verbatim `data_prep.make_splits` / `cascade.split_validation_3way`)

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

#%% md
## Run

#%% code
from xgboost import XGBClassifier
from src import graph_v2 as g

ALL = {}
ENC_CFGS = {
    "A_lr1e-3_wd1e-4_p0.2": dict(lr=1e-3, wd=1e-4, p=0.2, d=128, hid=128),
    "B_lr3e-4_wd1e-3_p0.3": dict(lr=3e-4, wd=1e-3, p=0.3, d=128, hid=128),
    "C_lr3e-4_wd1e-2_p0.4_d256": dict(lr=3e-4, wd=1e-2, p=0.4, d=256, hid=256),
}
N_BAGS_ENC, N_BAGS_RET = 3, 2


def rep(seed, name, row):
    print(f"[v2][seed {seed}] {name:44s} " + " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    ALL.setdefault(str(seed), {})[name] = row
    with open(f"{OUT}/results_v2_round2.json", "w") as f:
        json.dump(ALL, f, indent=1)


def scores(z_held, mv_pos, te_pos, y_mv, y_te):
    return {"model_val_auroc": g.auroc(y_mv, z_held[mv_pos]), "test_auroc": g.auroc(y_te, z_held[te_pos]),
            "test_auprc": float(average_precision_score(y_te, z_held[te_pos]))}


def bag(kind, cfg, n_bags, Xt, tr_idx, ev_idx, SL, seed):
    """Average logits over bags (each bag = a full 5-fold cross-fit, new folds + inits)."""
    oofs, helds, infos = [], [], []
    for b in range(n_bags):
        cf = g.crossfit(kind, Xt, Y, tr_idx, ev_idx, SL, cfg, n_folds=5,
                        seed=seed * 10 + b, device=DEV)
        oofs.append(cf["oof"][-1]); helds.append(cf["held"][-1])
        infos.append([round(i["best_epoch"], 1) for i in cf["info"]])
    return np.mean(oofs, 0), np.mean(helds, 0), infos


def cs_layer(base_oof, base_held, Z, tr_idx, ev_idx, ks=(16, 64)):
    """Leak-free C&S features (self excluded) + logistic combiner fit on train OOF rows."""
    y_tr = Y[tr_idx].astype(float)
    resid = y_tr - g.sigmoid(base_oof)
    F_tr, F_ev = [base_oof[:, None]], [base_held[:, None]]
    for k in ks:
        nbr_tr, sim_tr = g.knn_train_graph(Z, tr_idx, tr_idx, k, device=DEV)
        nbr_ev, sim_ev = g.knn_train_graph(Z, tr_idx, ev_idx, k, device=DEV)
        F_tr.append(g.cs_features(nbr_tr, sim_tr, nbr_tr, sim_tr, np.arange(len(tr_idx)), resid, y_tr))
        F_ev.append(g.cs_features(nbr_ev, sim_ev, nbr_tr, sim_tr, np.full(len(ev_idx), -1), resid, y_tr))
    F_tr, F_ev = np.concatenate(F_tr, 1), np.concatenate(F_ev, 1)
    sc = StandardScaler().fit(F_tr)
    best = None
    for C in (0.01, 0.1, 1.0):  # chosen by 5-fold CV AUROC on train OOF rows
        from sklearn.model_selection import cross_val_score
        cv = cross_val_score(LogisticRegression(C=C, max_iter=2000), sc.transform(F_tr), Y[tr_idx],
                             scoring="roc_auc", cv=5).mean()
        if best is None or cv > best[0]:
            best = (cv, C)
    lr = LogisticRegression(C=best[1], max_iter=2000).fit(sc.transform(F_tr), Y[tr_idx])
    return lr.decision_function(sc.transform(F_ev)), {"cs_cv_auroc": float(best[0]), "C": best[1],
                                                       "coef": [round(float(c), 3) for c in lr.coef_[0]]}


for SEED in SEEDS:
    t_seed = time.time()
    np.random.seed(SEED); torch.manual_seed(SEED)
    train, val, test = make_splits(Y, SEED)
    mv, _gsel, _gcert = split_validation_3way(val, Y, SEED)
    tr_idx, mv_idx, te_idx = np.flatnonzero(train), np.flatnonzero(mv), np.flatnonzero(test)
    ev_idx = np.concatenate([mv_idx, te_idx])
    mv_pos, te_pos = np.arange(len(mv_idx)), np.arange(len(mv_idx), len(ev_idx))
    y_mv, y_te = Y[mv_idx], Y[te_idx]
    S = lambda z: scores(z, mv_pos, te_pos, y_mv, y_te)

    # ---- reference: tuned XGBoost (native NaN, early-stopped on model_val) ----
    clf = XGBClassifier(n_estimators=4000, max_depth=5, learning_rate=0.02, subsample=0.8,
                        colsample_bytree=0.5, min_child_weight=2, reg_lambda=2.0, eval_metric="auc",
                        early_stopping_rounds=300, random_state=SEED, n_jobs=4, tree_method="hist",
                        device="cuda" if DEV.type == "cuda" else "cpu")
    clf.fit(X_RAW[train], Y[train], eval_set=[(X_RAW[mv], Y[mv])], verbose=False)
    pz = clf.predict_proba(X_RAW[ev_idx])[:, 1]
    rep(SEED, "REF_xgb_tuned", S(pz))

    Xp, SL = g.prep_views(X_RAW, VIEWS, train, "iqr", True)
    Xt = torch.tensor(Xp, device=DEV)

    # ---- encoder: pick regularisation config on 1-bag OOF AUROC ----
    oof_auc = {}
    for name, cfg in ENC_CFGS.items():
        cf = g.crossfit("encoder", Xt, Y, tr_idx, ev_idx, SL, cfg, n_folds=5, seed=SEED * 10, device=DEV)
        oof_auc[name] = cf["oof_auroc"]
        rep(SEED, f"enc_1bag[{name}]", {"oof_auroc": cf["oof_auroc"], **S(cf["held"][-1]),
                                         "best_epochs": [i["best_epoch"] for i in cf["info"]]})
    best_enc = max(oof_auc, key=oof_auc.get)
    enc_oof, enc_held, enc_ep = bag("encoder", ENC_CFGS[best_enc], N_BAGS_ENC, Xt, tr_idx, ev_idx, SL, SEED)
    rep(SEED, f"ENC_bag{N_BAGS_ENC}[{best_enc}] exit0",
        {"oof_auroc": g.auroc(Y[tr_idx], enc_oof), **S(enc_held)})

    # ---- G1 retrieval, same regularisation family, cross-fitted + bagged ----
    ret_cfg = dict(ENC_CFGS[best_enc], m=32, lr=3e-4, patience=10)
    ret_oof, ret_held, ret_ep = bag("retrieval", ret_cfg, N_BAGS_RET, Xt, tr_idx, ev_idx, SL, SEED)
    rep(SEED, f"G1_bag{N_BAGS_RET}[m=32] exit1", {"oof_auroc": g.auroc(Y[tr_idx], ret_oof), **S(ret_held)})

    # ---- neural ensemble (no XGBoost anywhere) ----
    ens_oof, ens_held = (enc_oof + ret_oof) / 2, (enc_held + ret_held) / 2
    rep(SEED, "ENC+G1 mean", {"oof_auroc": g.auroc(Y[tr_idx], ens_oof), **S(ens_held)})

    # ---- exit 1: leak-free Correct & Smooth on each base ----
    for bname, (bo, bh) in {"ENC": (enc_oof, enc_held), "G1": (ret_oof, ret_held),
                            "ENC+G1": (ens_oof, ens_held)}.items():
        z, info = cs_layer(bo, bh, Xp, tr_idx, ev_idx)
        rep(SEED, f"{bname} + C&S(feature-kNN) exit1", {**S(z), **info})

    print(f"[v2][seed {SEED}] done in {time.time()-t_seed:.0f}s", flush=True)

# ---- summary across seeds ----
names = sorted({n for s in ALL.values() for n in s})
print("\n[v2] SUMMARY test AUROC mean ± sd over seeds", SEEDS)
for n in names:
    v = [ALL[str(s)][n]["test_auroc"] for s in SEEDS if n in ALL[str(s)]]
    print(f"[v2] {n:46s} {np.mean(v):.4f} ± {np.std(v):.4f}  per-seed={[round(x, 4) for x in v]}")
print("[v2] DONE", flush=True)
