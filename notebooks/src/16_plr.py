#%% md
# 16 · PLR numeric embeddings for the encoder and G1 retrieval

Round 4 (notebook 15, base+temporal features, core+aux training): tuned
XGBoost 0.9082 ± 0.0029 vs encoder 0.9063 ± 0.0009 test AUROC -- the trees
used the trajectory features better. RESEARCH.md prompt 2: periodic numeric
embeddings (PLR, Gorishniy et al. 2022) are the model-side lever that most
consistently closes the MLP-vs-GBDT gap, and TabR's best configs use them.
Here every continuous column gets a PLR-lite embedding (per-feature learned
frequencies, shared Linear, ReLU) before its view MLP; binary columns and
missingness indicators pass through unchanged.

Same protocol as notebooks 14-15 (core splits/test unchanged, auxiliary
patients train-only, selection on core OOF rows). Features = base+temporal,
reused from notebook 15's output (no event-table rescans).

#%% code
import os, sys, json, time, glob, shutil, subprocess
import numpy as np, pandas as pd, torch
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
#%% embed src/temporal_block.py
#%% embed tests/test_graph_v2.py

#%% code
r = subprocess.run([sys.executable, "-m", "pytest", "-q", f"{WORK}/tests/test_graph_v2.py"],
                   cwd=WORK, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": WORK})
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0, "leakage tests failed -- refusing to run"
print("[v2] leakage tests passed", flush=True)

#%% md
## Cohorts: core (cached) + full (block caches reused from notebook 12's output)

#%% code
hits = glob.glob("/kaggle/input/**/processed/cohort_97de93de81.csv.gz", recursive=True)
assert hits, "attach kernel satgunsinghsodhi/graph-v2-label-aware-experiments (its processed/ output)"
src_dir = os.path.dirname(hits[0])
for f in os.listdir(src_dir):
    if f.startswith(("cohort_97de93de81", "block_")):
        shutil.copy(os.path.join(src_dir, f), f"{WORK}/processed/{f}")
print("[v2] cached files:", sorted(os.listdir(f"{WORK}/processed")), flush=True)


def _find(start, max_depth=6):
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
print("[v2] MIMIC root:", RAW, flush=True)

from src import mimic_iv


def cfg_for(required):
    return {"seed": 42, "data": {
        "source": "mimic", "task": "mortality", "raw_dir": RAW, "cache_dir": f"{WORK}/processed",
        "obs_window_hours": 24, "min_los_hours": 28, "max_patients": None, "cohort_seed": 42,
        "force_rebuild": False, "blocks": ["demographics", "comorbidity", "labs", "vitals", "treatment"],
        "require_nonempty_blocks": required}}


CFG_CORE, CFG_FULL = cfg_for(["vitals"]), cfg_for([])
assert mimic_iv.cohort_signature(CFG_CORE) == "97de93de81"
t0 = time.time()
Xc, Yc, Nc, Mc = mimic_iv.build_cohort(CFG_CORE)
Xf, Yf, Nf, Mf = mimic_iv.build_cohort(CFG_FULL)
print(f"[v2] core n={len(Yc)}  full n={len(Yf)}  ({time.time()-t0:.0f}s)", flush=True)
assert list(Nc) == list(Nf), "feature columns differ between core and full cohorts"

# locate core rows inside the full cohort (order-preserving filter, asserted)
pos = {s: i for i, s in enumerate(Mf["stay_id"])}
CORE_IN_FULL = np.array([pos[s] for s in Mc["stay_id"]])
assert np.all(np.diff(CORE_IN_FULL) > 0), "core rows are not an order-preserving subset of full"
assert np.array_equal(Yf[CORE_IN_FULL], Yc)
AUX = np.setdiff1d(np.arange(len(Yf)), CORE_IN_FULL)
X_RAW, Y = Xf, Yf.astype(np.int64)
VIEWS = {k: np.asarray(v) for k, v in Mf["views"].items()}
IS_CORE = np.zeros(len(Y), dtype=bool); IS_CORE[CORE_IN_FULL] = True
print(f"[v2] aux patients={len(AUX)} prevalence core={Yc.mean():.4f} aux={Y[AUX].mean():.4f}", flush=True)

#%% md
## Temporal block (labs ~2 min + chartevents ~4 min DuckDB scans), cached to output

#%% code
from src import temporal_block as tb
from src import mimic_concepts as mc
TPATH = f"{WORK}/processed/temporal_full_v1.parquet"
t0 = time.time()
base_df, _ = mimic_iv._base_cohort(mimic_iv.find_root(RAW), CFG_FULL)
base_df = base_df.reset_index(drop=True)
assert np.array_equal(base_df["stay_id"].to_numpy(), Mf["stay_id"]), "base cohort order != full cohort"
hit_t = glob.glob("/kaggle/input/**/temporal_full_v1.parquet", recursive=True)
if hit_t:
    TB = pd.read_parquet(hit_t[0])
else:
    lab_t = tb.temporal_block(mimic_iv.find_root(RAW), base_df, "hosp/labevents.csv",
                              "hadm_id", "charttime", "valuenum", mc.LAB_ITEMS, "labt")
    vit_t = tb.temporal_block(mimic_iv.find_root(RAW), base_df, "icu/chartevents.csv",
                              "stay_id", "charttime", "valuenum", mc.CHART_ITEMS, "vitt")
    TB = pd.concat([lab_t, vit_t], axis=1)
    TB.to_parquet(TPATH)
assert len(TB) == len(Y)
X_TEMP = TB.to_numpy(np.float32)
n_base = X_RAW.shape[1]
lab_cols = [i for i, c in enumerate(TB.columns) if c.startswith("labt_")]
vit_cols = [i for i, c in enumerate(TB.columns) if c.startswith("vitt_")]
X_BT = np.concatenate([X_RAW, X_TEMP], 1)
VIEWS_BT = {**VIEWS, "labs_t": n_base + np.array(lab_cols), "vitals_t": n_base + np.array(vit_cols)}
FEATS = {"base": (X_RAW, VIEWS), "base+temporal": (X_BT, VIEWS_BT)}
print(f"[v2] temporal block {X_TEMP.shape} nan-frac={np.isnan(X_TEMP).mean():.3f} "
      f"(core rows {np.isnan(X_TEMP[IS_CORE]).mean():.3f}) ({time.time()-t0:.0f}s)", flush=True)

#%% md
## Splits on the CORE cohort (verbatim notebook 10-13 logic), mapped into full indices

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
## Run: encoder without / with PLR (two frequency scales), bagged winner, G1 with PLR

#%% code
from xgboost import XGBClassifier
from src import graph_v2 as g

ALL = {}
ENC_CFGS = {
    "B_lr3e-4_wd1e-3_p0.3": dict(lr=3e-4, wd=1e-3, p=0.3, d=128, hid=128),
    "C_lr3e-4_wd1e-2_p0.4_d256": dict(lr=3e-4, wd=1e-2, p=0.4, d=256, hid=256),
}


def rep(seed, name, row):
    print(f"[v2][seed {seed}] {name:40s} " + " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    ALL.setdefault(str(seed), {})[name] = row
    with open(f"{OUT}/results_v2_round5.json", "w") as f:
        json.dump(ALL, f, indent=1)


PLR_CFGS = {"none": None,
            "plr_s0.05": {"n_freq": 16, "sigma": 0.05, "e": 8},
            "plr_s0.3": {"n_freq": 16, "sigma": 0.3, "e": 8}}
BASE_CFG = ENC_CFGS["C_lr3e-4_wd1e-2_p0.4_d256"]
X_RAW, VIEWS = FEATS["base+temporal"]

for SEED in SEEDS:
    t_seed = time.time()
    np.random.seed(SEED); torch.manual_seed(SEED)
    tr_c, va_c, te_c = make_splits(Yc.astype(np.int64), SEED)
    mv_c, _, _ = split_validation_3way(va_c, Yc.astype(np.int64), SEED)
    tr_core = CORE_IN_FULL[tr_c]
    mv_idx, te_idx = CORE_IN_FULL[mv_c], CORE_IN_FULL[te_c]
    ev_idx = np.concatenate([mv_idx, te_idx])
    mv_pos, te_pos = np.arange(len(mv_idx)), np.arange(len(mv_idx), len(ev_idx))
    S = lambda z: {"model_val_auroc": g.auroc(Y[mv_idx], z[mv_pos]),
                   "test_auroc": g.auroc(Y[te_idx], z[te_pos]),
                   "test_auprc": float(average_precision_score(Y[te_idx], z[te_pos]))}
    tr_idx = np.sort(np.concatenate([tr_core, AUX]))

    clf = XGBClassifier(n_estimators=4000, max_depth=5, learning_rate=0.02, subsample=0.8,
                        colsample_bytree=0.5, min_child_weight=2, reg_lambda=2.0, eval_metric="auc",
                        early_stopping_rounds=300, random_state=SEED, n_jobs=4, tree_method="hist",
                        device="cuda" if DEV.type == "cuda" else "cpu")
    clf.fit(X_RAW[tr_idx], Y[tr_idx], eval_set=[(X_RAW[mv_idx], Y[mv_idx])], verbose=False)
    rep(SEED, "REF_xgb_tuned[base+temporal]", S(clf.predict_proba(X_RAW[ev_idx])[:, 1]))

    tr_mask = np.zeros(len(Y), dtype=bool); tr_mask[tr_idx] = True
    Xp, SL, CONT = g.prep_views(X_RAW, VIEWS, tr_mask, "iqr", True, return_cont=True)
    Xt = torch.tensor(Xp, device=DEV)

    best = None
    for pname, plr in PLR_CFGS.items():
        cfg = dict(BASE_CFG, cont=CONT, plr=plr)
        cf = g.crossfit("encoder", Xt, Y, tr_idx, ev_idx, SL, cfg, n_folds=5, seed=SEED * 10,
                        device=DEV, score_mask=IS_CORE)
        rep(SEED, f"ENC[{pname}]", {"oof_auroc_core": cf["oof_auroc"], **S(cf["held"][-1]),
                                    "best_epochs": [i["best_epoch"] for i in cf["info"]]})
        if best is None or cf["oof_auroc"] > best[0]:
            best = (cf["oof_auroc"], pname, cf)
    helds = [best[2]["held"][-1]]
    for b in (1, 2):
        cfb = g.crossfit("encoder", Xt, Y, tr_idx, ev_idx, SL, dict(BASE_CFG, cont=CONT, plr=PLR_CFGS[best[1]]),
                         n_folds=5, seed=SEED * 10 + b, device=DEV, score_mask=IS_CORE)
        helds.append(cfb["held"][-1])
    enc_held = np.mean(helds, 0)
    rep(SEED, f"ENC_bag3[{best[1]}]", S(enc_held))

    rcfg = dict(BASE_CFG, cont=CONT, plr=PLR_CFGS[best[1]], m=32, lr=3e-4, patience=8)
    rf = g.crossfit("retrieval", Xt, Y, tr_idx, ev_idx, SL, rcfg, n_folds=5, seed=SEED * 10 + 5,
                    device=DEV, score_mask=IS_CORE)
    rep(SEED, f"G1[{best[1]}][m=32]", {"oof_auroc_core": rf["oof_auroc"], **S(rf["held"][-1]),
                                       "exit0_test_auroc": g.auroc(Y[te_idx], rf["held"][0][te_pos])})
    rep(SEED, f"ENC_bag3+G1[{best[1]}]", S((enc_held + rf["held"][-1]) / 2))
    del Xt
    torch.cuda.empty_cache()
    print(f"[v2][seed {SEED}] done in {time.time()-t_seed:.0f}s", flush=True)

names = sorted({n for s in ALL.values() for n in s})
print("\n[v2] SUMMARY test AUROC mean ± sd over seeds", SEEDS)
for n in names:
    v = [ALL[str(s)][n]["test_auroc"] for s in SEEDS if n in ALL[str(s)]]
    print(f"[v2] {n:46s} {np.mean(v):.4f} ± {np.std(v):.4f}  per-seed={[round(x, 4) for x in v]}")
print("[v2] DONE", flush=True)
