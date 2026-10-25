#%% md
# 18 · Final baseline benchmark (same seeds/splits as the graph-v2 rounds)

Re-runs every cheap baseline on seeds 42-44 with the exact core splits used by
notebooks 10-17, in two data settings:

- **core**: the original setting -- 17,663-patient cohort, 331 whole-window
  features, core-train rows only (what phase 1 reported).
- **best-data**: core-train + 28,988 auxiliary (vitals-missing) patients for
  training, base + temporal features (notebooks 14-15). Test rows unchanged.

Arms: phase-1 plain XGBoost config, tuned XGBoost (native NaN, early-stopped
on model_val), the 3-stage tabular cascade (`src/tabular.py`, temporal views
joined to their stage's block), and the corrected Arch4R `encoder_only`
(original model/preprocessing, mini-batched + early-stopped; core only).
Graph arms that are expensive to retrain (Architecture 2, Arch4R fixed_k /
adaptive_k / certified) are cited from phase 1 (5 seeds) in the docs.

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
#%% embed src/tabular.py
#%% embed src/instrument.py
#%% embed src/encoder_mv.py
#%% embed src/graph_density.py
#%% embed src/local_push.py
#%% embed src/arch4r.py
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
## Run

#%% code
from xgboost import XGBClassifier
from src import graph_v2 as g, tabular, arch4r
from src.encoder_mv import fit_view_preprocessors, observed_fraction
from src.local_push import make_checkpoints

ALL = {}
XGB_DEV = "cuda" if DEV.type == "cuda" else "cpu"


def rep(seed, name, row):
    print(f"[v2][seed {seed}] {name:40s} " + " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    ALL.setdefault(str(seed), {})[name] = row
    with open(f"{OUT}/results_final_baselines.json", "w") as f:
        json.dump(ALL, f, indent=1)


def impute_and_scale(X, train_mask):  # same maths as src/data_prep.impute_and_scale
    X = np.asarray(X, dtype=np.float32)
    missing = np.isnan(X)
    med = np.nanmedian(X[train_mask], 0)
    med = np.where(np.isnan(med), 0.0, med)
    Xi = np.where(missing, med, X)
    mu, sd = Xi[train_mask].mean(0), Xi[train_mask].std(0)
    Xi = (Xi - mu) / np.where(sd < 1e-6, 1.0, sd)
    keep = missing[train_mask].any(0)
    if keep.any():
        Xi = np.concatenate([Xi, missing[:, keep].astype(np.float32)], 1)
    return Xi.astype(np.float32)


X_B, VIEWS_B = FEATS["base+temporal"]
# cascade stages: temporal lab / vital views join their parent block's stage
CASCADE_VIEWS = {
    "core": dict(VIEWS),
    "best-data": {**{k: v for k, v in VIEWS_B.items() if k not in ("labs_t", "vitals_t")},
                  "labs": np.concatenate([VIEWS_B["labs"], VIEWS_B["labs_t"]]),
                  "vitals": np.concatenate([VIEWS_B["vitals"], VIEWS_B["vitals_t"]])},
}

for SEED in SEEDS:
    t_seed = time.time()
    np.random.seed(SEED); torch.manual_seed(SEED)
    tr_c, va_c, te_c = make_splits(Yc.astype(np.int64), SEED)
    mv_c, _, _ = split_validation_3way(va_c, Yc.astype(np.int64), SEED)
    tr_core = CORE_IN_FULL[tr_c]
    mv_idx, te_idx = CORE_IN_FULL[mv_c], CORE_IN_FULL[te_c]
    SETTINGS = {"core": (X_RAW, tr_core),
                "best-data": (X_B, np.sort(np.concatenate([tr_core, AUX])))}

    def T(p_te):
        return {"test_auroc": g.auroc(Y[te_idx], p_te),
                "test_auprc": float(average_precision_score(Y[te_idx], p_te))}

    for sname, (X, tr_idx) in SETTINGS.items():
        # rows used by this setting: train, then model_val, then test
        sub = np.concatenate([tr_idx, mv_idx, te_idx])
        trm = np.zeros(len(sub), dtype=bool); trm[:len(tr_idx)] = True
        te_sub = np.arange(len(tr_idx) + len(mv_idx), len(sub))
        Xi = impute_and_scale(X[sub], trm)
        ytr = Y[tr_idx]
        pos = (ytr == 1).sum()

        clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, scale_pos_weight=float((ytr == 0).sum() / pos),
                            eval_metric="logloss", random_state=SEED, n_jobs=4, tree_method="hist",
                            device=XGB_DEV)
        clf.fit(Xi[trm], ytr)
        rep(SEED, f"plain_xgboost_phase1cfg[{sname}]", T(clf.predict_proba(Xi[te_sub])[:, 1]))

        clf = XGBClassifier(n_estimators=4000, max_depth=5, learning_rate=0.02, subsample=0.8,
                            colsample_bytree=0.5, min_child_weight=2, reg_lambda=2.0, eval_metric="auc",
                            early_stopping_rounds=300, random_state=SEED, n_jobs=4, tree_method="hist",
                            device=XGB_DEV)
        clf.fit(X[tr_idx], ytr, eval_set=[(X[mv_idx], Y[mv_idx])], verbose=False)
        rep(SEED, f"xgboost_tuned[{sname}]", T(clf.predict_proba(X[te_idx])[:, 1]))

        cols, _names = tabular.stage_columns(CASCADE_VIEWS[sname], X.shape[1], Xi.shape[1],
                                             tabular.indicator_sources(X[sub], trm))
        res = tabular.fit_staged_boosting(Xi, Y[sub], trm, cols, seed=SEED, n_folds=5,
                                          device=XGB_DEV, verbose=False)
        row = T(res["margins"][-1][te_sub])
        row["stage_test_auroc"] = [round(g.auroc(Y[te_idx], res["margins"][k][te_sub]), 4)
                                   for k in range(len(cols))]
        rep(SEED, f"cascade_3stage[{sname}]", row)

    # corrected Arch4R encoder_only: original model + preprocessing, core only,
    # mini-batched AdamW + early stopping on model_val (phase 1 took 5 full-batch steps)
    raw_views = {v: Xc[:, np.asarray(idx)] for v, idx in Mc["views"].items()}
    pre = fit_view_preprocessors(raw_views, tr_c)
    vx, vo = pre.transform(raw_views), observed_fraction(raw_views)
    VX = {v: torch.tensor(a, device=DEV) for v, a in vx.items()}
    VO = {v: torch.tensor(a, dtype=torch.float32, device=DEV) for v, a in vo.items()}
    yc_t = torch.tensor(Yc, dtype=torch.float32, device=DEV)
    torch.manual_seed(SEED)
    m4 = arch4r.Arch4RModel({v: a.shape[1] for v, a in vx.items()},
                            {"fusion_dim": 64, "stage_hidden": 32}, "encoder_only").to(DEV)
    ck = make_checkpoints()
    fwd4 = lambda b, train=True: m4({v: t[b] for v, t in VX.items()},
                                    {v: t[b] for v, t in VO.items()}, None, ck)["stage_logits"][-1:]
    mvc, tec = np.flatnonzero(mv_c), np.flatnonzero(te_c)
    info = g.fit_minibatch(m4, fwd4, np.flatnonzero(tr_c), yc_t,
                           lambda: g.auroc(Yc[mvc], g.predict_rows(fwd4, m4, mvc)[-1]), seed=SEED)
    pt = g.predict_rows(fwd4, m4, tec)[-1]
    rep(SEED, "arch4r_encoder_only_corrected[core]",
        {"test_auroc": g.auroc(Yc[tec], pt), "test_auprc": float(average_precision_score(Yc[tec], pt)),
         "best_epoch": info["best_epoch"]})
    print(f"[v2][seed {SEED}] done in {time.time()-t_seed:.0f}s", flush=True)

names = sorted({n for s_ in ALL.values() for n in s_})
print("\n[v2] SUMMARY test AUROC mean ± sd over seeds", SEEDS)
for n in names:
    v = [ALL[str(s_)][n]["test_auroc"] for s_ in SEEDS if n in ALL[str(s_)]]
    print(f"[v2] {n:46s} {np.mean(v):.4f} ± {np.std(v):.4f}  per-seed={[round(x, 4) for x in v]}")
print("[v2] DONE", flush=True)
