# Graph v2 — pushing Arch4R past XGBoost (notebooks 12–18)

**Status (2026-09-27, final):** exploration paused pending a non-truncated MIMIC-IV
source. This file is the record of what was tried, what worked, and the
numbers of record for every baseline and every model.

## 1. Goal and constraints

- **Goal:** a graph / neural model that beats XGBoost on the project's
  in-hospital mortality task, target test AUROC > 0.91 (best model).
- **Constraints (user decisions):** no stacking/ensembling with XGBoost (the
  project already has an XGBoost-based cascade); any graph idea allowed; the
  Arch4R "weak encoder" rule may be relaxed; the phase-1 `encoder_only`
  baseline must be re-run correctly.
- **Literature ceiling** (`../RESEARCH.md`, prompt 1): for a leak-free, 24 h,
  structured-data MIMIC-IV setup the credible band is 0.87–0.90, with
  0.91–0.92 reachable only with richer temporal representations, dense chart
  coverage, pretraining or ensembling; >0.92–0.93 should trigger a leakage
  audit.

## 2. Protocol (identical across notebooks 12–18)

- Cohort `cohort_97de93de81` (17,663 first ICU stays, prevalence 10.78%),
  splits from `data_prep.make_splits(y, seed)` + `cascade.split_validation_3way`
  — the **same train / model_val / gate / test rows as notebooks 10–11**.
  Seeds 42, 43, 44.
- **Model selection never touches test.** Rounds 1 used `model_val`
  (1,413 rows); from round 2 on, 5-fold cross-fitting on the training split
  selects configs by out-of-fold (OOF) AUROC on ~10.6k core rows.
- **Leakage guards** (`tests/test_graph_v2.py`, executed at the top of every
  notebook, which refuses to run on failure): a training row's OOF score never
  comes from a model that saw its label; kNN / retrieval never lets a patient
  read its own label (self-edges removed, 2-hop self-returns subtracted);
  held-out patients only read training rows' labels.
- **Auxiliary patients** (rounds 3+): the 28,988 eligible stays the cohort
  drops only because the Kaggle mirror's `chartevents.csv` is truncated
  (`mimic_iv._drop_uncovered`). They are added to **training only**; test
  rows are unchanged; early stopping / selection score core rows only. XGBoost
  always receives exactly the same augmentation. Caveat: their vitals
  missingness is a download artefact, not a clinical decision (test rows never
  show that pattern).
- **Temporal features** (rounds 4+, `src/temporal_block.py`): per lab (28) and
  vital (14) concept, mean in each 6 h bin of the 24 h window, within-window
  slope, hours since last measurement — strictly inside `[intime, intime+24h)`,
  same plausibility ranges as `mimic_iv`. XGBoost gets them too.
- All compute on Kaggle T4 via self-contained notebooks (`#%% embed`, see
  `notebooks/README.md`); nothing needs secrets.

## 3. Round-by-round (test AUROC, mean ± SD over seeds 42–44 unless noted)

| round / notebook | change | best neural/graph model | tuned XGBoost (same data) | finding |
|---|---|---|---|---|
| 1 · nb12 (seed 42) | fixed encoder training; preprocessing ablation; G1 retrieval; G2 masked-label PPR | encoder 0.8981, G1 0.8991, G2 0.894 | 0.9018 | phase-1 `encoder_only` protocol = **0.7387** vs 0.8866 trained properly → the "+0.147 graph signal" was undertraining; everything peaks at epoch 2–3 on a noisy 1.4k-row val set |
| 2 · nb13 | 5-fold cross-fit + bagging; leak-free Correct & Smooth | encoder cfg C **0.9013 ± 0.0022** | 0.9025 ± 0.0042 | cross-fitting fixes selection; C&S / retrieval add ≤ 0 |
| 3 · nb14 | + 28,988 auxiliary training patients | encoder **0.9055 ± 0.0015** (best seed 0.9073) | 0.9051 ± 0.0027 | data helps every model ~+0.004; neural ≈ XGBoost |
| 4 · nb15 | + 252 temporal trajectory features | encoder **0.9063 ± 0.0009** (best seed 0.9078) | **0.9082 ± 0.0029** (seed 43: 0.9121) | trees use the trajectories better (+0.003 vs +0.0008) |
| 5 · nb16 | PLR numeric embeddings | encoder (no PLR) 0.9063; PLR σ=0.05 0.9054 ± 0.0019 (seed 43: **0.9080**); σ=0.3 0.9018 | 0.9082 ± 0.0029 | PLR does not help here |
| 6 · nb17 | masked-feature self-supervised pretraining (features of training rows only, no labels) | **encoder ⊕ G1, both pretrained: 0.9082 ± 0.0023** [0.9099]; G1 alone 0.9080 ± 0.0019 | 0.9082 ± 0.0029 | pretraining makes the retrieval exit add signal on every seed (+0.0011 / +0.0021 / +0.0012 over its own graph-free exit); ties tuned XGBoost |

Graph components: in rounds 1–5 the G1 retrieval exit differed from its own
graph-free exit by ≤ 0.0005, G2's diffusion checkpoints were flat and Correct
& Smooth changed −0.004…+0.000 — a well-trained encoder already captured the
neighbourhood signal (label-only kNN AUROC 0.84, adjusted homophily 0.25).
**Round 6 is the exception:** with a self-supervised-pretrained encoder the
retrieval exit adds +0.0015 on average over its own graph-free exit, positive
on all three seeds — the first consistent graph gain in the project.

## 4. Numbers of record — every baseline and every model

Test AUROC on the core test split. "core" = the original 17,663-patient
setting (331 features, core-train rows); "best-data" = core-train + 28,988
auxiliary patients for training, base + temporal features. Seeds 42–44 unless
stated; best single run in brackets.

### Baselines

| model | core | best-data | source |
|---|---|---|---|
| XGBoost, phase-1 config (400 trees, imputed + indicators) | 0.8939 ± 0.0015 · phase 1, 5 seeds: 0.8962 ± 0.0032 | 0.9006 ± 0.0034 | nb18 · nb10 |
| **XGBoost, tuned** (native NaN, early-stopped on model_val) | 0.9034 ± 0.0025 | **0.9082 ± 0.0040** [0.9137] | nb18 |
| 3-stage tabular cascade | 0.9012 ± 0.0045 · phase 1, 5 seeds: 0.9015 ± 0.0053 | 0.9057 ± 0.0039 [0.9110] | nb18 · nb10 |
| Architecture 2 graph-free encoder | 0.8572 ± 0.0134 (5 seeds) | — | nb10 |
| Architecture 2 PPR-push diffusion | 0.8852 ± 0.0044 (5 seeds) | — | nb10 |
| Arch4R `encoder_only`, **phase-1 protocol** (5 full-batch steps) | 0.7457 ± 0.0479 (5 seeds) | — | nb10 |
| Arch4R `encoder_only`, **corrected** (mini-batch + early stop) | 0.8919 ± 0.0048 | — | nb18 |
| Arch4R `fixed_k` (5 epochs) | 0.8917 ± 0.0049 (5 seeds) | — | nb10 |
| Arch4R `adaptive_k` (5 epochs) | 0.8925 ± 0.0038 (5 seeds) | — | nb10 |
| Arch4R `adaptive_k` certified gate | 0.8887 ± 0.0059 (5 seeds; 61.1% compute saved) | — | nb10 |
| Arch4R `adaptive_k`, 30 epochs, no early stop | 0.8580 (1 seed — overfits) | — | nb11 |

`xgboost_tuned[core]` is 0.9034 here (nb18, CPU) vs 0.9025 in nb13–14 (GPU
`hist`): same config, GPU/CPU histogram non-determinism.

### Graph v2 models (ours)

| model | core | best-data | source |
|---|---|---|---|
| **Encoder** (multi-view MLP, IQR + missingness indicators, 5-fold cross-fit, cfg C) | 0.9013 ± 0.0022 [0.9040] | **0.9063 ± 0.0009** [0.9078] | nb13 · nb15/16 |
| Encoder, 3-bag ensemble | 0.9003 ± 0.0021 (per seed 0.9021 / 0.9014 / 0.8974) | 0.9061 ± 0.0002 | nb13 · nb16 |
| Encoder + PLR (σ = 0.05) | — | 0.9054 ± 0.0019 [**0.9080**] | nb16 |
| G1 — TabR-style label-aware retrieval (m = 32) | 0.8995 ± 0.0016 | 0.9047 ± 0.0012 | nb13 · nb16 |
| Encoder ⊕ G1 (mean of logits; no XGBoost) | 0.9003 ± 0.0015 | 0.9059 ± 0.0006 | nb13 · nb16 |
| Encoder + leak-free Correct & Smooth | 0.8987 ± 0.0032 | — | nb13 |
| G2 — masked-label PPR diffusion | 0.8941 (seed 42) | — | nb12 |
| Encoder, masked-feature pretraining | — | 0.9065 ± 0.0024 (3-bag 0.9067) | nb17 |
| **G1 retrieval, pretrained encoder** | — | **0.9080 ± 0.0019** [0.9094] | nb17 |
| **Encoder (3-bag) ⊕ G1, both pretrained — best model** | — | **0.9082 ± 0.0023** [**0.9099**] | nb17 |

### Head-to-head (best-data, same seeds/splits)

| comparison | Δ test AUROC |
|---|---|
| best model (pretrained encoder ⊕ G1) vs Arch4R `adaptive_k` (phase 1) | **+0.0157** |
| best model vs the original 3-stage cascade (phase 1) | **+0.0067** |
| best model vs cascade given the same best-data (nb18) | +0.0025 |
| best model vs tuned XGBoost given the same best-data (paired, nb17) | +0.0000 (per seed +0.0049 / −0.0024 / −0.0027) |

**Bottom line.** The best model — a self-supervised-pretrained multi-view
encoder plus label-aware retrieval over training patients, no XGBoost
anywhere — reaches **0.9082 ± 0.0023 (best run 0.9099)**: above every original
baseline including the project's cascade, and **tied with tuned XGBoost given
identical data** (0.9082, beating it on 1 of 3 seeds). Nothing clears 0.91 on
the 3-seed mean; single runs reach 0.9099 (ours) and 0.9137 (XGBoost). This
matches the literature ceiling for a truncated-chart, 24 h, aggregate-feature
cohort.

## 5. Provenance

| source | location |
|---|---|
| phase-1 5-seed baselines (Arch2, Arch4R graph arms, certified gate) | `ARCH4R_HANDOVER.md` §9, `notebooks/10_arch4r_phase1.ipynb` |
| rounds 1–6 logs + JSON | `results/kaggle_runs/k12_v1` … `k17_v1` |
| final baseline benchmark | `results/kaggle_runs/k18_v1` |
| Kaggle kernels | `satgunsinghsodhi/graph-v2-*` (private) |

## 6. What would move the number

1. **A non-truncated MIMIC-IV** (the user is obtaining it): 62% of eligible
   stays currently have no vitals at all, capping both the cohort and the
   temporal features — the only lever that moved every model in rounds 3–4 was
   more/better data.
2. Once dense chart data exists: a DuETT-style event × time model with
   masked value/presence pretraining over all stays (the literature's only
   clean +0.026 win over XGBoost on MIMIC-IV mortality).
3. Re-test the graph components (G1 retrieval, masked-label PPR) on the full
   data — with dense vitals the similarity structure changes, so the null
   result here is specific to this truncated cohort.
