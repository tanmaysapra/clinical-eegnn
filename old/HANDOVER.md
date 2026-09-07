# Handover — Early-Exit GNN for Clinical Risk Stratification

Last updated: 2026-09-01. Status of the implementation against the phased plan,
plus everything a new session (or a new person) needs to pick this up.

**Read `CURRENT_ITERATION.md` first.** Every claim below has been through an
independent four-agent reproduction (§5o). Most held; four did not, and those
are corrected in place and marked **[REPRO 2026-09-01]**.

**§5p (2026-09-01) closes everything §5o left open** and adds the spec's
three-architecture controlled experiment (`ARCHITECTURES.md`). Headlines: the
tie-block hypothesis is **refuted**; `fit_gate`'s search is **exact**; the
seed-blind graph cache is real but **immaterial**; alpha=0.05 and the conformal
ablation **were never actually unrun**; and ACT vs discrete exit stages is
**not resolvable at 5 seeds**, which upgrades the negative result from
"this halting mechanism fails" to "early exit fails on this data".

---

## 1. TL;DR

**Updated 2026-09-01 after an independent reproduction pass — see §5o and
`CURRENT_ITERATION.md`. Read §5n for the result, §5o for what survived checking.**

The reproduction's verdict in one line: **the §5n pivot holds, the graph negative
hardened, and four recorded claims were wrong** — one code bug, one metric bug,
one statistic averaged over 2 seeds but printed as 5, one wording error. The
corrections are inline below.

The project's central mechanism now works, and it works because the thing being
cascaded over changed. §5g–§5j did not show early exit failing; they showed it
being asked to choose between options that were all identical (per-exit AUROC
flat from layer 1 to layer 10, so a trivial constant policy dominated the
learned one). §5n cascades over **information** instead of depth — bedside
observations, then charted orders, then the laboratory panel — and the picture
inverts:

- The exits differ: 0.8624 / 0.8733 / 0.8968 AUROC across three stages, stage 2
  minus stage 0 = +0.0343, **t = 17.0** over 5 paired seeds. The depth cascade's
  exits were flat within noise.
- **A monotone accuracy/compute trade-off curve exists**, which §5h explicitly
  said did not, and the gate is **no longer dominated** by the trivial "exit
  everyone immediately" and "always run everything" policies.
- The final stage is **indistinguishable from XGBoost** (+0.0002, t=0.12), so the
  staging is free — the strongest measured component is now the backbone rather
  than the baseline being apologised to.
- The safety rule carries a **distribution-free certificate**: thresholds chosen
  against a Clopper–Pearson bound realise miss rates of 0.0081 / 0.0316 against
  budgets of 0.02 / 0.05, and release 2.7% of the top severity decile against
  44.3% of everyone else (**16.7x**). **[REPRO 2026-09-01]** The 0.01 budget was
  dropped from this list: its release side fires in only 2/5 seeds and is
  uncertifiable at stage 0 (needs m >= 299 clean releases, has 98–220). See §5n.
- The headline saving is **46.2% of patients resolved without the model reading
  the lab block at all** — money and turnaround, not the 5 microseconds of
  compute, which would be a rerun of §5j's oversized-budget error.

The negative result also hardened, and is now worth publishing as one. The
patient-similarity graph adds nothing on top of the boosted score under four
independent framings (§5n): parameter-free residual propagation and a learned
residual GNN, over both a feature-space kNN graph and a graph built from the
booster's own leaf co-occurrence. Best variant: +0.0001 AUROC for a stage costing
5x the three tabular stages combined. §5f's diagnosis survives a far stronger test
than the one that produced it.

**The earlier record below is still accurate and is the reason §5n is framed the
way it is. Read §5b–§5m for how the conclusion was reached.**

---

**Updated 2026-08-28. §5b-§5j — the depth-cascade picture.**

The pipeline runs end to end on **real MIMIC-IV v3.1** (Kaggle mirror), and the
full notebook series 00-07 executes clean. 13/13 unit tests pass.

The headline finding is **negative and worth taking seriously**:

- On a 17,663-patient first-24h ICU mortality cohort, **XGBoost (0.8958 ± 0.0023)
  significantly beats every graph model** over 5 paired seeds — against SAS-GNN
  full depth (0.8888) t=3.68, against EEGNN (0.8861) t=4.75.
- **Layer-1 accuracy equals the full model at every depth budget** (e.g. 0.8901
  vs 0.8899 at a 2-layer budget), so propagation adds nothing.
- **The early-exit policy saves no compute at any sensible budget.** Realised
  depth equals the budget exactly at 2, 4 and 6 layers. It only engages at a
  budget of 10, and then by 20.4% ± 13.6% — the previously quoted 37% was a
  single favourable seed.
- Root cause (§5f): the kNN graph is **built from the feature matrix**, so it
  cannot add information beyond `X`. The relational structure is real (1-hop
  neighbour-averaging reaches 0.811 AUROC) but redundant with the features it
  came from.

So the project's central claim — adaptive depth buying compute savings at
clinical parity — **is not supported on this data as currently framed**. §7 lists
the three routes that could change that; only "edges from a source outside `X`"
looks capable of changing the conclusion rather than restating it.

Genuine positives from the run:

- The ETL is sound: leakage controls hold under independent verification, the
  automated leak detector passes, and SHAP on the surrogate ranks age, urine
  output, GCS, respiratory rate and BUN — real ICU medicine.
- §6.1's degenerate safety metric is **fixed and validated**, and immediately
  earned its keep: it exposed that the model is **twice as badly calibrated on
  the sickest decile** (ECE ~0.29 vs ~0.14) in all 13 sweep configurations — a
  deployment-blocking finding invisible in pooled ECE.
- Two real ETL bugs found and fixed (urine irrigation double-count; unbounded
  rate), plus a truncated `chartevents.csv` in the mirror caught before it could
  be learned as signal (§5b).

**§5i and §5j supersede the single-seed numbers** in §5f-§5h with 5-seed means,
standard deviations and paired t-tests, and correct two claims — notably that
GCN/GAT sit below logistic regression (they do not) and that SAS-GNN does not
beat a linear model (it does, significantly).

## 2. Environment (important — read before running)

The machine's system Python is 3.14, which is too new for torch/PyG. Work
happens in a dedicated conda env:

```bash
conda activate eegnn          # Python 3.11
# or call the interpreter directly:
/opt/homebrew/Caskroom/miniconda/base/envs/eegnn/bin/python
```

Installed: torch 2.13, torch_geometric 2.8, scikit-learn 1.9, numpy 2.0.2,
xgboost 2.1.4, shap 0.46.0, pandas, pyyaml, matplotlib, torchmetrics, pytest.

Recreate from scratch:
```bash
conda create -y -n eegnn python=3.11
conda activate eegnn
pip install -r requirements.txt
```

### Two platform footguns already solved (do not "fix" these back)

1. **SHAP × NumPy 2 clash** — the plan warned about this. `shap < 0.46` calls
   `np.obj2sctype`, removed in NumPy 2, and hard-crashes on import. Pinned to
   `shap==0.46.0` / `xgboost==2.1.4` in `requirements.txt`. This is the
   known-good pair; do not downpin.

2. **XGBoost × torch OpenMP clash on macOS/arm64** — they cannot coexist in one
   process. torch-then-xgboost **segfaults** (exit 139); xgboost-then-torch
   **deadlocks** at 0% CPU. `KMP_DUPLICATE_LIB_OK=TRUE` silences it but is
   documented-unsafe (silent numeric corruption); `OMP_NUM_THREADS=1` works but
   costs ~2.2× on torch. **Solution in the repo:** XGBoost runs in a separate
   process. `src/_omp_guard.run_xgb_worker()` shells out to
   `scripts/_xgb_worker.py`, which imports xgboost/shap and never imports torch.
   If you add any XGBoost/SHAP call, route it through that worker.

3. **Spurious NumPy matmul warnings** — NumPy 2 + Apple Accelerate emits
   divide-by-zero/overflow warnings from float32 matmul that are harmless
   (verified: results match manual dot products and stay finite). Filtered at
   the entry point.

---

## 3. How to run

```bash
cd clinical-eegnn
PY=/opt/homebrew/Caskroom/miniconda/base/envs/eegnn/bin/python

$PY scripts/run_experiment.py --stage smoke     --data mock   # P0: 1-epoch end-to-end
$PY scripts/run_experiment.py --stage baselines --data mock   # P2: SAS-GNN vs GCN/GAT/XGB/naive-EE
$PY scripts/run_experiment.py --stage eegnn     --data mock   # P3+P4: multi-exit + calibration + 3 metric families
$PY scripts/run_experiment.py --stage sweep     --data mock   # P4: trade-off curve + safety-ordering check
$PY scripts/run_experiment.py --stage full --seeds 3          # everything, mean±std over seeds
$PY -m pytest tests/ -q -W ignore::RuntimeWarning             # unit tests (10)
```

Useful overrides: `--ponder-cost`, `--halt-penalty`, `--max-layers`, `--epochs`,
`--seed`. Results land in `results/*.json` (keyed by config hash); figures in
`results/figures/`.

### The staged hybrid cascade (§5n) — this is the current model

```bash
# the headline run: 3 tabular stages, 5 seeds, certified + empirical gates
$PY scripts/run_cascade.py --source mimic --seeds 42 43 44 45 46 \
    --alphas 0.005 0.01 0.02 0.05 --ablate-conformal --graph-stage none \
    --out results/headline.json

# the graph-stage ablation: {Correct&Smooth, learned GNN} x {feature kNN, leaf graph}
$PY scripts/run_cascade.py --graph-stage cs  --graph leaf --out results/cs_leaf.json

$PY -m pytest tests/ -q -W ignore::RuntimeWarning        # 24 tests
```

On Kaggle add `--raw-dir /kaggle/input/datasets/akshaybe/updated-mimic-iv
--cache-dir /kaggle/working/processed`. A 5-seed run takes ~3 minutes on 4 CPUs
once the cohort is cached. Key flags: `--alphas` (safety budgets), `--beta`
(escalation budget), `--gamma` (1-gamma is the certificate's confidence),
`--graph-stage {none,cs,gnn}`, `--graph {knn,leaf}`, `--n-strata` (stratified
calibration; leave at the default, see §6.5).

---

## 4. What each file does

```
clinical-eegnn/
├── configs/default.yaml        single flat config (no config classes)
├── requirements.txt            pinned deps incl. the shap/xgboost fix
├── scripts/
│   ├── run_experiment.py       CLI entry + all stage functions (depth cascade)
│   ├── run_cascade.py          CLI entry for the staged hybrid cascade (§5n)
│   ├── plot_cascade.py         the §5n frontier figure + release-rate-by-stratum panel
│   └── _xgb_worker.py          torch-free XGBoost/SHAP subprocess (OpenMP isolation)
├── src/
│   ├── mock_data.py            synthetic EHR cohort (REWRITTEN — see §6.2)
│   ├── data_prep.py            cohort load, train-only imputation, stratified masks, MIMIC stub
│   ├── graph_build.py          kNN / threshold / multiview patient graphs; PyG Data; caching
│   ├── layers.py               SASGNNLayer (weight-tied ODE step) + ExitHead (Gumbel halt)
│   ├── model.py                EEGNN: encoder → shared backbone → per-node exit mixture
│   ├── losses.py               multi-exit focal + risk-weighted ponder + safety term
│   ├── train.py                transductive loop, Gumbel annealing, early stop on val AUROC
│   ├── evaluate.py             predictive / calibration / efficiency / safety metrics
│   ├── baselines.py            GCN, GAT, XGBoost (via worker), fixed-threshold EE
│   ├── explain.py              SHAP, GAT attention, exit-depth-vs-difficulty
│   ├── safe_exit.py            risk-gated exit over *depth* (§5m; superseded by cascade.py)
│   ├── tabular.py              THE STAGES: nested feature blocks, out-of-fold staged XGBoost (torch-free)
│   ├── cascade.py              THE GATE: isotonic calibration, Clopper-Pearson two-sided gate, routing (torch-free)
│   ├── hybrid.py               THE GRAPH STAGE: Correct & Smooth + learned residual GNN on top of the boosted score
│   ├── plots.py                reliability diagram, trade-off curve, exit dist, depth-vs-difficulty
│   └── _omp_guard.py           run_xgb_worker(): out-of-process xgboost
├── tests/test_pipeline.py      13 unit tests (leakage, graph, weight-sharing, gumbel grads, loss↓)
└── tests/test_cascade.py       11 unit tests (stage nesting, indicator block inheritance,
                                C&S label leakage, Clopper-Pearson, gate routing, calibration split)
```

---

## 5. Phase status vs the plan

| Phase | Deliverable | Acceptance | Status |
|---|---|---|---|
| P0 | Skeleton + mock + 1-epoch run | trains 1 epoch, no error | **PASS** |
| P1 | Data prep + kNN graph | node/edge counts, prevalence preserved | **PASS** |
| P2 | SAS-GNN + baselines | SAS-GNN full-depth AUROC ≥ GCN | **PASS** (old generator) |
| P3 | Exit heads + multi-exit | non-trivial exit distribution, AUROC≈full at lower depth | **RUNS**, see §6 |
| P4 | Calibration + eval | trade-off curve, ECE/Brier, high-risk analysis | **RUNS**, see §6.1 |
| P5 | Interpretability | SHAP + attention + depth-vs-difficulty | **IMPLEMENTED**, unvalidated |

### Numbers from the last full P2/P3/P4 run (OLD generator — will change)

| Model | Test AUROC | AUPRC |
|---|---|---|
| XGBoost | 0.781 | — |
| GCN | 0.747 | 0.263 |
| GAT | 0.767 | 0.313 |
| SAS-GNN full depth | 0.776 | 0.376 |
| Fixed-threshold EE | 0.735 | — |
| EEGNN (multi-exit) | 0.772 | 0.350 |

EEGNN reached 0.772 at avg depth 3.9/10 (≈61% compute saving), calibrated ECE
0.29 → 0.11 after temperature scaling (T≈0.19).

---

---

## 5b. MIMIC-IV is now wired up and validated (2026-08-27)

Real data runs. Work happens on a **Kaggle session** with the
`akshaybe/updated-mimic-iv` dataset attached (full MIMIC-IV v3.1, ~100 GB,
every table present including `chartevents` 42 GB and `labevents` 18 GB).

### Driving the remote kernel

`scripts/remote_kernel.py` talks to the session over its REST/WebSocket API —
no MCP server needed, and no GitHub push needed to get code onto it.

```bash
export KERNEL_URL='https://kkb-production.jupyter-proxy.kaggle.net/k/<session>/<token>/proxy'
python scripts/remote_kernel.py status
python scripts/remote_kernel.py push src configs scripts notebooks
python scripts/remote_kernel.py run -f some_script.py -t 900
```

The URL changes on **every** session restart, and `/kaggle/working` does not
survive a session ending — so `push` is expected after each reconnect. A dead
session answers 404 on every endpoint; `status` says so explicitly.

Remote env: python 3.12, torch 2.10+cu128, **2x Tesla T4**, 4 CPUs, 32 GB RAM,
20 GB writable, duckdb 1.3.2, xgboost 3.2, shap 0.51. `torch_geometric` is
**not** in the base image — notebook 00 installs it.

### Verified cohort

`_base_cohort` runs clean against the real v3.1 schema:

| step | n |
|---|---|
| icu stays | 94,458 |
| first stay per patient | 65,366 |
| adult | 65,366 |
| los >= 28h | 46,776 |
| died after window | 46,705 |
| discharged after window | **46,651** |

In-hospital mortality **11.02%**. Demographics (33 cols), comorbidity (19) and
treatment (27) blocks all build and read as clinically plausible (37% invasive
ventilation, 30% any vasopressor, median urine 64.6 mL/h). DuckDB scans at
~150 MB/s, so labs + vitals should be minutes, not hours.

All five blocks now build: **331 features** before the coverage filter. Labs
take ~4 min and vitals ~11 min on 4 CPUs. The leak detector passes (no feature
above 0.95 univariate AUROC) and the top predictors read as real ICU medicine —
GCS verbal 0.723, GCS eye 0.719, BUN last 0.704. Leakage control 3 excludes
**79.7%** of this cohort's diagnosis rows as outcome-contaminated.

### The Kaggle mirror's `chartevents.csv` is truncated (important)

`icu/chartevents.csv` (41.94 GB) **ends mid-record with no trailing newline**.
It contains only the first ~37.8% of ICU stays in `subject_id` order:

- vitals coverage is a clean prefix — covered `subject_id` 10000690..13799184,
  and only **28** of the ~29,000 uncovered patients fall below that cut point;
- heart rate, SBP, DBP, MAP and SpO2 all land on *exactly* 37.80% coverage,
  which is whole patients missing, not sparse charting;
- verified independently: of the uncovered stays, only 28 appear anywhere in
  `chartevents` at all, and join dtypes are clean, so this is not our bug.

`inputevents.csv` spans the full `subject_id` range and `labevents.csv` is
complete, so only the vitals block is affected.

**Why this matters more than ordinary missingness.** `impute_and_scale` appends
a missingness indicator per column *on purpose* — an ICU lab not drawn is a
clinical decision. When the missingness is a download artefact instead, those
indicators encode "this patient's `subject_id` sorts below the cut point", and
the model learns the defect as signal. Notebook 02 shows measurement intensity
alone is predictive, so the risk is real, not theoretical.

**Fix:** `data.require_nonempty_blocks: [vitals]` (new) drops patients whose
required block is entirely absent — data-driven, so it handles any partial
mirror rather than hardcoding a cut point.

| | before filter | after filter |
|---|---|---|
| patients | 46,651 | **17,663** |
| vitals missingness | 57.3% | **14.6%** |
| GCS coverage | 37.8% | **99.8%** |
| overall missingness | 25.6% | 14.7% |
| in-hospital mortality | 11.02% | 10.78% |

Note the subtlety that cost a run: the presence test cannot be
`notna().any(axis=1)`, because `_aggregate_events` zero-fills the `_count`
columns on purpose (a concept never measured has count 0, not a *missing*
count), so every row looks covered. `_block_present` keys off the count columns
instead. There is a regression test for this.

Univariate AUROCs barely move across the filter (GCS verbal 0.7226 -> 0.7226,
BUN last 0.7040 -> 0.7033), which is independent evidence that the dropped
patients were not clinically different.

`subject_id` is randomly assigned in MIMIC-IV, so the truncation is arbitrary
with respect to clinical variables, and the retained/dropped comparison bears
that out: mortality 10.76% vs 11.18%, identical median age (67) and median LOS
(64.5 vs 65.5 h). The notebook asserts this rather than assuming it.

**The alternative** is `require_nonempty_blocks: []` plus dropping `vitals` from
`data.blocks`: ~46.7k patients but no vitals — and vitals include GCS, which the
leak detector ranks as the single strongest univariate predictor (AUROC 0.72).
Worth revisiting if a complete mirror becomes available; that is the better fix.

### Changes made to support it

- `data_prep.load_cohort` now routes `source: mimic` to `mimic_iv.build_cohort`
  (it previously went to a stub CSV reader that never saw the ETL). The old
  reader survives as `source: csv` for externally-built cohorts.
- `configs/default.yaml` grew the mimic keys (`raw_dir`, `obs_window_hours`,
  `min_los_hours`, `max_patients`, `blocks`, ...); `run_experiment.py` grew
  `--raw-dir`, `--cache-dir`, `--max-patients`, `--obs-window`.
- **Per-block ETL caching.** Each feature block is written to parquet the moment
  it is built, keyed on `row_signature` (the fields deciding *which patients*)
  rather than `cohort_signature` (which includes the block list). Adding a block
  therefore reuses the multi-GB scans instead of invalidating them. This exists
  because a 12h session cap plus a 42 GB scan makes all-or-nothing fatal.
- **Two ETL version constants** in `mimic_iv.py`, both folded into the cache
  keys. `_BLOCK_VERSION` covers block-building semantics (itemid sets,
  aggregates, window logic) and bumping it invalidates the 18 GB and 42 GB
  scans — expensive, so do it deliberately. `_COHORT_VERSION` covers assembly
  semantics (cohort filters, packing) and is cheap: blocks are reused, only the
  assembled table rebuilds. These exist because a fix to `_drop_uncovered` was
  silently ignored on the next run — the config had not changed, so the stale
  assembled cohort was served from cache and the fix appeared not to work.
- **kNN no longer builds a dense `[n, n]` matrix.** At n=46,651 that is 8.7 GB
  and does not coexist with training. `build_knn_graph` now scores in row blocks
  (GPU when available), verified bit-identical to the dense path on both
  similarities. `_similarity` raises above 20k rather than silently thrashing.
- `build_multiview_graph` fusion is vectorised (was a Python dict loop over
  ~3.5M edges), and the multi-view graph now uses **real** view membership from
  `meta["views"]` instead of the placeholder contiguous column split.

### Two ETL bugs found and fixed

1. **Urine output counted irrigation fluid as urine.** `227488` is "GU Irrigant
   Volume In" — fluid instilled, not produced. MIMIC's own `urine_output`
   concept subtracts it from `227489`; the ETL summed both. This systematically
   overstated output for patients on bladder irrigation (a specific subgroup,
   so it is bias, not noise). Now subtracted.
2. **No physiologic bound on the derived rate**, only a loose 5000 mL per-event
   cap — which let `tx_urine_ml_per_hr` reach 8,323 mL/h. Per-event cap is now
   2500 mL and a rate above 1000 mL/h (or negative, post-subtraction) becomes
   NaN, i.e. missing, matching how out-of-range labs are handled.

### Documented limitation, not a bug

Comorbidities may only come from *closed prior* admissions (leakage control 3),
so ~60% of the cohort has an all-zero comorbidity block and apparent
prevalences run far below population rates (diabetes ~9%). The block partly
encodes "known frequent patient". The alternative — using this admission's
discharge codes — is what makes many published MIMIC mortality AUROCs
unreproducible, so this stays.

## 5c. Notebook series

`notebooks/` now holds the documented pipeline; see `notebooks/README.md`.
Cell sources live as plain text in `notebooks/src/*.py` so they diff like code.

`00_environment_and_data_audit`, `01_cohort_extraction`, `02_cohort_eda` and
`03_graph_construction` are written. **None has been executed end to end yet** —
the session died before labs/vitals finished. 04-07 are intentionally unwritten:
they depend on what 01-03 report on real data.

Notebook 03 carries the decisive test: **hop-AUROC**, the val AUROC of
neighbour-averaged training labels at 1/2/3/4/6/8 hops. If it rises past 1 hop,
depth genuinely buys accuracy and the exit policy has something to learn. If it
is flat, §6.2's synthetic failure has reproduced on real data and the project's
central claim needs restating. This is the number to look at first.

## 5d. First real-data results (2026-08-27)

Notebooks 00-03 executed end to end on the filtered 17,663-patient cohort.

### The graph carries signal — depth is worth studying

`hop-AUROC`: val AUROC of neighbour-averaged *training* labels at k hops
(val/test labels are never aggregated).

| hops | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| val AUROC | 0.811 | **0.836** | 0.830 | 0.821 | 0.811 | 0.804 |

It **rises from 1 to 2 hops, peaks at 2, then decays** — information genuinely
propagates, and over-smoothing sets in after that. This is the opposite of the
synthetic failure in §6.2, where layer 1 already saturated. Adaptive depth is a
coherent thing to study on this cohort.

Supporting numbers: edge homophily 0.862 vs a 0.808 chance baseline for this
class balance (excess +0.055); P(neighbour died | patient died) = 0.331 versus
0.077 for survivors, a 4.3x ratio.

### ...but the peak is at 2 hops, and `max_layers` is 10

This matters for how the efficiency claim is framed. With a useful depth of ~2-3
and a budget of 10, most of the propagation budget is dead weight, and a large
"depth saving %" becomes trivially achievable — it measures the oversized budget
rather than a clever policy. **Reduce `model.max_layers` to ~4-6 before quoting
any compute-saving figure**, and report saving against a budget justified by the
hop curve.

### Baseline floors from notebook 02 (val AUROC, logistic regression per block)

| block | cols | val AUROC |
|---|---|---|
| ALL | 542 | **0.870** |
| vitals | 84 | 0.818 |
| labs | 168 | 0.799 |
| treatment | 27 | 0.757 |
| demographics | 33 | 0.735 |
| comorbidity | 19 | 0.556 |

A linear model on all blocks reaches 0.870. That is the honest floor the GNN
must clear. Note the 1-hop neighbour-average alone reaches 0.811 with no
learning at all. Comorbidity at 0.556 is the leakage-control cost from §5b.

### Graph configuration findings

- **Larger k is better** for 1-hop signal: k=5 -> 0.761, k=15 -> 0.811,
  k=40 -> 0.841 (cosine). Cosine beats `euclidean_rbf` at every k.
- **Multi-view fusion helps**: fusing the five per-block kNN graphs gives 1-hop
  val AUROC 0.846 versus 0.811 for the single all-feature graph at k=15. Caveat:
  the fused graph has 1.97M edges against 426k, and single-graph k=40 (1.11M
  edges) reaches 0.841 — so much of the gain may be density rather than the
  fusion itself. Compare at matched edge counts before claiming fusion wins.
- Per-view 1-hop AUROC: vitals 0.786, labs 0.743, treatment 0.720,
  demographics 0.664, comorbidity 0.520.
- Graph is well formed: mean degree 24.1, no isolated nodes, 3 components with
  the largest holding 99.7% of patients.

## 5e. `criticalness` leaks the outcome — use `severity` for the stratum

`_criticalness` is `0.35*los + 0.35*burden + **1.2*died**`. The death flag is
weighted 1.2 against two standardised (unit-variance) terms, so it dominates:
**AUROC 0.969 against mortality** on this cohort.

That is fine for an auxiliary *training* target — it is a label. It is
disqualifying for an *evaluation stratum*: "top decile of criticalness" is
approximately "the patients who died", which reproduces the §6.1 degeneracy that
the `high_risk_analysis` rewrite exists to remove. The fix in §5b would have been
undone by the stratum it was given.

**New `mimic_iv._severity`**: a mini-SOFA from the observation window only —
vasopressors, invasive ventilation, CRRT, intervention count, lactate max,
creatinine max, BUN max, GCS motor/verbal, SBP min, SpO2 min, respiratory rate
max. Robust-z-scored (median/IQR, clipped at +/-4), oriented so larger is
sicker, averaged over available components. No outcome, no full-stay data.

`evaluate_model` and `explain.exit_depth_vs_difficulty` now prefer
`severity` > `true_risk` > `criticalness`, with the last documented as a
last resort. `criticalness` is unchanged and still available as the auxiliary
target.

Validated on the cohort:

| | criticalness | severity |
|---|---|---|
| AUROC vs mortality | 0.969 | **0.785** |
| correlation with criticalness | — | 0.628 |
| top-decile mortality | ~ the deaths | 0.388 (3.6x enriched over 0.108) |

Mortality across severity deciles: 0.014, 0.023, 0.028, 0.046, 0.062, 0.061,
0.110, 0.144, 0.202, 0.388 — **near**-monotone (deciles 5→6 invert, 0.0618 →
0.0611), strongly enriched at the top, and not identical to the label.
**[REPRO 2026-09-01]** This sentence previously read "monotone"; the inversion is
visible in the numbers printed immediately before it, and was found independently
by two reproduction agents via different code paths. The stratum remains usable —
the inversion is one adjacent pair deep inside the low-risk region. That combination is what makes the safety metric
informative rather than a restatement of the threshold rule.

Also fixed in notebook 02: the measurement-intensity AUROC was printed with a
sign flip (0.297 where it should read **0.703**). More measurement predicts
death — median 306 features measured for those who died versus 281 for those
who survived. Process-of-care signal is real here and needs stating whenever
feature importance is discussed.

## 5f. Baselines on real data — XGBoost wins (2026-08-27)

Test-set results on the 17,663-patient cohort, single seed, threshold at 90%
operating sensitivity:

| model | test AUROC | AUPRC |
|---|---|---|
| **XGBoost** | **0.8951** | 0.5592 |
| SAS-GNN full depth | 0.8855 | 0.5586 |
| logistic regression | 0.8803 | 0.5320 |
| GCN (2 layer) | 0.8659 | 0.5127 |
| GAT (2 layer) | 0.8658 | 0.5040 |
| neighbour average (no learning) | 0.8246 | 0.4523 |

**The tabular baseline beats every graph model.** (The single-seed run also
showed GCN/GAT below logistic regression — **§5i refutes that**; across 5 seeds
the three are indistinguishable, and SAS-GNN significantly beats logistic
regression.) This is §6.2's failure mode reproducing on real data —
but the reason is not the one §6.2 described, and the difference matters.

### The graph signal is real but redundant

Notebook 03 established genuine relational structure: 1-hop neighbour-averaging
reaches 0.811 val AUROC, rising to 0.836 at 2 hops. That is not nothing. Yet the
GNNs do not convert it into an edge over tabular models.

The likely reason is structural, and it is worth being blunt about it in the
write-up: **the kNN graph is built from the feature matrix itself**. A graph
derived from `X` cannot inject information that is not already in `X`. Its only
contribution is inductive bias — smoothing predictions over feature-space
neighbourhoods. With 17.6k patients and 542 features, gradient boosting extracts
more from those features directly than message passing recovers by smoothing
them.

This is a real limitation of the patient-similarity-graph framing, not a tuning
failure, and no amount of `k` or depth tuning escapes it. Options that would
change the picture, in rough order of honesty:

1. **Edges from a source outside `X`** — shared clinicians, units, transfer
   paths, temporal adjacency, or diagnosis-code overlap held out of the feature
   matrix. Then the graph carries genuinely new information.
2. **Temporal node features** (the plan's extension): sequences rather than
   first-24h aggregates, where a GNN over patient trajectories has more to do.
3. **Report the result as-is** and reframe the contribution as compute/latency
   under an accuracy *parity* constraint rather than an accuracy gain.

### What this means for the project's claim

The early-exit contribution can still stand, but it must be framed as **an
efficiency result at near-parity accuracy**, not as an accuracy win. Specifically
`sasgnn_full_depth` (0.8855) is within ~0.01 AUROC of XGBoost (0.8951) — close
enough that "matches a strong tabular baseline at a fraction of the propagation
cost" is defensible, while "beats it" is not.

Caveat on one row: `neighbour_average` shows sensitivity 1.000 / specificity
0.000 because its predicted scores are coarse (ratios over ~24 neighbours), so
the 90%-sensitivity threshold lands at the bottom of its range. Its AUROC is
meaningful; its threshold-based columns are not.

These numbers were single-seed; **§5i supersedes them** with 5-seed means and
paired tests, and corrects two claims.

## 5g. The multi-exit result — depth buys nothing (2026-08-27)

Notebook 05, single seed, `max_layers: 10`, `ponder_cost: 0.01`.

EEGNN test AUROC **0.8816** (AUPRC 0.5395), against SAS-GNN full depth 0.8855
and XGBoost 0.8951. Exit distribution is non-trivial and bimodal — 33.6% halt at
layer 4, 28.7% run to layer 10 — mean depth 6.30/10, "37.0% depth saved".

### The number that matters: per-exit AUROC is flat

| exit layer | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| test AUROC | 0.8823 | 0.8810 | 0.8768 | 0.8810 | 0.8817 | **0.8804** |

**Layer 1 is as accurate as layer 10.** Every exit sits at ~0.88, within noise of
each other and of the full-depth model. Propagation contributes essentially
nothing: the encoder MLP over 542 features already reaches the ceiling, which is
the same ceiling logistic regression hits (0.8803).

This makes the efficiency claim vacuous *in the direction opposite to the one we
worried about*. The problem is not that early exit costs accuracy — it is that a
trivial policy exiting **everyone at layer 1** would lose nothing and save 90%,
so a learned policy averaging depth 6.3 for a 37% saving is strictly worse than
the trivial baseline. Any compute-saving figure quoted from this configuration
would be measuring an oversized budget, not a useful policy.

Consistent with §5f: the graph adds nothing beyond the features it was built
from, so there is nothing for depth to accumulate.

### Training converges almost immediately

Best val AUROC at **epoch 10**, early stop at 61. A model that is effectively an
MLP on tabular features has little to learn from 300 epochs of propagation.

### The §6.1 safety-metric fix is working

| | value |
|---|---|
| stratum | top 10% of severity (n=354, prevalence 0.367) |
| high-risk miss rate | **0.0154** |
| low-risk miss rate | **0.1434** |

Two different, informative numbers where the old metric returned a constant
0.104 for every model. The model misses 1.5% of deaths among the sickest decile
versus 14.3% elsewhere — clinically the right direction, and now measurable.

### Two genuine problems it surfaces

1. **Safety ordering is inverted** (HANDOVER §6.3 reproduces on real data):
   high-risk mean depth 6.01 vs low-risk 6.33, gap **-0.32**. The sick exit
   *earlier*. Given per-exit AUROC is flat this is close to noise — depth is not
   meaningful here — but the loss is still not selective, as §6.3 predicted.
2. **Calibration is much worse on the sickest patients**: high-risk ECE
   **0.3008** vs low-risk 0.1363, after temperature scaling. A risk model that is
   least trustworthy exactly where the stakes are highest is a real deployment
   objection, and it is invisible in the pooled ECE of 0.152. This is a new
   finding, not previously recorded.

Global calibration: raw ECE 0.224 -> 0.152 after temperature scaling (T=0.377).
Still poor in absolute terms.

## 5h. Sweep and interpretability — the case closes (2026-08-27)

### AUROC is flat across the entire depth range

`ponder_cost` sweep, test AUROC against realised mean depth:

| mean depth | 1.00 | 1.70 | 4.80 | 6.30 | 8.13 | 9.53 | 9.63 |
|---|---|---|---|---|---|---|---|
| test AUROC | 0.8825 | 0.8823 | 0.8819 | 0.8816 | 0.8816 | 0.8867 | 0.8872 |

A 0.006 spread across a **10x** range of compute. The damning comparison:

- exit **everyone at layer 1**: AUROC 0.8825, 90% depth saved;
- learned policy at the default `ponder_cost: 0.01`: AUROC 0.8816, 37% saved.

The learned policy is **worse on both axes** than the trivial baseline. There is
no accuracy/compute trade-off to plot on this cohort because there is no
trade-off — depth is free to remove.

### The two loss levers are both global (confirms §6.3)

`ponder_cost` 0 -> 0.1 drives mean depth 8.13 -> 1.00.
`high_risk_halt_penalty` 0 -> 0.5 drives mean depth 1.00 -> 10.00.

They move the *same* global quantity in opposite directions. In (mean depth,
depth gap) space the two sweeps largely trace one curve, so the gap is mostly a
function of mean depth rather than something the safety term buys independently.
§6.3's diagnosis was correct.

Safety ordering is correct at only **4 of 13** configurations, and only at
shallow mean depths (~1-5). At every deep setting the sick exit earlier.

### The corrected safety metric behaves as intended

`high_risk_miss_rate` ranges 0.0154-0.0308 across the sweep — a range, where the
old definition returned exactly 0.104 at every one of those 13 points.

`high_risk_ece` sits at **0.25-0.30 in all 13 configurations** against a
low-risk ECE near 0.14. The model is roughly twice as badly calibrated on the
sickest decile, robustly, regardless of depth policy. This is the most
deployment-relevant finding in the whole run and it is invisible in pooled ECE.

### Interpretability (notebook 07)

SHAP on the XGBoost surrogate ranks: age (0.347), urine output (0.332), GCS
motor (0.308), GCS verbal (0.282), respiratory rate (0.233), CVICU admission
(0.161), BUN (0.144). That reads as real ICU medicine, which is the main
positive result of the session. Note `vit_gcs_eye_count` and
`vit_glucose_chart_count` appear in the top 25 — measurement intensity as
predicted in §5e.

**GAT attention is uniform**: mean 0.0395 on same-outcome edges vs 0.0418 on
different-outcome edges, ratio **0.944** — very slightly *against* useful
neighbours. The model is not using the graph for retrieval, only as a smoother.

**Depth does not track difficulty**:

| statistic | value | wanted |
|---|---|---|
| corr(depth, confidence margin) | +0.048 | negative |
| mean depth when correct / incorrect | 6.36 / 6.18 | incorrect higher |
| corr(depth, severity) | +0.102 | strongly positive |
| depth by severity quartile | 5.19, 6.47, 7.16, 6.38 | monotone rising |

Depth is essentially arbitrary — expected, given per-exit accuracy is flat.

## 5i. Multi-seed results — §7's standing requirement, discharged (2026-08-28)

Five seeds (42-46), same splits per seed so every comparison is **paired**.
`processed/multiseed.json` holds the per-seed values.

### Test AUROC, mean ± std over 5 seeds

| model | mean | std |
|---|---|---|
| **XGBoost** | **0.8958** | 0.0023 |
| SAS-GNN full depth | 0.8888 | 0.0056 |
| EEGNN multi-exit | 0.8861 | 0.0054 |
| logistic regression | 0.8744 | 0.0042 |
| GCN | 0.8717 | 0.0072 |
| GAT | 0.8701 | 0.0052 |

### Paired t-tests (df=4, critical |t| = 2.776)

| comparison | difference | t | verdict |
|---|---|---|---|
| XGBoost − SAS-GNN full | +0.0070 ± 0.0042 | 3.68 | **significant** |
| XGBoost − EEGNN | +0.0098 ± 0.0046 | 4.75 | **significant** |
| SAS-GNN full − logistic regression | +0.0144 ± 0.0091 | 3.57 | **significant** |
| SAS-GNN full − EEGNN | +0.0028 ± 0.0023 | 2.67 | not distinguishable |
| logistic regression − GCN | +0.0027 ± 0.0111 | 0.55 | not distinguishable |
| logistic regression − GAT | +0.0043 ± 0.0092 | 1.04 | not distinguishable |
| GCN − GAT | +0.0016 ± 0.0025 | 1.38 | not distinguishable |

### Two corrections to §5f, which was single-seed

1. **"GCN and GAT fall below plain logistic regression" does not hold.** t=0.55
   and t=1.04 — the three are indistinguishable across seeds. Seed 42 happened to
   be a favourable draw for logistic regression (0.8804 against a 0.8744 mean,
   +1.4 sd). Do not repeat that claim.
2. **SAS-GNN full depth *does* significantly beat logistic regression**
   (+0.0144, t=3.57). So the graph-plus-depth architecture adds real value over
   a linear model on the same features. What it does not do is beat gradient
   boosting. That is a materially fairer statement of the result than §5f's
   original framing, which understated what the architecture achieves.

**What survives unchanged:** XGBoost significantly beats every graph model
(t=3.68 and t=4.75). The headline conclusion stands.

## 5j. Depth budget — the saving was an artefact of the oversized budget

EEGNN trained at four budgets, 5 seeds each. This tests the §5g suspicion
directly.

| max_layers | test AUROC | realised depth | depth saved | layer-1 AUROC |
|---|---|---|---|---|
| 2 | 0.8899 ± 0.0023 | 2.000 ± 0.000 | **0.0%** | 0.8901 ± 0.0030 |
| 4 | 0.8890 ± 0.0059 | 4.000 ± 0.000 | **0.0%** | 0.8894 ± 0.0063 |
| 6 | 0.8882 ± 0.0061 | 6.000 ± 0.000 | **0.0%** | 0.8875 ± 0.0062 |
| 10 | 0.8861 ± 0.0054 | 7.96 ± 1.36 | 20.4% ± 13.6% | 0.8863 ± 0.0055 |

Three findings, all robust across seeds:

1. **At any sensible budget the policy saves nothing at all.** Realised depth
   equals the budget exactly at 2, 4 and 6 layers — every patient runs to full
   depth. Early exit only ever engages when the budget is far larger than the
   useful depth, i.e. it is cutting dead weight, not making clinical decisions.
2. **The single-run "37.0% depth saved" was a favourable draw.** The mean at
   budget 10 is 20.4% with a standard deviation of 13.6 — enormous relative
   variance. Do not quote 37%.
3. **Accuracy is slightly *better* at smaller budgets** (0.8899 at 2 layers vs
   0.8861 at 10), consistent with over-smoothing past the 2-hop peak from §5d.
   More depth is mildly harmful here.

And **layer-1 AUROC matches the full model at every budget** — 0.8901 vs 0.8899
at budget 2, 0.8863 vs 0.8861 at budget 10. The §5g finding was not a
single-run artefact.

### The high-risk calibration gap is robust

| max_layers | high-risk ECE | low-risk ECE | ratio |
|---|---|---|---|
| 2 | 0.2369 ± 0.0201 | 0.1057 ± 0.0099 | 2.24x |
| 4 | 0.2352 ± 0.0218 | 0.1048 ± 0.0107 | 2.24x |
| 6 | 0.2343 ± 0.0213 | 0.1086 ± 0.0072 | 2.16x |
| 10 | 0.2267 ± 0.0405 | 0.1117 ± 0.0170 | 2.03x |

Stable at roughly **2.2x worse calibration on the sickest decile**, across every
budget and every seed, after temperature scaling. Combined with §5h's stability
across 13 sweep configurations, this is the most robust and most
deployment-relevant finding of the whole project — and it is invisible in
pooled ECE.

## 5k. Where we sit against the reference literature (2026-08-28)

Read of `references/Bucket 1` (14 problem-statement papers) and `Bucket 2`
(11 implementation papers).

### The method paper is being applied outside its stated regime

`2505.18088` (Di Francesco et al., *Early-Exit Graph Neural Networks*) is the
source of SAS-GNN + EEGNN. It evaluates on **heterophilic graphs (Questions) and
long-range benchmarks (Peptides-func/struct, LRGB)** — never on EHR or patient
similarity. Two lines from it matter for us:

- *"Even when not outperforming the strongest baselines, EEGNNs consistently
  deliver favorable accuracy-efficiency trade-offs."* The authors do **not**
  claim accuracy wins. Our §5i result is the expected kind of outcome.
- *"When depth itself is unreliable, early exits merely act as a safeguard,
  dedicating time to easy inputs, but ignoring harder samples, which contrasts
  with the core purpose of early-exit."*

Our graph is **homophilous** (0.862 against a 0.808 chance baseline) and
**short-range** (hop-AUROC peaks at 2). That is the opposite of the regime the
method targets. So §5g/§5j are not evidence the method is broken — they are
evidence the premise it needs is absent in this task. Worth noting too that on
Peptides-func the paper's own model "predominantly exits after just two layers",
so shallow exit distributions are documented normal behaviour, not pathology.

### Our numbers are consistent with the well-controlled papers

| paper | task / graph | graph model | best non-graph | gap |
|---|---|---|---|---|
| `2411.19742` | heart failure, MIMIC-III, kNN patient similarity | GT **0.7925** | GBT / RF 0.7755 | +0.017 |
| `3765612.3767211` | ICU LOS, homogeneous similarity graph | SAGE_2L AUC **86.19** | XGBoost 85.88 | +0.3 pt |
| **ours** | mortality, MIMIC-IV, kNN patient similarity | SAS-GNN 0.8888 | XGBoost **0.8958** | −0.007 |

The two closest published setups find the graph model **essentially tied** with
tree ensembles. `2411.19742` says so explicitly: *"similar AUPRC values for GT
(0.5168) and RF (0.5132) suggest that RF, with appropriate threshold tuning,
could also yield a higher F1 and serve as a resource-efficient alternative."*
Our result is one small step further along the same line, not an outlier.

In absolute terms our pipeline is not underperforming: `2308.12575`
(hypergraph, ICU risk) reports 0.688-0.757 AUROC at 24 h and 0.787 at 48 h;
we get 0.896 at 24 h. Different cohort and task, so not a like-for-like
comparison, but it does rule out "our model is simply weak".

### The one paper that disagrees, and why to be careful with it

`2508.00615` (SBSCGM — the project's other reference) reports **AUC-ROC 0.942**
on 6,000 MIMIC-III ICU stays, against MLP 0.810, LR 0.799, RF 0.825.

Two reasons to treat that gap cautiously rather than as a target to chase:

1. **Their non-graph baselines are weak relative to ours.** Our logistic
   regression reaches 0.8744 and XGBoost 0.8958 — well above their LR (0.799)
   and RF (0.825). A +0.12 graph gain over RF is far larger than anything else
   in this literature, and the most likely explanation is baseline strength, not
   graph strength.
2. **Their multi-task target is `criticalness`.** We found (§5e) that a
   criticalness score blending LOS, intervention burden and discharge
   disposition correlates with mortality at **AUROC 0.969** — it is very nearly
   the label. Training multi-task against such a signal could inflate mortality
   AUROC. This is a question to check in their paper, not an established flaw,
   but it is exactly the failure mode we caught in our own code. Note
   `diagnostics-15-02825` reports 0.96 for mortality *and* explicitly records
   that a "leakage-allowed" variant produced >0.98, confirming censoring was
   needed — that paper was alert to the same risk.

### Where our implementation diverges from its own references

- **Hybrid similarity is specified but not implemented.** SBSCGM uses "a hybrid
  similarity measure combining cosine-based feature similarity and
  Jaccard-based structural similarity". We implement cosine and
  `euclidean_rbf` only. `mimic_iv._unpack` computes `meta["binary_cols"]` for
  exactly this purpose and **nothing consumes it** — grep confirms one
  definition, zero uses. Cheapest outstanding gap to close.
- **Static vs dynamic graph.** SBSCGM's graph "evolves with new ICU data"; ours
  is built once and cached.
- **Backbone.** SBSCGM fuses GCN + GraphSAGE + GAT; we use the SAS-GNN backbone
  from the other paper. The project is a hybrid of two papers and currently
  implements neither in full — worth stating plainly in any write-up.

### Bucket 2 speaks directly to our open problems

- `3469116.3470012` (Laskaridis et al., adaptive inference) on confidence-based
  exit policies: *"the overconfidence of certain deep models... Solutions
  include calibrating the softmax output values."* Our §5j finding — 2.2x worse
  calibration on the sickest decile — is precisely this failure, sitting on the
  stratum the exit policy most needs to be right about. It is a known,
  named problem with known mitigations, which makes it addressable rather than
  fatal.
- The same survey notes exit **number and placement** should be co-optimised
  with the policy. We attach a head at every layer and tune only the budget.
- `1-s2.0-S0169260722004151` (Deep Dynamic Patient Similarity, ICU) and
  `1-s2.0-S1386505619310925` both *learn* patient similarity rather than taking
  raw feature kNN. That is the principled version of §7's "edges from outside
  X" recommendation, and there is a directly applicable method for it.

## 5l. Scored against the actual project spec (2026-08-28)

Source: `../Early_Exit_GNN_Clinical_Review_1.md` (BCSE497J Project 1).

**Correction to how §5f-§5j were framed.** The spec's aim is *"to optimize the
trade-off between predictive accuracy and computational efficiency"* with
clinical safety constraints. It nowhere claims to beat state-of-the-art
accuracy. Framing "XGBoost wins on AUROC" as the headline failure was harsher
than the spec warrants — on the project's own terms, accuracy only has to be
*clinically adequate*, and 0.89 AUROC is that. The real scoring is below.

### Objective-by-objective

| # | Objective | Status | Evidence |
|---|---|---|---|
| 1 | Improve processing speed | **MET** (§5m) | safety gate saves 34.2% compute at alpha=0.01 with no accuracy change; the learned policy saved 0% |
| 2 | Ensure clinical safety | **MET** (§5m) | safety ordering correct 3/3 seeds; 4.7% of the top severity decile exits early vs 51.0% of the rest; miss-rate budget holds on held-out data |
| 3 | Patient similarity graphs | **met** | 5 blocks (demographics, comorbidity, labs, vitals, treatment), multi-view fusion wired to real view membership |
| 4 | SHAP explanations | **met** | age, urine output, GCS, respiratory rate, BUN — clinically sensible (§5h) |

### Objective 1 is the subtle one

The spec wants a model "fast and light enough for real-time hospital use". We
have that: 2 layers, 0.8899 AUROC, sub-second inference. **The objective is
satisfied.** What is *not* satisfied is the claim that adaptive depth is how you
get there — §5g/§5j show one layer is as accurate as ten, so there is nothing
for an adaptive policy to adapt to on this task. The speed win is real; the
attribution to early exit is not.

### Objective 2 is the one that actually matters

This is the project's stated novelty: *"Existing work improves interpretability
and survival prediction but never defines clinically safe early-stop rules for
low-risk patients."* Two findings block it:

1. **Safety ordering inverts.** The sick exit *earlier* at most operating
   points (§5h) — the exact opposite of the design intent.
2. **"Verified low-risk" is not verifiable with this model.** Deciding a patient
   is safe to exit early requires a calibrated risk estimate, and calibration
   on the top severity decile is 2.2x worse than elsewhere (§5j), stable across
   every seed and budget. You cannot build a safe early-stop rule on a
   confidence signal that is least reliable precisely where it matters.

Objective 2 is where the remaining work should go. It is also the most
publishable part: the corrected safety metric (§5e) plus the stratified
calibration finding is machinery the cited literature does not have.

### Two gaps against the spec's own text

- **C-index is specified and not implemented.** The abstract lists evaluation
  "across clinical metrics (AUROC, AUPRC, C-index)". `grep` finds no
  concordance index anywhere in `src/`. It needs a time-to-event formulation,
  which ties to the untouched survival/readmission extension in §7.
- **The literature review omits the closest prior art.** It reviews GNN-surv,
  ICU-LOS, CliqueFluxNet, DGAT and AGFN, but does **not** cite Di Francesco et
  al., *Early-Exit Graph Neural Networks* (`2505.18088`) — the paper whose
  SAS-GNN backbone and Gumbel exit heads this project implements. A Scopus
  reviewer will find it immediately. The research gap also needs restating: it
  currently says GNNs "lack confidence-based early exits", which that paper
  disproves. The defensible gap is narrower and still true — **early-exit GNNs
  exist, but have never been applied to clinical risk stratification under
  safety constraints, and no prior work defines or validates a safe early-stop
  rule.** That is exactly what §5e's metric does.

## 5m. The redesign: calibrated risk-gated early exit (2026-08-28)

Built to satisfy the spec's four objectives rather than the paper's benchmark.
New module `src/safe_exit.py`.

### The problem being solved

The learned Gumbel policy halts on raw **confidence**. Measurements say that is
unsafe here: calibration on the top severity decile is ~2.2x worse than
elsewhere (§5j) and the depth ordering *inverts* (§5h). Objective 2 — "only
verified low-risk patients exit early" — cannot be met by a confidence rule
whose confidence is least reliable at high risk.

### The rule

Decide on **calibrated risk**, and only ever exit downward:

    exit at layer L  <=>  p_calibrated(death at exit L) < tau_L

`tau` is fitted on validation, per layer, as the *largest* threshold whose
realised death rate among released patients stays within a safety budget
`alpha`. Layers are fitted in order, each seeing only the patients its
predecessors declined to release.

### Why this fixes the safety objective rather than dodging it

The gate only fires on **low** predicted risk, and the calibration failure is
concentrated at **high** risk. So the gate operates precisely in the region
where the model is trustworthy, and high-risk patients are routed to full depth
*by construction* rather than by hoping the policy learns to. The
miscalibration stops being a safety hazard without first having to be fixed.

It also makes safety a dial the clinician sets (`alpha`) rather than an emergent
property of a loss weight — and `alpha` is validated on held-out data, so the
guarantee is measured, not asserted.

Deliberately **not** a learned policy. A rule a clinician must audit should be a
threshold on a calibrated probability, not the argmax of a Gumbel head. Having
measured that the learned policy inverts the ordering, the simpler auditable
rule is the better trade. This is consistent with the adaptive-inference survey
in Bucket 2 (`3469116.3470012`), which notes confidence-based exits suffer from
overconfidence and prescribes calibrating the softmax as the fix.

### Synthetic validation (before touching real data)

Releases 72.6% of patients early at 47.8% compute saved, with a **held-out**
miss rate of 0.0092 against a 0.01 budget, and prevalence among retained
patients rising to 0.41 from a 0.11 base rate — i.e. it concentrates the sick in
the full-depth path, which is the intended behaviour.

### Also closed in this pass

- **`evaluate.c_index`** — the spec's third clinical metric, previously absent.
  Harrell's concordance, chunked so the n^2 pair matrix is never materialised.
  Time axis is ICU length of stay with in-hospital death as the event, so
  survivors are censored at discharge; that is an approximation and is
  documented as one.
- **Hybrid cosine + Jaccard similarity** (`graph.similarity: hybrid`), which
  SBSCGM specifies and which `meta["binary_cols"]` was already being computed
  for with zero consumers. Cosine over the continuous block, Jaccard over the
  binary block (comorbidity flags, interventions, care unit), mixed by
  `graph.hybrid_beta`.
- **`max_layers` reduced to 4** for these runs, per §5j.

### Results on the real cohort (3 seeds, max_layers=4, cosine graph)

| variant | AUROC | C-index | mean depth | compute saved | early-exit share | miss rate among exited | high-risk early-exit | depth gap | ordering OK |
|---|---|---|---|---|---|---|---|---|---|
| learned Gumbel | 0.8868 | 0.811 | 4.00 | **0.0%** | — | — | — | 0.00 | **0 / 3** |
| safe gate, alpha=0.005 | 0.8870 | 0.811 | 3.18 | 20.4% | 27.8% | **0.0041** | 0.9% | +0.88 | **3 / 3** |
| safe gate, alpha=0.01 | 0.8866 | 0.811 | 2.63 | **34.2%** | 46.4% | **0.0105** | 4.7% | +1.38 | **3 / 3** |
| safe gate, alpha=0.02 | 0.8866 | 0.811 | 2.16 | 46.1% | 61.8% | **0.0179** | 11.4% | +1.68 | **3 / 3** |
| XGBoost (reference) | 0.8954 | 0.816 | — | — | — | — | — | — | — |

Five things this establishes:

1. **The learned policy saves nothing at a sensible budget** — 0.0% at
   `max_layers=4`, confirming §5j on a third independent run.
2. **The gate saves 20-46% at no accuracy cost.** AUROC moves by 0.0004 across
   the whole safety range and matches the learned policy exactly.
3. **The safety budget holds on held-out data.** alpha 0.005 / 0.01 / 0.02
   realise 0.0041 / 0.0105 / 0.0179. The dial does what it says.
4. **The safety ordering is fixed.** Correct in 3/3 seeds at every alpha, versus
   0/3 for the learned policy. Depth gap goes from 0.00 to +1.38.
5. **Only low-risk patients are released.** At alpha=0.01, 4.7% of the top
   severity decile exits early against 51.0% of everyone else — a 10x
   separation, which is the objective stated almost verbatim.

Per-seed spread at alpha=0.01: miss rate 0.0128 / 0.0120 / 0.0066, compute
saved 35.7% / 38.2% / 28.8%.

**Hybrid similarity does not help.** 0.8853 against 0.8866 for cosine, and 14 s
against 6 s to build. It closes the gap with the SBSCGM specification and should
be reported, but it is not an improvement on this cohort.

### What this does *not* fix

`high_risk_ece` remains ~0.245. The gate **sidesteps** the calibration failure
rather than repairing it: by only ever releasing low-risk patients it avoids
relying on the miscalibrated region. That is sound for the exit decision, but if
the predicted probability is shown to a clinician as a *risk score* for the
sickest patients, it is still poorly calibrated and should not be presented as
trustworthy there. Fixing it properly needs a stratum-aware calibration step.

Also honest: the alpha=0.01 run realises 0.0105 on test, marginally above
budget. The threshold is fitted on validation, so this is an empirical guarantee,
not a finite-sample one. A conformal/split-calibration treatment would give a
distribution-free bound and is the natural next step.

Finally, the gate does **not** show that depth helps high-risk patients — §5j
says it does not. Its contribution is a compute saving with an *auditable
certificate* over who may be released, which is the safety objective. Do not
claim the retained patients benefit from the extra layers.

## 5n. The staged hybrid cascade — early exit stops needing an excuse (2026-08-29)

New modules `src/tabular.py`, `src/cascade.py`, `src/hybrid.py`; runner
`scripts/run_cascade.py`; notebook `08_hybrid_cascade`; 11 new tests
(24/24 pass). Results in `results/headline.json` (5 seeds, 3 stages) and
`results/{cs,gnn}_{knn,leaf}.json` (the graph-stage ablation, 3 seeds each).

### The diagnosis this is built on

§5g–§5j did not show early exit failing. They showed it being asked to choose
between options that were all the same: per-exit AUROC was flat from layer 1 to
layer 10, so the gate had nothing to trade, and at any sensible budget it saved
exactly 0%. The trivial "exit everyone at layer 1" policy **dominated** the
learned one on both axes. §5f explains why the exits were flat — the graph is
built from `X`, so propagation adds nothing to it.

The fix is not a better halting policy. It is to cascade over something whose
stages actually differ. So: cascade over **information**, not depth.

| stage | adds | cost to obtain | cols |
|---|---|---|---|
| 0 · bedside | demographics + vitals | nothing — already at the bedside | 187 |
| 1 · chart | + treatment + comorbidity | nothing, but needs the record linked | 234 |
| 2 · labs | + the laboratory panel | money and turnaround time | 542 |
| (3 · cohort) | + patient-graph residual | the whole cohort in memory | — |

Each stage is an independent XGBoost over its nested feature set, so the
project's strongest measured component is now the backbone rather than the
baseline being apologised to.

### The exits are no longer flat — this is the whole result

5 seeds, paired splits, test set:

| stage | AUROC | AUPRC | ECE | cum. cost |
|---|---|---|---|---|
| 0 · bedside | 0.8624 ± 0.0047 | 0.4522 ± 0.0252 | 0.0135 | 3.8 us |
| 1 · chart | 0.8733 ± 0.0075 | 0.4924 ± 0.0232 | 0.0138 | 8.3 us |
| 2 · labs | **0.8968 ± 0.0054** | 0.5483 ± 0.0176 | 0.0149 | 14.9 us |
| XGBoost reference | 0.8965 ± 0.0018 | — | — | — |

| paired comparison | difference | t (df=4) | verdict |
|---|---|---|---|
| stage 2 − stage 0 | +0.0343 ± 0.0045 | **17.04** | significant |
| stage 1 − stage 0 | +0.0109 ± 0.0048 | **5.13** | significant |
| stage 2 − XGBoost reference | +0.0002 ± 0.0045 | 0.12 | not distinguishable |

Set against the depth cascade's per-exit AUROC (0.8823 / 0.8810 / 0.8768 /
0.8810 / 0.8817 / 0.8804 — flat within noise), the contrast is the point. The
staging also costs nothing: stage 2 is indistinguishable from the incumbent
XGBoost, so the gradient is bought without giving anything up.

### A real accuracy/compute trade-off curve exists, for the first time

§5h's most damaging sentence was "there is no accuracy/compute trade-off to plot
on this cohort because there is no trade-off". There is one now. Certified
two-sided gate, 5 seeds:

| alpha | AUROC | compute saved | resolved before labs | realised miss rate | decision agreement |
|---|---|---|---|---|---|
| 0.005 | 0.8959 ± 0.0058 | 4.5% | 6.0% | (low side not certified) | 1.000 |
| 0.01 ⚠ | 0.8939 ± 0.0059 | 10.8% *(bimodal)* | 6.0% | 0.0074 ± 0.0011 *(n=2)* | 0.998 |
| 0.02 | 0.8920 ± 0.0063 | **34.5%** | **46.2%** | 0.0081 ± 0.0021 | 0.989 |
| 0.05 | 0.8776 ± 0.0031 | 56.6% | 75.8% | 0.0316 ± 0.0104 | 0.880 |
| always stage 0 | 0.8624 ± 0.0047 | 74.7% | 100% | — | — |
| always full | 0.8968 ± 0.0054 | 0.0% | 0% | — | — |

`results/figures/cascade_frontier.png` (`scripts/plot_cascade.py`) plots this
against the two constant policies, alongside the release rates by severity
stratum.

> **⚠ [REPRO 2026-09-01] The alpha=0.01 row is not a frontier point. Do not
> quote it.** An independent 5-seed rerun released **no patients at all at
> alpha=0.01 in 5/5 seeds** (saving 4.3%, indistinguishable from the 0.005 row).
> Re-reading this run's own `results/headline.json` shows why: the release side
> fired in only **2 of 5 seeds** here too —
> `released_frac = [0.0, 0.356, 0.366, 0.0, 0.0]`.
>
> - "10.8% saved" is the mean of a **bimodal** quantity (± 8.04). No seed is near
>   it; seeds sit at ~4% or ~19.5%.
> - "0.0074 ± 0.0011" is an **n=2 average** — the three non-releasing seeds
>   contribute `nan` — printed in a table whose every other row is n=5.
>
> The cause is arithmetic and is already derived two subsections below: a
> Clopper–Pearson bound needs m >= ln(0.05)/ln(1-alpha) clean releases, i.e.
> **m >= 299 at alpha=0.01**, and the stage-0 ranking never supplies more than a
> 220-long death-free prefix (98–220 across seeds). The two releasing seeds cleared
> the bar at *stage 1* (`taus_lo = [0.0, 0.0084, 0.0]`), which is a knife edge —
> a 2-thread vs 4-thread XGBoost numerical perturbation flips it.
>
> **alpha=0.005 and alpha=0.02 are unaffected**; the curve between them is real.
> Report alpha=0.01 as "release side fires in 2/5 seeds" or drop the row.

Monotone in both columns, and **the gate is no longer dominated**. §5h's
comparison — learned policy worse than the trivial baseline on accuracy *and*
on saving — does not reproduce: at alpha=0.02 the gate gives up 0.0048 AUROC
against always-full to save 34.5% of compute, and buys 0.0296 AUROC over
always-stage-0. Those are frontier points, not a policy losing to a constant.

### Lead with the lab panel, not the microseconds

Total inference cost is 14.9 us per patient. Saving 34% of that is 5 us and is
clinically meaningless as compute; quoting it as the headline would repeat §5j's
mistake of measuring an oversized budget. **The saving that matters is that
46.2% of patients are resolved without the model ever reading the lab block** —
money and hours of turnaround, not FLOPs.

Stated carefully: this is a claim about the model's *input requirement*, not a
counterfactual about clinical practice. Every patient in MIMIC had labs drawn;
what is measured is that for 46.2% of them the cascade reached a decision that
agrees with the full model 98.9% of the time without needing those columns.

### The safety objective, and a certificate rather than a hope

The gate is **two-sided**. Releasing downward on low calibrated risk is §5m's
rule. Escalating upward on high calibrated risk is new, and it answers the
criticism arXiv:2505.18088 levels at its own method — exits that "dedicate time
to easy inputs, but ignore harder samples". Here the ambiguous middle gets the
expensive stages by construction. The budgets are deliberately asymmetric:
alpha bounds the death rate among *released* patients (safety), beta bounds the
survival rate among *escalated* patients (resource). Escalated patients realise
62.0% mortality against a 10.8% base rate — 5.7x enrichment, so the high side is
firing on patients whose answer really is already decided.

This is the "I Don't Know" cascade (Wang et al., UAI 2018 — commit to a class
or defer to a costlier component) with the confidence score replaced by a
calibrated risk, which is what the adaptive-inference survey in Bucket 2
prescribes for exactly the overconfidence failure §5j measured.

Only low-risk patients are released, robustly:

| alpha | top severity decile released | everyone else released | separation |
|---|---|---|---|
| 0.01 | 1.1% | 15.9% | **14.8x** |
| 0.02 | 2.7% | 44.3% | **16.7x** |
| 0.05 | 17.9% | 75.5% | 4.2x |

§5m reached 10.9x (4.7% vs 51.0%); this is better, and it holds across 5 seeds.

**The certificate.** §5m picked the largest threshold whose *observed* miss rate
on validation was within budget, and alpha=0.01 then realised 0.0105 on test —
an empirical guarantee, slightly violated. Thresholds are now chosen against a
one-sided Clopper–Pearson upper bound at 95% confidence, fitted on a selection
split held out from the calibration split. Certified realisations: 0.0074
(budget 0.01), 0.0081 (0.02), 0.0316 (0.05) — all inside, with room.

The certificate is not free and the cost is informative:

- At **alpha=0.005 it certifies no releases at all** — the 6.0% resolved before
  labs at that budget are high-side *escalations*, which carry the loose beta
  budget rather than alpha. With 0 deaths among m releases the
  bound is 1 − 0.05^(1/m), so m >= 598 clean releases are needed; the 2.1k
  selection split cannot supply them. That is a concrete statement about how
  much validation data a 0.5% safety guarantee requires, and it is better to
  read it off the bound than to discover it in deployment.
  **[REPRO 2026-09-01]** The same bound gives **m >= 299 at alpha=0.01** and
  m >= 149 at alpha=0.02. The selection split cannot supply 299 either — its
  longest death-free prefix is 98–220 across seeds — so **alpha=0.01 is also
  uncertifiable at stage 0**, which is why the row above is bimodal. This
  paragraph originally applied the argument only to alpha=0.005; it kills
  alpha=0.01 too. alpha=0.02 clears comfortably (~1,000 certified releases in
  5/5 seeds).
- It is **markedly more stable**. At alpha=0.02 the certified gate saves
  34.5% ± 1.3 against the empirical gate's 46.5% ± 3.2, and at alpha=0.05 the
  empirical gate's saving swings 53.5% ± 26.8 across seeds.

The empirical rule happened not to be violated in this run (0.0082 / 0.0147 /
0.0473 against 0.01 / 0.02 / 0.05), so the certificate should be sold as
variance control plus a finite-sample bound, not as a rescue from an observed
failure.

### Calibration: the §7 open issue, answered — and not the way it was posed

HANDOVER §7 carried "calibration on the high-risk stratum (ECE ~0.29); the
stratum may need its own calibration". Both halves were tested.

| | pooled ECE | high-risk ECE | low-risk ECE | ratio |
|---|---|---|---|---|
| EEGNN + temperature scaling (§5j) | 0.152 | 0.227–0.237 | 0.105–0.112 | ~2.2x |
| cascade + global isotonic | **0.0164** | **0.0678** | 0.0138 | 4.9x |
| cascade + severity-stratified isotonic | 0.0255 | 0.0847 | 0.0226 | 3.8x |

Two conclusions, and the second is the useful one:

1. **Stratified calibration is not the fix.** It narrows the *ratio* to 3.8x but
   makes the absolute high-risk error worse (0.0847 vs 0.0678) and costs AUROC
   (0.8877 vs 0.8939) — per-stratum isotonic introduces ties that damage
   ranking. Do not pursue it further.
2. **Absolute high-risk calibration error falls ~3.5x** (0.23 → 0.068) while the
   ratio gets *worse*, because the low-risk stratum improved even more. The
   ratio was the wrong statistic to have been tracking; 0.068 and 0.23 are
   different deployment regimes and the ratio does not distinguish them.

Caveat on attribution: this comparison changes two things at once — the model
(EEGNN → boosted cascade) and the calibrator (temperature → isotonic on a
held-out split). Isotonic on held-out data is the more likely cause, since
temperature scaling is a one-parameter monotone map and cannot fix a
stratum-specific distortion, but the run does not separate the two.

### The graph stage does not earn its place — the sharpest test yet

Stage 3 was posed as a **residual** question rather than a rival one: hand the
graph the boosted margin and ask what a tree ensemble structurally cannot,
because it scores one row at a time — *are patients like this one systematically
under-predicted?* The answer lives in the neighbours' residuals, and a residual
is a function of the labels, not of `X`. That is the cheap route at §7's "edges
from a source outside `X`": the edges still come from `X`, but the signal riding
along them does not. (Correct & Smooth, Huang et al., ICLR 2021, used as a
cascade stage. It needs the out-of-fold discipline in `src/tabular` — in-sample
residuals are ~0 and there is nothing to propagate.)

Four variants, 3 seeds each, against the labs stage at **0.8985 ± 0.0048**:

| graph | correction | test AUROC | cost |
|---|---|---|---|
| cosine kNN over features | Correct & Smooth | 0.8973 ± 0.0050 | 14 us |
| booster leaf co-occurrence | Correct & Smooth | 0.8973 ± 0.0039 | 12 us |
| cosine kNN over features | learned residual GNN | 0.8976 ± 0.0056 | 76 us |
| booster leaf co-occurrence | learned residual GNN | 0.8986 ± 0.0033 | 86 us |

The best is +0.0001 AUROC for a stage that on its own costs 5x the three
tabular stages combined (85.7 us against 17.1). Nothing here is a result.

The **leaf graph** is worth recording as a method even though it failed: edges
from cosine similarity over the booster's own per-tree leaf assignments, so an
edge means "the fitted model treats these two patients alike" rather than "these
two rows are close" (`tabular.leaf_onehot`). It is genuinely not a function of
raw feature distance, and it still adds nothing.

So §5f's diagnosis now survives a much stronger test than the one that produced
it. The graph fails even when it is (a) asked only for a *correction* to a
strong model rather than to compete with it, (b) given edges from the model's
learned geometry instead of from `X`, and (c) allowed to learn the correction
rather than propagate it parameter-free. **Report the patient-similarity graph
as a measured negative on this cohort, not as an open question.**

Consequently the headline cascade is the **three tabular stages**. Crediting a
compute saving for skipping a stage that does nothing is precisely the error
§5j caught in the depth policy, and it is not repeated here.

### What this changes about the project's claims

- **Objective 1 (speed) is now attributable.** §5l had to concede that the speed
  win was real but "the attribution to early exit is not". It is now: the exits
  differ by 0.034 AUROC (t=17), the trade-off curve is monotone, and the gate is
  not dominated by a constant policy.
- **Objective 2 (safe early stop) is met with a certificate**, at 14.8–16.7x
  separation between the top severity decile and everyone else, with a
  distribution-free bound on the miss rate rather than an empirical one.
- **The graph contribution should be reported as a negative result**, and it is
  a clean and well-controlled one — arguably the most defensible finding in the
  project after the safety machinery.
- **The research gap needs restating again.** §5l narrowed it to "early-exit
  GNNs exist but have never been applied to clinical risk stratification under
  safety constraints". After this run the honest version drops the GNN: *early
  exit over a heterogeneous cascade of clinical information sources, gated on
  calibrated risk with a distribution-free safety certificate, is what is new;
  the graph is not part of the contribution on this cohort.*

### Two things measured and discarded, recorded so they are not retried

1. **Boosting continuation across stages.** One additive score, each stage
   handed its predecessor's margin as `base_margin`. Prettier ("an additive
   score is natively truncatable") and worse: mock cohort test AUROC 0.7954 /
   0.8087 / 0.7939 across stages against 0.7977 / 0.8286 / 0.8318 for
   independent nested models. The base margin on training rows is out-of-fold
   while the one at inference comes from a full-train model, so each
   continuation is calibrated against a weaker base than it is applied to; using
   the fold ensemble at inference narrowed the gap without closing it.
2. **Severity-stratified calibration** — see above.

---

## 5o. Independent reproduction of all four architectures (2026-09-01)

Four agents, one architecture each, in isolated folders (`experiments/{baselines,
eegnn,graph,cascade}/`), each a full repo copy pushed to its own remote root, all
sharing one read-only cohort cache built once. Brief: `experiments/AGENTS_BRIEF.md`.
Job: **re-run from scratch and check whether the recorded numbers hold** — not
tune, not extend. Per-agent detail in each folder's `REPORT.md`; raw JSON in
`results_repro/`.

Two agents (`baselines`, `eegnn`) finished. Two (`graph`, `cascade`) completed all
compute and died on an API session limit during write-up; their reports were
reconstructed by the orchestrator from raw JSON, with every number re-derived
rather than taken from agent prose. The Kaggle session has since ended, so two
follow-ups remain unrun (§7).

### Verdict summary

| architecture | claims checked | held | corrected |
|---|---|---|---|
| baselines + safe_exit | cohort, LR/neighbour-avg, hop-AUROC, fusion, ETL fixes | all | 1 significance verdict is knife-edge |
| EEGNN depth cascade | 23 | 20 | depth-gap sign, §5j mechanism, metric bug |
| graph residual stage | 6 | 6 | seed-blind graph cache bug found |
| staged info cascade | 16 | 15 | alpha=0.01 row |

### What held

- **The §5n pivot, which is the thing the project now rests on.** Stages
  0.8623 / 0.8753 / 0.8974 against recorded 0.8624 / 0.8733 / 0.8968; labs −
  bedside **+0.0351, t = 14.76** against recorded +0.0343, t = 17.04. Same
  conclusion, comfortably significant, 5/5 seeds same sign.
- **The cohort is reproducible to the patient.** All six filter-ladder steps
  identical (94,458 → 65,366 → 65,366 → 46,776 → 46,705 → 46,651), prevalence
  0.10780, 331 → 542 features. Deterministic baselines match to 4 dp, pinning the
  feature matrix, imputation and splits as bit-identical. `_severity` was
  re-derived from `X_raw` (r = 0.99999999999) and is verifiably outcome-free.
- **Flat depth exits.** L1 − L10 = +0.0011, t = 0.75. 0% saving in 15/15 runs.
- **The high-risk calibration gap**, hi > lo in **20/20** runs, paired t = **+29.7**.
- **alpha=0.02 as an operating point**: 37.5% saved / 50.2% resolved before labs
  against recorded 34.5% / 46.2%, realised miss 0.0110 vs 0.0081 — both inside budget.
- **§7's multi-view fusion question, closed negatively.** The 0.811 → 0.846 gain is
  *entirely edge count*. At matched density fusion loses every time (0.7656 vs
  0.8115 @0.42M; 0.8217 vs 0.8407 @1.1M; 0.8455 vs 0.8487 @1.9M). A single graph
  at k=70 beats fused k=15 with fewer edges.

### The graph negative got stronger

5 seeds (the §5n table was 3), paired against **the same run with the graph off**:

| variant | Δ vs graph-off | t |
|---|---|---|
| kNN / C&S (scale fit on val) | −0.00069 | −0.87 ns |
| leaf / C&S (scale fit on val) | −0.00060 | −1.32 ns |
| kNN / learned GNN | +0.00052 | +0.76 ns |
| leaf / learned GNN | +0.00017 | +0.28 ns |

None significant; both parameter-free variants are *negative* under honest
selection. Sweeping the C&S scale and selecting **on the test set itself** — i.e.
cheating — picks scale **0.0 in 4/5 seeds** (leaf) and 2/5 (kNN), with an oracle
ceiling of **+0.0003**. So the correct statement is stronger than "does not earn
its place": **no achievable setting of this stage helps, including one chosen with
the answers in hand.** Cost re-measured at 4.1x the three tabular stages (recorded
5x). Rank correlation base vs graph = 0.959.

### Four corrections

1. **Code bug — seed-blind graph cache.** `scripts/run_cascade.py:76` keys the kNN
   graph cache on `(k, similarity)` with **no seed**, while `X` is rebuilt per seed
   (`impute_and_scale(X_raw, train)`, fit on that seed's train split). Every seed
   after the first silently reuses seed 42's graph. Confirmed by SHA-1 over the
   sorted edge list: `c3b80d8c3ae0` **identical across all 5 seeds** for kNN,
   5 distinct hashes for leaf (which is built inside the loop, uncached). Effect:
   understated seed variance on kNN rows; conclusion unchanged, since the leaf
   variants are unaffected and show the same null. Fix and the `--per-seed-graph`
   control live in `experiments/graph/`.

2. **Metric bug — tie reported as inversion.** `src/evaluate.py:243` uses strict
   `>` for `safety_ordering_ok`, so a gap of exactly 0.00 (everyone at full depth,
   nothing to order) reports `False` and reads as an ordering *inversion*. §5m's
   "learned Gumbel 0/3 ordering OK" is entirely this tie artefact. Should be `>=`,
   or the degenerate case should be reported separately.

3. **"The sick exit earlier" is not a reliable property.** Recorded as −0.32 on a
   single seed. Re-measured across 5: seed 42 = −0.216 (same sign as recorded),
   seeds 43–46 = +0.597 / +0.627 / +0.792 / +0.649. Mean **+0.49 ± 0.40,
   t = +2.728** against critical 2.776 — the sign flips and the effect does not
   clear significance. **Third single-seed number in this project not to survive
   more seeds** (after §5f's GCN/GAT ordering and §5g's 37.0% saving).

4. **§5j's mechanism is wrong, though its conclusion is right.** §5j attributes
   the 0% saving to an oversized budget. In fact at budgets 2/4/6 the *maximum*
   halt-vs-continue margin over all 3,533 test nodes is **negative**
   (−0.262 / −0.208 / −0.220): the halt head is saturated shut by the default loss
   weights (`high_risk_halt_penalty: 0.05` = 5x `ponder_cost: 0.01`), not by budget
   size. At budget 4 with ponder 0.05 early exit *does* engage — 56.9% saved at no
   accuracy cost, +1.76 depth gap. This does **not** rescue the architecture (the
   best budget-2 config is still "exit everyone at layer 1"), so §5n's pivot stands;
   but the stated reason for the failure should be corrected.

Plus the two wording/statistics corrections already applied inline: the severity
decile curve is near-monotone with one inversion (§5e), and the alpha=0.01 row is
bimodal with an n=2 miss rate (§5n).

### Caveat on the reproduction itself

XGBoost was pinned to 2 threads (four agents on a shared 4-CPU box), which changes
histogram reduction order. Runs are therefore **not bit-identical** to the recorded
ones — per-stage AUROCs drift 0.0001–0.0020. Irrelevant for every headline claim;
**decisive for the knife-edge alpha=0.01 row**, and the honest reading there is
that the claim is fragile, not that one run is right and the other wrong. Wall-clock
`cost_us` figures are not comparable across the two settings at all; ratios are.

---

## 5p. The follow-ups closed, and the spec's controlled experiment (2026-09-01)

Everything §5o left blocked on compute has now run, on a rebuilt cohort
(identical: 17,663 / 331 → 542 / 0.10780). Detail in
`experiments/cascade/FOLLOWUP.md`, `experiments/graph/FOLLOWUP.md` and
`ARCHITECTURES.md`.

### The tie-block hypothesis is refuted

§5o and `CURRENT_ITERATION.md` both flagged isotonic tie blocks as "the most
likely full explanation of the alpha=0.01 2/5-seed behaviour". **It is not.**
The certificate scan was run tie-aware (one threshold per distinct calibrated
value) and tie-*optimistic* (allowed to cut inside a tie block, i.e. to cheat):

| alpha | clean releases needed | tie-aware certifiable | tie-optimistic |
|---|---|---|---|
| 0.005 | 598 | 0 0 0 0 0 | 0 0 0 0 0 |
| 0.01 | **299** | **0 0 0 0 0** | **0 0 0 0 0** |
| 0.02 | 149 | 896 1084 917 985 905 | 918 1100 972 996 1041 |
| 0.05 | 59 | 1409 1525 1318 1531 1651 | 1570 1680 1531 1605 1708 |

Ties are real and large — isotonic leaves only **20–45 distinct values** on the
stage-1 pool — but wherever the honest scan certifies nothing the cheating scan
also certifies nothing. Ties move the achievable set by ~10%, never from zero.

### `fit_gate`'s search is exact — a validation, not a defect

The audit found seed 43 could certify 857 stage-1 releases at alpha=0.01 where
`fit_gate` set `tau_lo = 0`, raising the worry that its 400-point quantile grid
misses thresholds. Run in one process on identical calibrated probabilities
(`experiments/cascade/scripts/gate_vs_exhaustive.py`), **the shipped gate matches
an exhaustive scan exactly at every alpha in every seed — 0/5 misses.**

### What the alpha=0.01 row is

| run | seeds firing | released when it fires |
|---|---|---|
| original record (4 threads) | 2/5 | ~36% |
| §5o reproduction (2 threads) | 0/5 | — |
| this session (rebuilt cohort) | 1/5 (seed 43) | 40.4% of sel |

Stage 0 can never certify at alpha=0.01 (needs 299 clean, longest prefix 132–250).
Stage 1 *can*, but only when its ranking holds **≤3 deaths among the cleanest
~857** — k=4 gives 0.01065, just over budget. A discrete event decided far below
the level of any modelling choice, which is why the row is bimodal rather than
noisy. **Demoted to a flagged footnote**, per the decision taken 2026-09-01.

### alpha=0.05 and `--ablate-conformal` were never unrun

§7 listed both as outstanding. They are in `results_repro/headline_repro.json` —
`ablate_conformal: true`, `alphas` includes 0.05, all five seeds populated. They
were computed and never analysed. **alpha=0.05 is a solid fourth operating
point**: released 0.694 ± 0.060, miss **0.0306 ± 0.0073** against budget 0.05,
**75.2% never read the lab panel**, AUROC 0.8768 ± 0.0038. Fires 5/5, inside
budget 5/5 — it fills the gap the demoted alpha=0.01 row leaves.

**The certificate ablation is a result in its own right.** At alpha=0.005 the
empirical gate fires in 5/5 seeds and **breaches the budget in 3 of them**; the
conformal gate refuses to fire at all. The certificate converts three silent
breaches into one explicit refusal. Where both fire, the certificate costs
**11.6 pp of release volume at alpha=0.02** (t = −7.18) and buys a miss rate of
0.0110 rather than 0.0168.

### The seed-blind graph cache: real, and immaterial

`run_cascade.py:76` is confirmed by edge hashing — the shipped kNN path yields
**1 distinct graph across 5 seeds**, the seed-aware key yields **5**; the leaf
path was already 5 (built inside the loop, uncached). Isolated within one
session (same cohort, splits, threads, libraries; only the cache key differs):

```
shipped (seed-blind)  0.90162 ± 0.00457
fixed  (seed-aware)   0.90158 ± 0.00458
paired diff  -0.00004 ± 0.00011,  t = -0.74     std ratio 1.003
```

§5o predicted the bug "understates seed variance". **It does not** — the error
bars are unchanged (ratio 1.003). A kNN graph over 542 standardised features is
stable across train splits, so seed 42's graph is functionally near-identical to
each seed's own; the hashes differ bit-for-bit, the graphs barely differ at all.
The null is unmoved and slightly harder: the best variant `gnn_knn` moves from
+0.00052 to **−0.00086** (t = −0.99, ns) once the graph is rebuilt per seed.

### The spec's controlled experiment, finally run

Architectures 2 (PPR-Push) and 3 (corrected Fixed-Block) were built —
`src/ppr_push.py`, `src/fixed_block.py`, driver `scripts/run_architectures.py`.
All three share cohort, splits, graph, calibration split, certified gate and
depth budget 4; only the exit mechanism varies. Full detail in `ARCHITECTURES.md`.

**Does ACT beat discrete exit stages?**

```
AUROC(ACT) - AUROC(fixed-block) = +0.0024 ± 0.0024,  t = +2.18   (crit 2.776)
```

Not resolvable at 5 seeds. **The negative result is therefore about early exit on
this data, not about the ACT halting mechanism** — two structurally different
exit designs fail the same way. That is a stronger claim than the project could
previously support, and §5o's caveat ("does not distinguish 'early exit fails
here' from 'this halting mechanism fails here'") can now be discharged.

**Flat exits reproduce in an architecture sharing none of Arch 1's choices:**

| architecture | last stage − first | t | |
|---|---|---|---|
| 1. Graph-ACT | −0.0005 ± 0.0007 | −1.73 | flat |
| 3. Fixed-Block | +0.0009 ± 0.0027 | +0.72 | flat |
| **2. PPR-Push** | **+0.0207 ± 0.0083** | **+5.61** | **differentiated** |

Architecture 2 breaks the flatness because its stage 0 is a *graph-free* encoder
(0.8658) and each push round adds information — §5n's "cascade over information,
not depth" reached from inside a GNN.

**Compute is measured for the first time.** The spec's §3.3 Fix 1 (active-subgraph
masking) was never implemented in Arch 1; `model.py` propagates every node at
every step and masks afterwards. Arch 3 implements it, with a test asserting
masked inference equals dense-plus-freeze to 1e-5:

| | alpha=0.02 | alpha=0.05 |
|---|---|---|
| Arch 3 **measured** edge work saved | **44.8% ± 10.7** | **67.6% ± 2.1** |
| Arch 3 gate's stage arithmetic | 42.3% ± 10.5 | 66.7% ± 2.3 |
| Arch 1 | no measured figure exists | no measured figure exists |

Architecture 2's tolerance sweep is the only compute claim here with a bound
behind it: **47.0% ± 9.8 of edges saved for −0.0022 AUROC**, 94.2% for −0.0148,
against a graph-free floor of 0.8658 ± 0.0080. Diffusion beats that floor by
**+0.0207** — consistent with §5n, where propagation on top of a *strong* boosted
score bought nothing: the graph adds information relative to a weak tabular model
and none relative to a strong one. The system still loses to XGBoost
(0.8865 vs 0.8958).

### A leak found in the new driver, and what it implies

The first architectures run returned all three at ~0.91 AUROC — above XGBoost, on
identical Arch 1 code. `graph_build.build_graph` keys its cache on `cfg["seed"]`
**and stores the split masks in the cached object**; the driver never set
`cfg["seed"]`, so every seed loaded seed 42's graph *and masks* while scoring on
its own test split. Fixed, with an assertion. Corrected Arch 1 lands at
0.8887 ± 0.0055 against the record's 0.8888 / 0.8889 — it reproduces.
**Same failure family as `run_cascade.py:76`, far worse consequence.**

---

## 6. OPEN ISSUES — start here next session

### 6.1 RESOLVED — degenerate safety metric
Fixed 2026-08-27. `high_risk_analysis` now stratifies on an independent severity
proxy (`mimic_iv._severity`, observation-window features only) rather than on the
label at the operating threshold, and returns nan instead of a misleading
constant when no proxy is supplied. Verified across a 13-point sweep: the miss
rate now ranges 0.0154-0.0308 where it was previously 0.104 everywhere. Two
regression tests cover it. See §5e, §5g, §5h.

### 6.4 ANSWERED — the graph is a measured negative, not an open question
Closed 2026-08-29 (§5n). Four framings, 3 seeds each, all against the labs stage
at 0.8985: Correct & Smooth over a feature kNN graph (0.8973), C&S over a
booster-leaf graph (0.8973), a learned residual GNN over the feature graph
(0.8976), and over the leaf graph (0.8986). The best is +0.0001 for a stage
costing 5x the three tabular stages combined.
The residual framing was the strongest available test — it asks the graph only
for a correction to a strong model, using neighbours' residuals, which are a
function of the labels rather than of `X` — and it fails too. Stop tuning the
graph; report it.

### 6.5 ANSWERED — stratified calibration is not the fix
Closed 2026-08-29 (§5n). Severity-stratified isotonic narrows the high/low ECE
*ratio* (4.9x -> 3.8x) but makes the absolute high-risk error worse (0.0678 ->
0.0847) and costs AUROC (0.8939 -> 0.8877). The real improvement came from
isotonic calibration on a held-out split instead of temperature scaling:
high-risk ECE 0.23 -> 0.068. Note the ratio was the wrong statistic to track.

### 6.2 ANSWERED — depth does not pay on real MIMIC data
The mock-generator rewrite was never re-run, because real data superseded it.
The answer on real data is unambiguous and negative:

- per-exit AUROC is flat from layer 1 to layer 10 (§5g);
- AUROC is flat across a 10x sweep of mean depth (§5h);
- XGBoost beats every graph model (§5f);
- GAT attention is uniform, so the graph is a smoother, not a retrieval
  mechanism (§5h).

Root cause in §5f: **the kNN graph is built from the feature matrix**, so it
cannot carry information beyond `X`. Notebook 03 shows real relational
structure (1-hop neighbour-average reaches 0.811 AUROC, 2-hop 0.836), but that
structure is redundant with the features it was derived from, and gradient
boosting extracts more from those features directly.

This is a property of the patient-similarity-graph framing, not a tuning
failure. Tuning `k`, depth, or the loss will not change it. See §7 for what
would.

### 6.3 CONFIRMED — the two loss levers are not independent
`losses.multi_exit_loss` was already changed so that the efficiency term is
risk-*weighted* — it only rewards shallow exits on `(1 - risk)` mass — and the
safety term is normalised by risk mass so its effective weight doesn't silently
scale with the ~12% prevalence. Keep that structure; the remaining problem is
selectivity, not term magnitude.

---

## 7. Not yet done / what would change the conclusion

### Routes out of the negative result — status after §5n

1. ~~**Edges from a source outside `X`.**~~ **Partly walked, and it did not
   help.** §5n took the cheap version: keep the edges but propagate *residuals*,
   which are a function of the labels rather than of `X`, and separately build
   edges from the booster's leaf co-occurrence rather than from feature
   distance. Neither moved the number. What remains untested is the expensive
   version — edges from a genuinely external source: shared caregiver
   (`caregiver_id` in chartevents), care-unit and transfer paths (`transfers`),
   temporal adjacency (patients in the ICU concurrently), or diagnosis-code
   overlap held out of the feature matrix. After four failures the prior on this
   working should be lower than it was, and the cost of the ETL is real.
2. **Temporal node features.** First-24h aggregates discard trajectory. A GNN
   over patient time series has more to do than one over static summaries.
   Untouched, and now the most promising of the three.
3. ~~**Report as-is** and reframe around compute at accuracy parity~~ —
   **done, and it is no longer weak.** §5h's objection was that
   layer-1-for-everyone dominated the learned policy. Over the information
   cascade it does not: the gate occupies genuine frontier points between the
   two trivial policies (§5n).

### Still outstanding

- ~~Multi-seed reporting~~ **done** — see §5i/§5j: 5 seeds, paired t-tests,
  two claims corrected. Notebooks 04-07 still display their single-seed run;
  re-executing them would refresh the stored outputs but the numbers of record
  are now in `processed/multiseed.json`.
- ~~**Calibration on the high-risk stratum** (ECE ~0.29)~~ **largely closed**
  in §5n: isotonic on a held-out split brings high-risk ECE to 0.068. Stratified
  calibration was tried and is counterproductive (§6.5). The residual 4.9x
  high/low ratio is now a small-absolute-numbers artefact rather than a
  deployment blocker, but it has not been driven to parity.
- ~~`max_layers: 10` is far past useful depth~~ **confirmed and quantified**
  in §5j. At budgets of 2, 4 and 6 the policy saves exactly nothing; the saving
  exists only to trim an oversized budget. `configs/default.yaml` still ships
  `max_layers: 10` — left as-is so §5j's comparison reproduces, but 4 is the
  defensible setting for any future work.
- **A complete `chartevents` mirror.** The current one is truncated (§5b),
  costing 62% of the cohort. Another Kaggle mirror with an intact file would
  restore ~46.7k patients and is worth ten minutes of searching.
- ~~**Multi-view graph** ... compare at matched density before claiming it wins.~~
  **ANSWERED NEGATIVELY [REPRO 2026-09-01]** (§5o). The gain was *entirely* edge
  count. At matched density fusion loses at every density tested: 0.7656 vs
  0.8115 @0.42M, 0.8217 vs 0.8407 @1.1M, 0.8455 vs 0.8487 @1.9M. A single graph
  at k=70 beats fused k=15 using fewer edges. Do not revisit.
- ~~**Two reproduction follow-ups, blocked on compute.**~~ **BOTH DONE
  [2026-09-01]** — see §5p.
  1. `--per-seed-graph`: the cache bug is confirmed by edge hashing (1 distinct
     graph across 5 seeds, vs 5 with the fix) and is **immaterial** — paired diff
     −0.00004 ± 0.00011, and the error bars do *not* widen (std ratio 1.003),
     contrary to the prediction. Null unmoved; `gnn_knn` moves +0.00052 → −0.00086.
  2. The tie-aware scan **refutes** the tie hypothesis: where the honest scan
     certifies nothing, the tie-*optimistic* scan certifies nothing either.
     Separately, `fit_gate`'s quantile grid was checked against an exhaustive
     scan and matches exactly (0/5 misses) — a validation of the shipped gate.
  3. alpha=0.05 and `--ablate-conformal` **were never unrun** — both are in
     `results_repro/headline_repro.json` and had simply never been analysed.
     alpha=0.05 is a solid 5/5-seed operating point; the conformal-vs-empirical
     ablation shows the empirical gate breaching the budget in 3/5 seeds at
     alpha=0.005 where the certificate refuses to fire.
- **Two known bugs, unfixed in canonical** (both diagnosed in §5o, fixes exist in
  the experiment folders): the seed-blind graph cache at `scripts/run_cascade.py:76`,
  and the strict-`>` tie handling at `src/evaluate.py:243`.
- ~~**Architectures 2 and 3 from the spec were never built**, so the controlled
  experiment never happened.~~ **BUILT AND RUN [2026-09-01]** — §5p and
  `ARCHITECTURES.md`. ACT vs discrete stages: +0.0024 ± 0.0024, t = +2.18, not
  resolvable at 5 seeds. The negative result is about early exit on this data,
  not about the ACT mechanism.
- **Survival/readmission tasks, multimodal nodes** — untouched extensions.
- **Architecture 2 is the one live thread.** It is the only design here whose
  stages are differentiated (+0.0207, t = +5.61) and the only one with a compute
  bound rather than an empirical saving. It still loses to XGBoost, but it is the
  only GNN route that has not been closed off.

## 8. Reproducibility

Every run prints and logs `config hash | git SHA | seed | device | stage` and
writes `results/<stage>_<confighash>.json`. Seeds set across torch/numpy/random.
Note: not yet a git repo — `git init` when ready so the SHA logging becomes
meaningful (currently logs `nogit`).
