# Early-Exit Models for Clinical Risk Stratification

Adaptive-inference models for ICU mortality prediction on MIMIC-IV v3.1, built around one question: **when is it safe to stop computing and commit to an answer for this patient?**

Cohort: 17,663 first ICU stays, first-24h features, in-hospital mortality (prevalence 10.8%).

## Data

This project is built on **MIMIC-IV v3.1, which is credentialed data under the [PhysioNet Data Use Agreement](https://physionet.org/content/mimiciv/)**. No patient-level data is in this repository and none ever should be: the raw tables, extracted cohorts and cached feature blocks are all gitignored. Only aggregate results (metric tables, figures) are committed, which the DUA permits.

To reproduce, obtain your own MIMIC-IV credentials and point `data.raw_dir` at your copy.

## Current work: Architecture 4R

The active line of work. A weak multi-view encoder, a **density-adaptive** per-view patient-similarity graph (every prior graph in this project used fixed-k feature-kNN), checkpointed local-push diffusion, and a certified (Clopper-Pearson) low-risk-only stopping gate — aimed at closing the measured AUROC gap between the project's one positive graph result (Architecture 2, below) and the tabular cascade. Full technical record, including the bugs found and fixed getting it running on real GPU hardware, is in **[`ARCH4R_HANDOVER.md`](ARCH4R_HANDOVER.md)**.

**Phase 0 (integrity/replication gate): PASSED**, on the real 17,663-patient cohort — reproduced Architecture 2's diffusion gain at +0.0248 to +0.0266 across runs (recorded band: [+0.0041, +0.0373]).

**Phase 1 (the graph-and-gate experiment)**: runs all 8 arms below — 3 of them (`cascade_3stage`, `plain_xgboost`, `architecture2`) are the prior architectures, kept solely as comparison baselines, not as separate lines of work anymore. A single-seed confirmatory pass on the real cohort:

| arm | test AUROC |
|---|---|
| plain_xgboost (baseline) | 0.8932-0.8978 |
| cascade_3stage (baseline) | 0.8988-0.9004 |
| graph_free_encoder (baseline) | 0.8673-0.8681 |
| arch2_diffusion (baseline) | 0.8824-0.8833 |
| arch4r_encoder_only | 0.7293-0.7299 |
| arch4r_fixed_k | 0.8895-0.8903 |
| **arch4r_adaptive_k** | **0.8894-0.8899** |
| arch4r_adaptive_k_certified | 0.8819-0.8863 (55-58% compute saved) |

Read that as a pipeline/signal check, not a verdict: `arch4r_adaptive_k` clears `arch4r_encoder_only` by a large, real margin (graph structure helps), the certified policy trades under 1% AUROC for over half the compute, but `adaptive_k` doesn't yet beat `fixed_k`, and the certification check didn't clear at this seed/epoch budget. The statistically-real 5-seed verdict is the next milestone — see `ARCH4R_HANDOVER.md` for current status.

## Prior architectures — now comparison baselines only

Everything below was this project's own line of work before Architecture 4R, in order. Each is now used **only** as a comparison arm against Architecture 4R (see the table above), not as active research.

**The staged hybrid cascade** (notebook `08`) cascades over *information* rather than model depth — bedside vitals, then charted orders, then the lab panel — since clinical data doesn't arrive at once or at equal cost. At a 2% safety budget: 34.5% of compute saved, 46.2% of patients resolved without reading the lab block, 0.0081 realised miss rate. The main negative result from this line: a patient-similarity graph built from the feature matrix itself adds nothing on top of the boosted score (best variant: +0.0001 AUROC for a stage costing 5x the tabular stages combined).

**Architecture 1 (Graph-ACT / EEGNN)**, **Architecture 2 (PPR-Push)**, and **Architecture 3 (Fixed-Block)** are three structurally different early-exit mechanisms tested in a controlled experiment (`old/ARCHITECTURES.md`) against the same cohort/graph/gate. Architectures 1 and 3 both show flat per-exit accuracy (early exit buys nothing); Architecture 2 is the exception — its stages genuinely differentiate (+0.0207 AUROC, t=5.61) because its first stage is a graph-free encoder and each diffusion round adds real information. That's the finding Architecture 4R builds on directly.

Full technical record for all of this (including the negative results, the reproduction pass, and every correction made along the way) is in **[`old/HANDOVER.md`](old/HANDOVER.md)**; the three-architecture controlled experiment is in **[`old/ARCHITECTURES.md`](old/ARCHITECTURES.md)**.

![accuracy/compute frontier](results/figures/cascade_frontier.png)

## Layout

```
src/arch4r.py             Architecture 4R model (encoder + density-adaptive graph + diffusion + gate)
src/graph_density.py      density-adaptive per-view reference graph construction
src/local_push.py         checkpointed local-push diffusion
src/gate_lowrisk.py       the certified low-risk-only stopping gate
scripts/run_phase0_integrity.py   Arch4R Phase 0 stop-condition check
scripts/run_arch4r_phase1.py      Arch4R Phase 1 driver, all 8 comparison arms
scripts/analyse_arch4r_phase1.py  go/no-go criteria analysis
src/tabular.py            the staged cascade's stages (torch-free, comparison baseline)
src/cascade.py            calibration, the certified two-sided gate, routing (comparison baseline)
src/hybrid.py             the graph stage: Correct & Smooth + learned residual GNN (comparison baseline)
src/ppr_push.py           Architecture 2 / PPR-Push (comparison baseline)
src/fixed_block.py        Architecture 3 / Fixed-Block (comparison baseline)
src/model.py              Architecture 1 / Graph-ACT / EEGNN (comparison baseline)
src/mimic_iv.py           the MIMIC-IV ETL
notebooks/                the documented pipeline; 00-08 are the prior architecture line (comparison
                           baselines), 09-10 are Architecture 4R Phase 0/1 — see notebooks/README.md
results/                  aggregate metrics + figures (no patient data); results/checkpoints/ holds
                           trained Arch4R weights (gitignored, kept local-only, not pushed)
old/                       superseded status/handover docs and the prior architectures' result JSONs,
                           kept for provenance -- old/HANDOVER.md is the full technical record of
                           everything before Architecture 4R
ARCH4R_HANDOVER.md         the full technical record for the active work -- read this
```

## Running it

```bash
conda create -y -n eegnn python=3.11 && conda activate eegnn
pip install -r requirements.txt

python -m pytest tests/ -q                          # 108 tests

# Architecture 4R -- the active work
python scripts/run_phase0_integrity.py --source mock --seeds 42 43 44 45 46
python scripts/run_arch4r_phase1.py --source mock --seeds 42 43 44 45 46 --epochs 30

# the prior architectures, kept runnable as comparison baselines
python scripts/run_cascade.py --source mock
python scripts/run_architectures.py --source mock
```

Two platform footguns are already solved and documented in `old/HANDOVER.md` §2 — do not "fix" them back.
