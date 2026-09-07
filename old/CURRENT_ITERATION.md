# Current iteration — follow-ups closed, spec's experiment run

**Dates:** 2026-09-01
**Status:** the two blocked follow-ups are done; Architectures 2 and 3 built and run
**Previous iteration:** the four-agent reproduction (§5o), summarised at the bottom
**Full record:** `HANDOVER.md` §5p. Detail: `ARCHITECTURES.md`,
`experiments/cascade/FOLLOWUP.md`, `experiments/graph/FOLLOWUP.md`

---

## What this iteration was

§5o ended compute-blocked with two follow-ups written but unrun, three decisions
open, and the spec's central controlled experiment never performed. All of that
is now closed. Two of the three decisions were taken (demote the α=0.01 row;
build Architectures 2 and 3); the third — whether to land the two known bugs in
canonical — is still open, but is now informed by a measurement rather than a
guess.

## The five things that changed

### 1. The tie-block hypothesis is refuted

§5o called isotonic tie blocks "the most likely full explanation of the α=0.01
2/5-seed behaviour". They are not. Run tie-aware *and* tie-optimistic (allowed to
cut inside a tie block, i.e. to cheat), the stage-0 scan certifies **zero
releases at α=0.01 in all five seeds under both**. Ties are real and large —
isotonic leaves only 20–45 distinct calibrated values — but they move the
achievable release set by ~10%, never from zero.

### 2. `fit_gate` is exact — a validation of shipped code

The audit found seed 43 could certify 857 stage-1 releases where `fit_gate` set
`tau_lo = 0`, which raised a real worry that its 400-point quantile grid misses
thresholds on a heavily-tied distribution. Checked in one process on identical
inputs against an exhaustive scan over every achievable threshold: **0/5 misses,
at every α.** The grid leaves nothing on the table.

### 3. The α=0.01 row, fully explained

| run | seeds firing | released when it fires |
|---|---|---|
| original record (4 threads) | 2/5 | ~36% |
| §5o reproduction (2 threads) | 0/5 | — |
| this session (rebuilt cohort) | 1/5 | 40.4% of sel |

Stage 0 can never certify at α=0.01 (needs 299 clean releases; the longest clean
prefix available is 132–250). Stage 1 can, but only when its ranking holds **≤3
deaths among its cleanest ~857** — a fourth death takes the bound to 0.01065,
over budget. A discrete event decided by perturbations far below the level of any
modelling choice. That is why the row is bimodal rather than noisy. Demoted.

### 4. Two "unrun" items had already been run

`HANDOVER.md` §7 and `NEXT_SESSION.md` listed α=0.05 and `--ablate-conformal` as
outstanding. Both are in `results_repro/headline_repro.json` — the flag was set,
all five seeds populated. They had been computed and never analysed.

- **α=0.05 is a solid fourth operating point.** Released 0.694 ± 0.060, miss
  0.0306 ± 0.0073 against a 0.05 budget, **75.2% never read the lab panel**,
  AUROC 0.8768 ± 0.0038. Fires 5/5, inside budget 5/5. It fills the gap the
  demoted α=0.01 row leaves.
- **The certificate ablation is a result.** At α=0.005 the empirical gate fires
  in 5/5 seeds and **breaches the budget in 3 of them**; the conformal gate
  refuses to fire at all. The certificate converts three silent breaches into one
  explicit refusal. Where both fire, it costs 11.6 pp of release volume at α=0.02
  (t = −7.18) for a miss rate of 0.0110 rather than 0.0168. This is the cleanest
  available argument for the certificate and it was sitting unread in the JSON.

### 5. The seed-blind graph cache is real and immaterial

Confirmed by edge hashing: the shipped kNN path yields **1 distinct graph across
5 seeds**, the seed-aware key yields **5**. Isolated within one session, with only
the cache key differing:

```
shipped 0.90162 ± 0.00457   fixed 0.90158 ± 0.00458
paired diff −0.00004 ± 0.00011,  t = −0.74      std ratio 1.003
```

§5o predicted the bug "understates seed variance". **It does not** — the error
bars are unchanged. A kNN graph over 542 standardised features is stable across
train splits, so seed 42's graph is functionally near-identical to each seed's
own. The null is unmoved and marginally harder: `gnn_knn` goes +0.00052 → −0.00086.

## The spec's controlled experiment, finally run

Architectures 2 (PPR-Push) and 3 (corrected Fixed-Block) now exist —
`src/ppr_push.py`, `src/fixed_block.py`, `scripts/run_architectures.py`. All three
share cohort, splits, graph, calibration split, certified gate and depth budget 4.
Only the exit mechanism varies.

**Does ACT beat discrete exit stages?**

```
AUROC(ACT) − AUROC(fixed-block) = +0.0024 ± 0.0024,  t = +2.18   (crit 2.776)
```

Not resolvable at 5 seeds. **This upgrades the negative result.** §5o had to warn
that the project "does not distinguish 'early exit on GNNs fails here' from 'this
particular halting mechanism fails here'". It now does: two structurally
different exit designs — one continuous and learned with tied weights, one
discrete and certified with distinct weights and a consistency loss — fail the
same way.

**Flat exits reproduce in an architecture sharing none of Arch 1's choices:**

| architecture | last stage − first | t | |
|---|---|---|---|
| 1. Graph-ACT | −0.0005 ± 0.0007 | −1.73 | flat |
| 3. Fixed-Block | +0.0009 ± 0.0027 | +0.72 | flat |
| **2. PPR-Push** | **+0.0207 ± 0.0083** | **+5.61** | **differentiated** |

Architecture 2 breaks the flatness, because its stage 0 is a *graph-free* encoder
(0.8658) and each push round adds information. That is §5n's "cascade over
information, not depth" reached from inside a GNN rather than by leaving for
gradient boosting — and it is the one genuinely new lead here.

**Compute is measured for the first time.** The spec's §3.3 Fix 1 — active-subgraph
masking — was never implemented in Architecture 1, whose backbone propagates every
node at every step and masks afterwards. Architecture 3 implements it, with a test
asserting masked inference equals dense-plus-freeze to 1e-5, so the saving is a
cheaper evaluation of the same model:

| | α=0.02 | α=0.05 |
|---|---|---|
| Arch 3 **measured** edge work saved | **44.8% ± 10.7** | **67.6% ± 2.1** |
| Arch 3 gate's stage arithmetic | 42.3% ± 10.5 | 66.7% ± 2.3 |
| Arch 1 | no measured figure exists | no measured figure exists |

Architecture 2's tolerance sweep is the only compute claim in the project with a
bound behind it: **47.0% ± 9.8 of edges saved for −0.0022 AUROC**, 94.2% for
−0.0148. Diffusion beats its graph-free floor by **+0.0207** — consistent with
§5n, where propagation on a *strong* boosted score bought nothing: the graph adds
information relative to a weak tabular model and none relative to a strong one.
The whole system still loses to XGBoost (0.8865 vs 0.8958).

## A leak I introduced, caught, and fixed

The first architectures run returned all three at ~0.91 AUROC — above XGBoost, on
*identical* Architecture 1 code. `graph_build.build_graph` keys its cache on
`cfg["seed"]` and stores the split masks inside the cached object; the driver
never set `cfg["seed"]`, so all five seeds loaded seed 42's graph **and seed 42's
train/val/test masks** while scoring on their own test splits. Exactly one
`graph_*.pt` existed on the remote for five seeds.

Fixed, with an assertion comparing returned masks against freshly computed ones.
Corrected Architecture 1 lands at 0.8887 ± 0.0055 against the record's SAS-GNN
0.8888 and EEGNN-at-budget-4 0.8889 — it reproduces.

**This is the same failure family as `run_cascade.py:76`, with a far worse
consequence**, and it is the strongest argument for landing that fix.

## Open decisions

1. **Still yours: land the two known bugs in canonical?** Now informed —
   the graph cache fix moves AUROC by −0.00004 and widens nothing, so landing it
   costs almost nothing in reproducibility. The `evaluate.py:243` tie fix is
   unambiguous and its only affected claim is already documented as an artefact.
2. ~~Demote the α=0.01 row~~ — **decided and done**, with the mechanism now
   fully explained rather than hypothesised.
3. ~~Scope~~ — **decided and done**. Architectures 2 and 3 are built and run.

## Next iteration should probably

Either write up — the project now has a genuine negative (early exit buys nothing
here, established across two structurally different exit mechanisms), a genuine
positive (the staged information cascade, independently reproduced), and a
certificate whose value is demonstrated by ablation — or pursue **Architecture 2**,
the only design whose stages are differentiated and the only one with a compute
bound rather than an empirical saving. Everything else in the GNN direction is
closed.

---

## Previous iteration (2026-08-31 → 09-01), for context

Four agents independently reproduced all four architectures. The §5n staged
information cascade held (labs − bedside +0.0351, t = 14.76); the graph negative
hardened (best +0.0005 ns, oracle ceiling +0.0003); four recorded claims were
corrected. Full detail in `HANDOVER.md` §5o.
