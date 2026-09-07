# The three-architecture controlled experiment (2026-09-01)

The spec (`Top3_EarlyExit_GNN_Architectures_v2.md`) designed three architectures
and recommended Architecture 3 be kept "as a corrected, more conventional
fallback/ablation baseline (useful for isolating *does ACT actually beat
discrete exit stages* as a controlled experiment)". Only Architecture 1 was ever
built, so that experiment never ran, and the project's negative result could not
distinguish **early exit fails on this data** from **this halting mechanism
fails**.

All three now exist and run on one footing.

| | code | exit mechanism | compute |
|---|---|---|---|
| 1. Graph-ACT | `src/model.py` (pre-existing) | learned ACT halting, shared weight-tied operator | analytical only |
| 2. PPR-Push | `src/ppr_push.py` | local-push termination, decoupled encoder + diffusion | **measured** |
| 3. Fixed-Block | `src/fixed_block.py` | discrete per-block exits, certified gate, active-set masking | **measured** |

Held identical: cohort, per-seed splits, patient graph, calibration split,
certified two-sided gate (`src/cascade.py`), depth budget 4, training protocol.
Seeds 42–46, paired. Driver `scripts/run_architectures.py`, analysis
`scripts/analyse_architectures.py`, raw JSON `results/architectures.json`.

---

## 1. The spec's question, answered

```
AUROC(ACT) − AUROC(fixed-block) = +0.0024 ± 0.0024,  t = +2.18   (crit 2.776)
per-seed: [−0.0010, +0.0050, +0.0042, +0.0028, +0.0008]
```

**Not significant at 5 seeds.** ACT and a discrete four-stage cascade are
indistinguishable at matched budget on this data. The effect is small and
positive; with t = 2.18 against a critical 2.776 this sits close enough to the
boundary that it should be quoted as "ACT is ~0.002 above, not resolvable at
n=5" rather than as a tie — the same caution `REPORT.md` applies to the
SAS-GNN-vs-EEGNN comparison.

**Consequence for the write-up.** The negative result is now about early exit on
this data, not about the ACT halting mechanism. Two structurally different exit
designs — one continuous and learned, one discrete and certified, with distinct
weights and a consistency loss — fail the same way. That is a stronger and more
defensible claim than the one the project could previously support.

## 2. Why they fail is also now confirmed independently

| architecture | last stage − first | t | verdict |
|---|---|---|---|
| 1. Graph-ACT | −0.0005 ± 0.0007 | −1.73 | flat |
| 3. Fixed-Block | +0.0009 ± 0.0027 | +0.72 | flat |
| **2. PPR-Push** | **+0.0207 ± 0.0083** | **+5.61** | **differentiated** |

Stage profiles:

```
1. Graph-ACT     0.8892  0.8894  0.8885  0.8887
3. Fixed-Block   0.8854  0.8856  0.8858  0.8863
2. PPR-Push      0.8658  0.8792  0.8837  0.8865
```

The flat profile that killed the depth cascade (SS5g–SS5j) reproduces in an
architecture that shares none of Architecture 1's distinguishing choices — no
weight tying, no ACT, no ponder cost, exponential layer weights and a
consistency loss instead. **Flatness is a property of message-passing depth on
this cohort, not of the shared-weight operator.**

And Architecture 2 breaks it. Its stages genuinely improve, because its stage 0
is a *graph-free* encoder (0.8658, the measured floor) and each push round adds
information. This is SS5n's insight — cascade over information, not over depth —
reached from inside a GNN rather than by leaving for gradient boosting.

## 3. Compute, measured rather than asserted

This is the spec's SS1.1 / SS3.3 "Fix 1", which Architecture 1 never
implemented. `model.py` runs the backbone over every node and every edge at
every step and applies the halt mask *afterwards*, so its saving is exit-depth
arithmetic. Architectures 2 and 3 count the edges and node-updates they perform.

`tests/test_architectures.py::test_masking_is_numerically_free` asserts that
Architecture 3's masked inference equals dense-inference-plus-freeze to 1e-5, so
the saving is a cheaper evaluation of the same model, not a different model.

| architecture | alpha=0.02 | alpha=0.05 |
|---|---|---|
| 3. Fixed-Block, **measured edge work saved** | **44.8% ± 10.7** | **67.6% ± 2.1** |
| 3. Fixed-Block, gate's stage arithmetic | 42.3% ± 10.5 | 66.7% ± 2.3 |
| 1. Graph-ACT | no measured figure exists | no measured figure exists |

The measured and analytical figures agree to ~2 pp, which retrospectively
validates the analytical approach *for a discrete-stage architecture*. It says
nothing about Architecture 1, whose realised depth is 4.00/4 at every seed —
0% saving, as SS5j found.

### Architecture 2's work/accuracy curve

The only compute claim in this project with a bound behind it (local-push work
is bounded by source mass over `eps_push`, independent of N):

| eps_push | edges saved | mean exit round | AUROC | vs encoder floor |
|---|---|---|---|---|
| 1e-6 … 1e-2 | 0.0% | 4.00 | 0.8865 ± 0.0027 | +0.0207 |
| 0.03 | 1.5% ± 1.5 | 3.98 | 0.8865 ± 0.0027 | +0.0207 |
| 0.1 | **47.0% ± 9.8** | 3.12 | **0.8843 ± 0.0031** | +0.0185 |
| 0.3 | **94.2% ± 2.1** | 3.41 | 0.8717 ± 0.0070 | +0.0059 |

Graph-free encoder floor: **0.8658 ± 0.0080**. Nearly half the edge work is free
(−0.0022 AUROC); 94% of it costs 0.0148.

**Note the contrast with SS5n's graph stage.** There, propagation on top of a
boosted score bought nothing (best +0.0005, ns, oracle ceiling +0.0003). Here
diffusion buys **+0.0207 over a graph-free encoder of the same family**. Those
are consistent: the graph adds real information relative to a *weak* tabular
model, and nothing relative to a *strong* one. The whole system still loses to
XGBoost (0.8865 vs 0.8958), which is the finding that has never moved.

## 4. Under the shared certified gate

| alpha | architecture | AUROC | released | miss rate | fires | breaches | saved |
|---|---|---|---|---|---|---|---|
| 0.02 | 1. Graph-ACT | 0.8860 ± 0.0058 | 0.432 ± 0.115 | 0.0091 ± 0.0035 | 5/5 | 0/5 | 38.6% |
| 0.02 | 2. PPR-Push | 0.8774 ± 0.0041 | 0.347 ± 0.138 | 0.0105 ± 0.0046 | 5/5 | 0/5 | 30.9% |
| 0.02 | 3. Fixed-Block | 0.8842 ± 0.0053 | 0.471 ± 0.129 | 0.0122 ± 0.0042 | 5/5 | 0/5 | 42.3% |
| 0.05 | 1. Graph-ACT | 0.8866 ± 0.0052 | 0.797 ± 0.042 | 0.0359 ± 0.0053 | 5/5 | 0/5 | 65.9% |
| 0.05 | 2. PPR-Push | 0.8673 ± 0.0105 | 0.755 ± 0.027 | 0.0372 ± 0.0043 | 5/5 | 0/5 | 61.5% |
| 0.05 | 3. Fixed-Block | 0.8838 ± 0.0063 | 0.797 ± 0.019 | 0.0358 ± 0.0028 | 5/5 | 0/5 | 66.7% |

The certified gate holds its budget for all three architectures at both alphas,
with no breaches — the safety machinery transfers cleanly off the staged tabular
cascade it was built for.

Parameter counts: 39,952 (tied) / 39,042 (decoupled) / **52,432** (four distinct
blocks) — the structural cost the spec predicted for Architecture 3.

---

## A leak found and fixed in this driver

The first run of `run_architectures.py` returned all three architectures at
~0.91 AUROC, above XGBoost's 0.8958 and above the record's EEGNN (0.8861) on
*identical Architecture 1 code*. That was a bug, not a result.

`graph_build.build_graph` keys its on-disk cache on `cfg["seed"]` **and stores
the split masks inside the cached `Data` object**. The driver never set
`cfg["seed"]`, so all five seeds loaded seed 42's graph and seed 42's
train/val/test masks, while scoring used each seed's own test mask. Models
trained on seed 42's train split were scored on later seeds' test splits, which
overlap it. Exactly one `graph_*.pt` existed on the remote for five seeds.

Fixed by setting `cfg["seed"] = seed`, plus an assertion comparing the returned
masks against the freshly computed ones. The corrected numbers are the ones
above, and Architecture 1 now lands at 0.8887 ± 0.0055 against the record's
SAS-GNN 0.8888 and EEGNN-at-budget-4 0.8889 — i.e. it reproduces.

**This is the same failure family as the known seed-blind cache at
`run_cascade.py:76`, with a far worse consequence.** It is an argument for
landing that fix rather than leaving it in place.
