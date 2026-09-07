# Architecture 4R — implementation handover

**Status as of this handover (updated, session 3): Phase 0 PASSED on the
real MIMIC-IV cohort (17,663 patients), not just mock data. Phase 1's
single-seed confirmatory pass completed on the real cohort too, after
finding and fixing three real GPU bugs that never surfaced on the CPU-only
local test suite (session 2's tests all ran on CPU). The 5-seed run that
produces the statistically-real go/no-go verdict was triggered and, as of
this handover, in progress. Full detail in §9 below.**

**Status as of the previous handover (session 2): Phase 0 complete and
verified end-to-end. Phase 1, all 10 build steps complete and verified —
`src/arch4r.py`, `src/gate_lowrisk.py`, `cascade.split_validation_3way`,
`scripts/run_arch4r_phase1.py`, and `scripts/analyse_arch4r_phase1.py` all
exist, are wired together, and have been smoke-tested end-to-end on mock
data (driver runs all 8 comparison arms per seed without error; analysis
script correctly computes all 4 go/no-go criteria, including the correct
"FAIL" verdict on an intentionally undertrained toy run). 95 tests passing
(`tests/`, excluding `test_pipeline.py`, which nothing here touches). What's
left is a real, full-epoch, multi-seed MIMIC run — no further code is known
to be missing. Read section 4 in full before touching `local_push.py`,
`graph_density.py`, or `arch4r.py` — it now includes a load-bearing
performance fix (session 2, item 5) in addition to the original three
semantic corrections (session 1, items 1-4).**

Read this file first if you are continuing this work. It is self-contained —
you do not need to re-read the design documents or re-derive the plan to
pick this up, though pointers to both are given below if you want the full
reasoning behind a decision.

---

## 1. What this is, in three sentences

This project studies early-exit models for ICU mortality prediction on
MIMIC-IV. "Architecture 4R" is a new design — a weak multi-view encoder,
a density-adaptive per-view patient-similarity graph (untried construction;
everything before this used fixed-k feature-kNN), checkpointed local-push
diffusion, and a certified (Clopper-Pearson) low-risk-only stopping gate —
aimed at closing a measured 0.0109 AUROC gap between the project's one
positive graph result (Architecture 2, `src/ppr_push.py`) and its tabular
cascade. An approved implementation plan covers Phase 0 (integrity/
replication + a stop-condition gate) and Phase 1 (the minimum binary-only
graph-and-gate experiment) only — survival head, clique arm, interpretability,
high-risk exits, and 10-seed confirmation are Phase 2/3, explicitly deferred.

**Full plan document:** `C:\Users\Satgu\.claude\plans\robust-hugging-metcalfe.md`
(exists on the machine this was built on; read it if you want the full
file-by-file rationale, build order justification, and every open question —
this handover restates the load-bearing parts but the plan file is the
complete record).

**Design source documents** (external, produced by a separate design/review
process before implementation began, not needed to continue building —
referenced here only for provenance): two Architecture 4 / 4R specification
files existed in the user's Downloads folder during design; their content is
fully absorbed into the plan file and this handover.

---

## 2. What actually exists right now (verified, not just written)

Every item below has passing tests I ran myself this session — not just
code that "should work."

| # | File(s) | Status | Tests |
|---|---|---|---|
| 1 | `src/manifest.py` | Done | `tests/test_manifest.py` — 9 passing |
| 1 | `src/ledger.py` | Done | `tests/test_ledger.py` — 5 passing |
| 1 | `src/instrument.py` | Done | (exercised via local_push tests) |
| 3 | `scripts/run_phase0_integrity.py` | Done, **ran end-to-end** | see §3 |
| 4 | `src/encoder_mv.py` | Done | `tests/test_encoder_mv.py` — 6 passing |
| 5 | `src/graph_density.py` | Done, extended session 2 (`ReferenceGraph.train_graph_available`) | `tests/test_graph_density.py` — 16 passing |
| 6 | `src/local_push.py` | Done, perf-fixed session 2 (§4 item 5) | `tests/test_local_push.py` — 16 passing |
| 7 | `src/arch4r.py` | Done (session 2) | `tests/test_arch4r.py` — 12 passing |
| 8 | `src/cascade.py::split_validation_3way`, `src/gate_lowrisk.py` | Done (session 2) | `tests/test_cascade.py` (+2), `tests/test_gate_lowrisk.py` — 9 passing |
| 9 | `scripts/run_arch4r_phase1.py` | Done, smoke-tested end-to-end on mock data (session 2) | no dedicated pytest suite (same convention as `run_architectures.py`/`run_phase0_integrity.py`) — verified by running it, see §7 |
| 10 | `scripts/analyse_arch4r_phase1.py` | Done, smoke-tested against real driver output (session 2) | same as above |

Plus, pulled forward from build step 9 because Phase 0 needed it early:
- `scripts/_xgb_worker.py` — added a `mode="staged"` branch (backward
  compatible; `mode` absent = original "plain" behaviour unchanged).
- `src/_omp_guard.py` — added `run_xgb_staged_worker(...)`, and **fixed a
  pre-existing Windows-only bug**: `np.load()`'s `NpzFile` kept the output
  file handle open, so `TemporaryDirectory` cleanup failed with
  `PermissionError` on Windows (silent on macOS/Linux, which is why nobody
  hit it before). Fixed in both `run_xgb_worker` and the new
  `run_xgb_staged_worker` by using `with np.load(...) as d:`.

**Full regression check, last run session 2:**
```
python -m pytest tests/ -q --ignore=tests/test_pipeline.py
# 95 passed
```
(`test_pipeline.py` was not run — not touched by this work, no reason to
believe it regressed, just wasn't included in the sweep. Run it too before
trusting a "fully green" claim.)

### Local environment note

This machine did **not** have `torch_geometric` or `xgboost` installed at
the start of this session. Both were installed mid-session
(`pip install torch_geometric` → got 2.8.0.post1 against the already-
installed `torch==2.12.1+cpu`; `pip install xgboost==2.1.4` matching the
pinned version in `requirements.txt`) and everything above was verified
against those versions. If you're on a different machine/environment,
confirm `import torch_geometric` and `import xgboost` both work before
assuming any test failure is a real regression rather than a missing
dependency.

---

## 3. Phase 0 stop condition — already exercised, passed

```
python scripts/run_phase0_integrity.py --source mock --seeds 42 43 44 45 46 --epochs 80 --n-blocks 4
```
was run this session (output not committed — it's a scratch verification
run, deleted after inspection) and printed:
```
[phase0] STOP CONDITION: PASSED -- Phase 1 may proceed.
```
Manifest verification, graph-cache mask-equality assertions, Architecture 2
reproduction (graph-free floor vs. full diffusion), plain XGBoost, and the
3-stage cascade (via the new staged worker) all ran successfully end-to-end
on the mock cohort. The diffusion-gain-vs-recorded-band check is only
*enforced* when `--source mimic` (mock data has no reason to reproduce a
MIMIC-specific number) — on mock it's computed and reported but not gated.

**Before trusting a real MIMIC run's stop condition**, note open question
in the plan (§7, "open questions", item 6): the tolerance band for
"reproduces +0.0207 ± 0.0083" is implemented as a 95% CI overlap check
against `[0.0207 - 2×0.0083, 0.0207 + 2×0.0083]` — a reasonable but
not-independently-blessed choice. If it fires a false stop, that's the first
place to look.

---

## 4. Design corrections made *during* implementation (important — read before touching `local_push.py` or `graph_density.py`)

These are not in the original design docs. They were found while building
and testing, and the fixes are load-bearing.

1. **`local_push.py`'s diffusion semantics were initially wrong.** The first
   draft copied `ppr_push.py`'s "push the signal directly" trick (p[i] ends
   up being node i's own diffused representation). Architecture 4R's actual
   spec formula —
   `h_bar_q^(s) = sum_u p_q^(s)(u) * V*h_u^(0) / sum_u p_q^(s)(u)` —
   requires a genuine **per-query personalized-PageRank distribution over
   graph nodes** (`p_q(u)` = how much of query q's mass reached node u),
   which is a different, more expensive computation (one push per query,
   not one shared push for the whole batch). Caught before writing tests;
   `local_push.py` as it stands now does the correct per-query version.
   **Cost implication, not yet resolved:** running every training node as
   its own query (needed for transductive training) costs
   `O(n / (alpha*eps))` per epoch — for n≈17,663 at eps=3e-4, alpha=0.15,
   that's roughly 17,663 × ~22,000 ≈ 3.9×10^8 edge-relaxations *per epoch*.
   This is a real, unresolved wall-clock risk for Phase 1 on real MIMIC
   scale (was already flagged in the plan's open questions before this was
   confirmed as the actual mechanism). **First thing to benchmark once
   `arch4r.py`'s training loop exists.**

2. **Mass-conservation bug for edgeless (sink) nodes**, fixed in both
   `local_push_trace` and `replay_with_grad`: popping a node with zero
   outgoing edges was unconditionally applying `p[u] += alpha*r[u]` and
   then zeroing `r[u]`, silently discarding the `(1-alpha)` remainder that
   had nowhere to flow. Fixed by absorbing the *entire* `r[u]` into `p[u]`
   when a node has no outgoing edges. In the real pipeline this should never
   trigger — `graph_density.row_normalize_with_selfloop` guarantees every
   node has at least a self-loop — but `local_push.py` must not silently
   leak mass for an arbitrary graph, and the fix is cheap insurance.

3. **`graph_density.py`'s qualification threshold `T_v`** is *always* fit
   under the adaptive-k rule, even when building the fixed-k=16 ablation
   arm — resolves an open question from the plan (§7 item 5) explicitly:
   `T_v` is a property of the view's distance geometry, not of which `k` a
   given experimental arm happens to use, so sharing it keeps the fixed-k
   vs. adaptive-k comparison apples-to-apples. This is documented in the
   module's own docstring; don't "fix" it back to being per-arm without
   re-reading that reasoning.

4. **The GATv2-style edge scorer's "static attention" concern was checked
   and found not to apply** to the literal formula in the design doc (it
   uses `abs(z_i - z_j)`, which is provably non-separable — see the
   Architecture 4R design-review history if you want the four-point
   contradiction proof). The scorer as implemented
   (`graph_density.EdgeScorerGATv2`) is still the strictly-more-expressive
   GATv2-style version (nonlinearity before the collapse to a scalar via
   `a_v`), per the design's own decision to use it anyway for the modest
   extra parameter cost — not because the naive version was broken.

5. **(Session 2) `local_push.py`'s transductive-training cost was benchmarked
   before writing `arch4r.py`, per this session's own instructions, and
   found infeasible as originally written — but the root cause was two
   fixable implementation redundancies, not the per-query push mechanism
   itself.** Measured on synthetic graphs matching real density (out-degree
   ~50), extrapolated to real cohort scale (n≈17,663):
   - `local_push_trace` and `replay_with_grad` each rebuilt their entire
     adjacency structure from scratch on **every single query call**
     (`_CSRGraph.build` did an `O(E)` python loop keyed on `edge_weight`,
     which changes every training step, even though the *topology* — which
     edges exist — does not). Scoring every training node as its own query
     per epoch made this `O(n × E)` instead of `O(E + n × push_cost)`.
     **Fixed**: `_CSRGraph` is now topology-only (`(v, edge_k)` pairs, no
     baked-in weight), built once via the new public `local_push.build_graph
     (edge_index, n)` and passed as an optional `graph=` argument to both
     `local_push_trace` and `replay_with_grad`; `edge_weight` is looked up
     fresh per call from the array already passed in, at zero rebuild cost.
     `_CSRGraph` also now caches `out_deg` (previously recomputed with an
     `O(n)` python list comprehension on every call even with a shared
     `graph`) — a second instance of the same class of bug, same fix.
   - Even after that fix, `replay_with_grad` — the differentiable pass, run
     every training step — was still ~50-90x slower than the equivalent
     plain-numpy push, because it walked the trace one event at a time using
     individual `torch.tensor([...])` + `index_add`/`index_copy` calls, each
     registering its own autograd node (thousands of tiny ops per query).
     **Fixed**: `replay_with_grad` is now a thin wrapper around a custom
     `torch.autograd.Function` (`_ReplayTraceFn`) whose forward replays the
     trace in plain numpy (as cheap as `local_push_trace` itself) and whose
     backward is a single hand-derived reverse walk over the same trace —
     see `_ReplayTraceFn`'s docstring for the derivation (it's a genuine
     analytic VJP, not an approximation; verified via
     `torch.autograd.gradcheck` in `test_replay_backward_matches_finite_
     differences`, and the two pre-existing 1e-6 numerical-equivalence tests
     still pass unchanged). **Do not revert this to plain autograd-tracked
     ops** — that's the exact regression this fix exists to prevent, and the
     benchmark numbers below are why.
   - **Net effect, measured**: at real cohort scale, one full training epoch
     (every train node scored as its own query, gradients included) went
     from ~84 hours (naive, extrapolated) to ~17.6 min — making 10-30 epochs
     (~3-9 hours) genuinely tractable within one Kaggle T4 session, and 80
     epochs (~23.5 hours) tractable across a checkpointed multi-session run.
     This was a real, load-bearing finding, not a hypothetical one — the
     user's own instructions were to flag back if this looked infeasible
     rather than build the rest of the pipeline on an assumption that didn't
     hold; it did look infeasible at first, and the fix is why it now isn't.

6. **(Session 2) `graph_density.ReferenceGraph` gained a train-side
   `train_graph_available()` method**, additive only (new optional
   constructor args, new method, no existing signature touched). Needed
   because training is transductive (every train node is also scored as its
   own query against the same frozen graph, `arch4r.transductive_batch`), so
   a train node can be just as "cold start" as a held-out one — `build_
   reference_graph` already computed the qualified/reliable counts needed
   for this internally (to build `keep`/`T_v`) but never exposed them.
   `train_graph_available()` applies the exact same `graph_availability(...)`
   rule `attach()` uses for held-out queries, so both code paths agree.

7. **(Session 2) `gate_lowrisk.py`'s ownership boundary follows this
   handover, not the original plan file, where they briefly disagree.** The
   plan's reuse-map table says `cascade.py` gets a new `certify_low_risk_
   gate` sibling function; this handover's own §5 (written after that
   section of the plan) says `cascade.py` gets `split_validation_3way`
   *only*, and every other piece (calibration, selection, one-shot
   certification, routing) lives in the new `src/gate_lowrisk.py` module.
   Implemented per the handover's version — `cascade.py` was extended with
   exactly one new function. If you're reconciling against the plan file
   directly, this is not a discrepancy to "fix"; the handover was already
   the more specific/later instruction.

8. **(Session 2) Arch4RModel's `error_head` is a new, undocumented-in-the-
   plan design decision**, not specified in either the plan or the original
   handover: a small `nn.Linear(fusion_dim, 1)` over `h0` alone (graph-
   independent, always available even in `encoder_only`/cold-start
   fallback), returned as `out["error_head_raw"]`. The plan's own §5 step 7
   signature table lists `"error_head_raw"` as a required output key without
   specifying what produces it beyond "error-head calibration (isotonic/
   sigmoid fallback)" in `gate_lowrisk.py`'s docstring. This implementation
   treats it as a dedicated, always-computable uncertainty signal rather
   than an alias for the final checkpoint's stage logit. If a future design
   review specifies something more precise, this is the one place to check
   first — it's a reasonable default, not a verified-against-the-design
   choice.

9. **(Session 2) Two Phase-1 driver simplifications, documented in `scripts/
   run_arch4r_phase1.py`'s own module docstring, not silently assumed:**
   the reference graph's topology is built once per seed and never rebuilt
   mid-training (the design permits "at most one rebuild after epoch 10";
   this driver simply never rebuilds at all — a strict subset of what's
   allowed, chosen for a tractable minimum-viable driver), and there is no
   per-epoch validation-based early stopping for the three Arch4R arms
   (fixed `--epochs`, matching the wall-clock-predictability goal of the
   benchmark work in item 5). Both are real simplifications versus the full
   design, not bugs — revisit before Phase 2 if training dynamics warrant it.

10. **(Session 2) The Phase 1 driver caught a real instance of the project's
    own documented seed-cache bug class before it shipped**: `run_one_seed`
    initially never set `cfg["seed"] = seed` before calling `graph_build.
    build_graph`, so seed 43's run silently reused seed 42's cached graph
    and train/test masks — the exact same historical bug `ARCHITECTURES.md`
    documents and `run_architectures.py::run_seed`/`run_phase0_integrity.py
    ::run_one_seed` both already guard against. The seed-mask equality
    assertion (copied from those two scripts) caught it immediately on the
    first 2-seed smoke test. Fixed by copying the same `cfg = dict(cfg);
    cfg["seed"] = seed` pattern. If you ever see this assertion fire again
    anywhere in this project, it means a NEW caller forgot the same thing —
    it is never safe to relax or remove the assertion instead of fixing the
    caller.

---

## 5. What's left — steps 7-10 of the original 10-step build order (session 1 view — ALL NOW DONE, see §4 items 5-10 and the table in §2 for what actually shipped in session 2)

The plan's own step numbering (see the plan file for full detail); restated
here with what's now available to build against.

### Step 7 — `src/arch4r.py` (not started)

Wires `encoder_mv.MultiViewEncoder` + `graph_density` (topology, scorer,
reliability, fusion) + `local_push` (diffusion, stage context/representation)
into one `Arch4RModel` in three modes: `encoder_only`, `fixed_k`,
`adaptive_k`. Per the plan's signatures:

```python
class Arch4RModel(nn.Module):
    def __init__(self, view_in_dims: dict[str,int], cfg: dict, mode: str)
    def forward(self, view_x, view_observed, attached_graph, checkpoints) -> dict
        # -> {"h0":..., "stage_logits": [Tensor]*n_checkpoints,
        #     "error_head_raw":..., "ledger": OpCounter.snapshot()}

def arch4r_loss(out, y, mask, cfg, lam_surv=0.0) -> tuple[Tensor, dict]
```

Key wiring details that weren't fully pinned down before implementation
reached this point, now resolved and needing to be honored:
- `stage_representation`'s `w_int(h0_q * h_bar_q)` requires `V`'s output
  dimension to equal `h0`'s dimension (the fusion layer's output, 64 by
  default) — give `V` the same width as `fusion_dim`, not an independent
  hidden size.
- `encoder_only` mode must work with `attached_graph=None` — this is also
  the `G_q=0` cold-start fallback path (`graph_density.AttachedGraph
  .graph_available`), so `encoder_only` isn't just an ablation arm, it's a
  real code path other arms fall back into.
- Training is **transductive over the training reference graph** (per
  `graph_density.py`'s own docstring decision): train nodes are
  simultaneously the graph and, one at a time, the queries being scored —
  this is where the §4.1 cost concern above becomes concrete. Val/gate-
  selection/gate-certification/test nodes use `ReferenceGraph.attach(...)`
  instead, layered onto the same frozen train-train topology.
- `fixed_k` and `adaptive_k` arms must share every component (encoder
  weights, `EdgeScorerGATv2`, `ViewReliability`) except the `k` used in
  `graph_density.ViewGraphConfig` — this is required for
  `test_fixed_k_and_adaptive_k_share_every_component_except_k` (not yet
  written, see step 10 below) to mean anything.

### Step 8 — `src/cascade.py` extension + `src/gate_lowrisk.py` (not started)

Add to `cascade.py` (new functions only, do not touch existing signatures —
`fit_gate`/`split_validation` are what Architectures 1-3 reproduce against):
```python
def split_validation_3way(val_mask, y, seed, fracs=(...)) -> dict[str, np.ndarray]
    # -> {"model_val":..., "gate_selection":..., "gate_certification":...}
```

New module `gate_lowrisk.py` — full signatures are in the plan file §2
("`src/gate_lowrisk.py`"); the one thing to get right that the design's own
second review round caught (and that must not regress): **certification is
one-shot per seed.** `certify_policy` takes the full enabled-stage set and
the full certification mask once; if any Clopper-Pearson bound fails, the
policy is uncertified for that seed, full stop — **never** disable the
failing gate and re-simulate remaining gates against the same certification
labels. That exact "disable-and-repair" pattern was a real statistical bug
in an earlier draft of the Architecture 4R design (it breaks the CP bound's
validity because the population reaching later gates becomes conditioned on
which earlier gates failed, using labels that were supposed to stay
untouched). The regression test for this
(`test_gate_lowrisk.py::test_certification_is_one_shot_per_seed`) is
mandatory, not optional polish.

Also reuse `cascade.clopper_pearson_upper` directly — it's already exactly
the one-sided CP formula needed, verified by direct reading this session,
no changes required.

### Step 9 — `scripts/run_arch4r_phase1.py` (not started; its XGBoost-worker prerequisite IS done)

The staged/plain XGBoost worker plumbing this driver needs already exists
and is tested (§2 above). What's left is the actual driver: per-seed 5-way
split → build reference graph train-only → attach held-out patients → run
all 8 comparison arms (3-stage cascade, plain XGBoost, weak graph-free
encoder, Architecture-2 diffusion, Arch4R encoder-only, Arch4R fixed-k=16,
Arch4R adaptive-k full diffusion, Arch4R adaptive-k certified policy) → write
ledger rows + JSON, modeled on `scripts/run_architectures.py`'s
`{"config","args","runs":[...]}` shape. Must refuse to run unless
`results/phase0_integrity.json` reports `stop_condition_passed: true` (this
JSON contract already exists — `run_phase0_integrity.py` writes exactly this
field).

### Step 10 — `scripts/analyse_arch4r_phase1.py` + remaining tests (not started)

Analysis script computing the 4 go/no-go criteria (see plan file for exact
thresholds), reusing `analyse_architectures.py`'s `col()`/`ms()`/`paired_t()`
pattern plus a new `holm_correction()` helper. Remaining tests:
`test_gate_lowrisk.py` (6 tests specified in the plan, including the
one-shot-certification guard above) and `test_arch4r.py` (2 tests: encoder-
only has no graph dependency; fixed-k/adaptive-k share every component
except k).

---

## 6. Open risks, still unresolved (carried over from the plan; items 1 and 5 from the original list are now resolved, see below)

1. ~~Trace-and-replay's wall-clock cost~~ — **RESOLVED, session 2**: benchmarked
   and fixed, see §4 item 5. Real-scale training (every train node as its
   own query, gradients included) now costs ~17.6 min/epoch, not ~84
   hours/epoch. Not a risk anymore, but worth re-benchmarking on real MIMIC-
   scale hardware (this was measured on synthetic graphs matching the
   expected density, on CPU) before trusting the exact minute figures.
2. Exact-search cost at scale (5 views × many attach calls × gamma-grid
   candidates per seed) — **still open**, not exercised by the mock-data
   smoke test (`--n-patients 400`, far below real MIMIC scale); the Phase 1
   driver's own gamma is fixed via `configs/default.yaml`'s `arch4r.gamma`,
   not grid-searched, so this risk is dormant until gamma-grid selection
   (open question 3 below) is actually implemented.
3. Gamma-grid (γ ∈ {0.25, 0.5, 0.75}) selection ordering vs. the gate's own
   AUROC-loss constraint — needs pinning before it silently becomes an
   uncontrolled second multiple-comparisons problem.
4. Policy-hash lock semantics — what actually checks the hash post-lock.
5. Dependency lockfile still doesn't exist (`requirements.lock`) —
   `run_phase0_integrity.py` degrades gracefully (hashes `requirements.txt`
   instead, prints a warning, doesn't fail the run) but this weakens the
   manifest's dependency-drift guarantee. Cheap to fix: `pip freeze >
   requirements.lock` once the environment is final.

---

## 7. How to verify everything above yourself, right now

```bash
cd clinical-eegnn
python -m pytest tests/ -q --ignore=tests/test_pipeline.py    # expect 95 passed
python scripts/run_phase0_integrity.py --source mock --seeds 42 43 44 45 46 \
    --epochs 80 --n-blocks 4 --out results/phase0_integrity.json
# expect: [phase0] STOP CONDITION: PASSED
# (this file is NOT scratch anymore -- run_arch4r_phase1.py below refuses to
# run at all unless it exists here and says stop_condition_passed: true)

# Fast end-to-end smoke test of the Phase 1 driver + analysis script (tiny
# mock cohort, few epochs -- NOT meaningful numbers, just proof the plumbing
# runs). Real runs drop --n-patients/--epochs/--arch2-epochs back to their
# defaults (or larger) and use --source mimic on real cohort scale.
python scripts/run_arch4r_phase1.py --source mock --n-patients 400 \
    --seeds 42 43 --epochs 5 --arch2-epochs 10 \
    --out results/arch4r_phase1_smoke.json
python scripts/analyse_arch4r_phase1.py results/arch4r_phase1_smoke.json
rm -f results/arch4r_phase1_smoke.json   # scratch artifact, not a deliverable
```

Note: `results/phase0_integrity.json` was deliberately treated as scratch in
session 1 ("delete after inspection"); session 2 made it load-bearing, since
`run_arch4r_phase1.py` checks for it at startup and refuses to run otherwise
(`--skip-phase0-check` exists only for iterating on the driver itself — never
use it to skip the check for a real run). Regenerate it before a real Phase 1
run if it isn't already sitting in `results/`.

Nothing above has been committed to git — `git status --short` in
`clinical-eegnn/` will show every new/modified file listed in §2, plus
pre-existing uncommitted work from an earlier iteration (`old/ARCHITECTURES.md`,
`ppr_push.py`, `fixed_block.py`, etc. — those predate this session, do not
touch them as part of finishing this handover's work unless asked).

---

## 9. Session 3 — real Kaggle runs, three real bugs found and fixed

Everything in §1-7 above was verified on mock data only, on CPU, locally.
This session ran both Kaggle notebooks (`architecture-4r-phase-0-integrity`,
`architecture-4r-phase-1-experiment`) against the real credentialed
MIMIC-IV cohort on Kaggle's GPU runtime for the first time. That surfaced
three genuine bugs the CPU-only local suite could never catch, all now
fixed and covered by the regression suite (108 passing, up from 95).

### Phase 0: PASSED on real data

```
[mimic] wrote .../cohort_97de93de81.csv.gz  (17,663 patients, 331 features, prevalence 0.108)
[phase0] cohort n=17663 raw_features=331 prevalence=0.10780
reproduced diffusion gain: +0.0248 to +0.0266 (across runs) ± ~0.01   95% CI overlaps
recorded band (±2 SD):     [+0.0041, +0.0373]
[phase0] STOP CONDITION: PASSED -- Phase 1 may proceed.
```

The §3 open question (whether the ±2 SD tolerance band would false-stop) did
not fire — the real reproduction landed comfortably inside the recorded
band across every run.

### Three real bugs, all GPU-only (never surfaced on CPU)

1. **Device-placement bug in `scripts/run_arch4r_phase1.py`** — `torch.tensor(...)`
   calls building `view_x`/`view_observed` input tensors had no `device=`
   kwarg, so they defaulted to CPU even though the model was moved to `cuda`.
   `RuntimeError: Expected all tensors to be on the same device`. Fixed by
   threading `device` through `_torch_view_dict`/`_torch_obs_dict` and every
   call site (commit `465a431`).
2. **Device-placement bug in `src/graph_density.py`** — `row_normalize_with_selfloop`,
   `_top_k_per_src`, and the empty-edge branch of `fuse_views` built helper
   tensors (`self_w`, `self_idx`, `deg`, the empty-edges fallback) without a
   `device=` kwarg, surfacing only once the `fixed_k`/`adaptive_k` arms ran
   on GPU (`encoder_only` skips this code path entirely). Fixed by threading
   the source tensor's device through all four call sites (commit `bbf8703`).
3. **The real OOM cause, in `src/arch4r.py`'s per-query diffusion loop** —
   after fixing the two bugs above, `fixed_k`/`adaptive_k` still hit
   `CUDA out of memory` on a 16 GB T4 at ~13.6 GiB used. First attempted a
   query mini-batch fix (commit `6ce026a`, splitting the ~10,600 train
   queries into batches with a `backward()`/`step()` per batch) — this did
   **not** fix it; the crash recurred at the identical memory footprint
   regardless of batch size. Root cause: `Arch4RModel.forward`'s per-query
   loop called `stage_context(h0, snapshots[s], self.V)` once per
   `(query, checkpoint)` pair, and `stage_context` recomputes `V(h0)` — a
   full `[n_nodes, d_hidden]` projection — on every single call, even though
   `V(h0)` doesn't depend on the query or the checkpoint at all. With
   ~10,600 queries x 5 checkpoints, that's tens of thousands of redundant
   `[n_nodes, d_hidden]` activations all retained simultaneously for
   `backward()`. Fixed by adding `stage_context_from_Vh0(Vh0, p_stage)` to
   `src/local_push.py` (existing `stage_context` now delegates to it,
   unchanged public signature/tests) and computing `Vh0 = self.V(h0)` once
   per `forward()` call in `arch4r.py`, outside both loops (commit `ddb0565`).
   This was the actual fix; the mini-batching from `6ce026a` stays in place
   as an additional memory safety margin but wasn't load-bearing on its own.

**A platform dead end, not a code bug:** tried switching from Kaggle's T4 to
its P100 GPU option to work around the (mistakenly-diagnosed-as-hardware)
OOM before finding the real cause above. Kaggle's current PyTorch build has
dropped support for the P100's Pascal architecture (`sm_60`) entirely —
`AcceleratorError: CUDA error: no kernel image is available for execution
on the device`. Not fixable from this codebase; P100 is off the table on
Kaggle's current image. Stick to T4.

**Also learned the hard way:** `mcp__kaggle__save_notebook`-style API resaves
of a kernel's dataset attachment are unreliable — every API resave in this
session wiped `dataset_data_sources` from the kernel metadata even when the
field was explicitly included in the request, forcing a manual "Add Data" +
Save Version in the Kaggle UI to recover. A plain UI-triggered Save Version
(no API resave) never had this problem. Prefer that path when only the
underlying GitHub repo changed and the notebook's own cells don't need editing
(the notebook re-clones the repo fresh on every run regardless).

### Phase 1: real single-seed confirmatory pass, real 8-arm results

After all three fixes landed, a single-seed (seed 42), 5-epoch confirmatory
run completed cleanly on the real 17,663-patient cohort (`fixed_k`/`adaptive_k`
each trained in ~29 minutes on a T4):

| arm | test AUROC |
|---|---|
| plain_xgboost | 0.8932-0.8978 |
| cascade_3stage | 0.8988-0.9004 |
| graph_free_encoder | 0.8673-0.8681 |
| arch2_diffusion | 0.8824-0.8833 |
| arch4r_encoder_only | 0.7293-0.7299 |
| arch4r_fixed_k | 0.8895-0.8903 |
| arch4r_adaptive_k | 0.8894-0.8899 |
| arch4r_adaptive_k_certified | 0.8819-0.8863 (55-58% compute saved, ~0.4-0.7% AUROC lost) |

(Ranges cover two runs at the same config, seed 42 and seed 42 again after
the checkpoint-saving commit — GPU non-determinism accounts for the spread.)

All 4 go/no-go criteria report `FAIL` at n=1, but 3 of the 4 are `nan`-driven
artifacts of `analyse_arch4r_phase1.py` needing >=2 seeds to compute paired
statistics, not a real negative result — per-seed deltas are real and mostly
favorable (`adaptive_k - encoder_only` = +0.16, a large real graph-signal
gain; `adaptive_k - fixed_k` = essentially zero, a genuine "no advantage yet"
finding, not a stats artifact). Do not report this run's literal "NO-GO" as
the project's verdict — it's a pipeline/signal smoke test by design (see the
notebook's own framing), not the statistically-real result.

### Model checkpoints now persisted

`_save_checkpoint()` added to `run_one_seed` (commit `8151bbc`) writes each
arm's trained `state_dict` to `results/checkpoints/arch4r_<mode>_seed<seed>.pt`
after training — anything under `/kaggle/working` is captured as real Kaggle
kernel output, so these are downloadable per-run instead of discarded when
the session ends. `*.pt` stays gitignored (weights trained on credentialed
MIMIC-IV data are treated the same as raw/cached data in `.gitignore` — not
committed, kept local-only). Note: `.pt` files are themselves zip archives
internally (PyTorch's serialization format) — extracting them with a generic
archive tool that auto-unpacks nested zips corrupts them (loses the required
top-level folder wrapper); download and use as-is, don't extract further.

### The 5-seed run (the actual go/no-go verdict)

Notebook cell's `seeds` list bumped from `[42]` to `[42, 43, 44, 45, 46]`,
epochs left at 5 (not the full 30-epoch budget — 5 seeds x 30 epochs is
~29 hours, not tractable in one Kaggle session; 5 seeds x 5 epochs is
~5 hours and is). Source change is in `notebooks/src/10_arch4r_phase1.py`
(commit `b2b334d`) but note that pushing to GitHub alone does **not** change
what Kaggle runs — the notebook's own cell content is separate from the repo
and only updates via an explicit `save_notebook` API call or a manual edit
in the Kaggle UI. Triggered via manual UI edit + Save Version. As of this
handover, still running; check the kernel directly for the current
`current_version_number` and status before assuming this is done.

Once it completes: this is the run whose go/no-go verdict actually counts
(paired stats stop reading `nan` once seeds >= 2). Update this section (or
the top status line) with the real result once it lands, and pull the real
executed notebook + all 5 seeds' checkpoints the same way §7's smoke test
verifies the plumbing.
