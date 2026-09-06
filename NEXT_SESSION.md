# Paste this into the next session

I'm continuing the clinical-eegnn project. Read these first, in order:

1. `/Users/TanmaySapra/eegnn/clinical-eegnn/CURRENT_ITERATION.md` — start here.
   What the last iteration did, the verdict, the four corrections, and the gap
   against the original architecture spec.
2. `/Users/TanmaySapra/eegnn/clinical-eegnn/HANDOVER.md` — full record. §5n is the
   result; §5o is the reproduction. Corrections are tagged `[REPRO 2026-09-01]`.
3. `/Users/TanmaySapra/eegnn/experiments/{baselines,eegnn,graph,cascade}/REPORT.md`
   — per-architecture reproduction detail; raw JSON in each `results_repro/`.
4. `~/Downloads/Top3_EarlyExit_GNN_Architectures_v2.md` — the original spec, for
   context on what was designed vs. what got built.

## State

An independent four-agent reproduction of all four things in the codebase is
**complete**. The §5n staged information cascade holds (labs − bedside +0.0351,
t=14.76). The graph negative hardened (best +0.0005, ns; oracle ceiling +0.0003).
Four recorded claims were corrected — details in §5o.

Two follow-ups are **blocked on compute**, code already written:
- `experiments/graph/` — rerun the four variants with `--per-seed-graph` to
  quantify the seed-blind kNN cache bug.
- `experiments/cascade/` — tie-aware certificate scan (isotonic flat regions may
  release a whole tie block at once; likely explains the alpha=0.01 2/5-seed
  behaviour). This is a hypothesis with a test attached, NOT a finding. Also
  unrun: alpha=0.05, `--ablate-conformal`.

**The Kaggle kernel is dead** and the shared cohort cache went with it. To resume
remote work: restart the notebook, give me the new `KERNEL_URL`, and I'll rebuild
the cohort (~7 min; expect 17,663 patients / 542 features / 10.78% prevalence)
before running the two items above.

## Things that will NOT carry over

- The **cron job** that polled the kernel every 5 hours was session-only. It is
  gone. Ask me to recreate it if you still want the polling.
- The four **subagents** are gone. Their outputs persist in the folders above;
  the agents themselves cannot be resumed.

## Open decisions — mine to make, not yours

1. **Two known bugs are deliberately unfixed in canonical**, because fixing them
   changes numbers the record still quotes:
   - `scripts/run_cascade.py:76` — kNN graph cache key omits the seed, so seeds
     after the first reuse seed 42's graph (proved via edge hashing).
   - `src/evaluate.py:243` — strict `>` makes a legal 0.00 tie report as an
     ordering inversion.
   Fixes exist in the experiment folders. Decide whether to land them.
2. **The alpha=0.01 row** of the §5n trade-off table should be demoted or dropped
   (it's bimodal, fires in 2/5 seeds, and its miss rate is an n=2 average). It's
   currently flagged in place with a warning block, not removed.
3. **Scope.** Architectures 2 (PPR-Push) and 3 (corrected Fixed-Block) from the
   spec were never built, so the controlled experiment the spec was designed
   around never happened. Decide: build them, or write the project up as
   "one negative + one positive found by a different route" and stop.

## Working notes

- Local env: `/opt/homebrew/Caskroom/miniconda/base/envs/eegnn/bin/python` (3.11).
  24 unit tests expected to pass.
- `scripts/remote_kernel.py` has a `--root` flag (added last session) so parallel
  work doesn't collide on the remote.
- Thread count changes XGBoost numerics — reproduction ran at 2 threads, the
  original record at 4, which drifts per-stage AUROC by 0.0001–0.0020. Irrelevant
  to headline claims, decisive for knife-edge ones.
- **Stop trusting single-seed numbers.** Three have now failed to survive more
  seeds. Anything without a ± over >=5 seeds is provisional.
