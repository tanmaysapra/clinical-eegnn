# Notebooks

The pipeline split into a documented series. Each notebook is one stage, writes
its artefacts to `/kaggle/working/processed`, and the next one picks them up —
so a dropped session costs one stage, not the whole run.

Source of truth is `src/NN_name.py` (plain text, one cell per `#%% md` /
`#%% code` marker). Build the `.ipynb` with:

    python <scratch>/nbmake.py src/00_environment_and_data_audit.py 00_environment_and_data_audit.ipynb

Keeping cells as text means they diff and review like code instead of like JSON.

**The `.ipynb` files are build artefacts.** Rebuilding one from its source
discards any stored outputs, so edit the source, rebuild, then re-execute —
never hand-edit the `.ipynb`, and expect to re-run a notebook after touching its
source. Execution happens remotely via `nbclient` with `allow_errors=True`, so a
failing cell does not hide the cells after it; the runner reports every error in
one pass instead of stopping at the first.

| Notebook | Writes | Measured runtime |
|---|---|---|
| `00_environment_and_data_audit` | `audit.json` | 34 s |
| `01_cohort_extraction` | `block_*.parquet`, `cohort_*.csv.gz`, `cohort_summary.json` | 15.4 min cold / 5 min warm |
| `02_cohort_eda` | `eda_summary.json` | 31 s |
| `03_graph_construction` | `graph_*.pt`, `graph_summary.json` | 22 s |
| `04_baselines` | `baselines.json` | 51 s |
| `05_eegnn_multiexit` | `eegnn_model.pt`, `eegnn_results.json` | 22 s |
| `06_efficiency_safety_sweep` | `sweep_results.json` | 127 s (13 training runs) |
| `07_interpretability` | `interpretability.json` | 28 s |
| `08_hybrid_cascade` | — (results via `scripts/run_cascade.py`) | 70 s |

`08` currently ships **without stored outputs**: its source was edited after the
last execution and the Kaggle session ended before it could be re-run. It
executes clean (verified, `ERRORS: []`); re-run it on the next live session.
Its numbers are single-seed in any case — the numbers of record are the 5-seed
means in `old/results/headline.json`, summarised in `old/HANDOVER.md` §5n.

**Notebooks 00-08 are the prior architecture line (depth-cascade, EEGNN, the
staged hybrid cascade, and the Architecture 2/3 controlled experiment).**
Architecture 4R (notebooks 09-10) is the current active work; 00-08 now serve
as comparison baselines against it (see `README.md` and `ARCH4R_HANDOVER.md`),
not as a separately evolving line of work. Full history in `old/HANDOVER.md`
and `old/ARCHITECTURES.md`.

All nine execute clean end to end. Only `01` is expensive, and only on a cold
block cache — it scans an 18 GB and a 42 GB CSV. Everything downstream of it
runs in well under a minute, so iterating on the modelling notebooks is cheap.

Each notebook states in prose what result would *falsify* the design, not just
what would confirm it — 04 names the condition under which the graph is
worthless, 05 the condition under which the halting policy is vacuous, 06 the
condition under which the two loss levers are secretly one, and 08 names three
conditions, each of which has already sunk an earlier version of this project. That framing is
deliberate: the synthetic run in HANDOVER §6.2 looked fine on aggregate numbers
while the exit policy had quietly collapsed.

## Running them

These target a Kaggle session with the `akshaybe/updated-mimic-iv` dataset
attached. The session's proxy URL changes on every restart:

    export KERNEL_URL='https://kkb-production.jupyter-proxy.kaggle.net/k/<session>/<token>/proxy'
    python scripts/remote_kernel.py status
    python scripts/remote_kernel.py push src configs scripts notebooks

`/kaggle/working` does not survive a session ending, so `push` is expected to be
re-run after each reconnect. The per-block ETL cache means re-running `01` after
a reconnect rebuilds the blocks, but re-running it *within* a session is free.

## Why this split

Notebook 01 is the expensive one and 03 is the decisive one. 03 tests whether
the patient-similarity graph carries outcome signal at more than one hop — the
precondition for adaptive depth to mean anything. If that curve is flat, the
exit policy has nothing to learn, and finding that out before training is worth
an entire notebook.
