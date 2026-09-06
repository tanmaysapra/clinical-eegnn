#%% md
# 10 · Architecture 4R, Phase 1 — the graph-and-gate experiment

Runs all 8 comparison arms on the real MIMIC-IV cohort: the 3-stage tabular
cascade, plain XGBoost, the weak graph-free encoder, Architecture 2's
local-push diffusion, and Arch4R in its three ablation modes
(`encoder_only`, `fixed_k`, `adaptive_k`) plus the certified low-risk policy
built on top of `adaptive_k`.

**This run is deliberately a single-seed, few-epoch confirmatory pass, not
the 5-seed headline experiment the go/no-go criteria are meant to be read
from.** A from-scratch benchmark (`ARCH4R_HANDOVER.md` §4 item 5) put one
full training epoch (every train node scored as its own query, gradients
included) at roughly 17.6 minutes at ~17,663 patients -- tractable for a
handful of epochs on one seed in one Kaggle session, not yet for 5 seeds ×
30 epochs, which is a separate, longer job. Read the numbers below as "does
the pipeline produce a real, non-degenerate signal on real data", not as
the Phase 1 verdict.

Refuses to proceed if notebook `09`'s stop condition did not pass -- see the
next cell for how that gets checked here (Kaggle's own kernel-output
chaining if available, otherwise this notebook runs the same 5-seed check
itself before touching Phase 1).

#%% code
import glob, os, shutil, subprocess, sys

REPO = "/kaggle/working/clinical-eegnn"

if not os.path.isdir(REPO):
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception as e:
        raise RuntimeError(
            "Need a Kaggle Secret named GITHUB_TOKEN (a GitHub PAT with read "
            "access to tanmaysapra/clinical-eegnn) attached to this notebook "
            "-- Add-ons > Secrets in the notebook editor."
        ) from e
    os.environ["GITHUB_TOKEN"] = token
    subprocess.run(
        ["git", "clone", f"https://{token}@github.com/tanmaysapra/clinical-eegnn.git", REPO],
        check=True,
    )

os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")

exec(open(f"{REPO}/notebooks/src/_bootstrap.py").read())
ensure_pyg()
os.makedirs("results", exist_ok=True)

# See notebook 09's identical fix: attaching the dataset via the Kaggle
# API's kernel-metadata datasetDataSources field did not take effect at all
# (verified: no dataset_data_source in the saved kernel's metadata, empty
# /kaggle/input at runtime) -- download it directly via kagglehub instead.
def _find_mimic_root(start, max_depth=5):
    for dirpath, dirnames, _files in os.walk(start):
        if "hosp" in dirnames:
            base = os.path.basename(dirpath)
            return os.path.dirname(dirpath) if base in ("mimic-iv-3.1", "mimic-iv") else dirpath
        if dirpath[len(start):].count(os.sep) >= max_depth:
            dirnames[:] = []
    return None

_detected_raw = _find_mimic_root("/kaggle/input")
if _detected_raw is None:
    import kagglehub
    _downloaded = kagglehub.dataset_download("akshaybe/updated-mimic-iv")
    print(f"[phase1] downloaded dataset via kagglehub -> {_downloaded}")
    _detected_raw = _find_mimic_root(_downloaded, max_depth=3) or _downloaded

if _detected_raw != RAW:
    print(f"[phase1] RAW mount differs from _bootstrap.py's default: "
         f"using {_detected_raw} (not {RAW})")
    RAW = _detected_raw

#%% md
## Phase 0 gate

Looks for notebook `09`'s `phase0_integrity.json` among this kernel's
attached data sources first (cheap: reuses a run that already happened);
falls back to running the same 5-seed check itself if it isn't found or
didn't pass, rather than silently skipping it.

#%% code
found = None
for pattern in ("/kaggle/input/*/phase0_integrity.json",
               "/kaggle/input/*/results/phase0_integrity.json"):
    hits = glob.glob(pattern)
    if hits:
        found = hits[0]
        break

if found:
    shutil.copy(found, "results/phase0_integrity.json")
    print(f"[phase1] using phase0 result from attached kernel output: {found}")

ok, reason = None, None
if os.path.exists("results/phase0_integrity.json"):
    import run_arch4r_phase1 as p1
    ok, reason = p1._check_phase0_passed("results/phase0_integrity.json")

def _exec_code_cells(path, ns):
    """Run only the `#%% code` sections of a marker-split notebook source
    (`#%% md` sections are prose, not Python -- exec'ing the raw file whole
    would raise a SyntaxError on the very first markdown line)."""
    text = open(path).read()
    section, cells = None, []
    for line in text.splitlines():
        if line.strip() == "#%% md":
            section = "md"; continue
        if line.strip() == "#%% code":
            section = "code"; cells.append([]); continue
        if section == "code":
            cells[-1].append(line)
    for cell in cells:
        exec("\n".join(cell), ns)

if not ok:
    print(f"[phase1] no usable prior phase0 result ({reason}) -- running the "
         "5-seed check inline before Phase 1.")
    _exec_code_cells(f"{REPO}/notebooks/src/09_arch4r_phase0.py", globals())
    import run_arch4r_phase1 as p1
    ok, reason = p1._check_phase0_passed("results/phase0_integrity.json")

assert ok, f"phase0 gate failed even after running inline: {reason}"
print("[phase1] phase0 gate: PASSED, proceeding.")

#%% md
## Running Phase 1 (1 seed, modest epochs -- see the note at the top)

#%% code
import argparse
import json
import time

import torch

import run_arch4r_phase1 as p1
from src import data_prep

cfg = mimic_cfg()
args = argparse.Namespace(
    n_blocks=4, arch2_epochs=80, epochs=5, lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(f"device: {args.device}")

cohort = data_prep.load_cohort(cfg)
print(f"[phase1] cohort n={cohort[0].shape[0]} raw_features={cohort[0].shape[1]} "
     f"prevalence={cohort[1].mean():.5f}", flush=True)

seeds = [42]
runs = []
for seed in seeds:
    t0 = time.time()
    row = p1.run_one_seed(cfg, seed, args, cohort)
    runs.append(row)
    a = row["arms"]
    print(f"[phase1][seed {seed}] "
         f"cascade={a['cascade_3stage']['test_auroc']:.4f} "
         f"xgb={a['plain_xgboost']['test_auroc']:.4f} "
         f"graph_free={a['graph_free_encoder']['test_auroc']:.4f} "
         f"arch2={a['arch2_diffusion']['test_auroc']:.4f} "
         f"arch4r_enc={a['arch4r_encoder_only']['test_auroc']:.4f} "
         f"arch4r_fixed={a['arch4r_fixed_k']['test_auroc']:.4f} "
         f"arch4r_adaptive={a['arch4r_adaptive_k']['test_auroc']:.4f} "
         f"certified={a['arch4r_adaptive_k_certified']['certified']} "
         f"({time.time()-t0:.0f}s)", flush=True)

out_path = "results/arch4r_phase1.json"
with open(out_path, "w") as f:
    json.dump({"config": cfg, "args": vars(args), "runs": runs}, f, indent=1, default=str)
print(f"[phase1] wrote {out_path}")

#%% md
## The go/no-go read (provisional -- 1 seed, not 5)

#%% code
import analyse_arch4r_phase1 as an
an.main(out_path)

#%% md
## What this licenses saying

With one seed, none of the 4 go/no-go criteria can be trusted as a real
verdict -- `analyse_arch4r_phase1.py` itself will show `nan`/degenerate
paired statistics wherever a comparison structurally needs ≥2 seeds. What
this run *does* establish: whether the full pipeline (real cohort → 3-way
split → density-adaptive graph → local-push diffusion → certified gate)
executes end to end on real data and produces AUROC numbers in a sane
range (not chance, not degenerate). If the arm summary above shows
`arch4r_adaptive_k` clearly above chance and roughly in the neighbourhood
of `graph_free_encoder`/`arch2_diffusion`, that's the real signal to read
before committing to the full 5-seed, more-epochs run.
