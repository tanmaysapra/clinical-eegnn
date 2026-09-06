#%% md
# 09 · Architecture 4R, Phase 0 — integrity and replication

Architecture 4R's own design says Phase 1 (the density-adaptive graph +
certified low-risk gate) must not begin unless: (a) the seed-safe
manifest/cache integrity tests pass, and (b) Architecture 2's already-recorded
local-push diffusion gain (+0.0207 ± 0.0083 AUROC over its own graph-free
floor, `ARCHITECTURES.md` §2) still reproduces on this environment. This
notebook is that gate, on the real MIMIC-IV cohort, made checkable by a
printed stop condition rather than a note in a document.

Its output (`results/phase0_integrity.json`) is what notebook `10` requires
before it will run Phase 1 at all — this is not a formality, it's the same
refusal `scripts/run_arch4r_phase1.py` enforces from the command line.

### What would falsify this

If the reproduced Architecture-2 diffusion gain's 95% CI does not overlap the
recorded band, or the integrity tests fail, `STOP CONDITION: FAILED` prints
below and nothing downstream should be trusted until it's understood why.

#%% code
import base64, os, subprocess, sys

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

#%% code
# `_bootstrap.py`'s hardcoded RAW ("/kaggle/input/datasets/akshaybe/updated-
# mimic-iv") matches how the other notebooks' interactive "Add Data" sessions
# mounted it -- attaching the dataset via the Kaggle API's
# datasetDataSources instead mounts it directly under /kaggle/input/<slug>,
# with no "datasets/<owner>/" nesting. Auto-detect rather than hardcode a
# second guess: walk /kaggle/input for whichever directory actually has a
# "hosp" subdirectory (mimic_iv.find_root's own definition of a real root).
import os as _os

def _find_mimic_raw_dir(start="/kaggle/input", max_depth=5):
    for dirpath, dirnames, _files in _os.walk(start):
        if "hosp" in dirnames:
            base = _os.path.basename(dirpath)
            return _os.path.dirname(dirpath) if base in ("mimic-iv-3.1", "mimic-iv") else dirpath
        if dirpath[len(start):].count(_os.sep) >= max_depth:
            dirnames[:] = []
    return None

_detected_raw = _find_mimic_raw_dir()
if _detected_raw is None:
    raise FileNotFoundError(
        f"no MIMIC-IV 'hosp' directory found anywhere under /kaggle/input "
        f"(tried {RAW} via _bootstrap.py's default too) -- is the "
        "akshaybe/updated-mimic-iv dataset actually attached to this kernel?")
if _detected_raw != RAW:
    print(f"[phase0] RAW mount differs from _bootstrap.py's default: "
         f"using {_detected_raw} (not {RAW})")
    RAW = _detected_raw

#%% code
import argparse
import time

import numpy as np

import run_phase0_integrity as p0
from src import data_prep, manifest as manifest_mod

cfg = mimic_cfg()
args = argparse.Namespace(
    config="configs/default.yaml", raw_dir=RAW, cache_dir=CACHE,
    source="mimic", seeds=[42, 43, 44, 45, 46], n_blocks=4, epochs=80,
    device=None, out="results/phase0_integrity.json",
)
args.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"device: {args.device}")

#%% md
## Running the integrity tests, then 5 seeds on the real cohort

Mirrors `scripts/run_phase0_integrity.py::main` exactly, just without the
`sys.exit` (a failing exit would kill the whole Kaggle session instead of
just reporting the failure in this cell's output).

#%% code
print("[phase0] running seed-safe cache/manifest integrity tests ...", flush=True)
tests_ok, tests_output = p0._run_integrity_tests()
print(tests_output, flush=True)

cohort = data_prep.load_cohort(cfg)
X_raw = cohort[0]
print(f"[phase0] cohort n={X_raw.shape[0]} raw_features={X_raw.shape[1]} "
     f"prevalence={cohort[1].mean():.5f}", flush=True)

raw_dir = cfg["data"].get("raw_dir")
cohort_fingerprint = (
    manifest_mod.fingerprint_raw_table(raw_dir) if raw_dir and os.path.isdir(raw_dir)
    else manifest_mod.fingerprint_raw_table(X_raw)
)
lockfile_hash, lockfile_is_pinned = p0._lockfile_hash(REPO)

per_seed = []
for seed in args.seeds:
    t0 = time.time()
    row = p0.run_one_seed(cfg, seed, args, cohort, cohort_fingerprint, lockfile_hash)
    per_seed.append(row)
    print(f"[phase0][seed {seed}] graph_free={row['graph_free_encoder_auroc']:.4f} "
         f"diffusion={row['arch2_full_diffusion_auroc']:.4f} "
         f"gain={row['arch2_diffusion_gain']:+.4f} "
         f"xgb={row['plain_xgboost_auroc']:.4f} "
         f"cascade_final={row['cascade_final_auroc']:.4f} "
         f"({time.time()-t0:.0f}s)", flush=True)

#%% code
gains = [r["arch2_diffusion_gain"] for r in per_seed]
gain_mean, gain_sd, gain_ci = p0._paired_ci95(gains)
band = (p0.RECORDED_DIFFUSION_GAIN_MEAN - p0._BAND_WIDTH_SD * p0.RECORDED_DIFFUSION_GAIN_SD,
       p0.RECORDED_DIFFUSION_GAIN_MEAN + p0._BAND_WIDTH_SD * p0.RECORDED_DIFFUSION_GAIN_SD)
gain_overlaps_band = not (gain_ci[1] < band[0] or gain_ci[0] > band[1])

reasons = []
if not tests_ok:
    reasons.append("seed-safe cache/manifest integrity tests failed")
if len(args.seeds) < 5:
    reasons.append(f"only {len(args.seeds)} seeds (<5)")
if not gain_overlaps_band:
    reasons.append(
        f"reproduced diffusion-gain 95% CI {gain_ci} does not overlap the "
        f"recorded band {band}")
if not lockfile_is_pinned:
    reasons.append("dependency lockfile not pinned (weakens drift guarantee, not fatal)")

stop_condition_passed = tests_ok and len(args.seeds) >= 5 and gain_overlaps_band

report = {
    "config": cfg, "args": vars(args),
    "recorded_diffusion_gain": {"mean": p0.RECORDED_DIFFUSION_GAIN_MEAN,
                                "sd": p0.RECORDED_DIFFUSION_GAIN_SD},
    "diffusion_check_applicable": True,
    "reproduced_diffusion_gain": {"mean": gain_mean, "sd": gain_sd, "ci95": list(gain_ci),
                                  "overlaps_recorded_band": gain_overlaps_band},
    "integrity_tests_passed": tests_ok,
    "lockfile_is_pinned": lockfile_is_pinned,
    "per_seed": per_seed,
    "stop_condition_passed": stop_condition_passed,
    "reasons_if_failed": reasons,
}
import json
os.makedirs("results", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(report, f, indent=1, default=str)
print(f"[phase0] wrote {args.out}", flush=True)

print(f"\nreproduced diffusion gain: {gain_mean:+.4f} ± {gain_sd:.4f}   "
     f"95% CI [{gain_ci[0]:+.4f}, {gain_ci[1]:+.4f}]")
print(f"recorded band (±2 SD):     [{band[0]:+.4f}, {band[1]:+.4f}]")
if stop_condition_passed:
    print("\n[phase0] STOP CONDITION: PASSED -- Phase 1 may proceed.")
else:
    print("\n[phase0] STOP CONDITION: FAILED -- do NOT proceed to Phase 1.")
    for r in reasons:
        print(f"  - {r}")

#%% md
## What this licenses saying

If `STOP CONDITION: PASSED` printed above, the pipeline's basic integrity
(seed-keyed caches, no cross-split leakage) and Architecture 2's headline
result both hold on this environment against the real cohort, and notebook
`10` can proceed. If it printed `FAILED`, the specific line(s) under it say
which check broke -- most likely a genuinely different reproduced diffusion
gain (worth understanding before trusting anything built on top of it), not
a bug in this notebook.
