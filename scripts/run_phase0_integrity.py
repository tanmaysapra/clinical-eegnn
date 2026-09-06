#!/usr/bin/env python
"""Architecture 4R, Phase 0 -- integrity and replication, with a real stop gate.

Architecture 4R's own design says Phase 1 (the new density-adaptive graph +
certified gate) must not begin unless: (a) the seed-safe manifest/cache
integrity tests pass, and (b) Architecture 2's already-recorded local-push
diffusion gain (+0.0207 +/- 0.0083 AUROC over its own graph-free floor, on the
real MIMIC cohort -- `ARCHITECTURES.md` SS2) still reproduces. This script is
that gate, made checkable by an exit code rather than a note in a document.

    python scripts/run_phase0_integrity.py --source mock --seeds 42 43 44 45 46
    python scripts/run_phase0_integrity.py --source mimic --seeds 42 43 44 45 46 \
        --raw-dir /kaggle/input/datasets/akshaybe/updated-mimic-iv \
        --cache-dir /kaggle/working/shared_cache

On `--source mock` the diffusion-gain band is reported but not enforced --
+0.0207 is a MIMIC-cohort number and mock data has no reason to reproduce it.
Only a `mimic` run's gain is checked against the recorded band, since that is
the only setting the historical figure was ever measured on.

Exit code: 0 if the stop condition passes, 1 otherwise. Never proceed to
building Phase 1 code against a run that exited 1.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for run_architectures

from src import cascade, data_prep, graph_build, tabular, train as train_mod
from src import manifest as manifest_mod
from src._omp_guard import run_xgb_staged_worker, run_xgb_worker
from src.evaluate import auroc

import run_architectures as ra  # reuse _train_generic/run_arch2 verbatim -- see module docstring

RECORDED_DIFFUSION_GAIN_MEAN = 0.0207
RECORDED_DIFFUSION_GAIN_SD = 0.0083
_BAND_WIDTH_SD = 2.0  # tolerance band = recorded mean +/- this many recorded SDs


def _mock_views(n_raw):
    """Pseudo-blocks for the mock generator -- identical to `run_cascade.py`'s
    helper of the same name, duplicated rather than imported to keep this
    script's dependency surface limited to `src/`."""
    cut = np.array_split(np.arange(n_raw), 5)
    return dict(zip(
        ["demographics", "vitals", "treatment", "comorbidity", "labs"], cut))


def _seed_bundle(seed):
    """Deterministic, distinct sub-seed per manifest field -- never one seed
    reused across every role, so a manifest genuinely tests each of them."""
    return {name: seed + i for i, name in enumerate(manifest_mod.REQUIRED_SEED_FIELDS)}


def _lockfile_hash(repo_root):
    """Best-effort dependency hash; degrades loudly rather than silently.

    A pinned `requirements.lock` doesn't exist in this repo yet (open
    question in the Phase 0/1 plan). Until it does, fall back to hashing
    `requirements.txt` itself and say so -- this is a weaker check (open
    version ranges can still drift underneath an unchanged file) but is
    strictly better than crashing the whole integrity run over it.
    """
    lock_path = os.path.join(repo_root, "requirements.lock")
    try:
        return manifest_mod.dependency_lockfile_hash(lock_path), True
    except FileNotFoundError:
        req_path = os.path.join(repo_root, "requirements.txt")
        print(f"[phase0] WARNING: no {lock_path}; hashing requirements.txt "
              "instead -- open version ranges are not fully pinned by this.",
              flush=True)
        return manifest_mod.dependency_lockfile_hash(req_path), False


def run_one_seed(cfg, seed, args, cohort, cohort_fingerprint, lockfile_hash):
    X_raw, y, names, meta = cohort
    cfg = dict(cfg)
    cfg["seed"] = seed
    train_mod.set_seed(seed)
    device = torch.device(args.device)

    train, val, test = data_prep.make_splits(y, seed)
    X = data_prep.impute_and_scale(X_raw, train)

    # ------------------------------------------------------------ manifest
    partitions = {
        "train": np.flatnonzero(train), "val": np.flatnonzero(val),
        "test": np.flatnonzero(test),
    }
    build_kwargs = dict(
        cfg=cfg, cohort_fingerprint=cohort_fingerprint, partitions=partitions,
        seeds=_seed_bundle(seed), code_commit=manifest_mod.current_code_commit(_REPO_ROOT),
        lockfile_hash=lockfile_hash,
    )
    man = manifest_mod.build_manifest(**build_kwargs)
    fresh = manifest_mod.build_manifest(**build_kwargs)  # rebuilt from scratch
    manifest_mod.verify_manifest(man, fresh)  # raises on any drift

    # ---------------------------------------------------------------- graph
    data = graph_build.build_graph(X, y, (train, val, test), cfg, meta=meta,
                                    cache_dir=cfg["data"]["cache_dir"])
    for nm, want, got in (("train", train, data.train_mask),
                          ("val", val, data.val_mask),
                          ("test", test, data.test_mask)):
        assert np.array_equal(want, got.cpu().numpy()), (
            f"graph cache returned another seed's {nm} mask at seed {seed}")
    data = data.to(device)

    # ------------------------- Architecture 2: graph-free floor + diffusion
    ra_args = argparse.Namespace(
        ppr_alpha=0.15, eps_push=1e-4, n_blocks=args.n_blocks,
        epochs=args.epochs, lam_smooth=0.0, eps_sweep=[1e-4],
    )
    splits = (y, None, None, test)  # cal/sel unused by run_arch2 itself
    arch2 = ra.run_arch2(data, cfg, ra_args, splits, meta.get("severity"),
                         verbose=False)
    margins = arch2["margins"]
    full_diffusion_auroc = float(auroc(y[test], margins[-1][test]))
    graph_free_auroc = float(arch2["encoder_only_auroc"])
    diffusion_gain = full_diffusion_auroc - graph_free_auroc

    # ------------------------------ plain XGBoost + 3-stage cascade, out-of-process
    xgb_out = run_xgb_worker(X, y, (train, val, test), seed=seed)
    plain_xgb_auroc = float(auroc(y[test], xgb_out["p_test"]))

    views = meta.get("views") or _mock_views(X_raw.shape[1])
    cols, stage_names = tabular.stage_columns(
        views, X_raw.shape[1], X.shape[1],
        tabular.indicator_sources(X_raw, train))
    staged_out = run_xgb_staged_worker(X, y, (train, val, test), cols,
                                       seed=seed, n_folds=5)
    cascade_stage_auroc = [
        float(auroc(y[test], staged_out["margins"][s][test]))
        for s in range(len(cols))
    ]

    return {
        "seed": seed, "n": int(len(y)), "prevalence": float(y.mean()),
        "manifest_hash": man["manifest_hash"],
        "graph_free_encoder_auroc": graph_free_auroc,
        "arch2_full_diffusion_auroc": full_diffusion_auroc,
        "arch2_diffusion_gain": diffusion_gain,
        "plain_xgboost_auroc": plain_xgb_auroc,
        "cascade_stage_names": stage_names,
        "cascade_stage_auroc": cascade_stage_auroc,
        "cascade_final_auroc": cascade_stage_auroc[-1],
    }


def _run_integrity_tests():
    """The negative cache tests, run as a real subprocess so a failure here
    is exactly as fatal as a failure in the reproduction numbers below."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_manifest.py", "tests/test_ledger.py", "-q"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def _paired_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0, (mean, mean)
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    try:
        from scipy import stats
        crit = float(stats.t.ppf(0.975, df=n - 1))
    except Exception:
        crit = 2.776  # 5-seed (df=4) critical value used throughout this project
    return mean, sd, (mean - crit * se, mean + crit * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--source", default="mock", choices=["mock", "mimic", "csv"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/phase0_integrity.json")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(args.seeds) < 5:
        print(f"[phase0] WARNING: only {len(args.seeds)} seeds requested; "
              "this project's own discipline requires >=5 for any reported "
              "number to count as more than provisional.", flush=True)

    cfg = yaml.safe_load(open(os.path.join(_REPO_ROOT, args.config)))
    cfg["data"]["source"] = args.source
    if args.raw_dir:
        cfg["data"]["raw_dir"] = args.raw_dir
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir
    cfg["model"]["max_layers"] = args.n_blocks

    print("[phase0] running seed-safe cache/manifest integrity tests ...", flush=True)
    tests_ok, tests_output = _run_integrity_tests()
    print(tests_output, flush=True)

    cohort = data_prep.load_cohort(cfg)
    X_raw = cohort[0]
    print(f"[phase0] cohort n={X_raw.shape[0]} raw_features={X_raw.shape[1]} "
          f"prevalence={cohort[1].mean():.5f}", flush=True)

    if args.source == "mock":
        cohort_fingerprint = manifest_mod.fingerprint_raw_table(X_raw)
    else:
        raw_dir = cfg["data"].get("raw_dir")
        cohort_fingerprint = (
            manifest_mod.fingerprint_raw_table(raw_dir) if raw_dir and os.path.isdir(raw_dir)
            else manifest_mod.fingerprint_raw_table(X_raw)
        )
    lockfile_hash, lockfile_is_pinned = _lockfile_hash(_REPO_ROOT)

    per_seed = []
    for seed in args.seeds:
        t0 = time.time()
        row = run_one_seed(cfg, seed, args, cohort, cohort_fingerprint, lockfile_hash)
        per_seed.append(row)
        print(f"[phase0][seed {seed}] graph_free={row['graph_free_encoder_auroc']:.4f} "
              f"diffusion={row['arch2_full_diffusion_auroc']:.4f} "
              f"gain={row['arch2_diffusion_gain']:+.4f} "
              f"xgb={row['plain_xgboost_auroc']:.4f} "
              f"cascade_final={row['cascade_final_auroc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    gains = [r["arch2_diffusion_gain"] for r in per_seed]
    gain_mean, gain_sd, gain_ci = _paired_ci95(gains)

    band = (RECORDED_DIFFUSION_GAIN_MEAN - _BAND_WIDTH_SD * RECORDED_DIFFUSION_GAIN_SD,
            RECORDED_DIFFUSION_GAIN_MEAN + _BAND_WIDTH_SD * RECORDED_DIFFUSION_GAIN_SD)
    gain_overlaps_band = not (gain_ci[1] < band[0] or gain_ci[0] > band[1])

    reasons = []
    if not tests_ok:
        reasons.append("seed-safe cache/manifest integrity tests failed")
    if len(args.seeds) < 5:
        reasons.append(f"only {len(args.seeds)} seeds (<5)")
    diffusion_check_applicable = (args.source == "mimic")
    if diffusion_check_applicable and not gain_overlaps_band:
        reasons.append(
            f"reproduced diffusion-gain 95% CI {gain_ci} does not overlap the "
            f"recorded band {band} (mean {RECORDED_DIFFUSION_GAIN_MEAN} +/- "
            f"{_BAND_WIDTH_SD}x{RECORDED_DIFFUSION_GAIN_SD})"
        )
    if not lockfile_is_pinned:
        reasons.append(
            "dependency lockfile is not pinned (hashed requirements.txt "
            "instead of requirements.lock) -- not fatal, but weakens the "
            "manifest's dependency-drift guarantee"
        )
    stop_condition_passed = (
        tests_ok and len(args.seeds) >= 5
        and (not diffusion_check_applicable or gain_overlaps_band)
    )

    report = {
        "config": cfg, "args": vars(args),
        "recorded_diffusion_gain": {
            "mean": RECORDED_DIFFUSION_GAIN_MEAN, "sd": RECORDED_DIFFUSION_GAIN_SD,
            "source": "ARCHITECTURES.md SS2, real MIMIC cohort",
        },
        "diffusion_check_applicable": diffusion_check_applicable,
        "reproduced_diffusion_gain": {
            "mean": gain_mean, "sd": gain_sd, "ci95": list(gain_ci),
            "overlaps_recorded_band": gain_overlaps_band,
        },
        "integrity_tests_passed": tests_ok,
        "lockfile_is_pinned": lockfile_is_pinned,
        "per_seed": per_seed,
        "stop_condition_passed": stop_condition_passed,
        "reasons_if_failed": reasons,
    }

    out_path = os.path.join(_REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"[phase0] wrote {out_path}", flush=True)

    if stop_condition_passed:
        print("[phase0] STOP CONDITION: PASSED -- Phase 1 may proceed.", flush=True)
    else:
        print("[phase0] STOP CONDITION: FAILED -- do NOT proceed to Phase 1.",
              flush=True)
        for r in reasons:
            print(f"  - {r}", flush=True)
    sys.exit(0 if stop_condition_passed else 1)


if __name__ == "__main__":
    main()
