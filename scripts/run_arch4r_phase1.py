#!/usr/bin/env python
"""Architecture 4R, Phase 1 -- the minimum binary-only graph-and-gate experiment.

Per seed: split the cohort 5 ways (train / model-val / gate-selection /
gate-certification / test, via `data_prep.make_splits` + `cascade.
split_validation_3way`), build the density-adaptive per-view reference graph
train-only (`graph_density.build_reference_graph`, once for the fixed_k=16
ablation config and once for the adaptive_k config -- both share the same
adaptive-fit qualification threshold `T_v`, only `k` differs, per
`graph_density.py`'s own documented design decision), attach every held-out
patient inductively (`ReferenceGraph.attach`, never creating a held-out-to-
held-out edge), and run all 8 comparison arms:

    1. 3-stage tabular cascade (out-of-process XGBoost worker)
    2. plain XGBoost
    3. weak graph-free encoder            (Architecture 2's own floor)
    4. Architecture-2 local-push diffusion (`src/ppr_push.py`, unmodified)
    5. Arch4R encoder_only
    6. Arch4R fixed_k=16
    7. Arch4R adaptive_k, full diffusion (uncapped -- every checkpoint scored)
    8. Arch4R adaptive_k, certified low-risk policy (`src/gate_lowrisk.py`)

Refuses to run at all unless `results/phase0_integrity.json` reports
`stop_condition_passed: true` -- Phase 1 code must never be trusted against
a cohort/pipeline that hasn't cleared Phase 0's own integrity gate.

    python scripts/run_arch4r_phase1.py --source mock --seeds 42 43 44 45 46 \
        --epochs 30 --out results/arch4r_phase1.json
    python scripts/run_arch4r_phase1.py --source mimic --seeds 42 43 44 45 46 \
        --raw-dir /kaggle/input/datasets/akshaybe/updated-mimic-iv \
        --cache-dir /kaggle/working/shared_cache --epochs 30

Two documented Phase-1 simplifications (not in the original design, chosen
for a tractable minimum-viable driver -- revisit before Phase 2):
  - the reference graph's topology is built once and never rebuilt mid-
    training (the design permits "at most one rebuild after epoch 10"; this
    driver simply never rebuilds, since `local_push.build_graph(...)` being
    shared for the WHOLE training run, not just one epoch, is a strict
    subset of what the fix in `src/local_push.py` already supports).
  - no per-epoch validation-based early stopping for the Arch4R arms (each
    trains for a fixed `--epochs`); this keeps per-seed wall-clock
    predictable, at the cost of not stopping early on an obviously-converged
    run. Both are flagged here, not silently assumed.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for run_architectures

from src import arch4r, arch4r_train, cascade, data_prep, gate_lowrisk
from src import graph_density as gd
from src import train as train_mod
from src._omp_guard import run_xgb_staged_worker, run_xgb_worker
from src.encoder_mv import VIEW_OUT_DIMS, fit_view_preprocessors, observed_fraction
from src.evaluate import auroc
from src.local_push import algorithmic_bound, make_checkpoints

import run_architectures as ra  # reuse run_arch2 (Architecture 2 + its graph-free floor)


def _mock_views(n_raw):
    cut = np.array_split(np.arange(n_raw), 5)
    return dict(zip(
        ["demographics", "vitals", "treatment", "comorbidity", "labs"], cut))


def _check_phase0_passed(path):
    if not os.path.exists(path):
        return False, f"{path} does not exist -- run scripts/run_phase0_integrity.py first"
    with open(path) as f:
        report = json.load(f)
    if not report.get("stop_condition_passed"):
        return False, f"{path} reports stop_condition_passed=false: {report.get('reasons_if_failed')}"
    return True, None


# ------------------------------------------------------------- view wiring

def _build_view_arrays(X_raw, views, train_mask):
    """Fit train-only per-view preprocessors, transform the whole cohort, and
    compute the raw (pre-imputation) observed-fraction reliability signal --
    exactly the two `src.encoder_mv` primitives Arch4R needs, applied once
    per seed for every arm that touches the graph-adaptive encoder."""
    raw_views = {name: X_raw[:, idx] for name, idx in views.items()}
    pre = fit_view_preprocessors(raw_views, train_mask)
    view_x_full = pre.transform(raw_views)
    view_observed_full = observed_fraction(raw_views)
    return raw_views, view_x_full, view_observed_full


def _torch_view_dict(view_arrays, idx, device):
    return {name: torch.tensor(arr[idx], dtype=torch.float32, device=device)
           for name, arr in view_arrays.items()}


def _torch_obs_dict(view_observed_full, idx, device):
    return {name: torch.tensor(arr[idx], dtype=torch.float32, device=device)
           for name, arr in view_observed_full.items()}


# ------------------------------------------------------------ arch4r training

def _train_arch4r(mode, view_in_dims, cfg_arch4r, view_x_full, view_observed_full,
                  y, train_mask, ref_graph, epochs, lr, device):
    """Fixed-epoch transductive training loop for one Arch4R ablation arm.
    `ref_graph` is None for encoder_only (no graph dependency at all --
    trains over the whole cohort directly, matching arch4r.py's own no-
    graph-dependency contract).

    The per-epoch step itself lives in `src/arch4r_train.py` (shared with
    `notebooks/src/11_arch4r_fullrun.py`, which needs to stop after every
    epoch to checkpoint to Drive instead of running all `epochs` in one
    call) -- this function is now just that shared step called in a loop,
    behaviour unchanged from before the extraction."""
    model = arch4r.Arch4RModel(view_in_dims, cfg_arch4r, mode=mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoints = make_checkpoints(c=cfg_arch4r.get("checkpoint_scale", 1.0))
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    if mode == "encoder_only":
        n = len(y)
        view_x = _torch_view_dict(view_x_full, np.arange(n), device)
        view_observed = _torch_obs_dict(view_observed_full, np.arange(n), device)
        mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
        t0 = time.time()
        for _ in range(epochs):
            arch4r_train.train_encoder_epoch(
                model, opt, view_x, view_observed, mask_t, y_t, checkpoints, cfg_arch4r)
        return model, checkpoints, time.time() - t0

    train_idx = np.flatnonzero(train_mask)
    view_x_train = _torch_view_dict(view_x_full, train_idx, device)
    view_observed_train = _torch_obs_dict(view_observed_full, train_idx, device)
    full_batch = arch4r.transductive_batch(ref_graph)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.float32, device=device)

    # Scoring every train node as its own query in one forward() retains all
    # of their per-checkpoint diffusion activations simultaneously (nothing
    # is freed until the single end-of-epoch backward()) -- OOMs a 16 GB GPU
    # past a few thousand queries. Splitting the query set into mini-batches
    # (backward()+step() per batch, so a batch's graph is freed before the
    # next one is built) bounds peak memory to one batch's worth instead of
    # all of them -- the full reference-graph topology (`full_batch.
    # per_view_edges`) is still shared by every batch, only which nodes are
    # scored as queries changes.
    query_batch_size = cfg_arch4r.get("query_batch_size", 512)
    rng = np.random.default_rng(0)

    t0 = time.time()
    for _ in range(epochs):
        arch4r_train.train_graph_epoch(
            model, opt, full_batch, view_x_train, view_observed_train, y_train_t,
            checkpoints, cfg_arch4r, query_batch_size, rng)
    return model, checkpoints, time.time() - t0


def _eval_arch4r_graph_arm(model, checkpoints, view_x_full, view_observed_full,
                           ref_graph, train_mask, n_total, device):
    """Attach every held-out patient once and score every checkpoint. ->
    (margins [n_checkpoints, n_total] with NaN on train rows,
    graph_available [n_total] with False on train rows)."""
    n_train = int(train_mask.sum())
    train_idx = np.flatnonzero(train_mask)
    held_idx = np.flatnonzero(~train_mask)

    raw_query = {name: arr[held_idx] for name, arr in view_x_full.items()}
    # `view_x_full` is already preprocessed (train-fit transform applied to
    # the whole cohort upstream); attach() needs the SAME preprocessed
    # features the reference graph's own candidate search uses.
    query_reliability = {name: arr[held_idx] for name, arr in view_observed_full.items()}
    attached = ref_graph.attach(raw_query, query_reliability)
    batch = arch4r.attach_batch(ref_graph, attached, n_train)

    all_view_x = {name: torch.tensor(
        np.concatenate([arr[train_idx], arr[held_idx]]), dtype=torch.float32, device=device)
        for name, arr in view_x_full.items()}
    all_view_observed = {name: torch.tensor(
        np.concatenate([arr[train_idx], arr[held_idx]]), dtype=torch.float32, device=device)
        for name, arr in view_observed_full.items()}

    model.eval()
    with torch.no_grad():
        out = model(all_view_x, all_view_observed, batch, checkpoints)

    # 0.0-filled, not NaN, on train-row positions: every downstream mask
    # (model_val/gate_selection/gate_certification/test) is disjoint from
    # train by construction, so these entries are provably never read -- but
    # gate_lowrisk.calibrate_stages calls IsotonicRegression.predict on the
    # WHOLE array before masking, which raises on NaN input rather than
    # propagating it.
    margins = np.zeros((len(checkpoints), n_total))
    for s, logits in enumerate(out["stage_logits"]):
        margins[s, held_idx] = logits.cpu().numpy()
    graph_available = np.zeros(n_total, dtype=bool)
    graph_available[held_idx] = attached.graph_available
    return margins, graph_available


def _save_checkpoint(model, mode, seed, out_dir="results/checkpoints"):
    """Persist trained weights under `out_dir` so they show up as real
    Kaggle kernel output files (anything under /kaggle/working is captured),
    not just the summary metrics in arch4r_phase1.json."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"arch4r_{mode}_seed{seed}.pt")
    torch.save(model.state_dict(), path)
    return path


def _eval_arch4r_encoder_only(model, checkpoints, view_x_full, view_observed_full, n_total, device):
    view_x = _torch_view_dict(view_x_full, np.arange(n_total), device)
    view_observed = _torch_obs_dict(view_observed_full, np.arange(n_total), device)
    model.eval()
    with torch.no_grad():
        out = model(view_x, view_observed, None, checkpoints)
    margins = np.stack([lg.cpu().numpy() for lg in out["stage_logits"]])
    return margins, np.zeros(n_total, dtype=bool)  # cold-start everywhere, by construction


# --------------------------------------------------------------------- driver

def run_one_seed(cfg, seed, args, cohort, verbose=True):
    X_raw, y, names, meta = cohort
    # graph_build.build_graph's on-disk cache is keyed on cfg["seed"] --
    # leaving it at the config default would make every seed here reuse the
    # first seed's graph AND its train/val/test masks (the exact historical
    # bug documented in old/ARCHITECTURES.md; `run_architectures.py::run_seed`
    # and `run_phase0_integrity.py::run_one_seed` both guard against it the
    # same way). The assertion below is the safety net, not the fix.
    cfg = dict(cfg)
    cfg["seed"] = seed
    device = torch.device(args.device)
    train_mod.set_seed(seed)

    train, val, test = data_prep.make_splits(y, seed)
    v3 = cascade.split_validation_3way(val, y, seed, fracs=tuple(cfg["arch4r"]["val_fracs"]))
    model_val, gate_sel, gate_cert = v3["model_val"], v3["gate_selection"], v3["gate_certification"]

    views = meta.get("views") or _mock_views(X_raw.shape[1])
    view_in_dims = {name: len(idx) for name, idx in views.items()}
    raw_views, view_x_full, view_observed_full = _build_view_arrays(X_raw, views, train)

    n = len(y)
    arms = {}

    def _log(msg):
        if verbose:
            print(f"  [seed {seed}] {msg}", flush=True)

    # ----------------------------------------- arms 1-2: tabular baselines
    X_imputed = data_prep.impute_and_scale(X_raw, train)
    xgb_out = run_xgb_worker(X_imputed, y, (train, val, test), seed=seed)
    arms["plain_xgboost"] = {"test_auroc": float(auroc(y[test], xgb_out["p_test"]))}
    _log(f"plain_xgboost done (auroc={arms['plain_xgboost']['test_auroc']:.4f})")

    import src.tabular as tabular
    cols, stage_names = tabular.stage_columns(
        views, X_raw.shape[1], X_imputed.shape[1],
        tabular.indicator_sources(X_raw, train))
    staged_out = run_xgb_staged_worker(X_imputed, y, (train, val, test), cols,
                                       seed=seed, n_folds=5)
    arms["cascade_3stage"] = {
        "test_auroc": float(auroc(y[test], staged_out["margins"][-1][test])),
        "stage_auroc": [float(auroc(y[test], staged_out["margins"][s][test]))
                        for s in range(len(cols))],
        "stage_names": stage_names,
    }
    _log(f"cascade_3stage done (auroc={arms['cascade_3stage']['test_auroc']:.4f})")

    # ------------------------------------------ arms 3-4: Architecture 2
    from src import graph_build
    data = graph_build.build_graph(X_imputed, y, (train, val, test), cfg,
                                   meta=meta, cache_dir=cfg["data"]["cache_dir"])
    for nm, want, got in (("train", train, data.train_mask),
                          ("val", val, data.val_mask), ("test", test, data.test_mask)):
        assert np.array_equal(want, got.cpu().numpy()), (
            f"graph cache returned another seed's {nm} mask at seed {seed}")
    data = data.to(device)
    ra_args = argparse.Namespace(ppr_alpha=0.15, eps_push=1e-4, n_blocks=args.n_blocks,
                                 epochs=args.arch2_epochs, lam_smooth=0.0, eps_sweep=[1e-4])
    arch2 = ra.run_arch2(data, cfg, ra_args, (y, None, None, test),
                         meta.get("severity"), verbose=False)
    arms["graph_free_encoder"] = {"test_auroc": float(arch2["encoder_only_auroc"])}
    arms["arch2_diffusion"] = {
        "test_auroc": float(auroc(y[test], arch2["margins"][-1][test])),
    }
    _log(f"architecture2 done (graph_free={arms['graph_free_encoder']['test_auroc']:.4f} "
        f"diffusion={arms['arch2_diffusion']['test_auroc']:.4f})")

    # ------------------------------------------------- arms 5-7: Arch4R
    cfg_arch4r = dict(cfg["arch4r"])
    gd_cfg_fixed = gd.ViewGraphConfig(
        k_cand=cfg_arch4r["k_cand"], k0=cfg_arch4r["k0"], gamma=cfg_arch4r["gamma"],
        k_min_adaptive=cfg_arch4r["k_min_adaptive"], k_max_adaptive=cfg_arch4r["k_max_adaptive"],
        percentile_qualify=cfg_arch4r["percentile_qualify"], fixed_k=cfg_arch4r["fixed_k"])
    gd_cfg_adaptive = gd.ViewGraphConfig(
        k_cand=cfg_arch4r["k_cand"], k0=cfg_arch4r["k0"], gamma=cfg_arch4r["gamma"],
        k_min_adaptive=cfg_arch4r["k_min_adaptive"], k_max_adaptive=cfg_arch4r["k_max_adaptive"],
        percentile_qualify=cfg_arch4r["percentile_qualify"], fixed_k=None)

    train_views = {name: arr[train] for name, arr in view_x_full.items()}
    train_reliability = {name: arr[train] for name, arr in view_observed_full.items()}
    ref_fixed = gd.build_reference_graph(train_views, train_reliability, gd_cfg_fixed, seed=seed)
    ref_adaptive = gd.build_reference_graph(train_views, train_reliability, gd_cfg_adaptive, seed=seed)
    _log("reference graphs (fixed_k, adaptive_k) built")

    model_enc, ckpts, t_enc = _train_arch4r(
        "encoder_only", view_in_dims, cfg_arch4r, view_x_full, view_observed_full,
        y, train, None, args.epochs, args.lr, device)
    _save_checkpoint(model_enc, "encoder_only", seed)
    margins_enc, avail_enc = _eval_arch4r_encoder_only(model_enc, ckpts, view_x_full,
                                                       view_observed_full, n, device)
    arms["arch4r_encoder_only"] = {
        "test_auroc": float(auroc(y[test], margins_enc[-1][test])),
        "train_s": t_enc, "n_params": model_enc.n_trainable_params(),
    }
    _log(f"arch4r_encoder_only done (auroc={arms['arch4r_encoder_only']['test_auroc']:.4f}, "
        f"{t_enc:.0f}s train)")

    model_fixed, ckpts, t_fixed = _train_arch4r(
        "fixed_k", view_in_dims, cfg_arch4r, view_x_full, view_observed_full,
        y, train, ref_fixed, args.epochs, args.lr, device)
    _save_checkpoint(model_fixed, "fixed_k", seed)
    margins_fixed, avail_fixed = _eval_arch4r_graph_arm(
        model_fixed, ckpts, view_x_full, view_observed_full, ref_fixed, train, n, device)
    arms["arch4r_fixed_k"] = {
        "test_auroc": float(auroc(y[test], margins_fixed[-1][test])),
        "train_s": t_fixed, "n_params": model_fixed.n_trainable_params(),
        "graph_available_frac_test": float(avail_fixed[test].mean()),
    }
    _log(f"arch4r_fixed_k done (auroc={arms['arch4r_fixed_k']['test_auroc']:.4f}, "
        f"{t_fixed:.0f}s train)")

    model_adaptive, ckpts, t_adaptive = _train_arch4r(
        "adaptive_k", view_in_dims, cfg_arch4r, view_x_full, view_observed_full,
        y, train, ref_adaptive, args.epochs, args.lr, device)
    _save_checkpoint(model_adaptive, "adaptive_k", seed)
    margins_adaptive, avail_adaptive = _eval_arch4r_graph_arm(
        model_adaptive, ckpts, view_x_full, view_observed_full, ref_adaptive, train, n, device)
    arms["arch4r_adaptive_k"] = {
        "test_auroc": float(auroc(y[test], margins_adaptive[-1][test])),
        "stage_auroc": [float(auroc(y[test], margins_adaptive[s][test]))
                        for s in range(len(ckpts))],
        "train_s": t_adaptive, "n_params": model_adaptive.n_trainable_params(),
        "graph_available_frac_test": float(avail_adaptive[test].mean()),
    }
    _log(f"arch4r_adaptive_k done (auroc={arms['arch4r_adaptive_k']['test_auroc']:.4f}, "
        f"{t_adaptive:.0f}s train)")

    # ------------------------------------------- arm 8: certified policy
    stage_margins = [margins_adaptive[s] for s in range(len(ckpts))]
    p_cal, _ = gate_lowrisk.calibrate_stages(stage_margins, y, model_val,
                                             min_n=cfg_arch4r.get("gate_min_n", 100))
    policy = gate_lowrisk.select_policy(p_cal, y, gate_sel, eta=cfg_arch4r["gate_eta"],
                                        min_release=cfg_arch4r["gate_min_release"])
    cert = gate_lowrisk.certify_policy(p_cal, y, gate_cert, policy, avail_adaptive,
                                       eta=cfg_arch4r["gate_eta"], gamma=cfg_arch4r["gate_gamma"])
    stage_idx, risk, released = gate_lowrisk.route(p_cal, policy, avail_adaptive)

    cost_per_checkpoint = np.array(
        [0.0] + [algorithmic_bound(cfg_arch4r["alpha"], c.eps) for c in ckpts[1:]])
    spent = cost_per_checkpoint[stage_idx]
    compute_saved_pct = 100.0 * (1.0 - spent[test].mean() / cost_per_checkpoint[-1])

    arms["arch4r_adaptive_k_certified"] = {
        "certified": cert["certified"],
        "n_enabled": cert["n_enabled"],
        "delta_g": cert["delta_g"],
        "coverage_frac_test": float(released[test].mean()),
        "test_auroc": float(auroc(y[test], risk[test])),
        # Estimated from local_push's own per-eps algorithmic bound
        # (`local_push.algorithmic_bound`), NOT a per-query measured edge
        # count broken down by checkpoint -- the model's ledger only
        # aggregates edges across the whole diffusion pass, not per
        # checkpoint boundary. Labelled explicitly so this is never read as
        # a measured figure (see `src/instrument.py`'s own discipline note).
        "compute_saved_pct_bound_based": float(compute_saved_pct),
        "cert_stage_results": cert["stage_results"],
        "cert_aggregate": cert["aggregate"],
    }

    return {
        "seed": seed, "n": n, "prevalence": float(y.mean()),
        "arms": arms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--source", default="mock", choices=["mock", "mimic", "csv"])
    ap.add_argument("--n-patients", type=int, default=None,
                    help="override cfg.data.n_patients (mock source only, for fast smoke tests)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-blocks", type=int, default=4, help="Architecture 2's push-round budget")
    ap.add_argument("--arch2-epochs", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=30, help="Arch4R training epochs per arm")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--phase0-report", default="results/phase0_integrity.json")
    ap.add_argument("--skip-phase0-check", action="store_true",
                    help="DANGEROUS: only for iterating on this driver itself")
    ap.add_argument("--out", default="results/arch4r_phase1.json")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.skip_phase0_check:
        ok, reason = _check_phase0_passed(os.path.join(_REPO_ROOT, args.phase0_report))
        if not ok:
            print(f"[arch4r-phase1] REFUSING TO RUN: {reason}", flush=True)
            sys.exit(1)

    cfg = yaml.safe_load(open(os.path.join(_REPO_ROOT, args.config)))
    cfg["data"]["source"] = args.source
    if args.n_patients:
        cfg["data"]["n_patients"] = args.n_patients
    if args.raw_dir:
        cfg["data"]["raw_dir"] = args.raw_dir
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir
    cfg["model"]["max_layers"] = args.n_blocks

    cohort = data_prep.load_cohort(cfg)
    print(f"[arch4r-phase1] cohort n={cohort[0].shape[0]} raw_features={cohort[0].shape[1]} "
          f"prevalence={cohort[1].mean():.5f}", flush=True)

    runs = []
    for seed in args.seeds:
        t0 = time.time()
        row = run_one_seed(cfg, seed, args, cohort)
        runs.append(row)
        a = row["arms"]
        print(f"[arch4r-phase1][seed {seed}] "
              f"cascade={a['cascade_3stage']['test_auroc']:.4f} "
              f"xgb={a['plain_xgboost']['test_auroc']:.4f} "
              f"graph_free={a['graph_free_encoder']['test_auroc']:.4f} "
              f"arch2={a['arch2_diffusion']['test_auroc']:.4f} "
              f"arch4r_enc={a['arch4r_encoder_only']['test_auroc']:.4f} "
              f"arch4r_fixed={a['arch4r_fixed_k']['test_auroc']:.4f} "
              f"arch4r_adaptive={a['arch4r_adaptive_k']['test_auroc']:.4f} "
              f"certified={a['arch4r_adaptive_k_certified']['certified']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    out_path = os.path.join(_REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "args": vars(args), "runs": runs}, f, indent=1, default=str)
    print(f"[arch4r-phase1] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
