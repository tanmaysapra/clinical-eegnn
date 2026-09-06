#!/usr/bin/env python
"""The controlled experiment the spec was designed around, finally run.

`Top3_EarlyExit_GNN_Architectures_v2.md` specifies three architectures and
recommends building 1 as primary with 3 as "a corrected, more conventional
fallback/ablation baseline (useful for isolating 'does ACT actually beat
discrete exit stages' as a controlled experiment)". Only Architecture 1 was
ever built, so that comparison never happened, and the project's negative
result could not distinguish *early exit on GNNs fails here* from *this
particular halting mechanism fails here*.

This runs all three on one footing:

  Arch 1  Graph-ACT       shared weight-tied operator, learned ACT halting
  Arch 2  PPR-Push        decoupled encoder + local-push PPR, algorithmic exit
  Arch 3  Fixed-Block     distinct per-block weights, discrete exits,
                          active-subgraph masking, certified gate

Held identical across all three: the cohort, the seed's splits, the patient
graph, the certified two-sided gate (`src/cascade.py`), the calibration split,
and the depth budget. What varies is only the exit mechanism -- which is the
whole point.

Compute is *measured*, not inferred. Architectures 2 and 3 report the edges and
node-updates they actually performed; Architecture 1 reports both its
analytical figure (mean exit depth x per-layer cost, which is all its
implementation supports) and, for honesty, the fact that its backbone runs
dense regardless. See SS5o and `src/fixed_block.py` for why that distinction
matters.

    python scripts/run_architectures.py --source mimic --seeds 42 43 44 45 46 \
        --raw-dir /kaggle/input/datasets/akshaybe/updated-mimic-iv \
        --cache-dir /kaggle/working/shared_cache \
        --out results/architectures.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import cascade, data_prep, graph_build, train as train_mod
from src.evaluate import auroc, predictive_metrics
from src.fixed_block import FixedBlockEE, fixed_block_loss
from src.ppr_push import PPRPush, ppr_loss


# --------------------------------------------------------------------- helpers

def _metrics(y, p, mask):
    m = predictive_metrics(y[mask], p[mask])
    return {"auroc": float(m["auroc"]), "auprc": float(m["auprc"])}


def _gate_block(margins, y, cal, sel, test, alpha, beta, gamma, severity,
                cost_per_stage, conformal=True):
    """Calibrate, certify, route and score -- identical for all three."""
    p, _ = cascade.calibrate_stages(margins, y, cal)
    gate = cascade.fit_gate(p, y, sel, alpha=alpha, beta=beta, gamma=gamma,
                            conformal=conformal)
    stage, risk, side = cascade.route(p, gate)
    rep = cascade.evaluate_cascade(y, stage, risk, side, test, cost_per_stage,
                                   severity=severity)
    rep["taus_lo"] = gate["lo"].tolist()
    rep["taus_hi"] = gate["hi"].tolist()
    rep["mean_stage_test"] = float(stage[test].mean())
    rep["stage_hist_test"] = np.bincount(
        stage[test], minlength=margins.shape[0]).tolist()
    return rep


def _train_generic(model, data, cfg, loss_fn, epochs, patience, lr, verbose,
                   tag):
    """One transductive loop shared by Architectures 2 and 3.

    Deliberately mirrors `train.train_model`: same optimiser, same clipping,
    same early stopping on validation AUROC. If the training protocol differed
    between architectures the comparison would be confounded by it.
    """
    import copy
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=cfg["train"]["weight_decay"])
    best = {"auroc": -1.0, "epoch": -1,
            "state": copy.deepcopy(model.state_dict())}
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        loss, parts = loss_fn(out, data.y, data.train_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            ev = model(data.x, data.edge_index, data.edge_weight)
            lg = ev["block_logits"][-1] if "block_logits" in ev else ev["logits"]
            probs = torch.softmax(lg, dim=-1)[:, 1]
            va = auroc(data.y[data.val_mask].cpu().numpy(),
                       probs[data.val_mask].cpu().numpy())
        if va > best["auroc"]:
            best = {"auroc": va, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        if verbose and epoch % 25 == 0:
            print(f"    [{tag}] epoch {epoch:4d} loss {parts['total']:.4f} "
                  f"val_auroc {va:.4f}", flush=True)
        if epoch - best["epoch"] > patience:
            break
    model.load_state_dict(best["state"])
    return model, best


# ------------------------------------------------------------- architectures

def run_arch1(data, cfg, args, splits, severity, verbose):
    """Graph-ACT, as already built. Compute is analytical -- see the docstring."""
    y, cal, sel, test = splits
    t0 = time.time()
    model, _, best = train_mod.train_model(data, cfg, use_exits=True,
                                           epochs=args.epochs, verbose=False)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight, tau=1.0)
        margins = torch.stack(
            [lg[:, 1] - lg[:, 0] for lg in out["exit_logits"]]
        ).cpu().numpy().astype(np.float64)
        realised_depth = out["exit_layer"].float().cpu().numpy()
    L = cfg["model"]["max_layers"]
    return {
        "arch": "1_graph_act",
        "margins": margins,
        "train_s": time.time() - t0,
        "best_val_auroc": best["auroc"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
        # Analytical only. `model.py` propagates every node at every step and
        # masks afterwards, so no measured edge count exists for this arch.
        "mean_realised_depth": float(realised_depth[test].mean()),
        "analytical_depth_saved_pct": float(
            100.0 * (1.0 - realised_depth[test].mean() / L)),
        "measured_edge_work_saved_pct": None,
        "compute_is_measured": False,
    }


def run_arch3(data, cfg, args, splits, severity, verbose):
    """Fixed-Block with active-set masking. Compute is measured."""
    y, cal, sel, test = splits
    n = data.x.size(0)
    t0 = time.time()
    model = FixedBlockEE(data.x.shape[1], cfg, n_blocks=args.n_blocks).to(data.x.device)
    model, best = _train_generic(
        model, data, cfg,
        lambda o, yy, m: fixed_block_loss(o, yy, m, cfg, beta=args.block_beta),
        args.epochs, cfg["train"]["patience"], cfg["train"]["lr"], verbose, "arch3")

    cache = model.cache_norm(data.edge_index, data.edge_weight, n,
                             data.x.dtype, data.x.device)
    margins = model.block_margins(data.x, data.edge_index, data.edge_weight,
                                  norm_cache=cache).cpu().numpy().astype(np.float64)
    return {
        "arch": "3_fixed_block",
        "margins": margins,
        "train_s": time.time() - t0,
        "best_val_auroc": best["auroc"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "_model": model, "_cache": cache,
        "compute_is_measured": True,
    }


def run_arch2(data, cfg, args, splits, severity, verbose):
    """PPR-Push. Compute is measured, and bounded by the push tolerance."""
    y, cal, sel, test = splits
    t0 = time.time()
    model = PPRPush(data.x.shape[1], cfg, alpha=args.ppr_alpha,
                    eps_push=args.eps_push,
                    max_rounds=args.n_blocks).to(data.x.device)
    model, best = _train_generic(
        model, data, cfg,
        lambda o, yy, m: ppr_loss(o, yy, m, cfg, lam_smooth=args.lam_smooth,
                                  edge_index=data.edge_index,
                                  edge_weight=data.edge_weight),
        args.epochs, cfg["train"]["patience"], cfg["train"]["lr"], verbose, "arch2")

    snaps = tuple(range(1, args.n_blocks))
    margins, ledger = model.stage_margins(data.x, data.edge_index,
                                          data.edge_weight, snapshots=snaps)
    ledger = {k: (v.tolist() if torch.is_tensor(v) else v)
              for k, v in ledger.items() if k != "exit_round"}

    # The tolerance sweep IS Architecture 2's efficiency result. A single
    # eps_push reports whatever saving that one setting happens to give; the
    # curve is what shows whether work can be traded for accuracy at all, and
    # it is the only compute claim in this project with a bound behind it.
    sweep = []
    with torch.no_grad():
        for eps in args.eps_sweep:
            o = model(data.x, data.edge_index, data.edge_weight, eps_push=eps)
            pr = torch.softmax(o["logits"], dim=-1)[:, 1].cpu().numpy()
            L = o["ledger"]
            sweep.append({
                "eps_push": eps,
                "edges_touched": int(L["edges_touched"]),
                "edge_work_saved_pct": float(L["edge_work_saved_pct"]),
                "mean_exit_round": float(L["mean_exit_round"]),
                "rounds_used": int(L["rounds_used"]),
                "test_auroc": float(auroc(y[test], pr[test])),
            })
        # The graph-free floor: what the encoder alone achieves. If diffusion
        # buys nothing over this, Architecture 2's propagation is as redundant
        # as the graph stage in SS5n.
        enc = torch.softmax(
            model(data.x, data.edge_index, data.edge_weight)["encoder_logits"],
            dim=-1)[:, 1].cpu().numpy()
    return {
        "eps_sweep": sweep,
        "encoder_only_auroc": float(auroc(y[test], enc[test])),
        "arch": "2_ppr_push",
        "margins": margins.cpu().numpy().astype(np.float64),
        "train_s": time.time() - t0,
        "best_val_auroc": best["auroc"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "ledger": ledger,
        "measured_edge_work_saved_pct": ledger["edge_work_saved_pct"],
        "compute_is_measured": True,
    }


# --------------------------------------------------------------------- driver

def run_seed(cfg, seed, args, cohort, verbose=True):
    X_raw, y, names, meta = cohort
    # `graph_build.build_graph` keys its on-disk cache on cfg["seed"] AND stores
    # the split masks inside the cached Data object. Leaving cfg["seed"] at the
    # config default makes every seed reuse the first seed's graph *and* its
    # train/val/test masks -- i.e. train on one seed's train split and score on
    # another's test split, which overlap. Same family as the seed-blind cache
    # at run_cascade.py:76, with a much worse consequence.
    cfg["seed"] = seed
    train_mod.set_seed(seed)
    device = torch.device(args.device)

    train, val, test = data_prep.make_splits(y, seed)
    cal, sel = cascade.split_validation(val, y, seed)
    X = data_prep.impute_and_scale(X_raw, train)
    severity = meta.get("severity")

    data = graph_build.build_graph(X, y, (train, val, test), cfg, meta=meta,
                                   cache_dir=cfg["data"]["cache_dir"])
    # Cheap invariant that would have caught the above immediately.
    for nm, want, got in (("train", train, data.train_mask),
                          ("val", val, data.val_mask),
                          ("test", test, data.test_mask)):
        assert np.array_equal(want, got.cpu().numpy()), (
            f"graph cache returned another seed's {nm} mask at seed {seed}")
    data = data.to(device)
    splits = (y, cal, sel, test)

    # Equal per-stage cost in the gate's ledger: with the exit mechanism as the
    # only difference, a stage is a stage. The architectures' real cost
    # differences live in their measured work ledgers, reported separately.
    n_stages = args.n_blocks
    cost_per_stage = np.ones(n_stages, dtype=float)

    out = {"seed": seed, "n": int(len(y)), "prevalence": float(y.mean()),
           "n_blocks": args.n_blocks, "archs": {}}

    runners = [("1_graph_act", run_arch1), ("2_ppr_push", run_arch2),
               ("3_fixed_block", run_arch3)]
    for name, fn in runners:
        if args.only and name not in args.only:
            continue
        if verbose:
            print(f"  [seed {seed}] {name} ...", flush=True)
        r = fn(data, cfg, args, splits, severity, verbose)
        margins = r.pop("margins")
        model = r.pop("_model", None)
        cache = r.pop("_cache", None)

        # Full-depth quality, before any gating.
        p_full = 1.0 / (1.0 + np.exp(-margins[-1]))
        r["full_depth"] = _metrics(y, p_full, test)
        r["stage_auroc"] = [float(auroc(y[test], margins[s][test]))
                            for s in range(margins.shape[0])]
        # The contrast that killed the depth cascade: are the stages different?
        r["stage_gap_last_minus_first"] = r["stage_auroc"][-1] - r["stage_auroc"][0]

        r["gates"] = {}
        for a in args.alphas:
            rep = _gate_block(margins, y, cal, sel, test, a, args.beta,
                              args.gamma, severity, cost_per_stage)
            # Architecture 3 can now realise the routing it was given and
            # report what that actually cost -- the spec's Fix 1, measured.
            if model is not None:
                stage_all = np.full(len(y), n_stages - 1, dtype=int)
                p_cal, _ = cascade.calibrate_stages(margins, y, cal)
                g = cascade.fit_gate(p_cal, y, sel, alpha=a, beta=args.beta,
                                     gamma=args.gamma)
                stage_all, _, _ = cascade.route(p_cal, g)
                eb = torch.as_tensor(stage_all, dtype=torch.long, device=device)
                mm = model.forward_masked(data.x, data.edge_index,
                                          data.edge_weight, exit_block=eb,
                                          norm_cache=cache)
                rep["measured_edge_work_saved_pct"] = mm["edge_work_saved_pct"]
                rep["measured_node_work_saved_pct"] = mm["node_work_saved_pct"]
                rep["edges_processed"] = mm["edges_processed"]
            r["gates"][f"alpha={a}"] = rep
        out["archs"][name] = r
        if verbose:
            print(f"    {name}: full-depth AUROC {r['full_depth']['auroc']:.4f}  "
                  f"stage gap {r['stage_gap_last_minus_first']:+.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-dir"); ap.add_argument("--cache-dir")
    ap.add_argument("--source", default="mimic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.02, 0.05])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--n-blocks", type=int, default=4,
                    help="depth budget, shared by all three architectures")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--block-beta", type=float, default=0.5,
                    help="exponential exit-weight beta for Architecture 3")
    ap.add_argument("--ppr-alpha", type=float, default=0.15)
    ap.add_argument("--eps-push", type=float, default=1e-4)
    ap.add_argument("--eps-sweep", type=float, nargs="+",
                    default=[1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1],
                    help="Architecture 2's work/accuracy curve")
    ap.add_argument("--lam-smooth", type=float, default=0.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/architectures.json")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = yaml.safe_load(open(args.config))
    cfg["data"]["source"] = args.source
    if args.raw_dir:
        cfg["data"]["raw_dir"] = args.raw_dir
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir
    cfg["model"]["max_layers"] = args.n_blocks

    cohort = data_prep.load_cohort(cfg)
    print(f"[cohort] n={cohort[0].shape[0]} raw_features={cohort[0].shape[1]} "
          f"prevalence={cohort[1].mean():.5f}", flush=True)

    runs = []
    for s in args.seeds:
        t0 = time.time()
        runs.append(run_seed(cfg, s, args, cohort))
        print(f"[seed {s}] done in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"config": cfg, "args": vars(args), "runs": runs},
              open(args.out, "w"), indent=1, default=str)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
