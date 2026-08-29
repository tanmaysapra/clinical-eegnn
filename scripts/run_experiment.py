#!/usr/bin/env python
"""CLI entry point: parse config, build the graph, dispatch to a stage.

    python scripts/run_experiment.py --stage smoke   --data mock
    python scripts/run_experiment.py --stage baselines
    python scripts/run_experiment.py --stage eegnn
    python scripts/run_experiment.py --stage full --seeds 3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

# NumPy 2.0 on Apple Accelerate emits spurious divide-by-zero/overflow warnings
# from float32 matmul. Verified harmless: results match a manual dot product and
# stay finite. Remove this once NumPy/Accelerate fix it upstream.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                        category=RuntimeWarning)
import torch  # noqa: E402
import yaml  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src.baselines import (GAT, GCN, fixed_threshold_ee, train_plain_gnn,  # noqa: E402
                           xgboost_baseline)
from src.data_prep import config_hash, impute_and_scale, load_cohort, make_splits
from src.graph_build import build_graph, describe
from src.train import get_device, set_seed, train_model


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in overrides.items():
        if value is None:
            continue
        section, _, field = key.partition(".")
        if field:
            cfg[section][field] = value
        else:
            cfg[key] = value
    return cfg


def prepare(cfg):
    """Cohort -> splits -> imputation -> graph. Returns (data, X, y, masks, names)."""
    X_raw, y, names, meta = load_cohort(cfg)
    masks = make_splits(y, cfg["seed"])
    X = impute_and_scale(X_raw, masks[0])
    data = build_graph(X, y, masks, cfg, meta=meta)
    return data, X, y, masks, names


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


# ------------------------------------------------------------------- stages

def stage_smoke(cfg):
    """P0 acceptance: build a graph and train one epoch end to end."""
    data, *_ = prepare(cfg)
    print("graph:", describe(data))
    model, history, _ = train_model(data, cfg, epochs=1, verbose=True, log_every=1)
    print(f"P0 OK -- loss after 1 epoch: {history[-1]['total']:.4f}, "
          f"val_auroc {history[-1]['val_auroc']:.4f}")
    return {"loss": history[-1]["total"], "val_auroc": history[-1]["val_auroc"]}


def stage_baselines(cfg):
    """P2 acceptance: SAS-GNN at full depth vs GCN / GAT / XGBoost / naive EE."""
    data, X, y, masks, _ = prepare(cfg)
    print("graph:", describe(data))
    device = get_device()
    data = data.to(device)
    results = {}

    print("\n[XGBoost]")
    _, xgb, _ = xgboost_baseline(X, y, masks, seed=cfg["seed"])
    results["xgboost"] = xgb
    print(f"  test AUROC {xgb['test_auroc']:.4f}")

    for name, cls in (("gcn", GCN), ("gat", GAT)):
        print(f"\n[{name.upper()}]")
        model = cls(data.x.shape[1], cfg["model"]["hidden_dim"], n_layers=2).to(device)
        model, _ = train_plain_gnn(model, data, cfg)
        with torch.no_grad():
            p = torch.softmax(model(data.x, data.edge_index, data.edge_weight), -1)[:, 1]
        m = data.test_mask.cpu().numpy()
        results[name] = ev.predictive_metrics(
            data.y.cpu().numpy()[m], p.cpu().numpy()[m],
            cfg["eval"]["operating_sensitivity"],
        )
        print(f"  test AUROC {results[name]['auroc']:.4f} "
              f"AUPRC {results[name]['auprc']:.4f}")

    print("\n[SAS-GNN full depth]")
    model, _, _ = train_model(data, cfg, use_exits=False)
    res = ev.evaluate_model(model, data, cfg, split="test")
    results["sasgnn_full"] = res["predictive"]
    print(f"  test AUROC {res['predictive']['auroc']:.4f} "
          f"AUPRC {res['predictive']['auprc']:.4f}")

    print("\n[fixed-threshold EE on GCN]")
    ee = fixed_threshold_ee(data, cfg)
    m = data.test_mask.cpu().numpy()
    results["fixed_threshold_ee"] = {
        **ev.predictive_metrics(data.y.cpu().numpy()[m], ee["probs"][m],
                                cfg["eval"]["operating_sensitivity"]),
        "avg_exit_depth": ee["avg_exit_depth"],
    }
    print(f"  test AUROC {results['fixed_threshold_ee']['auroc']:.4f} "
          f"avg depth {ee['avg_exit_depth']:.2f}")

    ok = results["sasgnn_full"]["auroc"] >= results["gcn"]["auroc"]
    print(f"\nP2 acceptance (SAS-GNN >= GCN): {'PASS' if ok else 'FAIL'}")
    return results


def stage_eegnn(cfg):
    """P3 + P4: multi-exit training, calibration, and the three metric families."""
    data, *_ = prepare(cfg)
    print("graph:", describe(data))
    data = data.to(get_device())

    model, _, best = train_model(data, cfg, use_exits=True)
    print(f"\nbest val AUROC {best['auroc']:.4f} at epoch {best['epoch']}")

    temp = ev.calibrate_on_val(model, data)
    exit_temps = ev.calibrate_on_val(model, data, per_exit=True)
    print(f"temperature scaling: T = {temp:.3f} | per-exit T = "
          f"{[round(t, 2) for t in exit_temps]}")

    raw = ev.evaluate_model(model, data, cfg, split="test")
    cal = ev.evaluate_model(model, data, cfg, split="test", temperature=temp,
                            exit_temperatures=exit_temps)
    report(raw, cal, temp)
    return {"raw": raw, "calibrated": cal, "temperature": temp,
            "exit_temperatures": exit_temps}


def stage_sweep(cfg, out_dir):
    """P4: sweep the efficiency/safety pressures to draw the trade-off curve.

    `ponder_cost` buys shallower exits on low-risk patients and
    `high_risk_halt_penalty` keeps high-risk patients deep; the interesting
    operating points are where both are active, so we sweep them as a pair.
    """
    data, *_ = prepare(cfg)
    print("graph:", describe(data))
    data = data.to(get_device())

    pairs = [(0.0, 0.0), (0.1, 0.05), (0.3, 0.15), (0.3, 0.3),
             (0.5, 0.3), (0.3, 0.6), (0.6, 0.9), (0.1, 1.5)]
    points, rows = [], []
    for ponder, halt in pairs:
        cfg_i = json.loads(json.dumps(cfg))
        cfg_i["train"]["ponder_cost"] = ponder
        cfg_i["train"]["high_risk_halt_penalty"] = halt
        set_seed(cfg_i["seed"])

        model, _, _ = train_model(data, cfg_i, use_exits=True, verbose=False)
        temp = ev.calibrate_on_val(model, data)
        res = ev.evaluate_model(model, data, cfg_i, split="test", temperature=temp)

        row = {
            "ponder_cost": ponder,
            "halt_penalty": halt,
            "auroc": res["predictive"]["auroc"],
            "auprc": res["predictive"]["auprc"],
            "avg_depth": res["efficiency"]["avg_exit_depth"],
            "hi_depth": res["safety"]["high_risk_mean_depth"],
            "lo_depth": res["safety"]["low_risk_mean_depth"],
            "hi_miss": res["safety"]["high_risk_miss_rate"],
            "ece": res["calibration"]["ece"],
        }
        rows.append(row)
        points.append({"avg_depth": row["avg_depth"], "auroc": row["auroc"],
                       "label": f"{ponder}/{halt}"})
        print(f"  ponder {ponder:<4} halt {halt:<4} | AUROC {row['auroc']:.4f} "
              f"| depth {row['avg_depth']:5.2f} (hi {row['hi_depth']:5.2f} / "
              f"lo {row['lo_depth']:5.2f}) | hi_miss {row['hi_miss']:.3f}")

    print("\nSafety ordering (want hi_depth > lo_depth -- sick patients go deeper):")
    for r in rows:
        ok = "OK  " if r["hi_depth"] > r["lo_depth"] else "INV "
        print(f"  {ok} ponder {r['ponder_cost']:<4} halt {r['halt_penalty']:<4} "
              f"delta {r['hi_depth'] - r['lo_depth']:+.2f}")

    from src.plots import tradeoff_curve
    print(f"\nwrote {tradeoff_curve(points, out_dir)}")
    return {"sweep": rows}


def stage_full(cfg, seeds):
    """Every headline number as mean +/- std over >= 3 seeds."""
    runs = []
    for i in range(seeds):
        cfg_i = json.loads(json.dumps(cfg))
        cfg_i["seed"] = cfg["seed"] + i
        set_seed(cfg_i["seed"])
        print(f"\n{'='*60}\nseed {cfg_i['seed']}\n{'='*60}")
        runs.append({"baselines": stage_baselines(cfg_i), "eegnn": stage_eegnn(cfg_i)})

    print(f"\n{'='*60}\nAGGREGATE over {seeds} seeds (mean +/- std)\n{'='*60}")
    summary = {}
    for model_name in runs[0]["baselines"]:
        vals = [r["baselines"][model_name].get("auroc",
                r["baselines"][model_name].get("test_auroc")) for r in runs]
        summary[model_name] = (float(np.mean(vals)), float(np.std(vals)))
    vals = [r["eegnn"]["calibrated"]["predictive"]["auroc"] for r in runs]
    summary["eegnn"] = (float(np.mean(vals)), float(np.std(vals)))
    depths = [r["eegnn"]["calibrated"]["efficiency"]["avg_exit_depth"] for r in runs]

    for name, (mu, sd) in summary.items():
        print(f"  {name:22s} AUROC {mu:.4f} +/- {sd:.4f}")
    print(f"  {'eegnn avg depth':22s} {np.mean(depths):.2f} +/- {np.std(depths):.2f} "
          f"/ {cfg['model']['max_layers']}")
    return {"runs": runs, "summary": summary}


def report(raw, cal, temp):
    print("\n--- predictive (test, calibrated) ---")
    for k, v in cal["predictive"].items():
        print(f"  {k:22s} {v:.4f}")
    print("\n--- calibration ---")
    print(f"  {'ece (raw)':22s} {raw['calibration']['ece']:.4f}")
    print(f"  {'ece (T=%.2f)' % temp:22s} {cal['calibration']['ece']:.4f}")
    print(f"  {'brier (calibrated)':22s} {cal['calibration']['brier']:.4f}")
    print("\n--- efficiency ---")
    for k, v in cal["efficiency"].items():
        print(f"  {k:22s} {v:.4g}")
    print("\n--- safety ---")
    for k, v in cal["safety"].items():
        print(f"  {k:22s} {v:.4g}")
    print("\n--- exit distribution (test nodes per layer) ---")
    print(f"  {cal['exit_distribution']}")
    print("\n--- per-exit ---")
    print(f"  {'layer':>6} {'n':>7} {'auroc':>8} {'auprc':>8} {'ece':>8}")
    for e in cal["per_exit"]:
        print(f"  {e['layer']:>6} {e['n_exiting']:>7} {e['auroc']:>8.4f} "
              f"{e['auprc']:>8.4f} {e['ece']:>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", default="smoke",
                    choices=["smoke", "baselines", "eegnn", "sweep", "full"])
    ap.add_argument("--data", dest="data.source", default=None,
                    choices=["mock", "mimic", "csv"])
    ap.add_argument("--task", dest="data.task", default=None)
    ap.add_argument("--raw-dir", dest="data.raw_dir", default=None,
                    help="dir holding mimic-iv-3.1/{hosp,icu} (mimic only)")
    ap.add_argument("--cache-dir", dest="data.cache_dir", default=None)
    ap.add_argument("--max-patients", dest="data.max_patients", type=int, default=None,
                    help="label-stratified cohort subsample (mimic only)")
    ap.add_argument("--obs-window", dest="data.obs_window_hours", type=float,
                    default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", dest="train.epochs", type=int, default=None)
    ap.add_argument("--ponder-cost", dest="train.ponder_cost", type=float, default=None)
    ap.add_argument("--halt-penalty", dest="train.high_risk_halt_penalty",
                    type=float, default=None)
    ap.add_argument("--max-layers", dest="model.max_layers", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3, help="stage=full only")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "stage", "seeds", "out")}
    cfg = load_config(args.config, overrides)
    set_seed(cfg["seed"])

    cfg_hash = config_hash(cfg)
    print(f"config {cfg_hash} | git {git_sha()} | seed {cfg['seed']} | "
          f"device {get_device()} | stage {args.stage}")

    t0 = time.time()
    if args.stage == "smoke":
        results = stage_smoke(cfg)
    elif args.stage == "baselines":
        results = stage_baselines(cfg)
    elif args.stage == "eegnn":
        results = stage_eegnn(cfg)
    elif args.stage == "sweep":
        results = stage_sweep(cfg, os.path.join(args.out, "figures"))
    else:
        results = stage_full(cfg, args.seeds)
    print(f"\nelapsed {time.time() - t0:.1f}s")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.stage}_{cfg_hash}.json")
    with open(path, "w") as f:
        json.dump({"config": cfg, "config_hash": cfg_hash, "git": git_sha(),
                   "stage": args.stage, "results": results}, f, indent=2, default=str)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
