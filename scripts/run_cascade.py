#!/usr/bin/env python
"""Run the staged hybrid cascade end to end and write results as JSON.

    python scripts/run_cascade.py --seeds 42 43 44 --out results/cascade.json

Every comparison is paired: one set of splits per seed, shared by every model
and every gate setting, so differences are not split noise.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import cascade, data_prep, graph_build, hybrid, tabular
from src.cascade import split_validation
from src.evaluate import expected_calibration_error, predictive_metrics


def _mock_views(n_raw):
    """Pseudo-blocks for the mock generator, which has no real view membership.

    Smoke-path only -- the real cohort supplies `meta["views"]` from the ETL.
    """
    cut = np.array_split(np.arange(n_raw), 5)
    return dict(zip(["demographics", "vitals", "treatment", "comorbidity", "labs"],
                    [c.tolist() for c in cut]))


def _graph_cache_key(cfg, seed):
    """Cache key for the kNN graph -- must include the seed. `cache` is one
    dict shared by every seed in a run (see `main`), and `make_splits` gives
    each seed a different train mask, so the graph itself is seed-dependent."""
    return ("graph", seed, cfg["graph"]["k"], cfg["graph"]["similarity"])


def run_seed(cfg, seed, args, cache):
    t0 = time.time()
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = seed

    if "cohort" not in cache:
        cache["cohort"] = data_prep.load_cohort(cfg)
    X_raw, y, names, meta = cache["cohort"]
    n = len(y)

    train, val, test = data_prep.make_splits(y, seed)
    cal, sel = split_validation(val, y, seed)
    X = data_prep.impute_and_scale(X_raw, train)
    severity = meta.get("severity")

    views = meta.get("views") or _mock_views(X_raw.shape[1])
    cols, stage_names = tabular.stage_columns(
        views, X_raw.shape[1], X.shape[1],
        tabular.indicator_sources(X_raw, train))
    print(f"[seed {seed}] stages: " +
          ", ".join(f"{nm}({len(c)})" for nm, c in zip(stage_names, cols)), flush=True)

    # ---------------------------------------------------------- tabular stages
    boost = tabular.fit_staged_boosting(X, y, train, cols, seed=seed,
                                        n_folds=args.folds, device=args.xgb_device)
    margins = list(boost["margins"])
    cost_us = list(boost["cost_us"])
    stage_names = list(stage_names)

    # ------------------------------------------------------------ graph stage
    tg = time.time()
    if args.graph_stage == "none":
        edge_index = edge_weight = None
    elif args.graph == "leaf":
        # Edges from the booster's own partition of the space, not from X.
        Z = tabular.leaf_onehot(
            tabular.leaf_indices(boost["models"][-1], X, cols[-1]))
        edge_index, edge_weight = graph_build.build_knn_graph(
            Z, cfg["graph"]["k"], "cosine")
    else:
        gkey = _graph_cache_key(cfg, seed)
        if gkey not in cache:
            cache[gkey] = graph_build.build_knn_graph(
                X, cfg["graph"]["k"], cfg["graph"]["similarity"],
                binary_cols=meta.get("binary_cols"))
        edge_index, edge_weight = cache[gkey]
    graph_build_s = time.time() - tg

    base_p = 1.0 / (1.0 + np.exp(-margins[-1]))
    tg = time.time()
    graph_note, t_graph = {}, 0.0
    if args.graph_stage == "none":
        pass
    elif args.graph_stage == "cs":
        # The correction scale is an ordinary hyper-parameter, not a safety
        # threshold, so it is fitted on the whole validation split; the 1.4k
        # calibration half alone made it swing between 0.0 and 0.9 across seeds.
        scale, _ = hybrid.fit_cs_scale(edge_index, edge_weight, base_p, y, train,
                                       val, n, hops=args.cs_hops, device=args.device)
        p_graph = hybrid.correct_and_smooth(edge_index, edge_weight, base_p, y,
                                            train, n, scale=scale,
                                            hops=args.cs_hops, device=args.device)
        m_graph = np.log(p_graph / (1 - p_graph))
        graph_note = {"cs_scale": scale}
    else:
        m_graph, best_val, _ = hybrid.train_residual_gnn(
            X, margins[-1], y, edge_index, edge_weight, train, val, n,
            device=args.device, seed=seed)
        graph_note = {"gnn_val_auroc": best_val}
    if args.graph_stage != "none":
        t_graph = time.time() - tg
        margins.append(m_graph)
        # Cost of the graph stage per patient, over the nodes that could use it.
        cost_us.append(1e6 * t_graph / n)
        stage_names.append("cohort")

    margins = np.stack(margins)
    cost_us = np.array(cost_us)

    # --------------------------------------------------------- calibration
    p_glob, _ = cascade.calibrate_stages(margins, y, cal, severity, n_strata=1)
    p_strat, _ = cascade.calibrate_stages(margins, y, cal, severity,
                                          n_strata=args.n_strata)

    out = {"seed": seed, "n": int(n), "prevalence": float(y.mean()),
           "stage_names": stage_names, "stage_cols": [int(len(c)) for c in cols],
           "cost_us": cost_us.tolist(), "graph": graph_note,
           "graph_build_s": graph_build_s, "stages": {}, "gates": {},
           "baselines": {}}

    # Per-stage standalone quality: the thing the depth cascade never had.
    for s, nm in enumerate(stage_names):
        out["stages"][nm] = {
            **predictive_metrics(y[test], p_glob[s][test],
                                 cfg["eval"]["operating_sensitivity"]),
            "ece": expected_calibration_error(y[test], p_glob[s][test]),
            "cum_cost_us": float(cost_us[: s + 1].sum()),
        }

    # ------------------------------------------------------------- baselines
    p_ref = tabular.reference_xgboost(X, y, train, seed=seed, device=args.xgb_device)
    out["baselines"]["xgboost_reference"] = {
        **predictive_metrics(y[test], p_ref[test], cfg["eval"]["operating_sensitivity"]),
        "ece": expected_calibration_error(y[test], p_ref[test]),
    }
    # The two trivial policies the learned depth policy lost to (HANDOVER 5h).
    for nm, fixed in (("always_stage0", 0), ("always_full", len(stage_names) - 1)):
        stage = np.full(n, fixed)
        out["baselines"][nm] = cascade.evaluate_cascade(
            y, stage, p_glob[fixed], np.zeros(n, int), test, cost_us,
            severity=severity, p_full=p_glob[-1],
            operating_sensitivity=cfg["eval"]["operating_sensitivity"],
            top_frac=cfg["eval"]["high_risk_top_frac"])

    # ------------------------------------------------------------- the gates
    for cal_name, p in (("global", p_glob), ("stratified", p_strat)):
        for alpha in args.alphas:
            for two_sided in (True, False):
                for conformal in ([True, False] if args.ablate_conformal else [True]):
                    beta = args.beta if two_sided else 0.0
                    g = cascade.fit_gate(p, y, sel, alpha=alpha, beta=beta,
                                         gamma=args.gamma, conformal=conformal)
                    stage, risk, side = cascade.route(p, g)
                    key = (f"{cal_name}|alpha={alpha}|"
                           f"{'two' if two_sided else 'one'}sided|"
                           f"{'conformal' if conformal else 'empirical'}")
                    out["gates"][key] = {
                        **cascade.evaluate_cascade(
                            y, stage, risk, side, test, cost_us, severity=severity,
                            p_full=p[-1],
                            operating_sensitivity=cfg["eval"]["operating_sensitivity"],
                            top_frac=cfg["eval"]["high_risk_top_frac"]),
                        "taus_lo": g["lo"].tolist(), "taus_hi": g["hi"].tolist(),
                    }
    out["wall_s"] = time.time() - t0
    print(f"[seed {seed}] done in {out['wall_s']:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-dir"); ap.add_argument("--cache-dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--beta", type=float, default=0.5,
                    help="max survival rate among escalated patients")
    ap.add_argument("--gamma", type=float, default=0.05,
                    help="1-gamma is the confidence of the safety certificate")
    ap.add_argument("--n-strata", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--graph-stage", choices=["cs", "gnn", "none"], default="cs")
    ap.add_argument("--graph", choices=["knn", "leaf"], default="knn",
                    help="knn: cosine over features. leaf: cosine over the "
                         "booster's leaf codes, i.e. edges from outside X.")
    ap.add_argument("--cs-hops", type=int, default=2)
    ap.add_argument("--ablate-conformal", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--xgb-device", default="cpu")
    ap.add_argument("--source", default="mimic")
    ap.add_argument("--out", default="results/cascade.json")
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = yaml.safe_load(open(args.config))
    cfg["data"]["source"] = args.source
    if args.raw_dir:
        cfg["data"]["raw_dir"] = args.raw_dir
    if args.cache_dir:
        cfg["data"]["cache_dir"] = args.cache_dir

    cache = {}
    runs = [run_seed(cfg, s, args, cache) for s in args.seeds]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"config": cfg, "args": vars(args), "runs": runs},
              open(args.out, "w"), indent=1, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
