"""Three metric families: predictive, efficiency, safety.

Predictive metrics lead with AUPRC as well as AUROC (the label is imbalanced),
and report sensitivity/specificity at a *fixed operating sensitivity* because
that is the operating point a clinician actually picks.

Safety metrics only mean something once confidence is calibrated, so
temperature scaling runs on the validation split before any of them are trusted.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)


# ---------------------------------------------------------------- predictive

def auroc(y, p):
    if len(np.unique(y)) < 2 or not np.isfinite(p).all():
        return float("nan")
    return float(roc_auc_score(y, p))


def threshold_at_sensitivity(y, p, target_sensitivity):
    """Lowest threshold whose recall still meets the target sensitivity."""
    pos = p[y == 1]
    if len(pos) == 0:
        return 0.5
    return float(np.quantile(pos, 1.0 - target_sensitivity))


def predictive_metrics(y, p, operating_sensitivity=0.90):
    if not np.isfinite(p).all():
        raise ValueError("non-finite predicted probabilities -- refusing to score")
    thr = threshold_at_sensitivity(y, p, operating_sensitivity)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "auroc": auroc(y, p),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "threshold": thr,
    }


def c_index(risk, time, event, chunk=2048):
    """Harrell's concordance index -- the spec's third clinical metric.

    Comparable pairs are those where the earlier of the two had the event; a
    pair is concordant when the patient who failed sooner carried the higher
    predicted risk. Ties in risk count a half.

    On this cohort the survival framing is an approximation and should be
    reported as one: `time` is ICU length of stay and `event` is in-hospital
    death, so survivors are treated as censored at ICU discharge. That matches
    the discrete-time setup in the GNN-surv reference rather than a true
    time-to-death analysis.

    Computed in row blocks so the n^2 pair matrix is never materialised.
    """
    risk = np.asarray(risk, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event).astype(bool)
    n = len(risk)
    conc = perm = 0.0

    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        # i indexes the block, j every patient; keep pairs where i had the
        # event and failed strictly earlier than j.
        ti, ri, ei = time[sl, None], risk[sl, None], event[sl, None]
        comparable = ei & (ti < time[None, :])
        if not comparable.any():
            continue
        higher = (ri > risk[None, :]) & comparable
        tied = (ri == risk[None, :]) & comparable
        conc += float(higher.sum()) + 0.5 * float(tied.sum())
        perm += float(comparable.sum())

    return float(conc / perm) if perm > 0 else float("nan")


# --------------------------------------------------------------- calibration

def expected_calibration_error(y, p, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def reliability_curve(y, p, n_bins=15):
    """(bin_confidence, bin_accuracy, bin_count) for reliability diagrams."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        sel = idx == b
        conf.append(float(p[sel].mean()) if sel.any() else np.nan)
        acc.append(float(y[sel].mean()) if sel.any() else np.nan)
        cnt.append(int(sel.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def fit_temperature(logits, y, lo=-3.0, hi=3.0, n_grid=60, n_refine=4):
    """Temperature scaling fit on the validation split. Returns scalar T.

    Coarse-to-fine grid search over log T rather than LBFGS: the objective is
    1-D and smooth, and LBFGS was diverging to NaN on the near-separable exit
    logits this model produces. A grid cannot diverge.
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)

    def nll(log_t):
        loss = F.cross_entropy(logits / float(np.exp(log_t)), y)
        return float(loss) if torch.isfinite(loss) else float("inf")

    best = 0.0
    for _ in range(n_refine):
        grid = np.linspace(lo, hi, n_grid)
        losses = [nll(g) for g in grid]
        best = float(grid[int(np.argmin(losses))])
        step = (hi - lo) / n_grid
        lo, hi = best - step, best + step

    T = float(np.exp(best))
    return T if np.isfinite(T) and T > 0 else 1.0


def calibration_metrics(y, p, n_bins=15):
    return {
        "ece": expected_calibration_error(y, p, n_bins),
        "brier": float(brier_score_loss(y, p)),
    }


# ---------------------------------------------------------------- efficiency

def efficiency_metrics(model, data, out, mask):
    """Average realised depth plus a FLOP estimate and wall-clock latency."""
    depth = out["exit_layer"][mask].float()
    hidden = model.hidden_dim
    n_edges = data.edge_index.shape[1]
    n_nodes = int(mask.sum())

    # Per propagation step: one dense hidden x hidden matmul per node plus one
    # message per edge. Cheap approximation, but comparable across models.
    per_step = 2 * hidden * hidden + 2 * hidden * (n_edges / max(data.num_nodes, 1))
    mean_depth = float(depth.mean())

    t0 = time.perf_counter()
    with torch.no_grad():
        model.eval()
        model(data.x, data.edge_index, data.edge_weight, tau=0.5, hard=True)
    latency = (time.perf_counter() - t0) / max(n_nodes, 1)

    return {
        "avg_exit_depth": mean_depth,
        "max_depth": model.max_layers,
        "depth_saving_pct": 100.0 * (1.0 - mean_depth / model.max_layers),
        "flops_per_patient": float(mean_depth * per_step),
        "latency_s_per_patient": float(latency),
    }


# -------------------------------------------------------------------- safety

def high_risk_analysis(y, p, exit_layer, max_layers, operating_sensitivity=0.90,
                       risk=None, top_frac=0.10, n_bins=15):
    """Does the model spend depth where being wrong is most costly?

    The stratum must be defined **independently of the model's own decisions**.
    An earlier version defined "high risk" as `y == 1` and then measured the
    miss rate at a threshold chosen to hit `operating_sensitivity` -- which
    makes the miss rate identically `1 - operating_sensitivity` for every
    model, carrying no information (HANDOVER 6.1). It sat at a constant 0.104
    across an entire sweep for exactly that reason.

    Passing `risk` -- a severity proxy such as `criticalness`, derived from the
    full stay and never a model input -- fixes this. The stratum becomes the
    top `top_frac` of severity, fixed before any prediction is made, so miss
    rate, calibration and depth inside it are all free to vary with the model.

    Without `risk` the old label-based split is kept for the depth ordering,
    but the degenerate rate is reported as nan rather than as a constant that
    looks like a result.
    """
    thr = threshold_at_sensitivity(y, p, operating_sensitivity)
    pred = (p >= thr).astype(int)

    if risk is not None:
        risk = np.asarray(risk, dtype=np.float64)
        hi = risk >= np.quantile(risk, 1.0 - top_frac)
        stratum = f"top {top_frac:.0%} of severity"
    else:
        hi = y == 1
        stratum = "label (degenerate -- pass `risk` for a real stratum)"
    lo = ~hi

    def _miss(sel):
        # Of the deaths inside this stratum, how many did the model not flag?
        deaths = sel & (y == 1)
        if not deaths.any():
            return float("nan")
        return float((pred[deaths] == 0).mean())

    def _ece(sel):
        return (expected_calibration_error(y[sel], p[sel], n_bins)
                if sel.sum() > n_bins else float("nan"))

    hi_depth = float(exit_layer[hi].mean()) if hi.any() else float("nan")
    lo_depth = float(exit_layer[lo].mean()) if lo.any() else float("nan")

    return {
        "stratum": stratum,
        "high_risk_n": int(hi.sum()),
        "high_risk_prevalence": float(y[hi].mean()) if hi.any() else float("nan"),
        # Informative only when `risk` is supplied; nan otherwise, by design.
        "high_risk_miss_rate": _miss(hi) if risk is not None else float("nan"),
        "low_risk_miss_rate": _miss(lo) if risk is not None else float("nan"),
        "high_risk_ece": _ece(hi),
        "low_risk_ece": _ece(lo),
        "high_risk_mean_depth": hi_depth,
        "low_risk_mean_depth": lo_depth,
        # The safety ordering: the sick must not be the ones cut short.
        "depth_gap": hi_depth - lo_depth,
        "safety_ordering_ok": bool(hi_depth > lo_depth),
        "high_risk_early_exit_rate": float((exit_layer[hi] <= max_layers / 2).mean())
        if hi.any() else float("nan"),
        "low_risk_early_exit_rate": float((exit_layer[lo] <= max_layers / 2).mean())
        if lo.any() else float("nan"),
    }


# ------------------------------------------------------------------ top level

def evaluate_model(model, data, cfg, split="test", temperature=None,
                   exit_temperatures=None):
    """Full evaluation on one split. Returns a nested dict of all three families."""
    model.eval()
    mask = getattr(data, f"{split}_mask")
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight, tau=0.5, hard=True)

    # Severity proxy for the safety stratum, in order of preference:
    #   severity     -- MIMIC, built from observation-window features only;
    #   true_risk    -- mock generator's latent risk;
    #   criticalness -- last resort. It weights the death flag at 1.2 and tracks
    #                   mortality at AUROC ~0.97, so a stratum built from it is
    #                   close to "the patients who died" and reproduces the very
    #                   degeneracy `high_risk_analysis` exists to avoid.
    risk = None
    for attr in ("severity", "true_risk", "criticalness"):
        if hasattr(data, attr):
            risk = getattr(data, attr).cpu().numpy()
            break

    logits = out["final_logits"]
    if temperature is not None:
        logits = logits / temperature
    p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    y = data.y.cpu().numpy()
    m = mask.cpu().numpy()

    res = {
        "predictive": predictive_metrics(y[m], p[m], cfg["eval"]["operating_sensitivity"]),
        "calibration": calibration_metrics(y[m], p[m], cfg["eval"]["n_bins_ece"]),
        "efficiency": efficiency_metrics(model, data, out, mask),
        "safety": high_risk_analysis(
            y[m], p[m], out["exit_layer"].cpu().numpy()[m],
            model.max_layers, cfg["eval"]["operating_sensitivity"],
            risk=risk[m] if risk is not None else None,
            top_frac=cfg["eval"].get("high_risk_top_frac", 0.10),
            n_bins=cfg["eval"]["n_bins_ece"],
        ),
    }

    # Per-exit predictive + calibration numbers -- shows whether shallow exits
    # are individually trustworthy or only work as a mixture.
    per_exit = []
    for i, layer in enumerate(out["exit_layers"]):
        lg = out["exit_logits"][i]
        if exit_temperatures is not None:
            lg = lg / exit_temperatures[i]
        pe = torch.softmax(lg, dim=-1)[:, 1].cpu().numpy()
        per_exit.append({
            "layer": layer,
            "n_exiting": int((out["exit_layer"].cpu().numpy()[m] == layer).sum()),
            **predictive_metrics(y[m], pe[m], cfg["eval"]["operating_sensitivity"]),
            **calibration_metrics(y[m], pe[m], cfg["eval"]["n_bins_ece"]),
        })
    res["per_exit"] = per_exit
    res["exit_distribution"] = {
        int(l): int((out["exit_layer"].cpu().numpy()[m] == l).sum())
        for l in out["exit_layers"]
    }
    return res


def calibrate_on_val(model, data, per_exit=False):
    """Fit temperature scaling on the validation split.

    A confidence-driven exit policy is only safe if confidence tracks true
    probability, so `per_exit=True` fits one temperature per head -- shallow
    heads are typically the worst calibrated and are exactly the ones the policy
    relies on to halt early.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight, tau=0.5, hard=True)
    m = data.val_mask
    y = data.y[m].cpu()
    if not per_exit:
        return fit_temperature(out["final_logits"][m].cpu(), y)
    return [fit_temperature(lg[m].cpu(), y) for lg in out["exit_logits"]]
