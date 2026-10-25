"""Clinically safe early exit: a calibrated, risk-gated halting rule.

Why this exists
---------------
The learned Gumbel halting policy in `model.EEGNN` decides to stop on raw
*confidence*. Two measurements say that is unsafe here:

  * confidence is badly calibrated exactly where it matters -- expected
    calibration error on the top severity decile runs ~2.2x the error on
    everyone else, stably across seeds and depth budgets (HANDOVER 5j);
  * the resulting depth ordering *inverts*, so the sickest patients exit
    earliest, at most operating points (HANDOVER 5h).

That fails the project's stated safety objective, which asks that "only
verified low-risk patients exit early, while complex cases are fully analysed".

The fix
-------
Decide on **calibrated risk**, not confidence, and only ever exit downward:

    exit at layer L  <=>  p_calibrated(death | exit L) < tau_L

Two properties follow, and both are the point:

1. **The gate operates where the model is trustworthy.** It only fires on
   low-risk predictions, and the calibration failure is concentrated at high
   risk. High-risk patients are never gated out; they run to full depth by
   construction. The miscalibration stops being a safety problem instead of
   having to be fixed first.
2. **The safety level is a dial, not a hope.** `tau` is fitted on validation as
   the *largest* threshold whose realised miss rate among exited patients stays
   within `alpha`. Larger alpha buys more compute saving; the clinician sets it.

This is deliberately not a learned policy. A safety rule that a clinician has
to audit should be a threshold on a calibrated probability, not the argmax of a
Gumbel head -- and having measured that the learned policy inverts the ordering,
a simpler auditable rule is the right trade.
"""

import numpy as np
import torch

from src.evaluate import expected_calibration_error, fit_temperature


# --------------------------------------------------------------- calibration

@torch.no_grad()
def per_exit_probabilities(model, data, temperatures=None):
    """Calibrated P(death) at every exit head, for every node. -> [n, n_exits]

    Temperatures are fitted per head because shallow heads are the worst
    calibrated and are exactly the ones a low-risk gate relies on.
    """
    model.eval()
    out = model(data.x, data.edge_index, data.edge_weight, tau=0.5, hard=True)
    logits = out["exit_logits"]
    if temperatures is None:
        temperatures = [1.0] * len(logits)
    probs = [
        torch.softmax(lg / float(t), dim=-1)[:, 1].cpu().numpy()
        for lg, t in zip(logits, temperatures)
    ]
    return np.stack(probs, axis=1), list(out["exit_layers"])


def fit_exit_temperatures(model, data):
    """One temperature per exit head, fitted on the validation split only."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight, tau=0.5, hard=True)
    m = data.val_mask
    y = data.y[m].cpu()
    return [float(fit_temperature(lg[m].cpu(), y)) for lg in out["exit_logits"]]


# ----------------------------------------------------------------- the gate

def fit_safety_gate(p_exit, y, val_mask, alpha=0.01, grid=200, min_exit_frac=0.0):
    """Per-layer risk thresholds, fitted on validation.

    For each layer in turn, take the patients who have not already exited and
    find the largest `tau` such that the death rate among those the gate would
    release stays at or below `alpha`. Scanning layers in order matters: each
    layer only ever sees the patients its predecessors declined to release.

    `alpha` is the safety budget -- the share of early-exited patients allowed
    to be deaths. It is the clinician's dial, not a tuned hyper-parameter.

    Returns (taus, info).
    """
    n, n_exits = p_exit.shape
    taus = np.zeros(n_exits, dtype=np.float64)
    unresolved = val_mask.copy()
    info = []

    for j in range(n_exits):
        # The last head is the full-depth fallback: it never gates.
        if j == n_exits - 1:
            taus[j] = 0.0
            info.append({"layer_idx": j, "tau": 0.0, "released": 0,
                         "miss_rate": float("nan"), "note": "fallback, no gate"})
            break

        pool = np.flatnonzero(unresolved)
        if pool.size == 0:
            taus[j] = 0.0
            info.append({"layer_idx": j, "tau": 0.0, "released": 0,
                         "miss_rate": float("nan"), "note": "pool empty"})
            continue

        p = p_exit[pool, j]
        yy = y[pool]
        best_tau, best_stats = 0.0, None
        # Candidate thresholds from the observed risk distribution; a grid over
        # quantiles rather than over [0,1] keeps resolution where the mass is.
        for q in np.linspace(0.0, 1.0, grid)[1:]:
            tau = float(np.quantile(p, q))
            rel = p < tau
            if rel.sum() == 0:
                continue
            miss = float(yy[rel].mean())
            if miss <= alpha and rel.mean() >= min_exit_frac:
                best_tau = tau
                best_stats = {"released": int(rel.sum()), "miss_rate": miss,
                              "released_frac": float(rel.mean())}
        taus[j] = best_tau
        released = p < best_tau
        unresolved[pool[released]] = False
        info.append({
            "layer_idx": j, "tau": best_tau,
            "released": int(released.sum()),
            "miss_rate": float(yy[released].mean()) if released.any() else float("nan"),
            **(best_stats or {}),
        })
    return taus, info


def apply_safety_gate(p_exit, taus, exit_layers):
    """Route every node through the gate. -> (exit_layer, risk, exit_index)

    A node exits at the first layer whose calibrated risk clears that layer's
    threshold; anything never released runs to full depth and is scored there.
    """
    n, n_exits = p_exit.shape
    exit_idx = np.full(n, n_exits - 1, dtype=int)
    resolved = np.zeros(n, dtype=bool)
    for j in range(n_exits - 1):
        fires = (~resolved) & (p_exit[:, j] < taus[j])
        exit_idx[fires] = j
        resolved |= fires
    risk = p_exit[np.arange(n), exit_idx]
    depth = np.array([exit_layers[j] for j in exit_idx], dtype=float)
    return depth, risk, exit_idx


# ------------------------------------------------------------- the report

def evaluate_gate(y, depth, risk, exit_idx, mask, max_layers, severity=None,
                  top_frac=0.10, n_bins=15):
    """Everything the safety objective needs, on one split."""
    from src.evaluate import predictive_metrics

    y, depth, risk, exit_idx = y[mask], depth[mask], risk[mask], exit_idx[mask]
    early = exit_idx < (exit_idx.max() if exit_idx.max() > 0 else 1)
    full = ~early

    res = {
        **predictive_metrics(y, risk),
        "ece": expected_calibration_error(y, risk, n_bins),
        "mean_depth": float(depth.mean()),
        "compute_saved_pct": 100.0 * (1.0 - depth.mean() / max_layers),
        "early_exit_frac": float(early.mean()),
        # The safety guarantee, measured on held-out data.
        "miss_rate_among_exited": float(y[early].mean()) if early.any() else float("nan"),
        "deaths_among_exited": int(y[early].sum()) if early.any() else 0,
        "prevalence_among_retained": float(y[full].mean()) if full.any() else float("nan"),
    }

    if severity is not None:
        sev = severity[mask]
        hi = sev >= np.quantile(sev, 1.0 - top_frac)
        lo = ~hi
        res.update({
            "high_risk_n": int(hi.sum()),
            "high_risk_early_exit_rate": float(early[hi].mean()),
            "low_risk_early_exit_rate": float(early[lo].mean()),
            "high_risk_mean_depth": float(depth[hi].mean()),
            "low_risk_mean_depth": float(depth[lo].mean()),
            "depth_gap": float(depth[hi].mean() - depth[lo].mean()),
            # >= not >: a tie is not an ordering violation, only
            # hi < lo (the sick cut short) is. See src/evaluate.py's
            # high_risk_analysis.
            "safety_ordering_ok": bool(depth[hi].mean() >= depth[lo].mean()),
            "high_risk_ece": (expected_calibration_error(y[hi], risk[hi], n_bins)
                              if hi.sum() > n_bins else float("nan")),
            "low_risk_ece": (expected_calibration_error(y[lo], risk[lo], n_bins)
                             if lo.sum() > n_bins else float("nan")),
        })
    return res
