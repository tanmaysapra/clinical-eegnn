"""Interpretability: feature attribution, attention, and exit-depth analysis.

The headline story is `exit_depth_vs_difficulty`: if the learned policy is doing
what we claim, propagation depth becomes a readable proxy for case difficulty --
the model spends compute exactly where the case is ambiguous.
"""

import numpy as np
import torch

from src.evaluate import threshold_at_sensitivity


def shap_feature_importance(X, y, masks, feature_names, seed=42, max_display=20):
    """SHAP values from the XGBoost surrogate -- which labs/vitals drive risk.

    Uses the tree model rather than the GNN because TreeExplainer is exact and
    cheap; the GNN's own attributions come from the attention/depth analyses.
    Computed in the xgboost worker process (see src/_omp_guard).
    """
    from src.baselines import xgboost_baseline

    out, _, _ = xgboost_baseline(X, y, masks, seed=seed, want_shap=True)
    values = out["shap_values"]

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)[:max_display]
    return {
        "shap_values": values,
        "ranking": [(feature_names[i], float(mean_abs[i])) for i in order],
    }


def attention_weights(model, data):
    """Per-edge attention from the final GAT layer, if the model exposes it."""
    att = getattr(model, "last_attention", None)
    if att is None:
        return None
    edge_index, alpha = att
    return {
        "edge_index": edge_index.detach().cpu().numpy(),
        "alpha": alpha.detach().cpu().numpy().mean(axis=-1),
    }


def exit_depth_vs_difficulty(model, data, cfg, split="test"):
    """Do harder patients actually use more layers?

    Difficulty is measured three ways so the story does not rest on one proxy:
      * predicted-probability margin (|p - 0.5|) -- model-internal ambiguity,
      * correctness at the clinical operating point,
      * true latent risk, when the mock generator supplies it.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight, tau=0.5, hard=True)

    m = getattr(data, f"{split}_mask").cpu().numpy()
    p = torch.softmax(out["final_logits"], dim=-1)[:, 1].cpu().numpy()[m]
    depth = out["exit_layer"].cpu().numpy()[m].astype(float)
    y = data.y.cpu().numpy()[m]

    margin = np.abs(p - 0.5)
    thr = threshold_at_sensitivity(y, p, cfg["eval"]["operating_sensitivity"])
    correct = ((p >= thr).astype(int) == y)

    res = {
        "corr_depth_vs_margin": float(np.corrcoef(depth, margin)[0, 1]),
        "mean_depth_correct": float(depth[correct].mean()) if correct.any() else np.nan,
        "mean_depth_incorrect": float(depth[~correct].mean()) if (~correct).any() else np.nan,
        "depth_by_quartile_of_margin": [
            float(depth[q].mean())
            for q in _quartile_masks(margin)
        ],
    }
    if hasattr(data, "true_risk"):
        risk = data.true_risk.cpu().numpy()[m]
        # Ambiguity peaks at risk = 0.5 -- distance from it is "how clear-cut".
        res["corr_depth_vs_true_ambiguity"] = float(
            np.corrcoef(depth, -np.abs(risk - 0.5))[0, 1]
        )
    res["depth"] = depth
    res["margin"] = margin
    return res


def _quartile_masks(v):
    edges = np.quantile(v, [0.25, 0.5, 0.75])
    return [v <= edges[0],
            (v > edges[0]) & (v <= edges[1]),
            (v > edges[1]) & (v <= edges[2]),
            v > edges[2]]
