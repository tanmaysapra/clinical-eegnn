"""Multi-exit, cost-sensitive training objective.

    L_total = sum_l w_l * L_task(exit_l)        # supervise every exit
            + ponder_cost * L_efficiency        # reward earlier halting
            + high_risk_halt_penalty * L_safety # be reluctant to exit the sick

L_task is focal loss with an asymmetric cost matrix: a false negative on a
high-risk patient costs `cost_ratio` times a false positive. L_safety adds the
clinical asymmetry to the *halting* policy itself -- shallow exits on nodes the
model scores as high-risk are penalised, so efficiency gains have to come from
confidently-easy low-risk patients.
"""

import torch
import torch.nn.functional as F


def focal_loss(logits, target, gamma=2.0, class_weight=None, reduction="mean"):
    """Focal loss for the imbalanced (~12% positive) mortality label."""
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, target.view(-1, 1)).squeeze(1)
    p_t = logp_t.exp()
    loss = -((1.0 - p_t) ** gamma) * logp_t
    if class_weight is not None:
        loss = loss * class_weight[target]
    if reduction == "mean":
        return loss.mean()
    return loss


def exit_weights(n_exits, scheme):
    """w_l over exits, normalised to sum to 1."""
    if scheme in ("uniform", "distill"):
        w = torch.ones(n_exits)
    elif scheme == "increasing":
        w = torch.arange(1, n_exits + 1, dtype=torch.float32)
    else:
        raise ValueError(f"unknown exit_weight_scheme: {scheme}")
    return w / w.sum()


def multi_exit_loss(out, y, mask, cfg):
    """Total loss over the supervised nodes. Returns (loss, components)."""
    tcfg = cfg["train"]
    device = y.device
    y_m = y[mask]

    # Cost matrix -> class weights: missing a positive is cost_ratio x worse.
    class_weight = torch.tensor([1.0, float(tcfg["cost_ratio"])], device=device)

    logits_list = [lg[mask] for lg in out["exit_logits"]]
    w = exit_weights(len(logits_list), tcfg["exit_weight_scheme"]).to(device)

    l_task = torch.zeros((), device=device)
    for i, logits in enumerate(logits_list):
        l_task = l_task + w[i] * focal_loss(
            logits, y_m, gamma=tcfg["focal_gamma"], class_weight=class_weight
        )

    # Distillation: shallow exits mimic the deepest exit's distribution.
    l_distill = torch.zeros((), device=device)
    if tcfg["exit_weight_scheme"] == "distill" and len(logits_list) > 1:
        teacher = F.softmax(logits_list[-1].detach(), dim=-1)
        for logits in logits_list[:-1]:
            l_distill = l_distill + F.kl_div(
                F.log_softmax(logits, dim=-1), teacher, reduction="batchmean"
            )
        l_distill = l_distill / (len(logits_list) - 1)
        l_task = l_task + tcfg["distill_weight"] * l_distill

    # Detached risk keeps the two policy terms below a pressure on halting
    # rather than a back door into the classifier.
    risk = F.softmax(out["final_logits"][mask], dim=-1)[:, 1].detach()
    depth_frac = out["expected_depth"][mask] / cfg["model"]["max_layers"]

    # Efficiency: buy depth reductions *only* on low-risk patients. An unweighted
    # mean here rewards halting the sick early too, which is precisely the
    # failure mode we are trying to design out.
    l_eff = ((1.0 - risk) * depth_frac).sum() / (1.0 - risk).sum().clamp_min(1e-8)

    # Safety: penalise shallow halts on high-risk nodes. Normalising by the risk
    # mass makes this a conditional expectation over high-risk patients, so its
    # weight does not silently scale with prevalence (~12% here, which otherwise
    # shrank this term by ~8x and let the ponder cost win).
    l_safety = (risk * (1.0 - depth_frac)).sum() / risk.sum().clamp_min(1e-8)

    total = (
        l_task
        + tcfg["ponder_cost"] * l_eff
        + tcfg["high_risk_halt_penalty"] * l_safety
    )
    return total, {
        "task": l_task.detach().item(),
        "distill": l_distill.detach().item(),
        "efficiency": l_eff.detach().item(),
        "safety": l_safety.detach().item(),
        "total": total.detach().item(),
    }
