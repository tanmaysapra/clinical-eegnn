"""The exit policy: a calibrated, two-sided, conformally-certified gate.

Torch-free, like `src/tabular` -- it works on probabilities, not models.

What changed, and why
---------------------
The learned Gumbel policy halts on raw *confidence*, and three measurements say
that cannot work here (HANDOVER 5h/5j): calibration on the top severity decile
is ~2.2x worse than elsewhere, the depth ordering inverts so the sickest exit
first, and per-exit accuracy is flat so the policy has nothing to trade. The
gate in `src/safe_exit` fixed the safety half by deciding on calibrated risk
and only ever releasing downward.

This module keeps that and adds the three things it was missing.

**1. Two-sided exits.** A patient the model puts at p = 0.93 does not need the
lab panel and the cohort graph either -- the decision is already made, and more
information is very unlikely to unmake it. Exiting only downward silently
assumes compute is only ever wasted on the well, which is the opposite of the
criticism the early-exit GNN paper levels at its own method ("early exits merely
act as a safeguard, dedicating time to easy inputs, but ignoring harder
samples"). Here the *ambiguous middle* gets the expensive stages, by
construction, and the budgets on the two sides are deliberately asymmetric:
releasing a patient who dies is a safety failure (alpha, small), escalating one
who survives is a resource cost (beta, generous). That asymmetry is the clinical
asymmetry, written down.

This is the "I Don't Know" cascade of Wang et al. (UAI 2018) -- commit to a
class or defer to the next, costlier component -- with the confidence score
replaced by a calibrated risk, which is what the adaptive-inference survey
(Laskaridis et al.) prescribes for exactly the overconfidence failure we
measured.

**2. A finite-sample certificate.** `safe_exit` picked the largest threshold
whose *observed* miss rate on validation was within budget, and the alpha=0.01
setting then realised 0.0105 on test -- an empirical guarantee, slightly
violated. Thresholds are now chosen against a one-sided Clopper-Pearson upper
bound on the miss rate, on a selection split held out from the calibration
split, so the budget holds with confidence 1-gamma rather than on average.

**3. Severity-stratified calibration.** The open issue from HANDOVER 7. A single
global calibrator cannot fix a calibration error that is concentrated in one
stratum; fitting isotonic regression within severity strata can. Whether it
actually does is measured, not assumed.
"""

import numpy as np


# ------------------------------------------------------------------- splits

def split_validation_3way(val_mask, y, seed, fracs=(0.4, 0.3, 0.3)):
    """Cut validation into model-val / gate-selection / gate-certification,
    class-stratified, disjoint -- the Architecture 4R low-risk gate's own 3-
    way split (`src.gate_lowrisk`), sibling to `split_validation` above, not
    a replacement: Architectures 1-3 keep calling the 2-way version
    unchanged.

    Three disjoint jobs need three disjoint partitions here, not two: fitting
    the stage-score calibrator (`model_val`), picking candidate thresholds
    and which checkpoints to enable as gates (`gate_selection`), and the
    one-shot Clopper-Pearson certification of the resulting policy
    (`gate_certification`). Reusing `gate_selection` rows for certification
    would let the certificate be optimistic about the very thresholds it was
    used to choose -- the same reason `split_validation` keeps calibration
    and selection disjoint, extended to a third job.
    """
    assert abs(sum(fracs) - 1.0) < 1e-9, "fracs must sum to 1.0"
    rng = np.random.default_rng(seed + 992)
    model_val = np.zeros_like(val_mask)
    gate_selection = np.zeros_like(val_mask)
    gate_certification = np.zeros_like(val_mask)
    idx = np.flatnonzero(val_mask)
    f_model, f_sel, _f_cert = fracs
    for cls in (0, 1):
        c = idx[y[idx] == cls]
        c = rng.permutation(c)
        cut1 = int(round(f_model * len(c)))
        cut2 = cut1 + int(round(f_sel * len(c)))
        model_val[c[:cut1]] = True
        gate_selection[c[cut1:cut2]] = True
        gate_certification[c[cut2:]] = True
    parts = (model_val, gate_selection, gate_certification)
    assert not (parts[0] & parts[1]).any() and not (parts[1] & parts[2]).any() \
        and not (parts[0] & parts[2]).any()
    assert (parts[0] | parts[1] | parts[2]).sum() == val_mask.sum()
    return {"model_val": model_val, "gate_selection": gate_selection,
           "gate_certification": gate_certification}


def split_validation(val_mask, y, seed, cal_frac=0.4):
    """Cut validation into calibration and threshold-selection splits.

    Fitting the calibrator and then choosing thresholds on the same rows makes
    the safety guarantee optimistic; the two jobs get disjoint data. The
    selection half gets the larger share because it carries the conformal
    bound, and a Clopper-Pearson bound at alpha=0.01 needs ~300 clean releases
    before it can certify anything at all.

    Calibrating on the training rows' out-of-fold margins instead would give
    far more data, and is legitimate for the tabular stages -- but not for the
    graph stage, where a training node's 2-hop neighbourhood can return its own
    residual to it. Held-out data avoids having to defend that.
    """
    rng = np.random.default_rng(seed + 991)
    cal = np.zeros_like(val_mask)
    sel = np.zeros_like(val_mask)
    idx = np.flatnonzero(val_mask)
    for cls in (0, 1):
        c = idx[y[idx] == cls]
        c = rng.permutation(c)
        cut = int(round(cal_frac * len(c)))
        cal[c[:cut]] = True
        sel[c[cut:]] = True
    assert not (cal & sel).any() and (cal | sel).sum() == val_mask.sum()
    return cal, sel


# ---------------------------------------------------------------- calibration

def _isotonic(m, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(m, y)
    return iso


class Calibrator:
    """Margin -> calibrated P(death), fitted on a held-out calibration split.

    `strata` (optional, e.g. severity) fits one isotonic map per quantile
    stratum. The point is HANDOVER 7's open issue: pooled calibration error
    hides a stratum that is twice as badly calibrated, and a single global map
    has no way to correct it because it is one monotone function of the score.
    """

    def __init__(self, n_strata=1, min_per_stratum=300):
        self.n_strata = int(n_strata)
        self.min_per_stratum = int(min_per_stratum)

    def fit(self, margin, y, strata=None):
        self.global_ = _isotonic(margin, y)
        self.edges_, self.models_ = None, None
        if self.n_strata > 1 and strata is not None:
            q = np.linspace(0, 1, self.n_strata + 1)[1:-1]
            self.edges_ = np.quantile(strata, q)
            self.models_ = []
            for b in range(self.n_strata):
                sel = self._bin(strata) == b
                # Too few points to fit a stratum-specific map -> reuse global,
                # rather than fitting a noisy one and calling it a fix.
                self.models_.append(
                    _isotonic(margin[sel], y[sel])
                    if sel.sum() >= self.min_per_stratum and 0 < y[sel].sum() < sel.sum()
                    else self.global_
                )
        return self

    def _bin(self, strata):
        return np.searchsorted(self.edges_, strata, side="right")

    def predict(self, margin, strata=None):
        if self.models_ is None or strata is None:
            return self.global_.predict(margin)
        out = np.empty(len(margin), dtype=np.float64)
        b = self._bin(strata)
        for k, mdl in enumerate(self.models_):
            sel = b == k
            if sel.any():
                out[sel] = mdl.predict(margin[sel])
        return out


def calibrate_stages(margins, y, cal_mask, strata=None, n_strata=1):
    """One calibrator per stage. -> [n_stages, n] calibrated probabilities.

    Per-stage rather than global because the stages have different score
    distributions -- stage 0 is a weaker model and is systematically the worse
    calibrated, and it is the stage the gate leans on hardest.
    """
    p = np.empty_like(np.asarray(margins, dtype=np.float64))
    cals = []
    s_cal = None if strata is None else strata[cal_mask]
    for s in range(len(margins)):
        c = Calibrator(n_strata=n_strata).fit(margins[s][cal_mask], y[cal_mask], s_cal)
        p[s] = c.predict(margins[s], strata)
        cals.append(c)
    return p, cals


# ------------------------------------------------------- the safety certificate

def clopper_pearson_upper(k, n, gamma=0.05):
    """One-sided exact upper confidence bound on a binomial rate.

    `k` events in `n` trials -> the true rate is at most this with probability
    1-gamma. Choosing thresholds against the bound rather than the point
    estimate is what turns "the miss rate was within budget on validation" into
    "the miss rate is within budget, with 1-gamma confidence".
    """
    from scipy.stats import beta as beta_dist
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - gamma, k + 1, n - k))


# --------------------------------------------------------------- the two-sided gate

def fit_gate(p, y, sel_mask, alpha=0.01, beta=0.5, gamma=0.05, conformal=True,
             grid=400, min_release=30):
    """Per-stage (tau_lo, tau_hi), fitted on the selection split.

    Stages are fitted in order and each one only ever sees the patients its
    predecessors declined to resolve, so the guarantee composes down the
    cascade rather than being restated at every stage.

      low side  -- release if p < tau_lo. Budget `alpha` on the death rate
                   among released patients. This is the safety dial.
      high side -- escalate if p > tau_hi. Budget `beta` on the *survival* rate
                   among escalated patients. This is a resource dial, and it is
                   loose on purpose: the cost of escalating someone who lives is
                   not the cost of releasing someone who dies.

    The final stage never gates; it is the fallback that catches everyone the
    cascade could not resolve.
    """
    n_stages = len(p)
    lo = np.zeros(n_stages)
    hi = np.ones(n_stages)
    unresolved = sel_mask.copy()
    bound = ((lambda k, n: clopper_pearson_upper(k, n, gamma)) if conformal
             else (lambda k, n: (k / n if n else 1.0)))
    info = []

    for s in range(n_stages - 1):
        pool = np.flatnonzero(unresolved)
        if pool.size == 0:
            info.append({"stage": s, "note": "pool empty", "tau_lo": 0.0, "tau_hi": 1.0})
            continue
        ps, ys = p[s][pool], y[pool]
        qs = np.quantile(ps, np.linspace(0.0, 1.0, grid))

        best_lo = 0.0
        for t in qs:
            rel = ps < t
            k, m = int(ys[rel].sum()), int(rel.sum())
            if m >= min_release and bound(k, m) <= alpha:
                best_lo = float(t)

        best_hi = 1.0
        for t in qs[::-1]:
            esc = ps > t
            k, m = int((1 - ys[esc]).sum()), int(esc.sum())
            if m >= min_release and bound(k, m) <= beta:
                best_hi = float(t)

        # A degenerate fit would gate everyone out at stage 0; refuse it rather
        # than reporting a spectacular compute saving from a broken threshold.
        if best_lo >= best_hi:
            best_lo, best_hi = 0.0, 1.0

        lo[s], hi[s] = best_lo, best_hi
        fired = (ps < best_lo) | (ps > best_hi)
        unresolved[pool[fired]] = False
        rel, esc = ps < best_lo, ps > best_hi
        info.append({
            "stage": s, "tau_lo": best_lo, "tau_hi": best_hi,
            "pool": int(pool.size),
            "released": int(rel.sum()),
            "released_miss_rate": float(ys[rel].mean()) if rel.any() else float("nan"),
            "escalated": int(esc.sum()),
            "escalated_prevalence": float(ys[esc].mean()) if esc.any() else float("nan"),
        })
    info.append({"stage": n_stages - 1, "tau_lo": 0.0, "tau_hi": 1.0,
                 "note": "fallback, never gates"})
    return {"lo": lo, "hi": hi, "info": info}


def route(p, gate):
    """Send every patient through the cascade. -> (stage, risk, side).

    `side` is -1 released low, +1 escalated high, 0 ran the full cascade.
    """
    n_stages, n = p.shape
    stage = np.full(n, n_stages - 1, dtype=int)
    side = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    for s in range(n_stages - 1):
        rel = (~done) & (p[s] < gate["lo"][s])
        esc = (~done) & (p[s] > gate["hi"][s])
        stage[rel], side[rel] = s, -1
        stage[esc], side[esc] = s, +1
        done |= rel | esc
    risk = p[stage, np.arange(n)]
    return stage, risk, side


# --------------------------------------------------------------------- report

def evaluate_cascade(y, stage, risk, side, mask, cost_us, severity=None,
                     p_full=None, operating_sensitivity=0.90, top_frac=0.10,
                     n_bins=15):
    """Everything the efficiency and safety objectives need, on one split."""
    from src.evaluate import expected_calibration_error, predictive_metrics

    y, stage, risk, side = y[mask], stage[mask], risk[mask], side[mask]
    n_stages = len(cost_us)
    cum = np.cumsum(cost_us)                 # cost of having run stages 0..s
    spent = cum[stage]
    early = side != 0

    res = {
        **predictive_metrics(y, risk, operating_sensitivity),
        "ece": expected_calibration_error(y, risk, n_bins),
        "mean_cost_us": float(spent.mean()),
        "full_cost_us": float(cum[-1]),
        "compute_saved_pct": 100.0 * (1.0 - spent.mean() / cum[-1]),
        "mean_stage": float(stage.mean()),
        "early_exit_frac": float(early.mean()),
        "released_frac": float((side < 0).mean()),
        "escalated_frac": float((side > 0).mean()),
        # The safety guarantee, measured where it counts: held-out data.
        "miss_rate_among_released": (float(y[side < 0].mean()) if (side < 0).any()
                                     else float("nan")),
        "deaths_among_released": int(y[side < 0].sum()),
        "prevalence_among_escalated": (float(y[side > 0].mean()) if (side > 0).any()
                                       else float("nan")),
        "prevalence_among_retained": (float(y[~early].mean()) if (~early).any()
                                      else float("nan")),
        # How many patients never needed the lab panel to be resolved.
        "resolved_before_labs_pct": 100.0 * float((stage < n_stages - 2).mean()),
    }
    for s in range(n_stages):
        res[f"exit_frac_stage{s}"] = float((stage == s).mean())

    if p_full is not None:
        # Decision fidelity: would the full cascade have decided differently?
        pf = p_full[mask]
        thr = np.quantile(pf[y == 1], 1.0 - operating_sensitivity) if y.sum() else 0.5
        agree = (risk >= thr) == (pf >= thr)
        res["decision_agreement"] = float(agree.mean())
        res["decision_agreement_early"] = (float(agree[early].mean()) if early.any()
                                           else float("nan"))

    if severity is not None:
        sev = severity[mask]
        himask = sev >= np.quantile(sev, 1.0 - top_frac)
        lomask = ~himask
        res.update({
            "high_risk_n": int(himask.sum()),
            "high_risk_release_rate": float((side[himask] < 0).mean()),
            "low_risk_release_rate": float((side[lomask] < 0).mean()),
            "high_risk_mean_cost_us": float(spent[himask].mean()),
            "low_risk_mean_cost_us": float(spent[lomask].mean()),
            "cost_gap_us": float(spent[himask].mean() - spent[lomask].mean()),
            # >= not >: a tie (e.g. the always_stage0/always_full baselines,
            # where every patient spends the same cost) is not an ordering
            # violation. See src/evaluate.py's high_risk_analysis.
            "safety_ordering_ok": bool(spent[himask].mean() >= spent[lomask].mean()),
            "high_risk_ece": (expected_calibration_error(y[himask], risk[himask], n_bins)
                              if himask.sum() > n_bins else float("nan")),
            "low_risk_ece": (expected_calibration_error(y[lomask], risk[lomask], n_bins)
                             if lomask.sum() > n_bins else float("nan")),
        })
    return res
