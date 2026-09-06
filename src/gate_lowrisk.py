"""Architecture 4R's exit policy: a low-risk-only, one-shot-certified gate
over the diffusion checkpoints.

Low-risk-only, not two-sided
------------------------------
`src/cascade.py`'s `fit_gate` is two-sided (release + escalate) for the
tabular cascade. Phase 1 explicitly defers high-risk exits (see the
implementation plan's scope note) -- this gate only ever RELEASES: it exits
a patient early by declaring them low risk (safe to stop diffusing further),
with a certified upper bound on the miss rate (predicting low risk when the
patient actually dies) among released patients. It never escalates; a
patient who isn't released by any enabled checkpoint simply falls through to
the final checkpoint, which never gates (same convention as
`cascade.fit_gate`'s own fallback stage).

Why certification is a separate, disjoint, one-shot pass
------------------------------------------------------------
Three jobs need three disjoint splits (`cascade.split_validation_3way`):
fitting the calibrator (`model_val`), picking candidate thresholds and which
checkpoints to enable (`gate_selection`, a point-estimate best effort against
`eta`, not a guarantee), and certifying the resulting fixed policy exactly
once (`gate_certification`). `certify_policy` evaluates every originally
enabled stage plus the aggregate against the ORIGINAL policy in one pass; if
any Clopper-Pearson bound fails, the whole policy is uncertified for that
seed, full stop. It must never disable the failing stage and re-simulate the
remaining ones against the same certification labels -- that "disable-and-
repair" pattern was a real statistical bug in an earlier Architecture 4R
design round: removing a gate changes which patients reach the next one
(the pool that would have been released early now flows through), so
re-checking a different enabled set against labels that were supposed to
stay untouched invalidates the Clopper-Pearson bound's validity. The
discipline this module provides is structural (one pure, stateless function
call per policy, no internal retry loop); the caller (the Phase 1 driver)
is responsible for actually calling it exactly once per seed and treating
`certified=False` as terminal.

Cold-start patients can never be released
--------------------------------------------
A patient with `graph_available[i] == False` has identical stage scores at
every checkpoint (the `encoder_only` fallback in `src/arch4r.py` -- no
diffusion happened, so there is nothing for a deeper checkpoint to have
confirmed). The certification was computed on graph-available patients'
score distribution; a cold-start patient's score does not carry the same
guarantee, so both `certify_policy` and `route` mask them out of every
gate's eligible pool, unconditionally.
"""

import numpy as np

from src.cascade import clopper_pearson_upper


# ------------------------------------------------------------- calibration

def _isotonic(m, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(m, y)
    return iso


def _sigmoid(m, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(m.reshape(-1, 1), y)
    return lr


class ErrorCalibrator:
    """Raw stage margin -> calibrated P(death). Isotonic when there's enough
    data to trust a monotone nonparametric fit; a 1-D logistic (sigmoid) fit
    otherwise -- a real fallback model, not just reusing a coarser isotonic
    fit the way `cascade.Calibrator`'s per-stratum fallback does, since
    there is no coarser calibrator here to fall back to."""

    def __init__(self, min_n: int = 100):
        self.min_n = int(min_n)

    def fit(self, margin: np.ndarray, y: np.ndarray) -> "ErrorCalibrator":
        enough = len(margin) >= self.min_n and 0 < y.sum() < len(y)
        if enough:
            self.model_, self.kind_ = _isotonic(margin, y), "isotonic"
        else:
            self.model_, self.kind_ = _sigmoid(margin, y), "sigmoid"
        return self

    def predict(self, margin: np.ndarray) -> np.ndarray:
        margin = np.asarray(margin, dtype=np.float64)
        if self.kind_ == "isotonic":
            return self.model_.predict(margin)
        return self.model_.predict_proba(margin.reshape(-1, 1))[:, 1]


def calibrate_stages(stage_margins: list, y: np.ndarray, cal_mask: np.ndarray,
                     min_n: int = 100):
    """One `ErrorCalibrator` per checkpoint stage, fit on `cal_mask` (the
    `model_val` split) only. -> ([n_stages, n] calibrated probabilities,
    [ErrorCalibrator, ...])."""
    y = np.asarray(y)
    p = np.empty((len(stage_margins), len(y)), dtype=np.float64)
    cals = []
    for s, m in enumerate(stage_margins):
        m = np.asarray(m, dtype=np.float64)
        c = ErrorCalibrator(min_n=min_n).fit(m[cal_mask], y[cal_mask])
        p[s] = c.predict(m)
        cals.append(c)
    return p, cals


# ------------------------------------------------------ familywise budget

def familywise_delta_allocation(n_enabled: int, gamma: float = 0.05) -> float:
    """Bonferroni split of the family confidence budget `gamma` across every
    enabled per-stage certificate PLUS one aggregate certificate -- `N`
    enabled stages + 1 aggregate check share `gamma` so the whole family of
    simultaneous Clopper-Pearson bounds holds jointly with confidence
    `1 - gamma`."""
    return gamma / (n_enabled + 1)


# -------------------------------------------------------------- selection

def select_policy(p: np.ndarray, y: np.ndarray, sel_mask: np.ndarray,
                  eta: float = 0.02, grid: int = 400, min_release: int = 30) -> dict:
    """Choose a low (release) threshold per checkpoint stage on the
    gate-selection split. Low-risk-only: release if `p[s] < tau_lo[s]`,
    never escalate. Same pool-shrinking discipline as `cascade.fit_gate`'s
    low side -- each stage only ever sees patients earlier stages didn't
    already release -- but against a plain empirical miss-rate target
    (`eta`), since this is a best-effort SELECTION, not a certificate;
    `certify_policy`'s disjoint pass is what turns this into one. The final
    stage never gates (same fallback convention as `cascade.fit_gate`).
    """
    n_stages = len(p)
    lo = np.zeros(n_stages)
    enabled = np.zeros(n_stages, dtype=bool)
    unresolved = sel_mask.copy()
    info = []
    for s in range(n_stages - 1):
        pool = np.flatnonzero(unresolved)
        if pool.size == 0:
            info.append({"stage": s, "note": "pool empty", "tau_lo": 0.0, "enabled": False})
            continue
        ps, ys = p[s][pool], y[pool]
        qs = np.quantile(ps, np.linspace(0.0, 1.0, grid))
        best = 0.0
        for t in qs:
            rel = ps < t
            k, m = int(ys[rel].sum()), int(rel.sum())
            if m >= min_release and (k / m) <= eta:
                best = float(t)
        lo[s] = best
        enabled[s] = best > 0.0
        rel = ps < best
        if enabled[s]:
            unresolved[pool[rel]] = False
        info.append({
            "stage": s, "tau_lo": best, "enabled": bool(enabled[s]),
            "pool": int(pool.size), "released": int(rel.sum()),
            "released_miss_rate": float(ys[rel].mean()) if rel.any() else float("nan"),
        })
    info.append({"stage": n_stages - 1, "tau_lo": 0.0, "enabled": False,
                 "note": "fallback, never gates"})
    return {"lo": lo, "enabled": enabled, "info": info}


# ------------------------------------------------------------ certification

def certify_policy(p: np.ndarray, y: np.ndarray, cert_mask: np.ndarray, policy: dict,
                   graph_available: np.ndarray, eta: float = 0.02, gamma: float = 0.05) -> dict:
    """One-shot certification of `policy` (from `select_policy`) against a
    certification split disjoint from selection. See the module docstring
    for why this must never be called a second time with a modified
    `policy["enabled"]` set against the same `cert_mask`/`y`.

    `graph_available[i] == False` patients are removed from every stage's
    eligible pool before routing -- they can never be released here (see
    module docstring).
    """
    n_stages = len(p)
    enabled_stages = [s for s in range(n_stages) if policy["enabled"][s]]
    delta_g = familywise_delta_allocation(len(enabled_stages), gamma)

    lo = policy["lo"]
    n = len(y)
    eligible = np.asarray(cert_mask, dtype=bool) & np.asarray(graph_available, dtype=bool)
    unresolved = eligible.copy()
    stage_results = []
    any_fail = False
    all_released = np.zeros(n, dtype=bool)

    for s in range(n_stages - 1):
        if not policy["enabled"][s]:
            continue
        pool = np.flatnonzero(unresolved)
        rel = np.zeros(n, dtype=bool)
        if pool.size:
            rel_pool = p[s][pool] < lo[s]
            rel[pool[rel_pool]] = True
        k, m = int(y[rel].sum()), int(rel.sum())
        bound = clopper_pearson_upper(k, m, delta_g) if m > 0 else 0.0
        passed = (m == 0) or (bound <= eta)
        any_fail = any_fail or not passed
        stage_results.append({
            "stage": s, "released": m, "deaths_among_released": k,
            "cp_upper_bound": bound, "passed": bool(passed),
        })
        unresolved[rel] = False
        all_released |= rel

    k_agg, m_agg = int(y[all_released].sum()), int(all_released.sum())
    agg_bound = clopper_pearson_upper(k_agg, m_agg, delta_g) if m_agg > 0 else 0.0
    agg_passed = (m_agg == 0) or (agg_bound <= eta)
    any_fail = any_fail or not agg_passed

    return {
        "certified": bool(not any_fail),
        "delta_g": delta_g,
        "n_enabled": len(enabled_stages),
        "stage_results": stage_results,
        "aggregate": {"released": m_agg, "deaths_among_released": k_agg,
                     "cp_upper_bound": agg_bound, "passed": bool(agg_passed)},
        "coverage_frac": float(all_released[cert_mask].mean()) if np.any(cert_mask) else 0.0,
    }


# -------------------------------------------------------------------- route

def route(p: np.ndarray, policy: dict, graph_available: np.ndarray):
    """Route every patient through the certified low-risk gate.
    -> (stage, risk, released). `graph_available[i] == False` can never be
    released -- forced through to the fallback stage regardless of score
    (see module docstring)."""
    n_stages, n = p.shape
    graph_available = np.asarray(graph_available, dtype=bool)
    stage = np.full(n, n_stages - 1, dtype=int)
    released = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    for s in range(n_stages - 1):
        if not policy["enabled"][s]:
            continue
        rel = (~done) & graph_available & (p[s] < policy["lo"][s])
        stage[rel] = s
        released[rel] = True
        done |= rel
    risk = p[stage, np.arange(n)]
    return stage, risk, released
