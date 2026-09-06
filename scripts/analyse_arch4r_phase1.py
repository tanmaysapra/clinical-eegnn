#!/usr/bin/env python
"""Aggregate `run_arch4r_phase1.py` into the 4 go/no-go criteria.

Every comparison is paired by seed (all arms share the same seed's splits
and reference graph), reusing `analyse_architectures.py`'s `col()`/`ms()`/
`paired_t()` pattern plus a new `holm_correction()` helper for the two
primary AUROC-improvement comparisons.

The 4 criteria (see the implementation plan):
  1. Graph signal: adaptive_k beats encoder_only by >=0.005 AUROC, with a
     paired 95% CI that excludes 0.
  2. adaptive_k beats fixed_k under a paired comparison.
  3. The certified low-risk policy saves >=30% of full-diffusion edge
     traversals (bound-based estimate -- see the driver's own note on why
     this isn't a per-checkpoint MEASURED figure) at <=0.002 AUROC loss
     versus adaptive_k's uncapped full diffusion.
  4. Every seed's certification passes (every enabled gate + the aggregate
     clear their Clopper-Pearson bound at eta), with >=20% mean certified
     coverage on the test split.

    python scripts/analyse_arch4r_phase1.py results/arch4r_phase1.json
"""
import json
import sys

import numpy as np
from scipy import stats

CRIT = 2.776  # df=4 (5 seeds), two-sided 0.05 -- the project's standard fallback


def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else float("nan")), v.size


def paired_t(a, b):
    """-> (mean diff, sd, t, per-seed diffs, two-sided p-value)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan"), d, float("nan")
    s = d.std(ddof=1)
    t = d.mean() / (s / np.sqrt(d.size)) if s > 0 else float("inf")
    p = float(2.0 * stats.t.sf(abs(t), df=d.size - 1)) if np.isfinite(t) else 0.0
    return float(d.mean()), float(s), float(t), d, p


def paired_ci95(diffs):
    diffs = np.asarray(diffs, float)
    n = len(diffs)
    if n < 2:
        return float(diffs.mean()) if n else float("nan"), (float("nan"), float("nan"))
    mean = float(diffs.mean())
    se = diffs.std(ddof=1) / np.sqrt(n)
    crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else CRIT
    return mean, (mean - crit * se, mean + crit * se)


def holm_correction(pvalues: list, alpha: float = 0.05) -> list:
    """Holm-Bonferroni step-down. -> one bool per input p-value (reject H0),
    in the SAME order as `pvalues`. The family confidence budget `alpha` is
    shared across every comparison in `pvalues` -- pass exactly the primary
    comparisons you want jointly protected, not every number in the report.
    """
    pvalues = np.asarray(pvalues, float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if not np.isfinite(pvalues[idx]):
            break
        threshold = alpha / (m - rank)
        if pvalues[idx] <= threshold:
            reject[idx] = True
        else:
            break  # step-down: once one comparison fails, so do all the rest
    return reject.tolist()


def col(runs, arm, *path, default=None):
    out = []
    for r in runs:
        cur = r["arms"].get(arm)
        for k in path:
            if cur is None:
                break
            cur = cur.get(k) if isinstance(cur, dict) else None
        out.append(default if cur is None else cur)
    return out


def main(path):
    d = json.load(open(path))
    runs = d["runs"]
    seeds = [r["seed"] for r in runs]
    n_seeds = len(seeds)

    print(f"== {path}")
    print(f"seeds {seeds}   n={runs[0]['n']}   prevalence={runs[0]['prevalence']:.5f}")
    if n_seeds < 5:
        print(f"WARNING: only {n_seeds} seeds -- this project's discipline requires "
              ">=5 for any reported number to count as more than provisional.\n")

    print("## Full-diffusion test AUROC by arm")
    arm_order = ["plain_xgboost", "cascade_3stage", "graph_free_encoder",
                "arch2_diffusion", "arch4r_encoder_only", "arch4r_fixed_k",
                "arch4r_adaptive_k"]
    for arm in arm_order:
        m, s, k = ms(col(runs, arm, "test_auroc"))
        print(f"   {arm:>22} {m:.4f} ±{s:.4f}  (n={k})")
    cert_auroc = col(runs, "arch4r_adaptive_k_certified", "test_auroc")
    cm, cs, ck = ms(cert_auroc)
    print(f"   {'arch4r_adaptive_certified':>22} {cm:.4f} ±{cs:.4f}  (n={ck})")

    # --------------------------------------------------- criterion 1 and 2
    print("\n## Go/no-go criterion 1: graph signal (adaptive_k vs encoder_only)")
    adaptive = col(runs, "arch4r_adaptive_k", "test_auroc")
    encoder_only = col(runs, "arch4r_encoder_only", "test_auroc")
    m1, sd1, t1, per1, p1 = paired_t(adaptive, encoder_only)
    _, ci1 = paired_ci95(per1)
    c1_pass = (m1 >= 0.005) and (ci1[0] > 0)
    print(f"   AUROC(adaptive_k) - AUROC(encoder_only) = {m1:+.4f} ±{sd1:.4f}  "
         f"95% CI [{ci1[0]:+.4f}, {ci1[1]:+.4f}]  t={t1:+.2f}  p={p1:.4f}")
    print(f"   per-seed {np.round(per1, 4).tolist()}")
    print(f"   PASS (>=0.005 AUROC, CI excludes 0)" if c1_pass else "   FAIL")

    print("\n## Go/no-go criterion 2: adaptive_k beats fixed_k")
    fixed = col(runs, "arch4r_fixed_k", "test_auroc")
    m2, sd2, t2, per2, p2 = paired_t(adaptive, fixed)
    _, ci2 = paired_ci95(per2)
    c2_pass = (m2 > 0) and (ci2[0] > 0)
    print(f"   AUROC(adaptive_k) - AUROC(fixed_k) = {m2:+.4f} ±{sd2:.4f}  "
         f"95% CI [{ci2[0]:+.4f}, {ci2[1]:+.4f}]  t={t2:+.2f}  p={p2:.4f}")
    print(f"   per-seed {np.round(per2, 4).tolist()}")
    print(f"   PASS (adaptive_k > fixed_k, CI excludes 0)" if c2_pass else "   FAIL")

    reject = holm_correction([p1, p2], alpha=0.05)
    print(f"\n   Holm correction across these 2 primary comparisons (family alpha=0.05):")
    print(f"   criterion 1 significant at family level: {reject[0]}")
    print(f"   criterion 2 significant at family level: {reject[1]}")
    c1_pass = c1_pass and reject[0]
    c2_pass = c2_pass and reject[1]

    # ------------------------------------------------------- criterion 3
    print("\n## Go/no-go criterion 3: certified policy compute saving vs AUROC loss")
    saved = col(runs, "arch4r_adaptive_k_certified", "compute_saved_pct_bound_based")
    sm, ssd, _ = ms(saved)
    loss = np.asarray(adaptive, float) - np.asarray(cert_auroc, float)
    lm, lsd, _ = ms(loss.tolist())
    c3_pass = (sm >= 30.0) and (lm <= 0.002)
    print(f"   compute saved (bound-based estimate): {sm:.1f}% ±{ssd:.1f}")
    print(f"   AUROC loss (full diffusion - certified policy): {lm:+.4f} ±{lsd:.4f}")
    print(f"   PASS (>=30% saved, <=0.002 AUROC loss)" if c3_pass else "   FAIL")

    # ------------------------------------------------------- criterion 4
    print("\n## Go/no-go criterion 4: certification holds, with real coverage")
    certified_flags = col(runs, "arch4r_adaptive_k_certified", "certified")
    coverage = col(runs, "arch4r_adaptive_k_certified", "coverage_frac_test")
    cov_m, cov_sd, _ = ms(coverage)
    all_certified = all(bool(c) for c in certified_flags)
    c4_pass = all_certified and (cov_m >= 0.20)
    print(f"   certified every seed: {all_certified} ({sum(certified_flags)}/{n_seeds})")
    print(f"   mean certified coverage (test): {cov_m:.3f} ±{cov_sd:.3f}")
    print(f"   PASS (certified every seed, >=20% coverage)" if c4_pass else "   FAIL")

    # --------------------------------------------------------------- verdict
    print("\n## Overall Phase 1 go/no-go")
    criteria = [("graph signal", c1_pass), ("adaptive_k > fixed_k", c2_pass),
               ("certified compute/accuracy tradeoff", c3_pass),
               ("certification holds with real coverage", c4_pass)]
    for name, ok in criteria:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    go = all(ok for _, ok in criteria)
    print(f"\n   {'GO' if go else 'NO-GO'} -- Phase 2 is "
         f"{'warranted' if go else 'NOT warranted'} on this evidence.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/arch4r_phase1.json")
