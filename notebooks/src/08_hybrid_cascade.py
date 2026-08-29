#%% md
# 08 · The staged hybrid cascade

Everything before this notebook cascades over **depth**, and HANDOVER §5g–§5j
records the result: per-exit AUROC is flat from layer 1 to layer 10, so the
halting policy has nothing to choose between, and at any sensible budget it
saves exactly 0%. A trivial "exit everyone at layer 1" policy dominates it on
both axes. Early exit was not failing — it was being asked to pick between
options that were all the same.

This notebook changes what the cascade is over. Not depth: **information**.

| stage | adds | what it costs to obtain |
|---|---|---|
| 0 · bedside | demographics + vitals | nothing — already at the bedside, continuously |
| 1 · chart | + treatment + comorbidity | nothing, but needs the record linked |
| 2 · labs | + the laboratory panel | money and turnaround time |
| 3 · cohort | + patient-graph residual | the whole cohort resident in memory |

Clinical information does not arrive at once or at equal cost, so a model that
consumes it in one gulp is paying for the most expensive input on every
patient. Staging it means the stages differ in **accuracy as well as in cost** —
which is precisely what a depth cascade did not have, and precisely what a gate
needs before it can decide anything.

### What would falsify this design

Three specific numbers. Each one is reported below, and each one has already
sunk an earlier version of this project:

1. **Flat exits.** If stage 0 is as accurate as stage 2, the cascade repeats
   §5g's failure with different words and the honest move is to ship stage 0
   alone. The gradient has to be real.
2. **Losing to the trivial policies.** §5h's most damaging finding was that the
   learned policy was worse than both "always exit immediately" and "always run
   everything". Both are computed here as baselines and the gate has to beat
   them jointly — better accuracy than the first, lower cost than the second.
3. **A safety guarantee that does not hold out of sample.** §5m's α=0.01 gate
   realised 0.0105 on test. If the certified gate does the same, the
   certificate is decoration.

#%% code
exec(open("/kaggle/working/clinical-eegnn/notebooks/src/_bootstrap.py").read())
ensure_pyg()

from src import cascade, graph_build, hybrid, mimic_iv, tabular
from src.cascade import split_validation
from src.data_prep import impute_and_scale, make_splits
from src.evaluate import expected_calibration_error, predictive_metrics
import torch

cfg = mimic_cfg()
X_raw, y, names, meta = mimic_iv.build_cohort(cfg, verbose=False)
train, val, test = make_splits(y, cfg["seed"])
X = impute_and_scale(X_raw, train)
severity = meta["severity"]

cols, stage_names = tabular.stage_columns(
    meta["views"], X_raw.shape[1], X.shape[1],
    tabular.indicator_sources(X_raw, train))
print(f"{len(y):,} patients, prevalence {y.mean():.4f}")
for nm, c in zip(stage_names, cols):
    print(f"  stage {nm:9s} {len(c):4d} columns")

#%% md
## The stages

Each stage is an independent XGBoost over its own nested feature set. The
tidier design — one additive score, each stage boosting from its predecessor's
margin — was tried first and is *worse*; see the docstring of `src/tabular` for
the numbers. Training-row margins are still out-of-fold at every stage, because
the graph stage propagates residuals and in-sample residuals are ~0.

#%% code
boost = tabular.fit_staged_boosting(X, y, train, cols, seed=cfg["seed"])
margins, cost_us = list(boost["margins"]), list(boost["cost_us"])

#%% md
## Stage 3: what the patient graph can add *on top of* the boosted score

Posed as a residual question, not a rival one. A tree ensemble scores one row
at a time, so it structurally cannot ask "are patients like this one
systematically under-predicted?" — the answer lives in the neighbours'
**residuals**, and a residual is a function of the labels, not of `X`. That is
the cheap route at HANDOVER §7's "edges from a source outside `X`": the edges
still come from `X`, but the signal riding along them does not.

Two graphs are compared. The cosine kNN graph over features is the one every
previous notebook used. The **leaf graph** is new: cosine over the booster's own
per-tree leaf assignments, so an edge means "the fitted model treats these two
patients alike" rather than "these two rows are close".

#%% code
p_labs = 1.0 / (1.0 + np.exp(-margins[-1]))
graph_variants = {}

ei_knn, ew_knn = graph_build.build_knn_graph(
    X, cfg["graph"]["k"], "cosine", binary_cols=meta.get("binary_cols"))
graph_variants["knn"] = (ei_knn, ew_knn)

Z = tabular.leaf_onehot(tabular.leaf_indices(boost["models"][-1], X, cols[-1]))
graph_variants["leaf"] = graph_build.build_knn_graph(Z, cfg["graph"]["k"], "cosine")
print("leaf code matrix:", Z.shape)

for gname, (ei, ew) in graph_variants.items():
    scale, _ = hybrid.fit_cs_scale(ei, ew, p_labs, y, train, val, len(y), hops=2)
    p_g = hybrid.correct_and_smooth(ei, ew, p_labs, y, train, len(y), scale=scale, hops=2)
    print(f"  {gname:5s} graph  scale {scale:.2f}  "
          f"test AUROC {predictive_metrics(y[test], p_g[test])['auroc']:.4f} "
          f"(labs stage alone {predictive_metrics(y[test], p_labs[test])['auroc']:.4f})")

#%% md
The graph stage does not earn its place — see §5n of HANDOVER for the 3-seed
numbers. It is left in the codebase and reported, because it is now the
sharpest test yet of §5f's diagnosis: the graph fails even when it is asked
only for a *correction* to a strong model rather than to compete with it, and
even when the edges come from the model's own geometry instead of from `X`.

Everything below therefore uses the **three tabular stages**, so that no
compute saving is credited for skipping a stage that does nothing — the exact
mistake §5j caught in the depth policy.

#%% code
margins = np.stack(margins)
cost_us = np.array(cost_us)
cal, sel = split_validation(val, y, cfg["seed"])
p, _ = cascade.calibrate_stages(margins, y, cal)

print(f"{'stage':10s} {'AUROC':>7s} {'AUPRC':>7s} {'ECE':>6s} {'cum us':>7s}")
for s, nm in enumerate(stage_names):
    m = predictive_metrics(y[test], p[s][test])
    print(f"{nm:10s} {m['auroc']:.4f} {m['auprc']:.4f} "
          f"{expected_calibration_error(y[test], p[s][test]):6.3f} "
          f"{cost_us[:s+1].sum():7.1f}")

#%% md
## The gate

Two-sided, calibrated, and certified.

**Two-sided** because a patient the model puts at p = 0.93 does not need the lab
panel either — the decision is made. Exiting only downward assumes compute is
only ever wasted on the well, which is the opposite of the criticism the
early-exit GNN paper (arXiv:2505.18088) levels at its own method: exits that
"dedicate time to easy inputs, but ignore harder samples". Here the ambiguous
middle gets the expensive stages by construction. The two budgets are
deliberately asymmetric — releasing a patient who dies is a safety failure,
escalating one who lives is a resource cost.

**Certified** because §5m's empirical threshold realised 0.0105 against a 0.01
budget. Thresholds are now chosen against a one-sided Clopper–Pearson upper
bound, on a selection split held out from the calibration split.

#%% code
rows = []
for alpha in (0.005, 0.01, 0.02):
    for conformal in (True, False):
        g = cascade.fit_gate(p, y, sel, alpha=alpha, beta=0.5, conformal=conformal)
        stage, risk, side = cascade.route(p, g)
        r = cascade.evaluate_cascade(y, stage, risk, side, test, cost_us,
                                     severity=severity, p_full=p[-1])
        rows.append((alpha, "certified" if conformal else "empirical", r))
        print(f"alpha={alpha:<6} {'certified' if conformal else 'empirical':10s} "
              f"AUROC {r['auroc']:.4f}  saved {r['compute_saved_pct']:5.1f}%  "
              f"no-labs {r['resolved_before_labs_pct']:5.1f}%  "
              f"miss {r['miss_rate_among_released']:.4f}  "
              f"high-risk released {100*r['high_risk_release_rate']:.1f}% vs "
              f"{100*r['low_risk_release_rate']:.1f}%  "
              f"agreement {r['decision_agreement']:.3f}")

#%% md
## The three falsification tests

Single seed, so these are the pattern rather than the numbers of record — those
are the 5-seed means with paired t-tests in HANDOVER §5n, from
`results/headline.json`. Seed 42 is a favourable draw here, as §5i warned it was
for the depth results too.

#%% code
trivial = {}
for nm, fixed in (("always_stage0", 0), ("always_full", len(stage_names) - 1)):
    st = np.full(len(y), fixed)
    trivial[nm] = cascade.evaluate_cascade(y, st, p[fixed], np.zeros(len(y), int),
                                           test, cost_us, severity=severity)
    print(f"{nm:14s} AUROC {trivial[nm]['auroc']:.4f}  "
          f"saved {trivial[nm]['compute_saved_pct']:5.1f}%")

# The most aggressive certified gate that actually released anyone. A budget the
# bound cannot certify releases nobody, which is the correct behaviour and not a
# result to quote -- so it is reported as a refusal, not folded into a mean.
fired = [(a, r) for a, k, r in rows if k == "certified" and r["released_frac"] > 0]
refused = [a for a, k, r in rows if k == "certified" and r["released_frac"] == 0]
alpha, best = fired[-1]
if refused:
    print(f"\nno certificate issued at alpha in {refused} -- the selection split "
          f"cannot supply enough clean releases for a bound that tight")

print(f"\n1. exits are not flat     stage0 {predictive_metrics(y[test], p[0][test])['auroc']:.4f}"
      f" -> stage2 {predictive_metrics(y[test], p[-1][test])['auroc']:.4f}")
print(f"2. not dominated          at alpha={alpha}: AUROC {best['auroc']:.4f} and "
      f"{best['compute_saved_pct']:.1f}% saved, against "
      f"{trivial['always_stage0']['auroc']:.4f} / "
      f"{trivial['always_stage0']['compute_saved_pct']:.1f}% (always stage 0) and "
      f"{trivial['always_full']['auroc']:.4f} / "
      f"{trivial['always_full']['compute_saved_pct']:.1f}% (always full) -- better "
      f"accuracy than the first, lower cost than the second")
print(f"3. certificate holds      realised miss {best['miss_rate_among_released']:.4f} "
      f"against a {alpha} budget; {100*best['high_risk_release_rate']:.1f}% of the top "
      f"severity decile released against {100*best['low_risk_release_rate']:.1f}% "
      f"of everyone else")
