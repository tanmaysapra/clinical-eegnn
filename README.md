# Early-Exit Models for Clinical Risk Stratification

Adaptive-inference models for ICU mortality prediction on MIMIC-IV v3.1, built
around one question: **when is it safe to stop computing and commit to an
answer for this patient?**

Cohort: 17,663 first ICU stays, first-24h features, in-hospital mortality
(prevalence 10.8%). Full technical record — including the negative results — is
in **[`HANDOVER.md`](HANDOVER.md)**.

## Data

This project is built on **MIMIC-IV v3.1, which is credentialed data under the
[PhysioNet Data Use Agreement](https://physionet.org/content/mimiciv/)**. No
patient-level data is in this repository and none ever should be: the raw
tables, extracted cohorts and cached feature blocks are all gitignored. Only
aggregate results (metric tables, figures) are committed, which the DUA permits.

To reproduce, obtain your own MIMIC-IV credentials and point `data.raw_dir` at
your copy.

## The current model: a staged hybrid cascade

Earlier versions cascaded over **GNN depth** and the result was negative — per-exit
accuracy was flat from layer 1 to layer 10, so the halting policy had nothing to
choose between and a trivial constant policy beat it (HANDOVER §5g–§5j).

The current model cascades over **information** instead. Clinical data does not
arrive at once or at equal cost, so the stages differ in accuracy *and* in cost:

| stage | adds | cost to obtain | test AUROC (5 seeds) |
|---|---|---|---|
| 0 · bedside | demographics + vitals | nothing — already at the bedside | 0.8624 ± 0.0047 |
| 1 · chart | + treatment + comorbidity | nothing, needs record linkage | 0.8733 ± 0.0075 |
| 2 · labs | + the laboratory panel | money and turnaround time | 0.8968 ± 0.0054 |

Each stage is an XGBoost over its nested feature set; the final stage is
statistically indistinguishable from a single XGBoost on everything
(+0.0002, t = 0.12), so the staging is free. A patient exits as soon as their
**calibrated** risk is decisive in either direction — released when confidently
low-risk, escalated when confidently high-risk — with thresholds chosen against a
**Clopper–Pearson bound**, so the safety budget is a finite-sample guarantee
rather than an empirical one.

At a 2% safety budget: **34.5% of compute saved, 46.2% of patients resolved
without the model reading the lab block**, 98.9% decision agreement with the full
cascade, and a realised miss rate of 0.0081. It releases 2.7% of the top severity
decile against 44.3% of everyone else — a **16.7× separation**.

![accuracy/compute frontier](results/figures/cascade_frontier.png)

## The main negative result

The patient-similarity graph adds nothing on top of the boosted score, under four
independent framings: parameter-free residual propagation and a learned residual
GNN, over both a feature-space kNN graph and a graph built from the booster's own
leaf co-occurrence. Best variant: **+0.0001 AUROC** for a stage costing 5× the
three tabular stages combined. See HANDOVER §5n.

## Layout

```
src/tabular.py           the stages (torch-free)
src/cascade.py           calibration, the certified two-sided gate, routing
src/hybrid.py            the graph stage: Correct & Smooth + learned residual GNN
src/mimic_iv.py          the MIMIC-IV ETL
scripts/run_cascade.py   CLI for the staged cascade
scripts/plot_cascade.py  the frontier figure
notebooks/               the documented pipeline, 00-08
results/                 aggregate metrics + figures (no patient data)
HANDOVER.md              the full technical record — read this
```

## Running it

```bash
conda create -y -n eegnn python=3.11 && conda activate eegnn
pip install -r requirements.txt

python -m pytest tests/ -q                          # 24 tests
python scripts/run_cascade.py --source mock         # end to end, no credentials
python scripts/run_cascade.py --source mimic --seeds 42 43 44 45 46 \
    --alphas 0.005 0.01 0.02 0.05 --ablate-conformal --graph-stage none \
    --out results/headline.json
```

Two platform footguns are already solved and documented in HANDOVER §2 — do not
"fix" them back.
