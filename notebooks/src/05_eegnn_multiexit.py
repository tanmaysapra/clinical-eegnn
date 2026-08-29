#%% md
# 05 · Early-Exit GNN

The model itself: a weight-tied SAS-GNN backbone with an exit head after every
layer, and a learned Gumbel-softmax halting policy that decides per patient how
deep to propagate.

Three claims to test, in order:

1. **The exit distribution is non-trivial.** If every patient exits at layer 1
   (or every patient runs to full depth), the halting policy has learned
   nothing and this is a fixed-depth GNN with extra steps.
2. **Accuracy at reduced average depth matches full depth.** The efficiency claim.
3. **The mixture is calibrated.** Halting early on a patient the model is
   overconfident about is worse than never halting — miscalibration and early
   exit compound.

HANDOVER §6.2 records claim 1 failing on synthetic data: the policy collapsed to
all-shallow, or under a strong halt penalty to all-deep, because depth bought no
accuracy. Whether it fails here was largely decided in notebook 03.

#%% code
exec(open("/kaggle/working/clinical-eegnn/notebooks/src/_bootstrap.py").read())
from src import mimic_iv, graph_build as gb
from src.data_prep import make_splits, impute_and_scale
from src.train import set_seed, get_device, train_model
from src.evaluate import evaluate_model, calibrate_on_val, reliability_curve
import torch, matplotlib.pyplot as plt

cfg = mimic_cfg()
set_seed(cfg["seed"])
X_raw, y, names, meta = mimic_iv.build_cohort(cfg, verbose=False)
train, val, test = make_splits(y, cfg["seed"])
X = impute_and_scale(X_raw, train)
data = gb.build_graph(X, y, (train, val, test), cfg, meta=meta,
                      cache_dir=CACHE).to(get_device())

print(gb.describe(data))
print("max_layers", cfg["model"]["max_layers"],
      "| ponder_cost", cfg["train"]["ponder_cost"],
      "| halt_penalty", cfg["train"]["high_risk_halt_penalty"])
print("has criticalness:", hasattr(data, "criticalness"))

#%% md
## Train

The Gumbel temperature anneals from `gumbel_tau_start` to `gumbel_tau_end`:
early on the halting decision is soft and gradients reach every exit head; by
the end it is nearly discrete, matching inference.

#%% code
set_seed(cfg["seed"])
t0 = time.time()
model, history, best = train_model(data, cfg, use_exits=True, verbose=True,
                                   log_every=25)
print(f"\ntrained in {time.time()-t0:.1f}s | best val AUROC "
      f"{best['auroc']:.4f} at epoch {best['epoch']}")

h = pd.DataFrame(history)
cols = [c for c in ["loss", "val_auroc", "avg_depth"] if c in h.columns]
fig, ax = plt.subplots(1, len(cols), figsize=(4.3 * len(cols), 3.2))
for a, c in zip(np.atleast_1d(ax), cols):
    a.plot(h["epoch"], h[c]); a.set_title(c); a.set_xlabel("epoch")
plt.tight_layout(); plt.show()

#%% md
## Calibrate, then evaluate

Temperature is fit on the **validation** split only. Fitting it on test would
report a calibration number no deployment could reproduce.

#%% code
T = calibrate_on_val(model, data)
T_per_exit = calibrate_on_val(model, data, per_exit=True)
print("global temperature:", round(float(T), 4))

res_raw = evaluate_model(model, data, cfg, split="test")
res = evaluate_model(model, data, cfg, split="test", temperature=T,
                     exit_temperatures=T_per_exit)

print("\npredictive :", {k: round(v, 4) for k, v in res["predictive"].items()
                         if isinstance(v, float)})
print("calib raw  :", {k: round(v, 4) for k, v in res_raw["calibration"].items()})
print("calib scaled:", {k: round(v, 4) for k, v in res["calibration"].items()})
print("efficiency :", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in res["efficiency"].items()})

#%% md
## Claim 1 — is the exit distribution non-trivial?

#%% code
dist = pd.Series(res["exit_distribution"]).sort_index()
share = dist / dist.sum()
print(pd.DataFrame({"n": dist, "share": share.round(4)}).to_string())

collapsed = share.max() > 0.95
print(f"\nmean exit depth {res['efficiency']['avg_exit_depth']:.2f} "
      f"of {cfg['model']['max_layers']}  "
      f"({res['efficiency']['depth_saving_pct']:.1f}% depth saved)")
print("VERDICT:", "COLLAPSED -- the policy is effectively fixed-depth"
      if collapsed else "non-trivial exit distribution")

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(dist.index, dist.values, color="#4C78A8")
ax.set_xlabel("exit layer"); ax.set_ylabel("patients")
ax.set_title("Exit distribution (test)")
plt.tight_layout(); plt.show()

#%% md
## Claim 2 — per-exit quality

Whether each exit is individually trustworthy, or whether shallow exits only
work because the mixture rescues them.

#%% code
pe = pd.DataFrame(res["per_exit"])
print(pe[["layer", "n_exiting", "auroc", "auprc", "ece"]].round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(pe["layer"], pe["auroc"], "o-", label="per-exit AUROC")
ax[0].axhline(res["predictive"]["auroc"], ls="--", c="#E45756", label="multi-exit mixture")
ax[0].set_xlabel("exit layer"); ax[0].set_ylabel("test AUROC"); ax[0].legend()
ax[0].set_title("Accuracy by exit depth")
ax[1].plot(pe["layer"], pe["ece"], "o-", color="#F58518")
ax[1].set_xlabel("exit layer"); ax[1].set_ylabel("ECE")
ax[1].set_title("Calibration by exit depth")
plt.tight_layout(); plt.show()

#%% md
If per-exit AUROC is already flat by layer 1–2, depth is not buying accuracy and
the efficiency result is trivially available to any shallow fixed-depth model —
a much weaker claim than the project set out to make. A rising curve that
plateaus is the shape that makes adaptive depth worthwhile.

## Claim 3 — calibration

#%% code
yt = data.y.cpu().numpy(); m = data.test_mask.cpu().numpy()
with torch.no_grad():
    out = model(data.x, data.edge_index, data.edge_weight, tau=0.5, hard=True)
p_raw = torch.softmax(out["final_logits"], -1)[:, 1].cpu().numpy()
p_cal = torch.softmax(out["final_logits"] / T, -1)[:, 1].cpu().numpy()

fig, ax = plt.subplots(figsize=(4.6, 4.4))
for p, lab in [(p_raw, f"raw (ECE {res_raw['calibration']['ece']:.3f})"),
               (p_cal, f"scaled (ECE {res['calibration']['ece']:.3f})")]:
    conf, acc, _cnt = reliability_curve(yt[m], p[m], cfg["eval"]["n_bins_ece"])
    ax.plot(conf, acc, "o-", label=lab)
ax.plot([0, 1], [0, 1], ls=":", c="grey")
ax.set_xlabel("predicted risk"); ax.set_ylabel("observed rate")
ax.set_title("Reliability"); ax.legend()
plt.tight_layout(); plt.show()

#%% md
## Safety — do the sick get more depth?

Using the corrected metric: the high-risk stratum is the **top decile of
`criticalness`**, a severity proxy fixed before any prediction is made. The
previous definition ("high risk" = died, measured at a threshold pinned to 90%
sensitivity) made the miss rate identically `1 - 0.90` for every model — it sat
at a constant 0.104 across an entire sweep and carried no information
(HANDOVER §6.1).

#%% code
safety = res["safety"]
for k, v in safety.items():
    print(f"  {k:28s} {round(v, 4) if isinstance(v, float) else v}")

print("\nSAFETY ORDERING:",
      "OK — high-risk patients propagate deeper" if safety["safety_ordering_ok"]
      else "INVERTED — the sick are being cut short, see HANDOVER §6.3")

#%% md
An inverted ordering is the failure mode flagged in HANDOVER §6.3: `ponder_cost`
and `high_risk_halt_penalty` both act on the *global* mean depth rather than
separating the low- and high-risk cohorts, so the safety term behaves as a depth
lever rather than a selective one. Notebook 06 sweeps that trade-off.

#%% code
torch.save({"state_dict": model.state_dict(), "cfg": cfg,
            "temperature": float(T),
            "exit_temperatures": [float(t) for t in np.atleast_1d(T_per_exit)]},
           f"{CACHE}/eegnn_model.pt")
json.dump({"predictive": res["predictive"], "calibration": res["calibration"],
           "calibration_raw": res_raw["calibration"], "efficiency": res["efficiency"],
           "safety": res["safety"], "per_exit": res["per_exit"],
           "exit_distribution": res["exit_distribution"]},
          open(f"{CACHE}/eegnn_results.json", "w"), indent=2, default=float)
print("wrote eegnn_model.pt and eegnn_results.json")

#%% md
**Next:** `06_efficiency_safety_sweep` — the accuracy/compute trade-off curve
and whether the safety ordering survives across operating points.
