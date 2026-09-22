#%% md
# 11 · Architecture 4R — full-epoch-budget training run

Trains **only Arch4R's own three modes** (`encoder_only`, `fixed_k`,
`adaptive_k`, plus the certified low-risk policy on top of `adaptive_k`) to
the design's full 30-epoch budget, for **one seed per notebook run**. It does
not retrain the comparison/baseline arms (`plain_xgboost`, `cascade_3stage`,
`graph_free_encoder`, `arch2_diffusion`) -- those reuse the frozen 5-seed
numbers already measured in notebook `10`'s phase-1 run
(`PHASE1_BASELINES` in `src/arch4r_fullrun.py`, sourced from
`ARCH4R_HANDOVER.md` section 9 / `results/arch4r_phase1.json`).

At ~5-6 min/epoch for the graph arms, 30 epochs x 3 modes does not reliably
fit in one Kaggle session, so this notebook checkpoints to Google Drive every
5 epochs (`src/gdrive.py`, `src/arch4r_fullrun.py`) and resumes automatically:
re-running the notebook picks the same seed back up at the exact epoch it
left off, and only draws a new random seed once the prior one has finished
all three modes. See `README_ARCH4R_FULLRUN.md` (next to this notebook) for
the one-time manual setup a fresh Kaggle kernel needs -- dataset, GitHub
token, and the three `GDRIVE_*` secrets -- since none of that can be attached
through the Kaggle MCP.

#%% code
import glob, os, shutil, subprocess, sys

REPO = "/kaggle/working/clinical-eegnn"

if not os.path.isdir(REPO):
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception as e:
        raise RuntimeError(
            "Need a Kaggle Secret named GITHUB_TOKEN (a GitHub PAT with read "
            "access to tanmaysapra/clinical-eegnn) attached to this notebook "
            "-- Add-ons > Secrets in the notebook editor."
        ) from e
    os.environ["GITHUB_TOKEN"] = token
    subprocess.run(
        ["git", "clone", f"https://{token}@github.com/tanmaysapra/clinical-eegnn.git", REPO],
        check=True,
    )

os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")

exec(open(f"{REPO}/notebooks/src/_bootstrap.py").read())
ensure_pyg()
ensure_gdrive_libs()
os.makedirs("results", exist_ok=True)
os.makedirs("results/checkpoints", exist_ok=True)

#%% md
## Dataset

Same resolution order as notebook `10`: a manually attached dataset under
`/kaggle/input` (preferred -- see `README_ARCH4R_FULLRUN.md`, this is faster
and more reliable than the fallback) is used if present; otherwise this
falls back to downloading it directly via `kagglehub`, exactly as `10` does
(attaching via the Kaggle API's `datasetDataSources` field does not take
effect -- verified in notebook `09`/`10`'s own sessions).

#%% code
def _find_mimic_root(start, max_depth=5):
    for dirpath, dirnames, _files in os.walk(start):
        if "hosp" in dirnames:
            base = os.path.basename(dirpath)
            return os.path.dirname(dirpath) if base in ("mimic-iv-3.1", "mimic-iv") else dirpath
        if dirpath[len(start):].count(os.sep) >= max_depth:
            dirnames[:] = []
    return None

_detected_raw = _find_mimic_root("/kaggle/input")
if _detected_raw is None:
    import kagglehub
    _downloaded = kagglehub.dataset_download("akshaybe/updated-mimic-iv")
    print(f"[fullrun] downloaded dataset via kagglehub -> {_downloaded}")
    _detected_raw = _find_mimic_root(_downloaded, max_depth=3) or _downloaded

if _detected_raw != RAW:
    print(f"[fullrun] RAW mount differs from _bootstrap.py's default: "
         f"using {_detected_raw} (not {RAW})")
    RAW = _detected_raw

#%% md
## Phase 0 gate

Refuses to proceed unless notebook `09`'s stop condition passed -- identical
check to notebook `10`.

#%% code
found = None
for pattern in ("/kaggle/input/*/phase0_integrity.json",
               "/kaggle/input/*/results/phase0_integrity.json"):
    hits = glob.glob(pattern)
    if hits:
        found = hits[0]
        break

if found:
    shutil.copy(found, "results/phase0_integrity.json")
    print(f"[fullrun] using phase0 result from attached kernel output: {found}")

import run_arch4r_phase1 as p1

ok, reason = None, None
if os.path.exists("results/phase0_integrity.json"):
    ok, reason = p1._check_phase0_passed("results/phase0_integrity.json")

def _exec_code_cells(path, ns):
    """Run only the `#%% code` sections of a marker-split notebook source."""
    text = open(path).read()
    section, cells = None, []
    for line in text.splitlines():
        if line.strip() == "#%% md":
            section = "md"; continue
        if line.strip() == "#%% code":
            section = "code"; cells.append([]); continue
        if section == "code":
            cells[-1].append(line)
    for cell in cells:
        exec("\n".join(cell), ns)

if not ok:
    print(f"[fullrun] no usable prior phase0 result ({reason}) -- running the "
         "5-seed check inline before training.")
    _exec_code_cells(f"{REPO}/notebooks/src/09_arch4r_phase0.py", globals())
    ok, reason = p1._check_phase0_passed("results/phase0_integrity.json")

assert ok, f"phase0 gate failed even after running inline: {reason}"
print("[fullrun] phase0 gate: PASSED, proceeding.")

#%% md
## Google Drive checkpoint client

Reads `GDRIVE_CLIENT_ID` / `GDRIVE_CLIENT_SECRET` / `GDRIVE_REFRESH_TOKEN`
(OAuth as your own Google account -- not a service account, see
`src/gdrive.py`'s module docstring for why) and `GDRIVE_FOLDER_ID` (the Drive
folder these checkpoints live under) from Kaggle Secrets.

#%% code
from src import gdrive

from kaggle_secrets import UserSecretsClient
_secrets = UserSecretsClient()

drive = gdrive.DriveClient(
    client_id=_secrets.get_secret("GDRIVE_CLIENT_ID"),
    client_secret=_secrets.get_secret("GDRIVE_CLIENT_SECRET"),
    refresh_token=_secrets.get_secret("GDRIVE_REFRESH_TOKEN"),
    folder_id=_secrets.get_secret("GDRIVE_FOLDER_ID"),
)
print("[fullrun] Drive client connected.")

#%% md
## Training config

30 epochs is the design's own full-epoch budget (`ARCH4R_HANDOVER.md`
section 9); checkpoints replace in place on Drive every 5 epochs.

#%% code
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from src import arch4r, arch4r_fullrun as fr, arch4r_train, cascade, data_prep
from src import gate_lowrisk, graph_density as gd, train as train_mod
from src.evaluate import auroc
from src.local_push import algorithmic_bound

FULL_EPOCHS = fr.FULL_EPOCHS            # 30
CHECKPOINT_EVERY = fr.CHECKPOINT_EVERY  # 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("results/checkpoints")
print(f"device: {DEVICE}  |  full_epochs={FULL_EPOCHS}  |  checkpoint_every={CHECKPOINT_EVERY}")

#%% md
## Resume state

Looks for `arch4r_fullrun_state.json` on Drive. None found, or the recorded
seed already finished all three modes -> start a brand-new random seed.
Otherwise resume that exact seed/mode/epoch.

#%% code
STATE_NAME = "arch4r_fullrun_state.json"
_state_hits = drive.list(pattern=rf"^{re.escape(STATE_NAME)}$")
_state_file_id = _state_hits[0][0] if _state_hits else None

_prior_state = None
if _state_file_id is not None:
    _local_state_path = CKPT_DIR / STATE_NAME
    drive.download(_state_file_id, _local_state_path)
    _prior_state = json.loads(_local_state_path.read_text())

state, is_resume = fr.plan_next(_prior_state)
seed = state["seed"]
print(f"[fullrun] {'resuming' if is_resume else 'starting fresh'} -- seed={seed} "
     f"mode={state['current_mode']} epoch={state['current_epoch']} "
     f"modes_done={state['modes_done']}")


def _save_state_to_drive(state, prev_state_file_id):
    """Upload the new state.json, THEN delete the old one -- never the other
    way around, so a crash mid-replace leaves Drive holding a valid (if
    stale) state rather than none at all."""
    local_path = CKPT_DIR / STATE_NAME
    local_path.write_text(json.dumps(state, indent=1))
    new_id = drive.upload(local_path, filename=STATE_NAME)
    if prev_state_file_id is not None and prev_state_file_id != new_id:
        drive.delete(prev_state_file_id)
    return new_id

#%% md
## Cohort, splits, reference graphs (built once for this seed)

Same shape as `run_arch4r_phase1.py::run_one_seed`'s setup, minus the tabular
and Architecture-2 arms this notebook doesn't retrain.

#%% code
cfg = mimic_cfg()
cfg = dict(cfg)
cfg["seed"] = seed  # graph_build's on-disk cache is keyed on this -- see
                    # ARCH4R_HANDOVER.md section 4 item 10 for the historical
                    # bug this guards against.
train_mod.set_seed(seed)

X_raw, y, names, meta = data_prep.load_cohort(cfg)
print(f"[fullrun] cohort n={X_raw.shape[0]} raw_features={X_raw.shape[1]} "
     f"prevalence={y.mean():.5f}", flush=True)

train_mask, val_mask, test_mask = data_prep.make_splits(y, seed)
v3 = cascade.split_validation_3way(val_mask, y, seed, fracs=tuple(cfg["arch4r"]["val_fracs"]))
model_val, gate_sel, gate_cert = v3["model_val"], v3["gate_selection"], v3["gate_certification"]

views = meta.get("views") or p1._mock_views(X_raw.shape[1])
view_in_dims = {name: len(idx) for name, idx in views.items()}
_, view_x_full, view_observed_full = p1._build_view_arrays(X_raw, views, train_mask)
n_total = len(y)

cfg_arch4r = dict(cfg["arch4r"])
gd_cfg_fixed = gd.ViewGraphConfig(
    k_cand=cfg_arch4r["k_cand"], k0=cfg_arch4r["k0"], gamma=cfg_arch4r["gamma"],
    k_min_adaptive=cfg_arch4r["k_min_adaptive"], k_max_adaptive=cfg_arch4r["k_max_adaptive"],
    percentile_qualify=cfg_arch4r["percentile_qualify"], fixed_k=cfg_arch4r["fixed_k"])
gd_cfg_adaptive = gd.ViewGraphConfig(
    k_cand=cfg_arch4r["k_cand"], k0=cfg_arch4r["k0"], gamma=cfg_arch4r["gamma"],
    k_min_adaptive=cfg_arch4r["k_min_adaptive"], k_max_adaptive=cfg_arch4r["k_max_adaptive"],
    percentile_qualify=cfg_arch4r["percentile_qualify"], fixed_k=None)

train_views = {name: arr[train_mask] for name, arr in view_x_full.items()}
train_reliability = {name: arr[train_mask] for name, arr in view_observed_full.items()}
ref_fixed = gd.build_reference_graph(train_views, train_reliability, gd_cfg_fixed, seed=seed)
ref_adaptive = gd.build_reference_graph(train_views, train_reliability, gd_cfg_adaptive, seed=seed)
REF_GRAPHS = {"encoder_only": None, "fixed_k": ref_fixed, "adaptive_k": ref_adaptive}
print("[fullrun] reference graphs (fixed_k, adaptive_k) built")

#%% md
## Train one mode to the full epoch budget, checkpointing to Drive every 5 epochs

Reuses `src/arch4r_train.py`'s per-epoch step -- the exact same training math
`scripts/run_arch4r_phase1.py` uses, just called one epoch at a time instead
of in one `for _ in range(epochs)` block, so there's a point to checkpoint
at.

#%% code
def _train_mode_to_full_budget(mode, seed, start_epoch, state, prev_ckpt_id, prev_state_id):
    model = arch4r.Arch4RModel(view_in_dims, cfg_arch4r, mode=mode).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    from src.local_push import make_checkpoints
    checkpoints = make_checkpoints(c=cfg_arch4r.get("checkpoint_scale", 1.0))

    if start_epoch > 0:
        ckpt_name = fr.checkpoint_name(seed, mode, start_epoch)
        hits = drive.list(pattern=rf"^{re.escape(ckpt_name)}$")
        assert hits, f"state.json says {mode} is at epoch {start_epoch} but " \
                     f"{ckpt_name} is missing from Drive"
        local_ckpt = CKPT_DIR / ckpt_name
        drive.download(hits[0][0], local_ckpt)
        fr.load_checkpoint(local_ckpt, model, opt, map_location=DEVICE)
        prev_ckpt_id = hits[0][0]
        print(f"[fullrun][{mode}] resumed from epoch {start_epoch}")

    ref_graph = REF_GRAPHS[mode]
    mode_idx = fr.MODES.index(mode)

    if mode == "encoder_only":
        view_x = p1._torch_view_dict(view_x_full, np.arange(n_total), DEVICE)
        view_observed = p1._torch_obs_dict(view_observed_full, np.arange(n_total), DEVICE)
        mask_t = torch.tensor(train_mask, dtype=torch.bool, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    else:
        train_idx = np.flatnonzero(train_mask)
        view_x_train = p1._torch_view_dict(view_x_full, train_idx, DEVICE)
        view_observed_train = p1._torch_obs_dict(view_observed_full, train_idx, DEVICE)
        full_batch = arch4r.transductive_batch(ref_graph)
        y_train_t = torch.tensor(y[train_idx], dtype=torch.float32, device=DEVICE)
        query_batch_size = cfg_arch4r.get("query_batch_size", 512)

    t0 = time.time()
    for epoch in range(start_epoch + 1, FULL_EPOCHS + 1):
        if mode == "encoder_only":
            loss = arch4r_train.train_encoder_epoch(
                model, opt, view_x, view_observed, mask_t, y_t, checkpoints, cfg_arch4r)
        else:
            # Reseeded fresh each epoch from (seed, mode, epoch) -- all ints,
            # so this is deterministic across process restarts (unlike a
            # single rng object mutated across epochs, which can't survive a
            # session boundary) while still giving a different query shuffle
            # every epoch, matching what the continuously-mutating rng in
            # scripts/run_arch4r_phase1.py::_train_arch4r does within one
            # uninterrupted call.
            epoch_rng = np.random.default_rng([seed, mode_idx, epoch])
            loss = arch4r_train.train_graph_epoch(
                model, opt, full_batch, view_x_train, view_observed_train, y_train_t,
                checkpoints, cfg_arch4r, query_batch_size, epoch_rng)

        if epoch % CHECKPOINT_EVERY == 0 or epoch == FULL_EPOCHS:
            ckpt_name = fr.checkpoint_name(seed, mode, epoch)
            local_ckpt = CKPT_DIR / ckpt_name
            fr.save_checkpoint(local_ckpt, model, opt, seed, mode, epoch)
            new_ckpt_id = drive.upload(local_ckpt, filename=ckpt_name)

            state = fr.advance_epoch(state, epoch)
            new_state_id = _save_state_to_drive(state, prev_state_id)
            if prev_ckpt_id is not None and prev_ckpt_id != new_ckpt_id:
                drive.delete(prev_ckpt_id)
            prev_ckpt_id, prev_state_id = new_ckpt_id, new_state_id
            print(f"[fullrun][{mode}] epoch {epoch}/{FULL_EPOCHS}  loss={loss:.4f}  "
                 f"checkpointed to Drive  ({time.time()-t0:.0f}s elapsed)", flush=True)

    return model, checkpoints, state, prev_ckpt_id, prev_state_id

#%% md
## Run every mode not yet finished for this seed

#%% code
models, checkpoints_by_mode = {}, {}
prev_state_id = _state_file_id

for mode in fr.MODES:
    if mode in state["modes_done"]:
        print(f"[fullrun] {mode} already complete for seed {seed}, skipping")
        continue

    # `state["current_epoch"]` is only nonzero for the one mode actually
    # in-progress (fresh modes always start at 0, and `complete_mode` resets
    # it to 0 when moving on) -- so keying off `current_mode` alone is
    # enough, whether or not this run resumed anything from Drive at all.
    start_epoch = state["current_epoch"] if mode == state["current_mode"] else 0
    # prev_ckpt_id resets to None for every mode -- each mode's checkpoint
    # file is named for that mode (`arch4r_fullrun_seed<seed>_<mode>_epoch
    # <N>.pt`), so there is never a live prior checkpoint to replace except
    # this same mode's own earlier one (which `_train_mode_to_full_budget`
    # discovers itself when start_epoch > 0). Carrying a previous mode's
    # file id in here would delete ITS final checkpoint the moment this
    # mode uploads its own first one.
    prev_ckpt_id = None
    model, checkpoints, state, prev_ckpt_id, prev_state_id = _train_mode_to_full_budget(
        mode, seed, start_epoch, state, prev_ckpt_id, prev_state_id)
    models[mode], checkpoints_by_mode[mode] = model, checkpoints

    state = fr.complete_mode(state)
    prev_state_id = _save_state_to_drive(state, prev_state_id)
    print(f"[fullrun] {mode} complete for seed {seed}")

#%% md
## Evaluate every trained mode on the held-out test split

#%% code
results = {"seed": seed, "n": n_total, "prevalence": float(y.mean()), "full_epochs": FULL_EPOCHS,
          "arms": {}}

if "encoder_only" in models:
    margins, avail = p1._eval_arch4r_encoder_only(
        models["encoder_only"], checkpoints_by_mode["encoder_only"],
        view_x_full, view_observed_full, n_total, DEVICE)
    results["arms"]["arch4r_encoder_only"] = {
        "test_auroc": float(auroc(y[test_mask], margins[-1][test_mask])),
        "n_params": models["encoder_only"].n_trainable_params(),
    }

margins_by_graph_mode = {}
for mode, ref_graph in (("fixed_k", ref_fixed), ("adaptive_k", ref_adaptive)):
    if mode not in models:
        continue
    margins, avail = p1._eval_arch4r_graph_arm(
        models[mode], checkpoints_by_mode[mode], view_x_full, view_observed_full,
        ref_graph, train_mask, n_total, DEVICE)
    margins_by_graph_mode[mode] = margins
    results["arms"][f"arch4r_{mode}"] = {
        "test_auroc": float(auroc(y[test_mask], margins[-1][test_mask])),
        "n_params": models[mode].n_trainable_params(),
        "graph_available_frac_test": float(avail[test_mask].mean()),
    }

# --------------------------------------------------- certified low-risk gate
if "adaptive_k" in margins_by_graph_mode:
    ckpts = checkpoints_by_mode["adaptive_k"]
    _, avail_adaptive = p1._eval_arch4r_graph_arm(
        models["adaptive_k"], ckpts, view_x_full, view_observed_full,
        ref_adaptive, train_mask, n_total, DEVICE)
    stage_margins = [margins_by_graph_mode["adaptive_k"][s] for s in range(len(ckpts))]

    p_cal, _ = gate_lowrisk.calibrate_stages(stage_margins, y, model_val,
                                             min_n=cfg_arch4r.get("gate_min_n", 100))
    policy = gate_lowrisk.select_policy(p_cal, y, gate_sel, eta=cfg_arch4r["gate_eta"],
                                        min_release=cfg_arch4r["gate_min_release"])
    cert = gate_lowrisk.certify_policy(p_cal, y, gate_cert, policy, avail_adaptive,
                                       eta=cfg_arch4r["gate_eta"], gamma=cfg_arch4r["gate_gamma"])
    stage_idx, risk, released = gate_lowrisk.route(p_cal, policy, avail_adaptive)

    cost_per_checkpoint = np.array(
        [0.0] + [algorithmic_bound(cfg_arch4r["alpha"], c.eps) for c in ckpts[1:]])
    spent = cost_per_checkpoint[stage_idx]
    compute_saved_pct = 100.0 * (1.0 - spent[test_mask].mean() / cost_per_checkpoint[-1])

    results["arms"]["arch4r_adaptive_k_certified"] = {
        "certified": cert["certified"], "n_enabled": cert["n_enabled"],
        "coverage_frac_test": float(released[test_mask].mean()),
        "test_auroc": float(auroc(y[test_mask], risk[test_mask])),
        "compute_saved_pct_bound_based": float(compute_saved_pct),
    }

out_path = f"results/arch4r_fullrun_seed{seed}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"[fullrun] wrote {out_path}")

#%% md
## Comparison: this seed's full-epoch Arch4R vs. the frozen phase-1 baselines

#%% code
print(f"{'arm':32s}  {'test_auroc':>10s}  source")
print("-" * 70)
for name, row in fr.PHASE1_BASELINES.items():
    print(f"{name:32s}  {row['test_auroc_mean']:.4f}±{row['test_auroc_sd']:.4f}  "
         f"phase-1, 5-seed mean")
for name, row in results["arms"].items():
    print(f"{name:32s}  {row['test_auroc']:>10.4f}  seed {seed}, {FULL_EPOCHS} epochs")

#%% md
## Archive this seed's checkpoints + results to a dated Drive subfolder

Only once every mode is done -- `state["seed_complete"]` is `True` here iff
the loop above didn't skip anything (i.e. this run actually finished the
last remaining mode rather than just resuming a still-in-progress one).

#%% code
if state.get("seed_complete"):
    from datetime import datetime as _dt
    date_str = _dt.utcnow().strftime("%Y-%m-%d")
    archive_id = drive.get_or_create_folder(date_str)

    for mode in fr.MODES:
        hits = drive.list(pattern=rf"^arch4r_fullrun_seed{seed}_{mode}_epoch\d+\.pt$")
        for file_id, _name in hits:
            drive.move(file_id, archive_id)

    results_name = f"arch4r_fullrun_seed{seed}.json"
    results_id = drive.upload(out_path, filename=results_name)
    drive.move(results_id, archive_id)

    print(f"[fullrun] seed {seed} complete -- archived to Drive/{date_str}/")
else:
    print(f"[fullrun] seed {seed} not yet complete this session -- "
         "re-run the notebook to continue from the last checkpoint.")

print("\n✓ Notebook run complete.")
