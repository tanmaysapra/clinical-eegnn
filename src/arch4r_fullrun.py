"""State/resume planning, checkpoint I/O, and frozen phase-1 baseline numbers
for `notebooks/src/11_arch4r_fullrun.py` -- the notebook that trains all
three Arch4R modes to the full epoch budget for one seed at a time, resuming
across Kaggle sessions via Google Drive (see `src/gdrive.py`).

Kept separate from the notebook itself (which only orchestrates: talks to
Drive, drives the GPU training loop, prints progress) so the resume logic --
the part that is easy to get subtly wrong -- has a normal pytest suite
(`tests/test_arch4r_fullrun.py`) instead of only being exercised by a live
Kaggle run.
"""

import re

import numpy as np
import torch

MODES = ("encoder_only", "fixed_k", "adaptive_k")
FULL_EPOCHS = 30          # the design's own "full-epoch headline experiment"
                          # budget -- see ARCH4R_HANDOVER.md section 9.
CHECKPOINT_EVERY = 5      # epochs between Drive checkpoint replaces.

# Frozen reference numbers from the phase-1 driver's 5-seed real MIMIC-IV run
# (`results/arch4r_phase1.json`, seeds 42-46; see ARCH4R_HANDOVER.md section 9,
# "The real result"). This notebook does not retrain any of these arms -- it
# only trains Arch4R's own three modes, to a full epoch budget the 5-seed
# phase-1 run itself did not use (5 epochs, not 30 -- see the same section).
PHASE1_BASELINES = {
    "plain_xgboost":      {"test_auroc_mean": 0.8962, "test_auroc_sd": 0.0032},
    "cascade_3stage":     {"test_auroc_mean": 0.9015, "test_auroc_sd": 0.0053},
    "graph_free_encoder": {"test_auroc_mean": 0.8572, "test_auroc_sd": 0.0134},
    "arch2_diffusion":    {"test_auroc_mean": 0.8852, "test_auroc_sd": 0.0044},
    # Arch4R's own arms from the same 5-seed, 5-epoch phase-1 run -- kept only
    # as a "did the full epoch budget actually help" reference point, since
    # this notebook re-trains these three itself rather than reusing them.
    "arch4r_encoder_only_5ep":         {"test_auroc_mean": 0.7457, "test_auroc_sd": 0.0479},
    "arch4r_fixed_k_5ep":              {"test_auroc_mean": 0.8917, "test_auroc_sd": 0.0049},
    "arch4r_adaptive_k_5ep":           {"test_auroc_mean": 0.8925, "test_auroc_sd": 0.0038},
    "arch4r_adaptive_k_certified_5ep": {"test_auroc_mean": 0.8887, "test_auroc_sd": 0.0059},
}


# --------------------------------------------------------------- resume state

def new_random_seed(rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.integers(0, 2**31 - 1))


def fresh_state(seed):
    return {
        "seed": seed, "modes_done": [], "current_mode": MODES[0],
        "current_epoch": 0, "seed_complete": False,
    }


def plan_next(state):
    """Given the manifest loaded from Drive (or None if nothing was found),
    decide what to run next. Returns `(state, is_resume)`.

    A missing state, or one whose seed already finished all three modes,
    starts a brand-new random seed. An in-progress state resumes the exact
    same seed/mode/epoch it left off at -- never re-randomizes mid-run."""
    if state is None or state.get("seed_complete"):
        return fresh_state(new_random_seed()), False
    return dict(state), True


def advance_epoch(state, epoch):
    state = dict(state)
    state["current_epoch"] = epoch
    return state


def complete_mode(state):
    """Call once the current mode finishes `FULL_EPOCHS`. Moves on to the
    next un-done mode (epoch reset to 0), or marks the whole seed complete
    once all three modes are done."""
    state = dict(state)
    modes_done = list(state["modes_done"]) + [state["current_mode"]]
    state["modes_done"] = modes_done
    remaining = [m for m in MODES if m not in modes_done]
    if remaining:
        state["current_mode"] = remaining[0]
        state["current_epoch"] = 0
    else:
        state["seed_complete"] = True
    return state


# ---------------------------------------------------------------- checkpoints

_CKPT_RE = re.compile(r"^arch4r_fullrun_seed(\d+)_([a-z_]+)_epoch(\d+)\.pt$")


def checkpoint_name(seed, mode, epoch):
    return f"arch4r_fullrun_seed{seed}_{mode}_epoch{epoch}.pt"


def parse_checkpoint_name(name):
    m = _CKPT_RE.match(name)
    if not m:
        return None
    return {"seed": int(m.group(1)), "mode": m.group(2), "epoch": int(m.group(3))}


def save_checkpoint(path, model, optimizer, seed, mode, epoch, extra=None):
    payload = {
        "seed": seed, "mode": mode, "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, optimizer, map_location="cpu"):
    # weights_only=True is safe here: the payload is only tensors/ints/strs/
    # dicts (model + optimizer state dicts), and these checkpoints are the
    # notebook's own output round-tripped through Drive, not third-party data.
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
