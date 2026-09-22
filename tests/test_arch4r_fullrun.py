import numpy as np
import torch

from src import arch4r_fullrun as fr


# ------------------------------------------------------------ resume planning

def test_plan_next_fresh_when_state_none():
    state, is_resume = fr.plan_next(None)
    assert is_resume is False
    assert state["current_mode"] == fr.MODES[0]
    assert state["current_epoch"] == 0
    assert state["modes_done"] == []
    assert state["seed_complete"] is False
    assert isinstance(state["seed"], int)


def test_plan_next_fresh_when_prior_seed_complete():
    prior = fr.fresh_state(seed=123)
    prior["seed_complete"] = True
    state, is_resume = fr.plan_next(prior)
    assert is_resume is False
    assert state["seed"] != 123  # drew a new seed rather than reusing the finished one


def test_plan_next_resumes_inprogress_state_with_same_seed():
    in_progress = fr.fresh_state(seed=777)
    in_progress["current_mode"] = "fixed_k"
    in_progress["current_epoch"] = 15
    in_progress["modes_done"] = ["encoder_only"]
    state, is_resume = fr.plan_next(in_progress)
    assert is_resume is True
    assert state["seed"] == 777
    assert state["current_mode"] == "fixed_k"
    assert state["current_epoch"] == 15
    assert state["modes_done"] == ["encoder_only"]


def test_complete_mode_advances_to_next_mode_and_resets_epoch():
    state = fr.fresh_state(seed=1)
    state["current_epoch"] = fr.FULL_EPOCHS
    state = fr.complete_mode(state)
    assert state["modes_done"] == ["encoder_only"]
    assert state["current_mode"] == "fixed_k"
    assert state["current_epoch"] == 0
    assert state["seed_complete"] is False


def test_complete_mode_marks_seed_complete_after_last_mode():
    state = fr.fresh_state(seed=1)
    state["modes_done"] = ["encoder_only", "fixed_k"]
    state["current_mode"] = "adaptive_k"
    state = fr.complete_mode(state)
    assert set(state["modes_done"]) == set(fr.MODES)
    assert state["seed_complete"] is True


# ---------------------------------------------------------------- checkpoints

def test_checkpoint_name_roundtrip():
    name = fr.checkpoint_name(seed=42, mode="adaptive_k", epoch=15)
    parsed = fr.parse_checkpoint_name(name)
    assert parsed == {"seed": 42, "mode": "adaptive_k", "epoch": 15}


def test_parse_checkpoint_name_rejects_unrelated_filename():
    assert fr.parse_checkpoint_name("best_model_epoch_3.pth") is None


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.no_grad():
        model.weight.fill_(0.5)
    path = tmp_path / "ckpt.pt"
    fr.save_checkpoint(path, model, opt, seed=9, mode="encoder_only", epoch=5)

    model2 = torch.nn.Linear(4, 1)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    ckpt = fr.load_checkpoint(path, model2, opt2)

    assert ckpt["seed"] == 9
    assert ckpt["mode"] == "encoder_only"
    assert ckpt["epoch"] == 5
    assert torch.allclose(model2.weight, model.weight)


# ------------------------------------------------------------------ baselines

def test_phase1_baselines_have_auroc_mean_and_sd():
    assert set(fr.PHASE1_BASELINES) >= {
        "plain_xgboost", "cascade_3stage", "graph_free_encoder", "arch2_diffusion",
    }
    for name, row in fr.PHASE1_BASELINES.items():
        assert 0.0 <= row["test_auroc_mean"] <= 1.0, name
        assert row["test_auroc_sd"] >= 0.0, name
