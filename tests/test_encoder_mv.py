import numpy as np
import torch

from src.encoder_mv import (
    MultiViewEncoder, VIEW_OUT_DIMS, fit_view_preprocessors, observed_fraction,
)

IN_DIMS = {"demographics": 5, "comorbidity": 8, "labs": 20, "vitals": 12, "treatment": 10}


def _mock_view_raw(n=100, seed=0):
    rng = np.random.default_rng(seed)
    raw = {v: rng.normal(size=(n, d)).astype(np.float64) for v, d in IN_DIMS.items()}
    # a few NaNs sprinkled in, plus patient 0 entirely missing "labs"
    for v, x in raw.items():
        mask = rng.random(x.shape) < 0.1
        x[mask] = np.nan
    raw["labs"][0, :] = np.nan
    return raw


def test_preprocessors_fit_train_only_and_transform_has_no_nans():
    raw = _mock_view_raw(n=200, seed=1)
    train_mask = np.zeros(200, dtype=bool)
    train_mask[:150] = True
    prep = fit_view_preprocessors(raw, train_mask)
    out = prep.transform(raw)
    for v, x in out.items():
        assert not np.isnan(x).any()
        assert x.shape == raw[v].shape


def test_preprocessors_perturbing_held_out_rows_does_not_change_fitted_stats():
    raw = _mock_view_raw(n=200, seed=2)
    train_mask = np.zeros(200, dtype=bool)
    train_mask[:150] = True
    prep_a = fit_view_preprocessors(raw, train_mask)

    raw_perturbed = {v: x.copy() for v, x in raw.items()}
    raw_perturbed["labs"][150:] = 999.0  # mutate only held-out rows
    prep_b = fit_view_preprocessors(raw_perturbed, train_mask)

    for v in raw:
        assert np.allclose(prep_a.stats[v].median, prep_b.stats[v].median)
        assert np.allclose(prep_a.stats[v].lo, prep_b.stats[v].lo)
        assert np.allclose(prep_a.stats[v].hi, prep_b.stats[v].hi)


def test_observed_fraction_matches_manual_nan_count():
    raw = _mock_view_raw(n=50, seed=3)
    frac = observed_fraction(raw)
    for v, x in raw.items():
        expected = 1.0 - np.isnan(x).mean(axis=1)
        assert np.allclose(frac[v], expected)
    assert frac["labs"][0] == 0.0  # patient 0's labs view is entirely missing


def _build_encoder_inputs(n=50, seed=4):
    raw = _mock_view_raw(n=n, seed=seed)
    train_mask = np.ones(n, dtype=bool)
    prep = fit_view_preprocessors(raw, train_mask)
    x = prep.transform(raw)
    frac = observed_fraction(raw)
    view_x = {v: torch.tensor(x[v], dtype=torch.float32) for v in x}
    view_frac = {v: torch.tensor(frac[v], dtype=torch.float32) for v in frac}
    return view_x, view_frac


def test_forward_shapes_and_missing_view_uses_learned_embedding_and_zero_reliability():
    view_x, view_frac = _build_encoder_inputs(n=50, seed=5)
    model = MultiViewEncoder(IN_DIMS, fusion_dim=64)
    out = model(view_x, view_frac)

    assert out["h0"].shape == (50, 64)
    for v, d in IN_DIMS.items():
        assert out["view_z"][v].shape[0] == 50
        assert out["view_z"][v].shape[1] == VIEW_OUT_DIMS[v]

    # patient 0's "labs" view was forced entirely missing above (frac == 0)
    assert view_frac["labs"][0].item() == 0.0
    expected_embed = model.missing_embed["labs"].detach()
    got = out["view_z"]["labs"][0].detach()
    assert torch.allclose(got, expected_embed, atol=1e-6)
    assert out["reliability_raw"]["labs"][0].item() == 0.0

    # a patient whose labs view IS observed must NOT get the missing embedding
    fully_observed = (view_frac["labs"] > 0.0).nonzero(as_tuple=True)[0][0].item()
    other = out["view_z"]["labs"][fully_observed].detach()
    assert not torch.allclose(other, expected_embed, atol=1e-6)


def test_rejects_unknown_view_name():
    import pytest
    with pytest.raises(ValueError, match="unknown view"):
        MultiViewEncoder({"not_a_real_view": 5})


def test_param_budget():
    """Architecture 4R caps encoder+fusion+heads at 300,000 trainable params.
    This test covers the encoder+fusion half of that budget on the real
    5-view dimension spec; head modules (arch4r.py/gate_lowrisk.py) must stay
    within the remainder once built."""
    model = MultiViewEncoder(IN_DIMS, fusion_dim=64)
    assert model.n_trainable_params() <= 300_000
