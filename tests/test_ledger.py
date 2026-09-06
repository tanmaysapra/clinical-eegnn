import numpy as np
import pytest

from src.ledger import PredictionLedger, recompute_aggregate


def _fake_metric(y, p_hat):
    y = np.asarray(y)
    p_hat = np.asarray(p_hat)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_p_hat": float(p_hat.mean()),
        "sum_p_hat": float(p_hat.sum()),
    }


def _build_ledger(n=50, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    p_hat = rng.random(n)
    ledger = PredictionLedger(seed=42, arch="arch4r_encoder_only")
    for i in range(n):
        ledger.add_row(
            patient_id=i, split="test", stage=0, p_hat=p_hat[i], y=y[i],
            route_flag="full_stage", edges_touched=0, nodes_visited=1,
            queue_pops=0, wall_clock_s=0.001,
        )
    return ledger, y, p_hat


def test_to_frame_has_expected_columns_and_row_count():
    ledger, y, p_hat = _build_ledger(n=10)
    df = ledger.to_frame()
    assert len(df) == 10
    assert set(["patient_id", "p_hat", "y", "stage", "route_flag"]).issubset(df.columns)


def test_save_and_load_round_trips(tmp_path):
    ledger, y, p_hat = _build_ledger(n=20)
    path = tmp_path / "ledger.csv"
    ledger.save(path)
    loaded = PredictionLedger.load(path)
    assert loaded.seed == 42
    assert loaded.arch == "arch4r_encoder_only"
    assert len(loaded.to_frame()) == 20


def test_load_rejects_mixed_seed_or_arch(tmp_path):
    ledger, _, _ = _build_ledger(n=5)
    path = tmp_path / "ledger.csv"
    ledger.save(path)
    import pandas as pd
    df = pd.read_csv(path)
    df.loc[0, "seed"] = 999  # corrupt: two seeds in one file
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly one"):
        PredictionLedger.load(path)


def test_aggregate_metrics_reproduce_exactly_from_ledger_rows(tmp_path):
    """Architecture 4R's mandatory integrity test #4: recomputing metrics from
    the immutable per-patient ledger must reproduce the aggregate exactly."""
    ledger, y, p_hat = _build_ledger(n=200, seed=7)

    live_metrics = _fake_metric(y, p_hat)

    path = tmp_path / "ledger.csv"
    ledger.save(path)
    reloaded_df = ledger.load(path).to_frame()
    ledger_metrics = recompute_aggregate(reloaded_df, _fake_metric)

    assert ledger_metrics == live_metrics


def test_recompute_aggregate_uses_only_ledger_rows_not_driver_arrays():
    """The recompute must not have any way to reach the original y/p_hat
    arrays directly -- only what was written into the ledger rows."""
    ledger, y, p_hat = _build_ledger(n=30, seed=1)
    df = ledger.to_frame()
    # Mutate the ledger's own stored copy and confirm the recompute reflects
    # that mutation rather than the untouched original arrays -- proof the
    # aggregate really is derived from the stored rows.
    df.loc[0, "p_hat"] = 0.999
    df.loc[0, "y"] = 1
    out = recompute_aggregate(df, _fake_metric)
    assert out["sum_p_hat"] == pytest.approx(df["p_hat"].sum())
