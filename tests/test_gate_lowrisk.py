import numpy as np
import pytest

from src import gate_lowrisk as gl


# ------------------------------------------------------------- calibration

def test_error_calibrator_uses_isotonic_with_enough_data():
    rng = np.random.default_rng(0)
    n = 500
    margin = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-margin))).astype(int)
    c = gl.ErrorCalibrator(min_n=100).fit(margin, y)
    assert c.kind_ == "isotonic"
    p = c.predict(margin)
    order = np.argsort(margin)
    assert (np.diff(p[order]) >= -1e-9).all(), "isotonic calibration must be monotone"


def test_error_calibrator_falls_back_to_sigmoid_with_too_little_data():
    rng = np.random.default_rng(1)
    n = 20  # well under min_n
    margin = rng.normal(size=n)
    y = (rng.random(n) < 0.3).astype(int)
    c = gl.ErrorCalibrator(min_n=100).fit(margin, y)
    assert c.kind_ == "sigmoid"
    p = c.predict(margin)
    assert ((p >= 0) & (p <= 1)).all()


def test_calibrate_stages_uses_only_the_cal_mask():
    rng = np.random.default_rng(2)
    n = 400
    margins = [rng.normal(size=n), rng.normal(size=n)]
    y = (rng.random(n) < 0.3).astype(int)
    cal = np.zeros(n, dtype=bool); cal[: n // 2] = True
    y2 = y.copy()
    y2[~cal] = 1 - y2[~cal]  # flip every label OUTSIDE the calibration split
    p1, _ = gl.calibrate_stages(margins, y, cal, min_n=50)
    p2, _ = gl.calibrate_stages(margins, y2, cal, min_n=50)
    assert np.allclose(p1, p2), "a label outside the calibration split moved a calibrated score"


# ------------------------------------------------------ familywise budget

def test_familywise_delta_allocation_sums_correctly():
    for n_enabled in (0, 1, 2, 4, 9):
        delta_g = gl.familywise_delta_allocation(n_enabled, gamma=0.05)
        assert delta_g == pytest.approx(0.05 / (n_enabled + 1))
        # N enabled stages + 1 aggregate check, each at delta_g, sum to gamma exactly.
        assert delta_g * (n_enabled + 1) == pytest.approx(0.05)


# -------------------------------------------------------------- selection

def _make_stage_scores(n, n_stages, seed):
    """Stage scores that get monotonically more separated at deeper
    checkpoints -- low scores increasingly reliable for the survivors."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    p = np.empty((n_stages, n))
    for s in range(n_stages):
        sharpness = 1.0 + s
        base = rng.normal(0, 1, size=n) * 0.3
        p[s] = np.clip(1 / (1 + np.exp(-sharpness * (base + 3 * (y - 0.15)))), 1e-4, 1 - 1e-4)
        # rescale so low-risk survivors cluster near 0
        p[s] = np.where(y == 0, p[s] * 0.1, p[s])
    return p, y


def test_select_policy_never_enables_the_final_stage():
    p, y = _make_stage_scores(2000, 4, seed=3)
    sel = np.ones(2000, dtype=bool)
    policy = gl.select_policy(p, y, sel, eta=0.02, min_release=20)
    assert policy["enabled"][-1] == False  # noqa: E712
    assert policy["lo"][-1] == 0.0


def test_select_policy_only_releases_within_eta_on_selection_split():
    p, y = _make_stage_scores(3000, 3, seed=4)
    sel = np.ones(3000, dtype=bool)
    policy = gl.select_policy(p, y, sel, eta=0.02, min_release=30)
    for info in policy["info"][:-1]:
        if info["enabled"]:
            assert info["released_miss_rate"] <= 0.02 + 1e-9


# ------------------------------------------------------------ certification

def _certifiable_setup(n=6000, n_stages=3, seed=5):
    p, y = _make_stage_scores(n, n_stages, seed)
    sel = np.zeros(n, dtype=bool); sel[: n // 2] = True
    cert = ~sel
    graph_available = np.ones(n, dtype=bool)
    policy = gl.select_policy(p, y, sel, eta=0.02, min_release=30)
    return p, y, cert, graph_available, policy


def test_certify_policy_passes_a_genuinely_safe_policy():
    p, y, cert, graph_available, policy = _certifiable_setup()
    result = gl.certify_policy(p, y, cert, policy, graph_available, eta=0.05, gamma=0.05)
    assert result["certified"] is True
    assert result["n_enabled"] == int(policy["enabled"].sum())
    assert len(result["stage_results"]) == result["n_enabled"]


def test_certification_is_one_shot_per_seed():
    """Simulate a failing bound at stage 0 (deliberately unsafe threshold),
    and confirm certify_policy (a) reports the whole policy uncertified in
    one pass -- not a partial/repaired result -- and (b) that naively
    "fixing" it by disabling stage 0 and re-certifying against the SAME
    certification split would change the pool reaching later stages,
    which is exactly why that pattern is forbidden (not just discouraged)."""
    n = 4000
    p, y = _make_stage_scores(n, n_stages=3, seed=6)
    cert = np.ones(n, dtype=bool)
    graph_available = np.ones(n, dtype=bool)
    # An unsafe policy: stage 0's threshold is set far too high (releases
    # most of the cohort, well past its actual low-risk cluster), so its
    # bound must fail; stage 1 is enabled too, with a threshold that would
    # be reasonable IF it only ever saw its own leftover pool.
    unsafe_policy = {"lo": np.array([0.9, 0.02, 0.0]),
                     "enabled": np.array([True, True, False])}

    result = gl.certify_policy(p, y, cert, unsafe_policy, graph_available,
                               eta=0.02, gamma=0.05)
    assert result["certified"] is False
    assert len(result["stage_results"]) == 2  # both originally-enabled stages
    assert result["stage_results"][0]["passed"] is False
    stage1_released_after_stage0 = result["stage_results"][1]["released"]

    # Now show why "disable the failing stage 0, re-certify stage 1 alone on
    # the SAME certification split" is invalid: stage 1 now sees the FULL
    # eligible pool instead of only stage 0's leftovers, so its released
    # count changes purely because of which earlier gate existed -- not
    # because of anything about stage 1's own threshold or the data. Reusing
    # the same certification labels for this changed pool is the exact bug
    # this module refuses to automate.
    repaired_policy = {"lo": np.array([0.0, 0.02, 0.0]),
                       "enabled": np.array([False, True, False])}
    repaired_result = gl.certify_policy(p, y, cert, repaired_policy, graph_available,
                                        eta=0.02, gamma=0.05)
    stage1_released_alone = repaired_result["stage_results"][0]["released"]
    assert stage1_released_alone != stage1_released_after_stage0


def test_graph_unavailable_patient_can_never_pass():
    n = 500
    rng = np.random.default_rng(7)
    y = (rng.random(n) < 0.1).astype(int)
    p = np.tile(np.full(n, 0.001), (2, 1))  # everyone looks maximally low-risk
    cert = np.ones(n, dtype=bool)
    graph_available = np.zeros(n, dtype=bool)
    graph_available[0] = False  # the one we check explicitly
    policy = {"lo": np.array([0.5, 0.0]), "enabled": np.array([True, False])}

    result = gl.certify_policy(p, y, cert, policy, graph_available, eta=0.5, gamma=0.05)
    assert result["stage_results"][0]["released"] == 0  # nobody eligible at all here

    graph_available_mixed = np.ones(n, dtype=bool)
    graph_available_mixed[0] = False
    stage, risk, released = gl.route(p, policy, graph_available_mixed)
    assert released[0] == False  # noqa: E712
    assert stage[0] == 1  # falls through to the never-gates final stage
    assert released[1:].all()  # every graph-available, extremely-low-risk patient is released


def test_route_releases_at_most_one_stage_per_patient():
    p, y, cert, graph_available, policy = _certifiable_setup()
    stage, risk, released = gl.route(p, policy, graph_available)
    n_stages = p.shape[0]
    assert ((stage >= 0) & (stage < n_stages)).all()
    assert (risk == p[stage, np.arange(len(y))]).all()
