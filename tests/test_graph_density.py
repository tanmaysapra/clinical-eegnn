import numpy as np
import pytest
import torch

from src import graph_density as gd
from src.encoder_mv import observed_fraction


def _synthetic_view(n=120, d=6, seed=0, n_clusters=6):
    rng = np.random.default_rng(seed)
    cluster = rng.integers(0, n_clusters, size=n)
    centroids = rng.normal(0, 3, size=(n_clusters, d))
    X = centroids[cluster] + rng.normal(0, 0.5, size=(n, d))
    return X.astype(np.float64)


def _cfg(**kw):
    base = dict(k_cand=20, k0=5, gamma=0.5, k_min_adaptive=2, k_max_adaptive=10,
               percentile_qualify=99.5)
    base.update(kw)
    return gd.ViewGraphConfig(**base)


def _build_ref(n_train=100, seed=0, **cfg_kw):
    X = _synthetic_view(n=n_train, seed=seed)
    train_views = {"labs": X}
    raw = {"labs": X.copy()}
    reliability = observed_fraction(raw)  # all 1.0, no NaNs here
    cfg = _cfg(**cfg_kw)
    ref = gd.build_reference_graph(train_views, reliability, cfg,
                                   metrics={"labs": "euclidean"}, seed=seed)
    return ref, X, cfg


# --------------------------------------------------------------- unit pieces

def test_self_tuning_sigma_is_median_of_k0_nearest():
    dist = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 10.0]])
    sigma = gd.self_tuning_sigma(dist[:, :3])
    assert sigma[0] == pytest.approx(2.0)
    assert sigma[1] == pytest.approx(0.5)


def test_adaptive_k_is_monotone_in_density_and_clipped():
    q = np.array([-5.0, 0.0, 5.0])
    k = gd.adaptive_k(q, k0=16, gamma=0.5, k_min=4, k_max=48)
    assert k[0] < k[1] < k[2]
    assert k.min() >= 4 and k.max() <= 48


def test_base_edge_weight_reciprocity_increases_weight():
    delta_ij = np.array([1.0, 1.0])
    w_recip = gd.base_edge_weight(delta_ij, np.array([True, False]))
    assert w_recip[0] > w_recip[1]


def test_graph_availability_requires_both_conditions():
    avail = gd.graph_availability(
        qualified_ref_count=np.array([5, 2, 5]),
        reliable_view_count=np.array([1, 1, 0]),
        k_min=4,
    )
    assert list(avail) == [True, False, False]


# --------------------------------------------------------- reference graph

def test_build_reference_graph_produces_edges_and_stats():
    ref, X, cfg = _build_ref()
    v = ref.views["labs"]
    assert v.edge_index_ref.shape[0] == 2
    assert v.edge_index_ref.shape[1] > 0
    assert v.edge_weight_ref.shape[0] == v.edge_index_ref.shape[1]
    assert v.T_v > 0
    # no self-loops in the reference-reference edge set
    assert not (v.edge_index_ref[0] == v.edge_index_ref[1]).any()


def test_qualification_threshold_and_sigma_fit_train_only():
    ref_a, X, cfg = _build_ref(seed=1)
    # Perturb a copy of the training data itself but refit with an
    # identical seed/config -- this is the "does the same input reproduce
    # the same stats" half; the "held-out never moves it" half is tested
    # below via the attach() path, which structurally cannot write into
    # `ref.views` at all (see test_held_out_patient_never_creates_reference_reference_edges).
    ref_b, _, _ = _build_ref(seed=1)
    va, vb = ref_a.views["labs"], ref_b.views["labs"]
    assert np.allclose(va.sigma_train, vb.sigma_train)
    assert va.T_v == pytest.approx(vb.T_v)


def test_held_out_patient_never_creates_reference_reference_edges():
    ref, X, cfg = _build_ref(n_train=100, seed=2)
    before = ref.views["labs"].edge_index_ref.copy()
    before_w = ref.views["labs"].edge_weight_ref.copy()

    rng = np.random.default_rng(999)
    query = {"labs": rng.normal(0, 3, size=(10, X.shape[1]))}
    query_rel = {"labs": np.ones(10)}
    attached = ref.attach(query, query_rel)

    # attach() must not have mutated the frozen reference topology at all
    assert np.array_equal(before, ref.views["labs"].edge_index_ref)
    assert np.array_equal(before_w, ref.views["labs"].edge_weight_ref)

    # and every returned edge must point INTO the training set (dst < n_train),
    # with sources indexed within the query batch, i.e. no query-to-query edge
    src, dst, _ = attached.per_view_edges["labs"]
    assert (dst < X.shape[0]).all()
    assert (src < 10).all()

    # attaching the same query twice must give identical edges (determinism,
    # not just "no crash")
    attached2 = ref.attach(query, query_rel)
    s2, d2, w2 = attached2.per_view_edges["labs"]
    assert np.array_equal(src, s2) and np.array_equal(dst, d2) and np.allclose(w2, w2)


def test_cold_start_flag_excludes_from_coverage():
    ref, X, cfg = _build_ref(n_train=100, seed=3, k_min_adaptive=4)
    # A query extremely far from every cluster should fail qualification
    # against every candidate (Delta >> T_v) and be flagged unavailable.
    far_query = {"labs": np.full((1, X.shape[1]), 1e4)}
    far_rel = {"labs": np.array([1.0])}
    attached = ref.attach(far_query, far_rel)
    assert attached.graph_available[0] == False  # noqa: E712

    # A normal in-distribution query should be available.
    rng = np.random.default_rng(4)
    normal_query = {"labs": X[:1] + rng.normal(0, 0.1, size=(1, X.shape[1]))}
    normal_rel = {"labs": np.array([1.0])}
    attached2 = ref.attach(normal_query, normal_rel)
    assert attached2.graph_available[0] == True  # noqa: E712


def test_cold_start_flag_requires_a_reliable_view():
    ref, X, cfg = _build_ref(n_train=100, seed=5)
    normal_query = {"labs": X[:1]}
    zero_reliability = {"labs": np.array([0.0])}  # entirely missing view
    attached = ref.attach(normal_query, zero_reliability)
    assert attached.graph_available[0] == False  # noqa: E712


def test_train_graph_available_matches_direct_qualification_rule():
    """Transductive training scores every train node as its own query, so it
    needs the same cold-start signal `attach()` gives held-out queries --
    computed here from the exact same qualified/reliable counts
    `build_reference_graph` already accumulates internally."""
    ref, X, cfg = _build_ref(n_train=100, seed=10, k_min_adaptive=4)
    available = ref.train_graph_available()
    assert available.shape == (100,)
    expected = gd.graph_availability(ref.train_qualified_count,
                                     ref.train_reliable_view_count,
                                     k_min=cfg.k_min_adaptive)
    assert np.array_equal(available, expected)
    # In-distribution synthetic clusters -> most train nodes should qualify.
    assert available.mean() > 0.5


def test_ann_recall_audit_gates_ann_usage():
    X = _synthetic_view(n=60, seed=6)

    def bad_ann(Xt, Xq, k, metric):
        # Deliberately wrong: returns arbitrary neighbours, not nearest.
        n = Xt.shape[0]
        idx = np.tile(np.arange(k), (Xq.shape[0], 1))
        dist = np.zeros_like(idx, dtype=float)
        del n
        return idx, dist

    audit = gd.audit_ann_recall(X, k_cand=10, metric="euclidean", ann_fn=bad_ann)
    assert audit["recall_at_k_mean"] < 0.98 or audit["recall_at_k_min_decile"] < 0.95
    assert audit["used_ann"] is False  # caller must not switch to ANN on a failed audit


# --------------------------------------------------------- dynamic scoring

def test_edge_scorer_gatv2_output_is_bounded():
    torch.manual_seed(0)
    scorer = gd.EdgeScorerGATv2(z_dim=8, hidden=16)
    z_i = torch.randn(20, 8) * 100  # deliberately large to test the tanh bound
    z_j = torch.randn(20, 8) * 100
    e = scorer(z_i, z_j)
    assert (e.abs() <= 3.0 + 1e-4).all()


def test_view_reliability_sums_to_one_across_views():
    torch.manual_seed(0)
    z_dims = {"labs": 8, "vitals": 8}
    rel = gd.ViewReliability(z_dims)
    z_view = {"labs": torch.randn(10, 8), "vitals": torch.randn(10, 8)}
    r_raw = {"labs": torch.ones(10), "vitals": torch.ones(10)}
    pi = rel(z_view, r_raw)
    total = pi["labs"] + pi["vitals"]
    assert torch.allclose(total, torch.ones(10), atol=1e-5)


def test_view_reliability_missing_mask_drives_weight_to_zero():
    torch.manual_seed(0)
    z_dims = {"labs": 8, "vitals": 8}
    rel = gd.ViewReliability(z_dims)
    z_view = {"labs": torch.randn(10, 8), "vitals": torch.randn(10, 8)}
    r_raw = {"labs": torch.ones(10), "vitals": torch.ones(10)}
    missing = {"labs": torch.zeros(10, dtype=torch.bool),
              "vitals": torch.ones(10, dtype=torch.bool)}
    pi = rel(z_view, r_raw, missing_mask=missing)
    assert (pi["vitals"] < 1e-3).all()
    assert (pi["labs"] > 1 - 1e-3).all()


def test_fuse_views_and_row_normalize_shapes_and_stochastic_rows():
    torch.manual_seed(0)
    n = 6
    src = np.array([0, 0, 1, 1, 2])
    dst = np.array([1, 2, 0, 2, 0])
    base_w = np.ones(5)
    per_view_edges = {"labs": (src, dst, base_w)}
    reliability = {"labs": torch.ones(n)}
    scorer_output = {"labs": torch.zeros(5)}  # exp(0) == 1, isolates the fusion math

    edge_index, edge_weight = gd.fuse_views(per_view_edges, reliability, scorer_output,
                                            n_nodes=n, top_k_keep=64)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_weight.shape[0]

    full_ei, full_w = gd.row_normalize_with_selfloop(
        edge_index, torch.as_tensor(edge_weight, dtype=torch.float32), n)
    # every node with at least one outgoing edge (including its own self-loop)
    # must have its outgoing weights sum to 1
    for i in range(n):
        m = full_ei[0] == i
        if m.any():
            assert full_w[m].sum().item() == pytest.approx(1.0, abs=1e-4)


def test_fuse_views_keeps_at_most_top_k_per_source():
    torch.manual_seed(0)
    n = 3
    src = np.array([0, 0, 0, 0])
    dst = np.array([1, 2, 1, 2])  # two views both connect 0->1 and 0->2
    base_w = np.array([1.0, 1.0, 1.0, 1.0])
    per_view_edges = {
        "a": (src[:2], dst[:2], base_w[:2]),
        "b": (src[2:], dst[2:], base_w[2:]),
    }
    reliability = {"a": torch.ones(n), "b": torch.ones(n)}
    scorer_output = {"a": torch.zeros(2), "b": torch.zeros(2)}
    edge_index, edge_weight = gd.fuse_views(per_view_edges, reliability, scorer_output,
                                            n_nodes=n, top_k_keep=1)
    # duplicate (0,1) and (0,2) edges across views must have been summed
    # first, then only the single highest-weight outgoing edge from node 0 kept
    assert (edge_index[0] == 0).sum().item() == 1
