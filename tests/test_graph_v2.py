"""Leakage invariants for src/graph_v2.py (run inside the Kaggle notebooks)."""

import numpy as np
import torch

from src import graph_v2 as g


def _toy(n=60, d=5, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d)).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(float)
    return Z, y


def test_knn_train_graph_excludes_self():
    Z, _ = _toy()
    tr = np.arange(40)
    nbr, _ = g.knn_train_graph(Z, tr, tr, k=5)
    assert not (nbr == np.arange(40)[:, None]).any()


def test_cs_features_never_read_own_label_or_residual():
    Z, y = _toy()
    tr = np.arange(40)
    resid = y[tr] - 0.3
    nbr_tr, sim_tr = g.knn_train_graph(Z, tr, tr, k=6)
    q_pos = np.arange(40)
    base = g.cs_features(nbr_tr, sim_tr, nbr_tr, sim_tr, q_pos, resid, y[tr])
    for i in (0, 7, 23):
        r2, y2 = resid.copy(), y[tr].copy()
        r2[i] += 5.0
        y2[i] = 1 - y2[i]
        pert = g.cs_features(nbr_tr, sim_tr, nbr_tr, sim_tr, q_pos, r2, y2)
        assert np.allclose(pert[i], base[i]), f"row {i} reads its own label/residual"


def test_heldout_queries_read_only_training_rows():
    Z, y = _toy()
    tr, te = np.arange(40), np.arange(40, 60)
    nbr, _ = g.knn_train_graph(Z, tr, te, k=5)
    assert nbr.max() < len(tr) and nbr.min() >= 0


def test_stratified_folds_cover_every_row_once():
    y = np.array([0] * 30 + [1] * 10)
    f = g.stratified_folds(y, 5, 0)
    assert set(f) == set(range(5))
    for k in range(5):
        assert (y[f == k] == 1).sum() == 2


def test_retrieval_excludes_self_from_context():
    torch.manual_seed(0)
    sl = {"a": (0, 3), "b": (3, 5)}
    x = torch.randn(20, 5)
    yv = torch.zeros(20, dtype=torch.long)
    yv[0] = 1
    m = g.Retrieval(sl, m=19, d=8, hid=8)
    m.eval()
    # query 0 is pool member 0; with m = n_pool - 1 its context is everyone else,
    # so flipping its own label must not change its retrieval exit.
    a = m(x[:1], torch.tensor([0]), x, yv)[1]
    yv2 = yv.clone(); yv2[0] = 0
    b = m(x[:1], torch.tensor([0]), x, yv2)[1]
    assert torch.allclose(a, b)


def test_plr_encoder_shapes_and_grad():
    rng = np.random.default_rng(0)
    X = np.where(rng.random((50, 6)) < 0.2, np.nan, rng.normal(size=(50, 6)))
    X[:, 5] = (rng.random(50) < 0.5).astype(float)  # binary column
    Xp, sl, cont = g.prep_views(X, {"a": np.arange(4), "b": np.arange(4, 6)},
                                np.arange(50) < 40, "iqr", True, return_cont=True)
    assert cont["b"].sum() == 1  # column 4 continuous, column 5 binary, rest indicators
    m = g.EncoderOnly(sl, d=8, hid=8, cont=cont, plr={"n_freq": 4, "sigma": 0.1, "e": 3})
    out = m(torch.tensor(Xp))[0]
    assert out.shape == (50,)
    out.sum().backward()
    assert m.enc.plr["a"].c.grad is not None


def test_pretrain_then_load_into_crossfit_encoder():
    rng = np.random.default_rng(1)
    X = np.where(rng.random((80, 6)) < 0.2, np.nan, rng.normal(size=(80, 6)))
    y = (rng.random(80) < 0.3).astype(np.int64)
    tr = np.arange(60)
    Xp, sl = g.prep_views(X, {"a": np.arange(4), "b": np.arange(4, 6)}, np.arange(80) < 60, "iqr", True)
    Xt = torch.tensor(Xp)
    cfg = dict(d=8, hid=8, p=0.1)
    st, losses = g.pretrain_encoder(sl, Xt, tr, cfg, epochs=3, bs=16)
    assert len(losses) == 3 and np.isfinite(losses).all()
    cf = g.crossfit("encoder", Xt, y, tr, np.arange(60, 80), sl, dict(cfg, init_enc=st, max_ep=2),
                    n_folds=3, seed=0)
    assert cf["oof"].shape == (1, 60)
