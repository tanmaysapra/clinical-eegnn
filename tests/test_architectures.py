"""Tests for spec Architectures 2 (PPR-Push) and 3 (Fixed-Block, corrected).

The load-bearing test here is `test_masking_is_numerically_free`. Architecture
3's whole claim is that active-subgraph masking saves work *without* changing
the answer; if masking silently changed the arithmetic, the efficiency numbers
would be measuring a different model rather than a cheaper evaluation of the
same one.
"""

import numpy as np
import pytest
import torch
import yaml

from src import cascade
from src.fixed_block import FixedBlockEE, fixed_block_loss
from src.ppr_push import PPRPush, ppr_loss


CFG = yaml.safe_load(open("configs/default.yaml"))


def _toy(n=60, d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = (torch.rand(n, generator=g) < 0.3).long()
    # A symmetric ring-plus-chords graph: every node has degree >= 4, so the
    # normalisation is non-degenerate and masking has something to skip.
    src, dst = [], []
    for i in range(n):
        for off in (1, 2, 7):
            j = (i + off) % n
            src += [i, j]
            dst += [j, i]
    ei = torch.tensor([src, dst], dtype=torch.long)
    ew = torch.rand(ei.size(1), generator=g) * 0.5 + 0.5
    return x, y, ei, ew


# --------------------------------------------------------- Architecture 3

def test_fixed_block_has_distinct_weights_per_block():
    """The defining difference from Architecture 1's shared operator."""
    x, y, ei, ew = _toy()
    m = FixedBlockEE(x.size(1), CFG, n_blocks=4)
    ws = [b.W for b in m.blocks]
    assert len(ws) == 4
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            assert ws[i] is not ws[j]
            assert not torch.equal(ws[i], ws[j])
    # ...and the parameter count is genuinely higher than the tied model.
    from src.model import EEGNN
    tied = EEGNN(x.size(1), {**CFG, "model": {**CFG["model"], "max_layers": 4}})
    n_fb = sum(p.numel() for p in m.blocks.parameters())
    n_t = sum(p.numel() for p in tied.backbone.parameters())
    assert n_fb == 4 * n_t


def test_masking_is_numerically_free():
    """Masked inference == dense inference + freeze, to float tolerance.

    This is the claim spec SS3.3 Fix 1 rests on. Nodes that have exited still
    send messages, so restricting to edges *into* the active set changes what
    is computed, not what is produced.
    """
    x, y, ei, ew = _toy()
    m = FixedBlockEE(x.size(1), CFG, n_blocks=4).eval()
    n = x.size(0)
    cache = m.cache_norm(ei, ew, n, x.dtype, x.device)

    # Everyone runs to the last block: masked must equal the dense forward.
    full = torch.full((n,), 3, dtype=torch.long)
    masked = m.forward_masked(x, ei, ew, exit_block=full, norm_cache=cache)
    with torch.no_grad():
        dense = m.forward(x, ei, ew, norm_cache=cache)["block_logits"][-1]
    assert torch.allclose(masked["realised_logits"], dense, atol=1e-5), \
        (masked["realised_logits"] - dense).abs().max()

    # With real early exits, compare against an explicit freeze reference.
    g = torch.Generator().manual_seed(3)
    exit_block = torch.randint(0, 4, (n,), generator=g)
    got = m.forward_masked(x, ei, ew, exit_block=exit_block, norm_cache=cache)
    ref = _dense_with_freeze(m, x, cache, exit_block)
    assert torch.allclose(got["realised_logits"], ref, atol=1e-5), \
        (got["realised_logits"] - ref).abs().max()


@torch.no_grad()
def _dense_with_freeze(m, x, cache, exit_block):
    """Reference: run every block on every node, freeze exited representations."""
    import torch.nn.functional as F
    from src.fixed_block import _sas_step
    ei, norm = cache
    n = x.size(0)
    h = F.relu(m.encoder(x))
    realised = torch.zeros(n, m.n_classes)
    alive = torch.ones(n, dtype=torch.bool)
    for l, (blk, head) in enumerate(zip(m.blocks, m.exits)):
        h_new = _sas_step(blk, h, ei, norm)
        h = torch.where(alive.unsqueeze(1), h_new, h)
        done = alive & (exit_block <= l)
        if done.any():
            realised[done] = head.task(h[done])
        alive = alive & ~done
    if alive.any():
        realised[alive] = m.exits[-1].task(h[alive])
    return realised


def test_masking_actually_saves_measured_work():
    """The ledger must fall as the active set shrinks -- and be a count."""
    x, y, ei, ew = _toy()
    m = FixedBlockEE(x.size(1), CFG, n_blocks=4).eval()
    n = x.size(0)
    cache = m.cache_norm(ei, ew, n, x.dtype, x.device)

    late = m.forward_masked(x, ei, ew, norm_cache=cache,
                            exit_block=torch.full((n,), 3, dtype=torch.long))
    early = m.forward_masked(x, ei, ew, norm_cache=cache,
                             exit_block=torch.zeros(n, dtype=torch.long))
    assert late["edge_work_saved_pct"] == pytest.approx(0.0, abs=1e-6)
    # Everyone exits after block 0 -> only the first block's work is done.
    assert early["edge_work_saved_pct"] == pytest.approx(75.0, abs=1e-6)
    assert early["edges_processed"] < late["edges_processed"]
    assert early["nodes_updated"] < late["nodes_updated"]


def test_fixed_block_loss_decreases():
    x, y, ei, ew = _toy()
    m = FixedBlockEE(x.size(1), CFG, n_blocks=3)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    mask = torch.zeros(x.size(0), dtype=torch.bool)
    mask[: x.size(0) // 2] = True
    first = last = None
    for i in range(30):
        opt.zero_grad()
        out = m(x, ei, ew)
        loss, parts = fixed_block_loss(out, y, mask, CFG)
        loss.backward()
        opt.step()
        if i == 0:
            first = parts["total"]
        last = parts["total"]
    assert last < first


# --------------------------------------------------------- Architecture 2

def test_ppr_encoder_never_sees_the_graph():
    """Decoupling check: the encoder's output must not depend on the edges."""
    x, y, ei, ew = _toy()
    m = PPRPush(x.size(1), CFG).eval()
    with torch.no_grad():
        a = m.encoder(x)
        perm = torch.randperm(ei.size(1))
        b = m.encoder(x)                    # graph changed below, encoder is pure
        out1 = m(x, ei, ew)
        out2 = m(x, ei[:, perm], ew[perm])
    assert torch.equal(a, b)
    # Same edge set in a different order -> same diffusion, up to float order.
    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-4)


def test_push_terminates_and_shrinks_with_tolerance():
    """A looser tolerance must touch strictly fewer edges. This is the dial."""
    x, y, ei, ew = _toy()
    m = PPRPush(x.size(1), CFG, max_rounds=8).eval()
    with torch.no_grad():
        tight = m(x, ei, ew, eps_push=1e-6)["ledger"]
        loose = m(x, ei, ew, eps_push=1e-1)["ledger"]
    assert loose["edges_touched"] < tight["edges_touched"]
    assert loose["edge_work_saved_pct"] > tight["edge_work_saved_pct"]
    assert tight["rounds_used"] <= 8
    assert loose["mean_exit_round"] <= tight["mean_exit_round"]


def test_push_approximates_the_dense_ppr_fixed_point():
    """With a tight tolerance, push must match exact PPR.

    If it does not, the 'provable' efficiency bound is being bought with an
    approximation error nobody measured.
    """
    x, y, ei, ew = _toy(n=40)
    m = PPRPush(x.size(1), CFG, alpha=0.2, max_rounds=200).eval()
    n = x.size(0)
    from src.layers import _gcn_norm
    with torch.no_grad():
        norm, ein = _gcn_norm(ei, ew, n, x.dtype, x.device)
        z = m.encoder(x)
        A = torch.zeros(n, n)
        A.index_put_((ein[1], ein[0]), norm, accumulate=True)
        exact = m.alpha * torch.linalg.solve(
            torch.eye(n) - (1 - m.alpha) * A, z)
        deg = torch.zeros(n).scatter_add_(
            0, ein[0], torch.ones(ein.size(1)))
        approx, ledger, _ = m.diffuse(z, ein, norm, deg, eps_push=1e-9)
    err = (approx - exact).abs().max() / exact.abs().max()
    assert err < 0.02, f"push deviates from exact PPR by {err:.4f}"


def test_ppr_stage_margins_feed_the_shared_gate():
    """Architecture 2 must produce the same [n_stages, n] the gate expects."""
    x, y, ei, ew = _toy(n=400)
    m = PPRPush(x.size(1), CFG, max_rounds=6).eval()
    margins, ledger = m.stage_margins(x, ei, ew, snapshots=(1, 2, 4))
    assert margins.shape[0] == 4 and margins.shape[1] == x.size(0)

    yn = y.numpy()
    rng = np.random.default_rng(0)
    idx = rng.permutation(x.size(0))
    cal = np.zeros(x.size(0), bool); cal[idx[:150]] = True
    sel = np.zeros(x.size(0), bool); sel[idx[150:300]] = True
    p, _ = cascade.calibrate_stages(margins.numpy().astype(np.float64), yn, cal)
    gate = cascade.fit_gate(p, yn, sel, alpha=0.05, min_release=5)
    stage, risk, side = cascade.route(p, gate)
    assert stage.shape == (x.size(0),)
    assert set(np.unique(side)) <= {-1, 0, 1}
