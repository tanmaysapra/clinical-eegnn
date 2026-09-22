"""One-epoch training steps for Arch4RModel, factored out of
`scripts/run_arch4r_phase1.py::_train_arch4r` so both that script (which
trains a fixed `--epochs` in one call) and `notebooks/src/11_arch4r_fullrun.py`
(which needs to stop after every epoch to checkpoint to Drive) share exactly
one implementation of what one epoch does -- no duplicated training math.

`_train_arch4r`'s behaviour is unchanged: it now just calls these functions
inside its own `for _ in range(epochs)` loop instead of inlining them.
"""

import numpy as np
import torch

from src import arch4r


def train_encoder_epoch(model, opt, view_x, view_observed, mask_t, y_t,
                        checkpoints, cfg_arch4r, grad_clip=5.0):
    """One epoch for `mode="encoder_only"` -- full-batch, no graph, no
    query mini-batching (matches the original inline loop exactly)."""
    model.train()
    opt.zero_grad()
    out = model(view_x, view_observed, None, checkpoints)
    loss, _ = arch4r.arch4r_loss(out, y_t, mask_t, cfg_arch4r)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return float(loss.item())


def train_graph_epoch(model, opt, full_batch, view_x_train, view_observed_train,
                      y_train_t, checkpoints, cfg_arch4r, query_batch_size, rng,
                      grad_clip=5.0):
    """One epoch for `mode in ("fixed_k", "adaptive_k")` -- mini-batches the
    query set (every train node scored as its own query, split into chunks of
    `query_batch_size` with a `backward()`/`step()` per chunk) so peak memory
    is bounded to one chunk's worth of retained diffusion activations, per
    ARCH4R_HANDOVER.md section 4 item 5. `rng` is mutated in place (its
    `permutation` call advances state), matching the original loop's
    single shared `np.random.default_rng` across the whole training run."""
    model.train()
    n_train = full_batch.n_nodes
    order = rng.permutation(n_train)
    total_loss = 0.0
    n_chunks = 0
    for start in range(0, n_train, query_batch_size):
        chunk = order[start:start + query_batch_size]
        batch = arch4r.Arch4RGraphBatch(
            per_view_edges=full_batch.per_view_edges, n_nodes=n_train,
            query_idx=chunk, graph_available=full_batch.graph_available[chunk])
        mask_chunk = torch.ones(len(chunk), dtype=torch.bool, device=y_train_t.device)
        opt.zero_grad()
        out = model(view_x_train, view_observed_train, batch, checkpoints)
        loss, _ = arch4r.arch4r_loss(out, y_train_t[chunk], mask_chunk, cfg_arch4r)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total_loss += float(loss.item())
        n_chunks += 1
    return total_loss / max(n_chunks, 1)
