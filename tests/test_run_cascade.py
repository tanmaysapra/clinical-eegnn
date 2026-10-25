"""Unit tests for scripts/run_cascade.py's helpers.

The cache dict in `run_seed` is shared across every seed in a `--seeds` run
(see `main`), so any cache key that omits the seed silently reuses one
seed's cached object for every other seed.

Loaded by file path rather than `from scripts.run_cascade import ...`: an
unrelated `scripts` package installed in site-packages shadows this repo's
`scripts/` directory as a namespace package.
"""

import importlib.util
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "run_cascade.py")
_spec = importlib.util.spec_from_file_location("run_cascade", _PATH)
run_cascade = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_cascade)
_graph_cache_key = run_cascade._graph_cache_key


def test_graph_cache_key_is_seed_specific():
    """A seed omitted from the key means seed 43's graph build is skipped and
    seed 42's kNN graph is reused instead -- silently, since both branches
    return a same-shaped (edge_index, edge_weight) pair."""
    cfg = {"graph": {"k": 15, "similarity": "cosine"}}
    keys = {_graph_cache_key(cfg, seed) for seed in (42, 43, 44)}
    assert len(keys) == 3, "each seed must get its own graph-cache entry"
