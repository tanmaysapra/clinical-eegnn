"""Patient-similarity graph construction.

Nodes are patients, edges connect clinically similar patients. kNN is the
default; the multi-view fusion (diagnosis / labs / treatment) is the GCRS-style
hybrid graph and is built on top of the same primitives.

Self-loops are dropped everywhere -- SAS-GNN carries its own residual term.
"""

import os

import numpy as np
import torch
from torch_geometric.data import Data

from src.data_prep import config_hash


def _similarity(X, similarity):
    """Dense [n, n] similarity matrix. Fine up to ~20k patients."""
    X = np.asarray(X, dtype=np.float32)
    if similarity == "cosine":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        return (X / norm) @ (X / norm).T
    if similarity == "euclidean_rbf":
        sq = (X ** 2).sum(axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
        sigma2 = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        return np.exp(-d2 / (2.0 * sigma2))
    raise ValueError(f"unknown similarity: {similarity}")


def _symmetrise(edge_index, edge_weight, n):
    """Make the edge set undirected and deduplicate, keeping max weight."""
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    w = np.concatenate([edge_weight, edge_weight])

    keep = src != dst
    src, dst, w = src[keep], dst[keep], w[keep]

    key = src.astype(np.int64) * n + dst.astype(np.int64)
    order = np.lexsort((-w, key))
    key, src, dst, w = key[order], src[order], dst[order], w[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    return np.stack([src[first], dst[first]]), w[first]


def build_knn_graph(X, k, similarity="cosine"):
    """kNN over patient feature vectors. Returns (edge_index, edge_weight)."""
    S = _similarity(X, similarity)
    n = S.shape[0]
    np.fill_diagonal(S, -np.inf)
    k = min(k, n - 1)

    nbr = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = nbr.reshape(-1)
    w = S[rows, cols]

    # Cosine similarity can be negative; message passing wants non-negative
    # weights, so shift into [0, 1].
    if similarity == "cosine":
        w = (w + 1.0) / 2.0
    return _symmetrise(np.stack([rows, cols]), w.astype(np.float32), n)


def build_threshold_graph(X, threshold, similarity="cosine"):
    """Connect every pair above a similarity threshold. Denser than kNN."""
    S = _similarity(X, similarity)
    n = S.shape[0]
    np.fill_diagonal(S, -np.inf)
    rows, cols = np.nonzero(S >= threshold)
    w = S[rows, cols]
    if similarity == "cosine":
        w = (w + 1.0) / 2.0
    if len(rows) == 0:
        raise ValueError(f"threshold {threshold} produced an empty graph")
    return _symmetrise(np.stack([rows, cols]), w.astype(np.float32), n)


def build_multiview_graph(views, k, similarity="cosine", weights=None):
    """Fuse one kNN graph per view into a single weighted adjacency.

    `views` maps a view name to its feature block [n, d_view]. Each view's
    adjacency is built independently, row-normalised, then averaged -- so a
    pair of patients similar under several views gets a stronger edge.
    """
    names = list(views)
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}
    n = next(iter(views.values())).shape[0]

    fused = {}
    for name in names:
        ei, ew = build_knn_graph(views[name], k, similarity)
        deg = np.zeros(n, dtype=np.float32)
        np.add.at(deg, ei[0], ew)
        deg = np.where(deg < 1e-8, 1.0, deg)
        ew = ew / deg[ei[0]]
        for s, d, w in zip(ei[0], ei[1], ew):
            fused[(int(s), int(d))] = fused.get((int(s), int(d)), 0.0) + weights[name] * float(w)

    pairs = np.array(list(fused.keys()), dtype=np.int64).T
    w = np.array(list(fused.values()), dtype=np.float32)
    return pairs, w


def build_graph(X, y, masks, cfg, meta=None, cache_dir="data/processed"):
    """Assemble the PyG Data object, caching by config hash."""
    train_mask, val_mask, test_mask = masks
    key = config_hash({"data": cfg["data"], "graph": cfg["graph"], "seed": cfg["seed"]})
    path = os.path.join(cache_dir, f"graph_{key}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)

    method = cfg["graph"]["method"]
    if method == "knn":
        ei, ew = build_knn_graph(X, cfg["graph"]["k"], cfg["graph"]["similarity"])
    elif method == "threshold":
        ei, ew = build_threshold_graph(X, cfg["graph"]["threshold"], cfg["graph"]["similarity"])
    elif method == "multiview":
        ei, ew = build_multiview_graph(
            _split_views(X, cfg["graph"]["views"]),
            cfg["graph"]["k"],
            cfg["graph"]["similarity"],
        )
    else:
        raise ValueError(f"unknown graph.method: {method}")

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(ei, dtype=torch.long),
        edge_weight=torch.tensor(ew, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
    )
    if meta and "true_risk" in meta:
        data.true_risk = torch.tensor(meta["true_risk"], dtype=torch.float32)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(data, path)
    return data


def _split_views(X, view_names):
    """Partition feature columns into contiguous blocks, one per view.

    A placeholder for the mock path -- with real MIMIC data the view membership
    comes from the extraction schema, not a column split.
    """
    blocks = np.array_split(np.arange(X.shape[1]), len(view_names))
    return {name: X[:, block] for name, block in zip(view_names, blocks)}


def describe(data):
    """One-line summary used by the P1 acceptance check."""
    n = data.num_nodes
    deg = data.edge_index.shape[1] / max(n, 1)
    prev = {
        split: float(data.y[getattr(data, f"{split}_mask")].float().mean())
        for split in ("train", "val", "test")
    }
    return (
        f"nodes={n} edges={data.edge_index.shape[1]} avg_degree={deg:.2f} "
        f"features={data.x.shape[1]} prevalence={prev}"
    )
