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


_DENSE_LIMIT = 20_000       # n above which an [n, n] matrix stops being sane
_BLOCK = 2048               # rows scored per chunk in the blocked path


def _normalise(X, similarity, binary_cols=None):
    """Row-normalise for cosine; for hybrid normalise only the continuous block."""
    X = np.asarray(X, dtype=np.float32)
    if similarity == "cosine":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm < 1e-8, 1.0, norm)
    if similarity == "hybrid":
        cont, _ = _split_binary(X, binary_cols)
        X = X.copy()
        if cont.size:
            norm = np.linalg.norm(X[:, cont], axis=1, keepdims=True)
            X[:, cont] = X[:, cont] / np.where(norm < 1e-8, 1.0, norm)
        return X
    return X


def _split_binary(X, binary_cols):
    """(continuous indices, binary indices) for the hybrid kernel."""
    d = X.shape[1]
    if binary_cols is None:
        vals = X[: min(len(X), 4096)]
        is_bin = np.all((vals == 0.0) | (vals == 1.0), axis=0)
        binm = np.flatnonzero(is_bin)
    else:
        binm = np.asarray([c for c in binary_cols if c < d], dtype=int)
    cont = np.setdiff1d(np.arange(d), binm)
    return cont, binm


def _rbf_sigma2(X, seed=0, sample=4096):
    """Median pairwise squared distance, estimated on a subsample.

    The exact median needs all n^2 distances, which is the very thing the
    blocked path exists to avoid. A few thousand rows pin the median down far
    more tightly than the kernel width needs.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Z = X[idx]
    sq = (Z ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    off = d2[~np.eye(len(Z), dtype=bool)]
    return float(np.median(off)) if off.size and off.max() > 0 else 1.0


def _jaccard_block(A, B):
    """Jaccard overlap between binary rows. A,B are 0/1 matrices."""
    inter = A @ B.T
    a = A.sum(axis=1)[:, None]
    b = B.sum(axis=1)[None, :]
    union = a + b - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32),
                     where=union > 0)


def _score_block(A, B, similarity, sigma2, binary=None, beta=0.5):
    """Similarity of every row of A against every row of B."""
    if similarity == "hybrid":
        # SBSCGM (arXiv:2508.00615 sec. 3.1) combines cosine similarity over
        # the continuous features with Jaccard overlap over the binary ones --
        # comorbidity flags, interventions, care unit. Cosine alone treats a
        # shared rare intervention as just another dimension; Jaccard treats it
        # as shared structure, which is what "clinically similar" means here.
        cont, binm = binary
        cos = A[:, cont] @ B[:, cont].T if cont.size else np.zeros((len(A), len(B)), np.float32)
        jac = _jaccard_block(A[:, binm], B[:, binm]) if binm.size else np.zeros_like(cos)
        return (1.0 - beta) * cos + beta * jac
    if similarity == "cosine":
        return A @ B.T
    if similarity == "euclidean_rbf":
        d2 = np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                        - 2.0 * (A @ B.T), 0.0)
        return np.exp(-d2 / (2.0 * sigma2))
    raise ValueError(f"unknown similarity: {similarity}")


def _similarity(X, similarity):
    """Dense [n, n] similarity matrix. Only for small cohorts.

    Kept for the threshold graph and for tests; `build_knn_graph` uses the
    blocked path instead, because an [n, n] float32 at MIMIC scale (n ~ 47k) is
    8.7 GB and does not coexist with a training run.
    """
    X = _normalise(X, similarity)
    n = len(X)
    if n > _DENSE_LIMIT:
        raise MemoryError(
            f"dense similarity for n={n:,} needs {n * n * 4 / 1e9:.1f} GB. "
            f"Use build_knn_graph (blocked) or raise _DENSE_LIMIT deliberately."
        )
    sigma2 = _rbf_sigma2(X) if similarity == "euclidean_rbf" else 1.0
    return _score_block(X, X, similarity, sigma2)


def _topk_torch(X, k, similarity, sigma2, device):
    """Blocked top-k on the GPU. Same result as the numpy path, ~50x faster."""
    Xt = torch.as_tensor(X, device=device)
    idx_out, val_out = [], []
    for start in range(0, len(Xt), _BLOCK):
        block = Xt[start:start + _BLOCK]
        if similarity == "cosine":
            S = block @ Xt.T
        else:
            d2 = torch.cdist(block, Xt).pow_(2).clamp_(min=0.0)
            S = torch.exp(-d2 / (2.0 * sigma2))
        rows = torch.arange(start, min(start + _BLOCK, len(Xt)), device=device)
        S[torch.arange(len(block), device=device), rows] = -float("inf")
        v, i = torch.topk(S, k, dim=1)
        idx_out.append(i.cpu().numpy())
        val_out.append(v.cpu().numpy())
    return np.concatenate(idx_out), np.concatenate(val_out)


def _topk_numpy(X, k, similarity, sigma2, binary=None, beta=0.5):
    """Blocked top-k in numpy. Memory is O(_BLOCK * n), not O(n^2)."""
    idx_out, val_out = [], []
    for start in range(0, len(X), _BLOCK):
        block = X[start:start + _BLOCK]
        S = _score_block(block, X, similarity, sigma2, binary=binary, beta=beta)
        S[np.arange(len(block)), np.arange(start, start + len(block))] = -np.inf
        part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx_out.append(np.take_along_axis(part, order, axis=1))
        val_out.append(np.take_along_axis(vals, order, axis=1))
    return np.concatenate(idx_out), np.concatenate(val_out)


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


def build_knn_graph(X, k, similarity="cosine", device=None, binary_cols=None,
                    hybrid_beta=0.5):
    """kNN over patient feature vectors. Returns (edge_index, edge_weight).

    Scored in row blocks so peak memory is O(block * n) rather than O(n^2),
    which is what makes a 47k-patient cohort tractable.
    """
    binary = _split_binary(np.asarray(X, dtype=np.float32), binary_cols) \
        if similarity == "hybrid" else None
    X = _normalise(X, similarity, binary_cols)
    n = len(X)
    k = min(k, n - 1)
    sigma2 = _rbf_sigma2(X) if similarity == "euclidean_rbf" else 1.0

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # The hybrid kernel mixes a dense matmul with a Jaccard term over a
    # different column set; it runs on the numpy path rather than duplicating
    # that logic in torch for a graph this size.
    if device != "cpu" and similarity != "hybrid":
        nbr, w = _topk_torch(X, k, similarity, sigma2, device)
    else:
        nbr, w = _topk_numpy(X, k, similarity, sigma2, binary=binary,
                             beta=hybrid_beta)

    rows = np.repeat(np.arange(n), k)
    cols = nbr.reshape(-1)
    w = w.reshape(-1)

    # Cosine similarity can be negative; message passing wants non-negative
    # weights, so shift into [0, 1]. The hybrid score is already non-negative.
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

    keys, vals = [], []
    for name in names:
        ei, ew = build_knn_graph(views[name], k, similarity)
        deg = np.zeros(n, dtype=np.float32)
        np.add.at(deg, ei[0], ew)
        deg = np.where(deg < 1e-8, 1.0, deg)
        ew = ew / deg[ei[0]]
        keys.append(ei[0].astype(np.int64) * n + ei[1].astype(np.int64))
        vals.append(weights[name] * ew)

    # Sum duplicate (src, dst) pairs across views without a Python-level loop
    # over the ~3.5M fused edges.
    key = np.concatenate(keys)
    val = np.concatenate(vals).astype(np.float32)
    uniq, inv = np.unique(key, return_inverse=True)
    fused = np.zeros(len(uniq), dtype=np.float32)
    np.add.at(fused, inv, val)

    pairs = np.stack([uniq // n, uniq % n])
    return pairs, fused


def build_graph(X, y, masks, cfg, meta=None, cache_dir="data/processed"):
    """Assemble the PyG Data object, caching by config hash."""
    train_mask, val_mask, test_mask = masks
    key = config_hash({"data": cfg["data"], "graph": cfg["graph"], "seed": cfg["seed"]})
    path = os.path.join(cache_dir, f"graph_{key}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)

    method = cfg["graph"]["method"]
    binary_cols = (meta or {}).get("binary_cols")
    if method == "knn":
        ei, ew = build_knn_graph(X, cfg["graph"]["k"], cfg["graph"]["similarity"],
                                 binary_cols=binary_cols,
                                 hybrid_beta=cfg["graph"].get("hybrid_beta", 0.5))
    elif method == "threshold":
        ei, ew = build_threshold_graph(X, cfg["graph"]["threshold"], cfg["graph"]["similarity"])
    elif method == "multiview":
        ei, ew = build_multiview_graph(
            _views_from(X, cfg, meta),
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
    # Severity proxies used only to stratify the safety evaluation. Never
    # model inputs: the model reads `data.x` alone.
    if meta and "true_risk" in meta:
        data.true_risk = torch.tensor(meta["true_risk"], dtype=torch.float32)
    if meta and "criticalness" in meta:
        data.criticalness = torch.tensor(meta["criticalness"], dtype=torch.float32)
    if meta and "severity" in meta:
        data.severity = torch.tensor(meta["severity"], dtype=torch.float32)

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(data, path)
    return data


def _views_from(X, cfg, meta):
    """Feature blocks for the multi-view graph.

    The MIMIC ETL reports true view membership in `meta["views"]` (one entry
    per feature block it built), so use that when it is present. The contiguous
    `_split_views` fallback exists only for the mock cohort, which has no real
    block structure.

    Note the column indices come from the *raw* cohort, while `X` here has been
    through `impute_and_scale`, which appends missingness indicators at the
    right-hand end. Those extra columns belong to no view and are dropped.
    """
    views = (meta or {}).get("views")
    if not views:
        return _split_views(X, cfg["graph"]["views"])
    out = {}
    for name, cols in views.items():
        cols = [c for c in cols if c < X.shape[1]]
        if cols:
            out[name] = X[:, cols]
    if not out:
        raise ValueError("meta['views'] selected no columns")
    return out


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
