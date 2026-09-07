"""Architecture 4R's density-adaptive, per-view, inductive patient graph.

Why this is a new module rather than an extension of `src/graph_build.py`
--------------------------------------------------------------------------
`graph_build.py` is fully transductive (every split's rows sit in the graph
from construction time) and uses one shared `k`/similarity metric across
every view. Neither property is safe to reuse here: Architecture 4R's own
methodology requires held-out patients to attach only to a *frozen* training
reference graph (no held-out-to-held-out edge may ever exist), and requires
a distinct construction per view (self-tuning local scale, density-adaptive
neighbourhood size, a per-view qualification threshold). `graph_build.py`'s
`_topk_numpy`/`_topk_torch` scoring kernels are still fine primitives for the
raw distance computation and are used here for that purpose only.

Explicit design decision (open question in the Phase 0/1 implementation
plan): the per-view qualification threshold `T_v` is always fit under the
**adaptive** rule (each training node's own adaptive k_i), even when this
module is asked to build the fixed-k=16 ablation arm. `T_v` is a property of
the view's own distance geometry, not of which k a given experimental arm
happens to use, and sharing it keeps the fixed-k vs. adaptive-k comparison
apples-to-apples (only k differs; qualification, reciprocity radius, and
novelty ECDF are all identical between the two arms by construction).

Topology only, no encoder here
-------------------------------
Everything in this module (candidate search, sigma/rho, adaptive k,
qualification, reciprocity, novelty, cold start) depends only on
*preprocessed features* -- never on the encoder's learned codes `z`. The
encoder is trained jointly with the edge scorer, so `z` changes every
optimizer step; baking it into a frozen topology would either go stale
immediately or force a topology rebuild every step, which the training
procedure explicitly forbids (at most one rebuild, after epoch 10). The
learned, `z`-dependent edge scoring (GATv2-style scorer + view-reliability
fusion) lives at the bottom of this module and is meant to be called fresh
every forward pass against this module's frozen topology.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

VIEW_METRIC = {
    "demographics": "euclidean", "comorbidity": "cosine", "labs": "euclidean",
    "vitals": "euclidean", "treatment": "cosine",
}


@dataclass
class ViewGraphConfig:
    k_cand: int = 64
    k0: int = 16
    gamma: float = 0.5              # in {0.25, 0.5, 0.75}, selected on val
    k_min_adaptive: int = 4
    k_max_adaptive: int = 48
    percentile_qualify: float = 99.5
    fixed_k: int | None = None      # None -> adaptive; int (16) -> fixed-k ablation arm


# ------------------------------------------------------------- candidate search

def candidate_search(X_train: np.ndarray, X_query: np.ndarray, k_cand: int,
                      metric: str):
    """Exact k_cand-nearest-neighbour search, `X_query` against `X_train`.

    Returns (idx [n_query, k_cand], dist [n_query, k_cand]), sorted ascending.
    Exact by default -- brute-force is well within reach at this cohort size
    (~17k patients per view); `audit_ann_recall` gates any future ANN use.
    """
    nn_ = NearestNeighbors(n_neighbors=min(k_cand, len(X_train)), metric=metric,
                           algorithm="brute")
    nn_.fit(X_train)
    dist, idx = nn_.kneighbors(X_query)
    return idx, dist


def audit_ann_recall(X_train: np.ndarray, k_cand: int, metric: str, ann_fn) -> dict:
    """Compare an approximate search function against exact search.

    `ann_fn(X_train, X_train, k_cand, metric) -> (idx, dist)`, same contract
    as `candidate_search`. Returns recall@k_cand statistics; the caller (not
    this function) decides whether to actually use the ANN path based on the
    thresholds in the Architecture 4R spec (mean >= 0.98, min-decile >= 0.95).
    """
    exact_idx, _ = candidate_search(X_train, X_train, k_cand, metric)
    ann_idx, _ = ann_fn(X_train, X_train, k_cand, metric)
    n = exact_idx.shape[0]
    recall = np.array([
        len(set(exact_idx[i]) & set(ann_idx[i])) / k_cand for i in range(n)
    ])
    deciles = np.quantile(recall, np.linspace(0.0, 0.9, 10))
    return {
        "recall_at_k_mean": float(recall.mean()),
        "recall_at_k_min_decile": float(deciles.min()),
        "used_ann": False,  # the caller sets this True only after checking thresholds
    }


# ------------------------------------------------------- density / adaptive k

def self_tuning_sigma(dist_to_k0: np.ndarray) -> np.ndarray:
    """sigma_i = median distance to the nearest k0 candidates."""
    return np.median(dist_to_k0, axis=1)


def local_density(dist_candidates: np.ndarray, sigma_query: np.ndarray,
                   sigma_candidates: np.ndarray) -> np.ndarray:
    """rho_i^(v), self-tuning kernel over the full k_cand candidate set.

    `sigma_candidates` is [n_query, k_cand] -- sigma_j for each of node i's
    candidates j, gathered by the caller from the training set's own sigma.
    """
    denom = 2.0 * sigma_query[:, None] * sigma_candidates + 1e-8
    return np.exp(-(dist_candidates ** 2) / denom).sum(axis=1)


def standardize_log_density(rho: np.ndarray, train_median: float, train_mad: float) -> np.ndarray:
    log_rho = np.log(rho + 1e-8)
    denom = train_mad if train_mad > 1e-8 else 1.0
    return (log_rho - train_median) / denom


def adaptive_k(q: np.ndarray, k0: int, gamma: float, k_min: int, k_max: int) -> np.ndarray:
    k = np.round(k0 * np.exp(gamma * q)).astype(int)
    return np.clip(k, k_min, k_max)


def delta(dist: np.ndarray, sigma_i, sigma_j) -> np.ndarray:
    """Dimensionless self-tuned distance Delta_ij = d^2 / (2 sigma_i sigma_j + eps)."""
    return (dist ** 2) / (2.0 * sigma_i * sigma_j + 1e-8)


def fit_qualification_threshold(train_deltas_loo: np.ndarray, percentile: float) -> float:
    return float(np.percentile(train_deltas_loo, percentile))


def base_edge_weight(delta_ij: np.ndarray, reciprocal_mask: np.ndarray) -> np.ndarray:
    return np.exp(-delta_ij) * (0.5 + 0.5 * reciprocal_mask.astype(np.float64))


def graph_availability(qualified_ref_count: np.ndarray, reliable_view_count: np.ndarray,
                        k_min: int = 4) -> np.ndarray:
    return (qualified_ref_count >= k_min) & (reliable_view_count >= 1)


# --------------------------------------------------------- per-view reference

@dataclass
class _ViewReference:
    X_train: np.ndarray
    metric: str
    sigma_train: np.ndarray          # [n_train], leave-one-out
    k_train: np.ndarray              # [n_train], adaptive k actually used to build edges
    reciprocity_radius: np.ndarray   # [n_train], distance to the k_train-th neighbour
    T_v: float                       # qualification threshold, fit under the adaptive rule
    sigma_log_median: float
    sigma_log_mad: float             # (of standardized log-density, for reuse if needed)
    ecdf_values: np.ndarray          # sorted leave-one-out l_i^(v), for novelty percentile
    edge_index_ref: np.ndarray       # [2, E] train-train edges (row 0: src i, row 1: dst j)
    edge_weight_ref: np.ndarray      # [E] base weight b_ij^(v)
    r_min: float                     # reliability threshold for this view


class ReferenceGraph:
    """Frozen, train-only topology for every view, plus everything needed to
    inductively attach held-out patients without ever touching it."""

    def __init__(self, views: dict[str, _ViewReference], cfg: ViewGraphConfig,
                 train_qualified_count: np.ndarray | None = None,
                 train_reliable_view_count: np.ndarray | None = None):
        self.views = views
        self.cfg = cfg
        # Per-train-node coverage bookkeeping, same rule `attach()` applies to
        # held-out queries (`graph_availability`) -- needed because Arch4R's
        # training is transductive (every train node is also scored as its
        # own query against this same frozen graph), so a train node can be
        # just as "cold start" (too few qualifying same-view neighbours) as
        # a held-out one, and the fallback path must trigger for both the
        # same way. Optional/None only for backward compatibility with any
        # caller constructing a bare `ReferenceGraph` directly in a test.
        self.train_qualified_count = train_qualified_count
        self.train_reliable_view_count = train_reliable_view_count

    def train_graph_available(self, k_min: int | None = None) -> np.ndarray:
        """Per-train-node availability, computed the identical way `attach()`
        computes it for held-out queries -- see `graph_availability`."""
        if self.train_qualified_count is None or self.train_reliable_view_count is None:
            raise ValueError("ReferenceGraph was not built with train-side coverage "
                             "counts (use build_reference_graph, not a bare constructor)")
        k_min = self.cfg.k_min_adaptive if k_min is None else k_min
        return graph_availability(self.train_qualified_count, self.train_reliable_view_count,
                                  k_min=k_min)

    # ---------------------------------------------------------------- attach
    def attach(self, query_views: dict[str, np.ndarray],
               query_reliability: dict[str, np.ndarray]) -> "AttachedGraph":
        """Attach a batch of held-out queries as new source nodes.

        Never mutates `self.views` (no reference-reference edge, density
        statistic, or fitted parameter can move because of held-out data),
        and never connects one query to another -- only query -> train edges.
        """
        n_query = next(iter(query_views.values())).shape[0]
        per_view_edges = {}
        novelty_per_view = np.zeros((n_query, len(self.views)))
        reliable_mask = np.zeros((n_query, len(self.views)), dtype=bool)
        qualified_count = np.zeros(n_query, dtype=int)

        for vi, (name, ref) in enumerate(self.views.items()):
            Xq = query_views[name]
            k_cand = min(self.cfg.k_cand, len(ref.X_train))
            idx, dist = candidate_search(ref.X_train, Xq, k_cand, ref.metric)

            k0 = min(self.cfg.k0, k_cand)
            sigma_q = self_tuning_sigma(dist[:, :k0])
            sigma_cand = ref.sigma_train[idx]
            delta_ij = delta(dist, sigma_q[:, None], sigma_cand)

            # novelty: median Delta over the same k0 neighbours used for sigma_q
            l_q = np.median(delta_ij[:, :k0], axis=1)
            novelty_per_view[:, vi] = np.searchsorted(ref.ecdf_values, l_q) / max(
                len(ref.ecdf_values), 1)

            r_q = query_reliability[name]
            reliable = r_q >= ref.r_min
            reliable_mask[:, vi] = reliable

            k_use = (np.full(n_query, self.cfg.fixed_k) if self.cfg.fixed_k is not None
                    else None)  # adaptive k computed per-query below if needed
            if k_use is None:
                # Adaptive k for a query uses the *query's own* density, consistent
                # with how a training node's k_i was derived from its own rho_i.
                rho_q = local_density(dist, sigma_q, sigma_cand)
                q_std = standardize_log_density(rho_q, ref.sigma_log_median, ref.sigma_log_mad)
                k_use = adaptive_k(q_std, self.cfg.k0, self.cfg.gamma,
                                   self.cfg.k_min_adaptive, self.cfg.k_max_adaptive)
            k_use = np.minimum(k_use, k_cand)

            qualifies = delta_ij <= ref.T_v
            keep = np.zeros_like(qualifies)
            for r in range(n_query):
                keep[r, :k_use[r]] = qualifies[r, :k_use[r]]

            recip_radius = ref.reciprocity_radius[idx]
            reciprocal = dist <= recip_radius
            w = base_edge_weight(delta_ij, reciprocal)

            src = np.repeat(np.arange(n_query), k_cand)[keep.reshape(-1)]
            dst = idx.reshape(-1)[keep.reshape(-1)]
            wt = w.reshape(-1)[keep.reshape(-1)]
            per_view_edges[name] = (src, dst, wt)
            qualified_count += keep.sum(axis=1)

        novelty = np.zeros(n_query)
        for r in range(n_query):
            rel = reliable_mask[r]
            if not rel.any():
                novelty[r] = 1.0
                continue
            weights = np.array(list(query_reliability[name][r] for name in self.views))
            weights = weights * rel
            weights = weights / (weights.sum() + 1e-8)
            novelty[r] = float((weights * novelty_per_view[r]).sum())

        n_reliable_views = reliable_mask.sum(axis=1)
        available = graph_availability(qualified_count, n_reliable_views,
                                       k_min=self.cfg.k_min_adaptive)

        return AttachedGraph(
            per_view_edges=per_view_edges, novelty=novelty,
            graph_available=available, n_query=n_query,
        )


@dataclass
class AttachedGraph:
    """Per-view (query -> train) edges for one batch of held-out or training queries."""
    per_view_edges: dict[str, tuple]   # name -> (src[E], dst[E], base_weight[E])
    novelty: np.ndarray                # [n_query], in [0, 1]
    graph_available: np.ndarray        # [n_query], bool
    n_query: int


# ------------------------------------------------------------ building it all

def build_reference_graph(train_views: dict[str, np.ndarray],
                          train_reliability: dict[str, np.ndarray],
                          cfg: ViewGraphConfig, metrics: dict[str, str] | None = None,
                          seed: int = 42) -> ReferenceGraph:
    """Build the frozen train-only reference topology for every view.

    `train_views[name]` is that view's preprocessed (imputed/winsorized/
    scaled) feature matrix for training patients only. `train_reliability
    [name]` is the per-patient observed-feature fraction for that view
    (`src.encoder_mv.observed_fraction`, computed on the *raw*, pre-
    imputation array).
    """
    metrics = metrics or VIEW_METRIC
    views = {}
    n_train_total = next(iter(train_views.values())).shape[0]
    qualified_count = np.zeros(n_train_total, dtype=int)
    reliable_view_count = np.zeros(n_train_total, dtype=int)
    for name, X in train_views.items():
        metric = metrics.get(name, "euclidean")
        n = X.shape[0]
        k_cand = min(cfg.k_cand, n - 1)
        k0 = min(cfg.k0, k_cand)

        idx, dist = candidate_search(X, X, k_cand + 1, metric)
        idx, dist = _drop_self(idx, dist, np.arange(n))
        idx, dist = idx[:, :k_cand], dist[:, :k_cand]

        sigma = self_tuning_sigma(dist[:, :k0])
        sigma_cand = sigma[idx]
        rho = local_density(dist, sigma, sigma_cand)
        log_rho = np.log(rho + 1e-8)
        med, mad = float(np.median(log_rho)), float(_mad(log_rho))
        q_std = standardize_log_density(rho, med, mad)
        k_i = adaptive_k(q_std, cfg.k0, cfg.gamma, cfg.k_min_adaptive, cfg.k_max_adaptive)
        k_i = np.minimum(k_i, k_cand)

        delta_ij = delta(dist, sigma[:, None], sigma_cand)

        # T_v under the adaptive rule, always -- see module docstring.
        loo_deltas = np.concatenate([delta_ij[i, :k_i[i]] for i in range(n) if k_i[i] > 0])
        T_v = fit_qualification_threshold(loo_deltas, cfg.percentile_qualify)

        k_use = np.full(n, cfg.fixed_k) if cfg.fixed_k is not None else k_i
        k_use = np.minimum(k_use, k_cand)

        qualifies = delta_ij <= T_v
        keep = np.zeros_like(qualifies)
        for i in range(n):
            keep[i, :k_use[i]] = qualifies[i, :k_use[i]]

        # reciprocity radius: each node's own k_i-th qualified-candidate distance,
        # used both to compute train-train reciprocity here and to let a future
        # held-out query check reciprocity without rebuilding this graph.
        recip_radius = np.array([
            dist[i, k_i[i] - 1] if k_i[i] > 0 else 0.0 for i in range(n)
        ])
        reciprocal = dist <= recip_radius[idx]
        w = base_edge_weight(delta_ij, reciprocal)

        src = np.repeat(np.arange(n), k_cand)[keep.reshape(-1)]
        dst = idx.reshape(-1)[keep.reshape(-1)]
        wt = w.reshape(-1)[keep.reshape(-1)]

        l_i = np.median(delta_ij[:, :k0], axis=1)
        ecdf_values = np.sort(l_i)

        r = train_reliability[name]
        r_min = float(np.clip(np.percentile(r, 5), 0.20, 0.80))

        qualified_count += keep.sum(axis=1)
        reliable_view_count += (r >= r_min).astype(int)

        views[name] = _ViewReference(
            X_train=X, metric=metric, sigma_train=sigma, k_train=k_i,
            reciprocity_radius=recip_radius, T_v=T_v,
            sigma_log_median=med, sigma_log_mad=mad, ecdf_values=ecdf_values,
            edge_index_ref=np.stack([src, dst]), edge_weight_ref=wt, r_min=r_min,
        )
    return ReferenceGraph(views, cfg, train_qualified_count=qualified_count,
                          train_reliable_view_count=reliable_view_count)


def _drop_self(idx: np.ndarray, dist: np.ndarray, self_indices: np.ndarray):
    """Remove each row's self-match (present when querying a set against itself)."""
    n, k = idx.shape
    out_idx = np.empty((n, k - 1), dtype=idx.dtype)
    out_dist = np.empty((n, k - 1), dtype=dist.dtype)
    for i in range(n):
        mask = idx[i] != self_indices[i]
        # If self appears more than once isn't possible for a metric NN search,
        # but guard anyway by taking the first k-1 non-self entries.
        kept = np.flatnonzero(mask)[: k - 1]
        out_idx[i] = idx[i, kept]
        out_dist[i] = dist[i, kept]
    return out_idx, out_dist


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def novelty_score(*args, **kwargs):
    """Kept as a thin, direct entry point mirroring the spec's function name;
    the actual computation happens inline in `ReferenceGraph.attach` and
    `build_reference_graph` (train-side), since it needs per-view state that
    only those two call sites hold. See `_ViewReference.ecdf_values`."""
    raise NotImplementedError(
        "use ReferenceGraph.attach(...).novelty (held-out) or "
        "the leave-one-out l_i computed inside build_reference_graph (train)"
    )


# ------------------------------------------------------ dynamic edge scoring

class EdgeScorerGATv2(nn.Module):
    """e_ij^(v) = a_v^T LeakyReLU(W_v[z_i||z_j||abs(z_i-z_j)] + b_v), hidden width 16.

    GATv2-style: the nonlinearity sits *before* the collapse to a scalar via
    `a_v`, which is what actually breaks the "static attention" degeneracy
    (unlike collapsing to a scalar first and applying LeakyReLU after) -- see
    the Architecture 4R review history for the derivation.
    """

    def __init__(self, z_dim: int, hidden: int = 16):
        super().__init__()
        self.proj = nn.Linear(3 * z_dim, hidden)
        self.a = nn.Linear(hidden, 1, bias=False)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([z_i, z_j, (z_i - z_j).abs()], dim=-1)
        h = F.leaky_relu(self.proj(feat))
        e = self.a(h).squeeze(-1)
        return 3.0 * torch.tanh(e / 3.0)  # bounded before exponentiation


class ViewReliability(nn.Module):
    """pi_i^(v) = softmax_v(c_v^T z_i^(v) + beta_v*log(r_i^(v)+eps) + missing_mask)."""

    def __init__(self, z_dims: dict[str, int]):
        super().__init__()
        self.view_names = list(z_dims)
        self.c = nn.ModuleDict({v: nn.Linear(z_dims[v], 1, bias=False) for v in self.view_names})
        self.beta = nn.ParameterDict({v: nn.Parameter(torch.zeros(1)) for v in self.view_names})

    def forward(self, z_view: dict[str, torch.Tensor], reliability_raw: dict[str, torch.Tensor],
                missing_mask: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        scores = []
        for v in self.view_names:
            s = self.c[v](z_view[v]).squeeze(-1) + self.beta[v] * torch.log(
                reliability_raw[v] + 1e-4)
            if missing_mask is not None and v in missing_mask:
                s = s + torch.where(missing_mask[v], torch.full_like(s, -1e4), torch.zeros_like(s))
            scores.append(s)
        stacked = torch.stack(scores, dim=1)  # [n, n_views]
        pi = torch.softmax(stacked, dim=1)
        return {v: pi[:, i] for i, v in enumerate(self.view_names)}


def fuse_views(per_view_edges: dict[str, tuple], reliability: dict[str, torch.Tensor],
              scorer_output: dict[str, torch.Tensor], n_nodes: int,
              top_k_keep: int = 64):
    """Combine per-view (src, dst, base_weight) edges into one fused, sparse
    graph: A_ij = sum_v sqrt(pi_i^(v) pi_j^(v)) b_ij^(v) exp(e_ij^(v)).

    `reliability[v]` is [n_nodes] (pi^(v) for every node in this batch's
    node set, src and dst share the same indexing). `scorer_output[v]` is
    [E_v], the bounded GATv2 score already computed per view's edges.
    Returns (edge_index [2, E<=n_nodes*top_k_keep], edge_weight [E]).
    """
    all_src, all_dst, all_w = [], [], []
    for v, (src, dst, base_w) in per_view_edges.items():
        if len(src) == 0:
            continue
        src_t = torch.as_tensor(src, dtype=torch.long, device=reliability[v].device)
        dst_t = torch.as_tensor(dst, dtype=torch.long, device=reliability[v].device)
        base_w_t = torch.as_tensor(base_w, dtype=torch.float32, device=reliability[v].device)
        pi_src, pi_dst = reliability[v][src_t], reliability[v][dst_t]
        w = torch.sqrt(pi_src * pi_dst + 1e-12) * base_w_t * torch.exp(scorer_output[v])
        all_src.append(src_t); all_dst.append(dst_t); all_w.append(w)

    if not all_src:
        dev = next(iter(reliability.values())).device
        empty = torch.zeros((2, 0), dtype=torch.long, device=dev)
        return empty, torch.zeros(0, device=dev)

    src = torch.cat(all_src); dst = torch.cat(all_dst); w = torch.cat(all_w)
    # Sum duplicate (src, dst) pairs across views before top-k keep.
    key = src * n_nodes + dst
    uniq_key, inverse = torch.unique(key, return_inverse=True)
    w_summed = torch.zeros(len(uniq_key), dtype=w.dtype, device=w.device)
    w_summed.index_add_(0, inverse, w)
    src_u = (uniq_key // n_nodes)
    dst_u = (uniq_key % n_nodes)

    edge_index, edge_weight = _top_k_per_src(src_u, dst_u, w_summed, top_k_keep)
    return edge_index, edge_weight


def _top_k_per_src(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, k: int):
    """Keep at most the top-`k` highest-weight outgoing edges per source node."""
    w_np_detached = w.detach().cpu().numpy()
    perm = np.lexsort((-w_np_detached, src.detach().cpu().numpy()))  # sort by (src, -w)
    src_np, dst_np, w_np = src.cpu().numpy()[perm], dst.cpu().numpy()[perm], w[perm]
    keep = np.zeros(len(src_np), dtype=bool)
    count = {}
    for i, s in enumerate(src_np):
        c = count.get(s, 0)
        if c < k:
            keep[i] = True
            count[s] = c + 1
    kept_idx = np.flatnonzero(keep)
    edge_index = torch.stack([
        torch.as_tensor(src_np[kept_idx], dtype=torch.long, device=src.device),
        torch.as_tensor(dst_np[kept_idx], dtype=torch.long, device=src.device),
    ])
    return edge_index, w_np[kept_idx]


def row_normalize_with_selfloop(edge_index: torch.Tensor, weight: torch.Tensor, n: int):
    """Add a self-loop (weight = each node's median retained outgoing weight)
    and row-normalize: P = D^-1 A."""
    src, dst = edge_index[0], edge_index[1]
    self_w = torch.full((n,), 1.0, dtype=weight.dtype, device=weight.device)
    for i in range(n):
        m = src == i
        if m.any():
            self_w[i] = weight[m].median()
    self_idx = torch.arange(n, dtype=torch.long, device=edge_index.device)
    full_src = torch.cat([src, self_idx])
    full_dst = torch.cat([dst, self_idx])
    full_w = torch.cat([weight, self_w])

    deg = torch.zeros(n, dtype=weight.dtype, device=weight.device)
    deg.index_add_(0, full_src, full_w)
    deg = deg.clamp_min(1e-8)
    p_w = full_w / deg[full_src]
    return torch.stack([full_src, full_dst]), p_w
