"""Architecture 4R's weak multi-view encoder.

Why "weak" is load-bearing, not a hedge
---------------------------------------
Architecture 2 (`ppr_push.py`) already showed that local-push diffusion adds
real signal (+0.0207 AUROC) *on top of a shallow, graph-free encoder* -- and
adds nothing on top of an already-strong tabular score (the closed graph-
residual experiment, `old/ARCHITECTURES.md` SS2/SS3). So this encoder is
deliberately capped in capacity: no boosted-tree score, no leaf identifier,
no strong teacher logit, and no label-derived feature is allowed anywhere
near it. Making it stronger would put diffusion back in the regime where the
project has already measured that it buys nothing.

Five views, one encoder each: demographics, comorbidity, labs, vitals,
treatment -- the same five blocks `src/tabular.py`'s `STAGE_BLOCKS` groups
into stages, kept separate here because Architecture 4R's graph needs one
similarity structure per view, not one per cascade stage.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VIEW_OUT_DIMS = {
    "demographics": 16, "comorbidity": 32, "labs": 32,
    "vitals": 32, "treatment": 32,
}
VIEW_ORDER = tuple(VIEW_OUT_DIMS)  # fixed iteration order everywhere


# --------------------------------------------------------------- preprocessing

@dataclass
class _ViewStats:
    median: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    scale: np.ndarray


class ViewPreprocessors:
    """Per-view median/winsorize/scale, fit train-only -- one `_ViewStats` per view.

    Mirrors `src.data_prep.impute_and_scale`'s train-only discipline, but
    per-view and with winsorization (a self-tuned graph is more sensitive to
    a single extreme value than a tree ensemble is, so clipping matters more
    here than it does for `data_prep`'s tabular path).
    """

    def __init__(self, stats: dict[str, _ViewStats]):
        self.stats = stats

    def transform(self, view_raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for name, x in view_raw.items():
            s = self.stats[name]
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isnan(x), s.median, x)
            x = np.clip(x, s.lo, s.hi)
            out[name] = ((x - s.median) / s.scale).astype(np.float32)
        return out


def fit_view_preprocessors(view_raw: dict[str, np.ndarray], train_mask) -> ViewPreprocessors:
    """Fit medians/winsorization limits/scale on training rows only per view."""
    train_mask = np.asarray(train_mask, dtype=bool)
    stats = {}
    for name, x in view_raw.items():
        x = np.asarray(x, dtype=np.float64)
        xt = x[train_mask]
        median = np.nanmedian(xt, axis=0)
        median = np.where(np.isnan(median), 0.0, median)
        q1 = np.nanpercentile(xt, 25, axis=0)
        q3 = np.nanpercentile(xt, 75, axis=0)
        iqr = np.where(np.isnan(q3 - q1) | (q3 - q1 < 1e-6), 1.0, q3 - q1)
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        scale = np.where(iqr < 1e-6, 1.0, iqr)
        stats[name] = _ViewStats(median=median, lo=lo, hi=hi, scale=scale)
    return ViewPreprocessors(stats)


def observed_fraction(view_raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-patient, per-view observed-feature fraction r_i^(v) -- computed on
    the *raw* (pre-imputation) arrays, since after `ViewPreprocessors.transform`
    every NaN has already been filled in and this signal would be lost."""
    out = {}
    for name, x in view_raw.items():
        x = np.asarray(x, dtype=np.float64)
        out[name] = 1.0 - np.isnan(x).mean(axis=1)
    return out


# -------------------------------------------------------------------- encoder

class ViewEncoder(nn.Module):
    """z^(v) = LN(W2 GELU(W1 x + b1) + b2), one hidden layer, width 64."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.fc1(x)))
        return self.ln(self.fc2(h))


class MultiViewEncoder(nn.Module):
    """Five per-view encoders + one residual fusion layer -> h_i^(0).

    A view that is entirely missing for a patient (`reliability_raw[v][i] ==
    0`) has its encoded code replaced by a learned, view-specific missing
    embedding -- so a patient with a whole block absent gets a deterministic,
    trainable representation instead of an encoder output computed from
    imputed medians pretending to be data.
    """

    def __init__(self, in_dims: dict[str, int], fusion_dim: int = 64,
                 dropout: float = 0.15, out_dims: dict[str, int] | None = None):
        super().__init__()
        out_dims = out_dims or VIEW_OUT_DIMS
        self.view_names = [v for v in VIEW_ORDER if v in in_dims]
        missing = [v for v in in_dims if v not in VIEW_OUT_DIMS]
        if missing:
            raise ValueError(f"unknown view name(s), not in VIEW_OUT_DIMS: {missing}")

        self.encoders = nn.ModuleDict({
            v: ViewEncoder(in_dims[v], out_dims[v], dropout=dropout)
            for v in self.view_names
        })
        self.missing_embed = nn.ParameterDict({
            v: nn.Parameter(torch.zeros(out_dims[v])) for v in self.view_names
        })
        concat_dim = sum(out_dims[v] for v in self.view_names)
        self.fusion_skip = nn.Linear(concat_dim, fusion_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.fusion_ln = nn.LayerNorm(fusion_dim)
        self.fusion_dim = fusion_dim

    def forward(self, view_x: dict[str, torch.Tensor],
                view_observed_frac: dict[str, torch.Tensor]) -> dict:
        view_z, reliability_raw = {}, {}
        for v in self.view_names:
            r = view_observed_frac[v]  # [n], in [0, 1]
            z = self.encoders[v](view_x[v])
            missing_row = (r <= 0.0)
            if missing_row.any():
                z = torch.where(
                    missing_row.unsqueeze(1),
                    self.missing_embed[v].unsqueeze(0).expand(z.size(0), -1),
                    z,
                )
            view_z[v] = z
            reliability_raw[v] = r

        concat = torch.cat([view_z[v] for v in self.view_names], dim=1)
        h0 = self.fusion_ln(self.fusion_skip(concat) + self.fusion_mlp(concat))
        return {"h0": h0, "view_z": view_z, "reliability_raw": reliability_raw}

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
