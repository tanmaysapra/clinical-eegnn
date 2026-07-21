"""Synthetic EHR cohort generator.

Lets the whole pipeline run before PhysioNet credentialing lands. The generator
mimics the properties that actually stress the architecture: class imbalance,
missingness, informative features buried in noise, and -- critically -- a label
that is *not* a pure function of a patient's own features.

Why that last point matters: an i.i.d. tabular generator makes depth worthless.
Every patient is classifiable from their own row, layer 1 saturates AUROC, and
the exit policy degenerates to all-shallow, so the early-exit claim cannot be
tested at all. So the cohort is built in two blocks:

  * `ctx_*`   -- phenotype features. Noisy but always observed, they place a
                 patient near their clinical cluster and therefore drive the
                 kNN graph.
  * clinical  -- individual severity features (lactate, creatinine, ...) that
                 carry the personal component of risk.

Risk = individual severity + a cluster-level effect. For **easy** patients the
severity block is clean, so one layer suffices. For **hard** patients it is
corrupted, so their risk is only recoverable by aggregating similar patients --
which takes propagation depth. That is the easy/hard split early exit exists to
exploit.
"""

import numpy as np

# Names borrowed from the real MIMIC feature block so downstream SHAP plots and
# the mock path look the same.
_SEVERITY_NAMES = [
    "age",
    "lactate_max",
    "creatinine_max",
    "wbc_max",
    "sbp_min",
    "spo2_min",
    "gcs_min",
    "bun_max",
    "resp_rate_max",
    "temp_max",
]


def _solve_intercept(logits, prevalence, tol=1e-4, max_iter=200):
    """Bisect for c such that mean(sigmoid(logits - c)) == prevalence."""
    lo, hi = logits.min() - 20.0, logits.max() + 20.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate = (1.0 / (1.0 + np.exp(-(logits - mid)))).mean()
        if abs(rate - prevalence) < tol:
            return mid
        # Higher intercept -> lower rate, so the search is monotone decreasing.
        lo, hi = (mid, hi) if rate > prevalence else (lo, mid)
    return 0.5 * (lo + hi)


def make_mock_cohort(n_patients, n_features=40, missing_rate=0.2, seed=42,
                     prevalence=0.12, n_clusters=12, hard_fraction=0.35,
                     cluster_weight=1.0, corruption=3.0):
    """Generate a synthetic patient cohort.

    Returns (X, y, feature_names, meta). `meta` carries mock-only ground truth
    (true risk, cluster id, hard flag) for difficulty analysis -- never for
    training.
    """
    rng = np.random.default_rng(seed)
    n_sev = min(len(_SEVERITY_NAMES), n_features // 2)
    n_ctx = n_features - n_sev

    # -- phenotype block: cluster centroids + noise. Drives the kNN graph. ----
    cluster = rng.integers(0, n_clusters, size=n_patients)
    centroids = rng.normal(0.0, 2.0, size=(n_clusters, n_ctx))
    X_ctx = centroids[cluster] + rng.normal(0.0, 0.7, size=(n_patients, n_ctx))

    # -- severity block: the individual component of risk ---------------------
    X_sev_true = rng.standard_normal((n_patients, n_sev))
    beta = rng.normal(0.0, 1.0, size=n_sev)
    individual = X_sev_true @ beta
    individual = individual / (individual.std() + 1e-8)

    cluster_effect = rng.normal(0.0, 1.0, size=n_clusters)[cluster]

    logits = 1.5 * (individual + cluster_weight * cluster_effect)

    # -- easy vs hard ---------------------------------------------------------
    # Hard patients' severity readings are corrupted, so their own row no longer
    # identifies their risk; only their neighbourhood does. Their *label* still
    # follows the uncorrupted risk, which is what forces deeper propagation.
    hard = rng.random(n_patients) < hard_fraction
    X_sev = X_sev_true.copy()
    X_sev[hard] = rng.normal(0.0, corruption, size=(int(hard.sum()), n_sev))

    intercept = _solve_intercept(logits, prevalence)
    risk = 1.0 / (1.0 + np.exp(-(logits - intercept)))
    y = (rng.random(n_patients) < risk).astype(np.int64)

    X = np.concatenate([X_sev, X_ctx], axis=1).astype(np.float32)

    if missing_rate > 0:
        mask = rng.random(X.shape) < missing_rate
        X[mask] = np.nan

    feature_names = (
        list(_SEVERITY_NAMES[:n_sev]) + [f"ctx_{j}" for j in range(n_ctx)]
    )
    meta = {
        "true_risk": risk.astype(np.float32),
        "cluster": cluster,
        "hard": hard,
    }
    return X, y, feature_names, meta
