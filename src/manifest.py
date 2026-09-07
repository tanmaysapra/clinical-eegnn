"""The seed-keyed integrity manifest for Architecture 4R.

Why this exists
----------------
This exact failure has already cost the project real results twice: a cache
keyed without the seed silently reused seed 42's graph (and, in the worse
case, its split masks) for every other seed, inflating `run_architectures.py`'s
first pass to ~0.91 AUROC before being caught (see `old/ARCHITECTURES.md`, "A leak
found and fixed in this driver"). `run_architectures.py::run_seed` now guards
against exactly this by setting `cfg["seed"] = seed` explicitly and asserting
every loaded mask matches a fresh computation.

A manifest generalises that one assertion into a single, content-addressed
object covering every seed-dependent artefact in the Architecture 4R pipeline
(split, init, dataloader, graph search, calibration, certification), plus the
raw data, cohort/feature/outcome definitions, and code/dependency identity.
Loading requires exact equality against a freshly computed manifest -- there
is no seedless, wildcard, or "latest" fallback lookup anywhere in this module.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REQUIRED_SEED_FIELDS = (
    "split", "init", "dataloader", "graph_search", "calibration", "certification",
)


# ------------------------------------------------------------------ hashing

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return _sha256_bytes(blob)


def fingerprint_raw_table(raw_dir_or_data) -> str:
    """Content fingerprint of a raw data source.

    A directory (e.g. the MIMIC-IV raw CSV tree) is fingerprinted by the
    sorted list of (relative path, byte size) for every file under it --
    cheap, and sufficient to detect a swapped or re-downloaded table without
    reading tens of gigabytes on every run. Anything array-like (a DataFrame
    or ndarray, for a mock/CSV cohort) is fingerprinted over its raw bytes.
    """
    if isinstance(raw_dir_or_data, (str, Path)):
        root = Path(raw_dir_or_data)
        if not root.exists():
            raise FileNotFoundError(f"raw data path does not exist: {root}")
        entries = sorted(
            (str(p.relative_to(root)), p.stat().st_size)
            for p in root.rglob("*") if p.is_file()
        )
        return _sha256_json(entries)
    if hasattr(raw_dir_or_data, "to_numpy"):  # pandas DataFrame/Series
        arr = raw_dir_or_data.to_numpy()
    else:
        arr = np.asarray(raw_dir_or_data)
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def hash_patient_ids(ids) -> str:
    """Order-independent fingerprint of a patient-ID set (one per split)."""
    arr = np.asarray(sorted(np.asarray(ids).tolist()))
    return _sha256_bytes(arr.tobytes()) if arr.size else _sha256_bytes(b"")


def hash_array(arr) -> str:
    """Fingerprint of any numpy-convertible object (weights, thresholds, ...)."""
    return _sha256_bytes(np.ascontiguousarray(np.asarray(arr)).tobytes())


def current_code_commit(repo_dir: str | Path | None = None) -> str:
    """Best-effort git commit hash; `"unknown"` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
            text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def dependency_lockfile_hash(lockfile_path: str | Path) -> str:
    """Hash of a pinned dependency lockfile.

    `requirements.txt` in this repo uses open version ranges for everything
    except xgboost/shap/numpy, so it is not itself a lockfile -- generate one
    (e.g. `pip freeze > requirements.lock`) before relying on this for a real
    run. Raises rather than silently hashing an unpinned file, since a
    manifest built against a moving target is not an integrity check.
    """
    path = Path(lockfile_path)
    if not path.exists():
        raise FileNotFoundError(
            f"no dependency lockfile at {path}. Generate one (e.g. "
            "`pip freeze > requirements.lock`) before building a manifest -- "
            "hashing an open-range requirements.txt would not catch a "
            "dependency change."
        )
    return _sha256_bytes(path.read_bytes())


# --------------------------------------------------------------- the manifest

def build_manifest(cfg, cohort_fingerprint: str, partitions: dict,
                    seeds: dict, code_commit: str, lockfile_hash: str,
                    extra_hashes: dict | None = None) -> dict:
    """Assemble the full manifest dict for one seed's run.

    `partitions` maps partition name -> patient-id array (train, model_val,
    gate_selection, gate_certification, test); each is hashed via
    `hash_patient_ids`. `seeds` must carry every key in
    `REQUIRED_SEED_FIELDS` -- this is checked here (build time) and again in
    `load_manifest` (load time), so a manifest can never be built or loaded
    with a silently-missing seed field.
    """
    missing = [k for k in REQUIRED_SEED_FIELDS if k not in seeds]
    if missing:
        raise ValueError(f"seeds dict is missing required fields: {missing}")

    manifest = {
        "schema_version": 1,
        "cohort_fingerprint": cohort_fingerprint,
        "cohort_query_hash": _sha256_json(cfg.get("data", {})),
        "feature_schema_hash": _sha256_json(cfg.get("arch4r", {}).get("views", {})),
        "outcome_definition_hash": _sha256_json(cfg.get("data", {}).get("outcome", "mortality")),
        "seeds": {k: int(seeds[k]) for k in REQUIRED_SEED_FIELDS},
        "partition_id_hashes": {
            name: hash_patient_ids(ids) for name, ids in partitions.items()
        },
        "code_commit": code_commit,
        "dependency_lockfile_hash": lockfile_hash,
        "extra_hashes": dict(extra_hashes or {}),
    }
    manifest["manifest_hash"] = _sha256_json(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    )
    return manifest


def save_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_manifest(path: str | Path) -> dict:
    """Load a manifest, raising if any required seed field is absent.

    There is deliberately no fallback here: a manifest missing a seed field,
    or a path that doesn't exist, is a hard failure, not a "rebuild from
    scratch" signal -- silently rebuilding would be exactly the kind of
    seedless-cache behaviour this module exists to prevent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")
    manifest = json.loads(path.read_text())
    seeds = manifest.get("seeds", {})
    missing = [k for k in REQUIRED_SEED_FIELDS if k not in seeds]
    if missing:
        raise ValueError(
            f"manifest at {path} is missing required seed field(s) {missing} "
            "-- refusing to load. No wildcard or seedless fallback exists."
        )
    return manifest


def verify_manifest(manifest: dict, fresh: dict) -> None:
    """Assert byte-for-byte equality between a loaded and a freshly built manifest.

    Raises `AssertionError` listing *every* mismatched top-level field (not
    just the first one found), so a genuine drift is diagnosable in one shot
    rather than requiring repeated re-runs to find each mismatch in turn.
    """
    mismatches = []
    keys = set(manifest) | set(fresh)
    for key in sorted(keys):
        if key == "manifest_hash":
            continue  # a derived field; comparing it is redundant with the rest
        if manifest.get(key) != fresh.get(key):
            mismatches.append((key, manifest.get(key), fresh.get(key)))
    if mismatches:
        lines = "\n".join(
            f"  - {k}: loaded={loaded!r} fresh={new!r}" for k, loaded, new in mismatches
        )
        raise AssertionError(
            f"manifest verification failed, {len(mismatches)} field(s) differ:\n{lines}"
        )
