import pytest

from src import manifest as M


def _partitions():
    return {
        "train": list(range(0, 60)),
        "model_val": list(range(60, 70)),
        "gate_selection": list(range(70, 80)),
        "gate_certification": list(range(80, 90)),
        "test": list(range(90, 100)),
    }


def _seeds(split=42):
    return {
        "split": split, "init": 1, "dataloader": 2,
        "graph_search": 3, "calibration": 4, "certification": 5,
    }


def _cfg():
    return {"data": {"source": "mock"}, "arch4r": {"views": {"demographics": [0, 1]}}}


def _build(split_seed=42):
    return M.build_manifest(
        cfg=_cfg(),
        cohort_fingerprint="deadbeef",
        partitions=_partitions(),
        seeds=_seeds(split=split_seed),
        code_commit="abc123",
        lockfile_hash="lockhash",
    )


def test_build_manifest_rejects_missing_seed_field():
    seeds = _seeds()
    del seeds["certification"]
    with pytest.raises(ValueError, match="missing required fields"):
        M.build_manifest(
            cfg=_cfg(), cohort_fingerprint="deadbeef", partitions=_partitions(),
            seeds=seeds, code_commit="abc123", lockfile_hash="lockhash",
        )


def test_manifest_is_reproducible_from_identical_inputs():
    m1 = _build()
    m2 = _build()
    assert m1 == m2
    assert m1["manifest_hash"] == m2["manifest_hash"]


def test_changing_only_the_seed_invalidates_every_seed_dependent_cache():
    m1 = _build(split_seed=42)
    m2 = _build(split_seed=43)
    assert m1["seeds"]["split"] != m2["seeds"]["split"]
    assert m1["manifest_hash"] != m2["manifest_hash"], (
        "changing only the split seed must change the manifest hash -- "
        "otherwise a cache keyed on this manifest would be reused across seeds"
    )
    with pytest.raises(AssertionError, match="seeds"):
        M.verify_manifest(m1, m2)


def test_deleting_a_seed_field_fails_loading(tmp_path):
    good = _build()
    del good["seeds"]["certification"]
    path = tmp_path / "manifest.json"
    import json
    path.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="missing required seed field"):
        M.load_manifest(path)


def test_load_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        M.load_manifest(tmp_path / "does_not_exist.json")


def test_save_then_load_round_trips(tmp_path):
    m = _build()
    path = tmp_path / "manifest.json"
    M.save_manifest(m, path)
    loaded = M.load_manifest(path)
    assert loaded == m


def test_verify_manifest_passes_on_identical_fresh_computation():
    m1 = _build()
    m2 = _build()
    M.verify_manifest(m1, m2)  # must not raise


def test_verify_manifest_reports_every_mismatch_not_just_the_first():
    m1 = _build(split_seed=42)
    m2 = _build(split_seed=43)
    m2["code_commit"] = "different"
    try:
        M.verify_manifest(m1, m2)
        assert False, "expected AssertionError"
    except AssertionError as e:
        msg = str(e)
        assert "seeds" in msg and "code_commit" in msg


def test_hash_patient_ids_is_order_independent():
    a = M.hash_patient_ids([3, 1, 2])
    b = M.hash_patient_ids([1, 2, 3])
    c = M.hash_patient_ids([1, 2, 3, 4])
    assert a == b
    assert a != c
