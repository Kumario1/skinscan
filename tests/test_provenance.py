from datetime import datetime, timezone

from src.pipeline.provenance import (
    build_provenance, canonical_json_bytes, catalog_bundle_identity,
    compute_replay_key, file_identity,
    read_legacy_artifact, sha256_file, validate_artifact_freshness,
)


def base_inputs(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    return {
        "source_image_sha256": sha256_file(image),
        "dataset": {"name": "fixture", "sample_id": "sample", "split": "valid",
                    "split_proof": "synthetic"},
        "input_profile": {"skin_type": "oily", "pregnancy_status": "not_pregnant"},
        "effective_config": {"threshold": 0.5, "api_token": "must-not-leak"},
        "models": {"detector": {"sha256": "detector-hash"},
                   "classifier": {"state": "not_applicable", "sha256": None}},
        "catalog": {"state": "available", "sha256": "catalog-hash"},
        "ranker": {"state": "none", "sha256": None},
        "policies": {"triage": {"sha256": "triage"}, "therapy": {"sha256": "therapy"}},
    }


def build(tmp_path, **overrides):
    inputs = base_inputs(tmp_path)
    inputs.update(overrides)
    return build_provenance(
        inputs,
        clock=lambda: datetime(2026, 7, 13, tzinfo=timezone.utc),
        git_reader=lambda: {"git_commit": "abc", "dirty": False},
    )


def test_canonical_hash_is_independent_of_dict_insertion_order():
    assert canonical_json_bytes({"b": 2, "a": 1}) == canonical_json_bytes({"a": 1, "b": 2})
    assert compute_replay_key({"b": 2, "a": 1}) == compute_replay_key({"a": 1, "b": 2})


def test_generated_timestamp_and_attempt_id_do_not_change_replay_key():
    one = compute_replay_key({"x": 1, "generated_at": "a", "attempt_id": "1"})
    two = compute_replay_key({"attempt_id": "2", "generated_at": "b", "x": 1})
    assert one == two


def test_each_semantic_input_class_changes_replay_key(tmp_path):
    base = base_inputs(tmp_path)
    original = compute_replay_key(base)
    mutations = [
        {**base, "source_image_sha256": "changed"},
        {**base, "input_profile": {"skin_type": "dry"}},
        {**base, "effective_config": {"threshold": 0.7}},
        {**base, "catalog": {"sha256": "changed"}},
        {**base, "models": {"detector": {"sha256": "changed"}}},
        {**base, "policies": {"therapy": {"sha256": "changed"}}},
    ]
    assert all(compute_replay_key(value) != original for value in mutations)


def test_dirty_code_is_recorded_and_blocks_release(tmp_path):
    envelope = build_provenance(
        base_inputs(tmp_path), clock=lambda: "now",
        git_reader=lambda: ("abc", True),
    )
    assert envelope["code"]["dirty"] is True
    assert "code_dirty_or_unknown" in envelope["release_eligibility"]["reasons"]


def test_missing_artifact_identity_is_explicit(tmp_path):
    assert file_identity(tmp_path / "missing.bin")["state"] == "missing"
    assert file_identity(None) == {"state": "unavailable", "sha256": None}


def test_catalog_bundle_identity_changes_with_implicit_tier2_bytes(tmp_path):
    primary = tmp_path / "catalog.json"
    tier2 = tmp_path / "catalog_tier2.json"
    primary.write_text("[]")
    before = catalog_bundle_identity(primary)
    tier2.write_text('[{"product_id":"tier2"}]')
    after = catalog_bundle_identity(primary)
    assert before["sha256"] != after["sha256"]
    assert after["tier2"]["state"] == "available"


def test_envelope_sanitizes_config_and_is_fresh(tmp_path):
    envelope = build(tmp_path)
    assert "api_token" not in envelope["semantic_inputs"]["effective_config"]
    assert validate_artifact_freshness(envelope) == []


def test_mutated_artifact_is_rejected_as_stale(tmp_path):
    envelope = build(tmp_path)
    envelope["semantic_inputs"]["input_profile"]["skin_type"] = "dry"
    assert "stale_replay_key" in validate_artifact_freshness(envelope)


def test_forged_top_level_envelope_cannot_bypass_semantic_inputs(tmp_path):
    envelope = build(tmp_path)
    envelope["dataset"] = {"name": "fake", "sample_id": "sample", "split": "valid",
                           "split_proof": "forged"}
    envelope["code"] = {"git_commit": "abc", "dirty": True}
    envelope["models"] = {"detector": {"sha256": "forged"}}
    reasons = validate_artifact_freshness(envelope)
    assert "envelope_dataset_mismatch" in reasons
    assert "envelope_code_mismatch" in reasons
    assert "envelope_models_mismatch" in reasons


def test_legacy_artifact_is_labeled_and_not_comparable():
    legacy = read_legacy_artifact({"schema_version": "2.0", "x": 1})
    assert legacy["artifact_status"] == "legacy"
    assert legacy["comparable_to_v3"] is False
    assert validate_artifact_freshness({"schema_version": "2.0"}) == [
        "legacy_schema_not_comparable"
    ]
