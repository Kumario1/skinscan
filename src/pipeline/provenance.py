"""Deterministic v3 provenance and replay/freshness checks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping


VOLATILE_REPLAY_KEYS = {
    "generated_at", "output_dir", "attempt_id", "diagnostic_image_encoding",
    "latency_ms",
}
SECRET_KEY_PARTS = {"password", "secret", "token", "api_key", "authorization"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _semantic(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in VOLATILE_REPLAY_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def compute_replay_key(semantic_inputs: object) -> str:
    return hashlib.sha256(canonical_json_bytes(_semantic(semantic_inputs))).hexdigest()


def sanitized_config(value: object) -> object:
    if isinstance(value, Mapping):
        clean: dict[str, object] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in SECRET_KEY_PARTS):
                continue
            clean[str(key)] = sanitized_config(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [sanitized_config(item) for item in value]
    return value


def file_identity(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {"state": "unavailable", "sha256": None}
    path = Path(path)
    if not path.exists():
        return {"state": "missing", "sha256": None, "name": path.name}
    if not path.is_file():
        return {"state": "not_a_file", "sha256": None, "name": path.name}
    return {"state": "available", "sha256": sha256_file(path), "name": path.name}


def catalog_bundle_identity(
    path: str | Path | None,
    tier2_path: str | Path | None = None,
    drug_path: str | Path | None = None,
) -> dict[str, object]:
    """Identify every explicitly consumed catalog in the recommendation bundle."""
    primary = file_identity(path)
    if tier2_path is None and path is not None:
        tier2_path = Path(path).with_name("catalog_tier2.json")
    tier2 = file_identity(tier2_path)
    drug = file_identity(drug_path)
    identities = {"primary": primary, "tier2": tier2, "drug": drug}
    return {
        "state": primary["state"],
        "sha256": compute_replay_key(identities),
        **identities,
    }


def _now(clock: Callable[[], object]) -> str:
    value = clock()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _git(git_reader: Callable[[], object]) -> dict[str, object]:
    value = git_reader()
    if isinstance(value, Mapping):
        return {
            "git_commit": value.get("git_commit") or value.get("commit") or "unknown",
            "dirty": value.get("dirty") if isinstance(value.get("dirty"), bool) else None,
        }
    if isinstance(value, tuple) and len(value) == 2:
        return {"git_commit": str(value[0]), "dirty": bool(value[1])}
    return {"git_commit": "unknown", "dirty": None}


def build_provenance(
    inputs: Mapping[str, object],
    *,
    clock: Callable[[], object],
    git_reader: Callable[[], object],
    schema_version: str = "3",
) -> dict[str, object]:
    """Build one envelope shared verbatim by analysis and routine artifacts."""
    code = _git(git_reader)
    normalized = dict(inputs)
    normalized["schema_version"] = schema_version
    normalized["code"] = code
    if "effective_config" in normalized:
        normalized["effective_config"] = sanitized_config(normalized["effective_config"])
    semantic_inputs = _semantic(normalized)
    replay_key = compute_replay_key(semantic_inputs)

    blockers: list[str] = []
    dataset = semantic_inputs.get("dataset", {}) if isinstance(semantic_inputs, dict) else {}
    split = dataset.get("split") if isinstance(dataset, dict) else None
    if split in {None, "unknown", "train"}:
        blockers.append(f"dataset_split_{split or 'missing'}")
    if code["dirty"] is not False:
        blockers.append("code_dirty_or_unknown")
    models = semantic_inputs.get("models", {}) if isinstance(semantic_inputs, dict) else {}
    detector = models.get("detector", {}) if isinstance(models, dict) else {}
    if not isinstance(detector, dict) or not detector.get("sha256"):
        blockers.append("detector_identity_unknown")
    for gate in (
        "clinician_policy_approval", "adequate_calibration_cohort",
        "external_clinical_review_set", "verified_real_catalog_overlay",
    ):
        blockers.append(gate)

    envelope: dict[str, object] = {
        "schema_version": schema_version,
        "generated_at": _now(clock),
        "source_image_sha256": semantic_inputs.get("source_image_sha256"),
        "dataset": semantic_inputs.get("dataset", {}),
        "code": code,
        "models": semantic_inputs.get("models", {}),
        "config_sha256": compute_replay_key(semantic_inputs.get("effective_config", {})),
        "catalog": semantic_inputs.get("catalog", {"state": "unavailable", "sha256": None}),
        "ranker": semantic_inputs.get("ranker", {"state": "none", "sha256": None}),
        "policies": semantic_inputs.get("policies", {}),
        "input_profile": semantic_inputs.get("input_profile", {}),
        "semantic_inputs": semantic_inputs,
        "replay_key": replay_key,
        "release_eligibility": {"eligible": not blockers, "reasons": blockers},
    }
    return envelope


def validate_artifact_freshness(
    artifact: Mapping[str, object],
    current_inputs: Mapping[str, object] | None = None,
) -> list[str]:
    reasons: list[str] = []
    artifact_schema = str(artifact.get("schema_version"))
    if artifact_schema not in {"3", "4"}:
        return ["legacy_schema_not_comparable"]
    replay_key = artifact.get("replay_key")
    semantic_inputs = artifact.get("semantic_inputs")
    if not isinstance(replay_key, str):
        reasons.append("replay_key_missing")
    if not isinstance(semantic_inputs, Mapping):
        reasons.append("semantic_inputs_missing")
    elif replay_key != compute_replay_key(semantic_inputs):
        reasons.append("stale_replay_key")
    if isinstance(semantic_inputs, Mapping):
        expected_fields = {
            "source_image_sha256": semantic_inputs.get("source_image_sha256"),
            "dataset": semantic_inputs.get("dataset", {}),
            "code": semantic_inputs.get("code", {}),
            "models": semantic_inputs.get("models", {}),
            "catalog": semantic_inputs.get(
                "catalog", {"state": "unavailable", "sha256": None}
            ),
            "ranker": semantic_inputs.get("ranker", {"state": "none", "sha256": None}),
            "policies": semantic_inputs.get("policies", {}),
            "input_profile": semantic_inputs.get("input_profile", {}),
        }
        for field_name, expected in expected_fields.items():
            if _semantic(artifact.get(field_name)) != _semantic(expected):
                reasons.append(f"envelope_{field_name}_mismatch")
        expected_config_hash = compute_replay_key(semantic_inputs.get("effective_config", {}))
        if artifact.get("config_sha256") != expected_config_hash:
            reasons.append("envelope_config_sha256_mismatch")
    if current_inputs is not None:
        current = dict(current_inputs)
        current.setdefault("schema_version", artifact_schema)
        if isinstance(semantic_inputs, Mapping) and "code" in semantic_inputs:
            current.setdefault("code", semantic_inputs["code"])
        if replay_key != compute_replay_key(current):
            reasons.append("current_inputs_mismatch")
    return list(dict.fromkeys(reasons))


def read_legacy_artifact(value: Mapping[str, object]) -> dict[str, object]:
    """Display-only reader; callers must not compare it with v3 semantics."""
    return {
        "artifact_status": "legacy",
        "source_schema_version": value.get("schema_version", "unknown"),
        "comparable_to_v3": False,
        "payload": dict(value),
    }
