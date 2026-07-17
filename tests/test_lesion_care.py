import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.recommendation.lesion_care import (
    LESION_TYPES,
    authorize_mvp_fixture_inputs,
    build_care_pathways,
    build_lesion_findings,
    decide_exact_label_care,
    load_lesion_care_policy,
)
from src.recommendation.schema import UserProfile


def _profile(**overrides):
    value = {
        "age_years": 25,
        "pregnancy_status": "not_pregnant",
        "allergies": [],
        "sensitivity_conditions": [],
        "current_actives": [],
        "current_medications": [],
        "treatment_history": [],
        "acne_duration_weeks": 12,
        "painful_or_deep_lesions": False,
        "prior_scarring": False,
        "spot_new_or_changing": False,
        "spot_bleeding_itching_or_painful": False,
        "spot_bleeding": False,
        "spot_itching": False,
        "spot_painful": False,
        "spot_other_symptoms": False,
        "active_acne_controlled": True,
        "scar_duration_months": 12,
        "pregnancy_or_hormonal_medication_onset": False,
        "abcde_change_present": False,
        "wound_closed": True,
        "scar_diagnosis_confirmed_by_clinician": True,
    }
    value.update(overrides)
    return UserProfile.from_dict(value)


@pytest.fixture(scope="module")
def policy():
    return load_lesion_care_policy(
        "lesion_care_policy.proposed.json",
        report_path="LESION_CARE_EVIDENCE_REPORT.md",
        environment="development",
        input_types=("synthetic_profile", "fixture_image"),
    )


def _paths(label, profile, policy):
    observations = [SimpleNamespace(label=label, score=0.83, region="left_cheek")]
    findings = build_lesion_findings(observations, evidence_source="fixture")
    return findings, build_care_pathways(findings, profile.to_dict(), policy)


def test_policy_gate_is_exactly_ten_labels_with_sources(policy):
    assert policy.scope_authorized
    assert tuple(policy.labels) == LESION_TYPES
    assert all(row["source_ids"] for row in policy.labels.values())
    assert not policy.labels["nevus"]["care_path"]["retail_target_actives"]
    assert not policy.labels["other"]["care_path"]["retail_target_actives"]


def test_policy_fails_closed_outside_synthetic_scope():
    policy = load_lesion_care_policy(
        "lesion_care_policy.proposed.json",
        report_path="LESION_CARE_EVIDENCE_REPORT.md",
        environment="production",
        input_types=("real_user",),
    )
    assert not policy.scope_authorized
    findings, pathways = _paths("papule", _profile(), policy)
    papule = next(row for row in pathways if row["lesion_type"] == "papule")
    assert papule["status"] == "deferred"
    assert papule["retail_target_actives"] == []


def test_policy_edit_invalidates_the_manifest_bound_audit(tmp_path):
    value = json.loads(Path("lesion_care_policy.proposed.json").read_text())
    value["labels"][0]["care_path"]["reason_codes"].append("tampered")
    path = tmp_path / "lesion_care_policy.proposed.json"
    path.write_text(json.dumps(value))
    policy = load_lesion_care_policy(
        path,
        report_path="LESION_CARE_EVIDENCE_REPORT.md",
        manifest_path="lesion_care_source_manifest.json",
        environment="development",
        input_types=("synthetic_profile", "fixture_image"),
    )
    assert policy.audit_approved is False
    assert policy.scope_authorized is False
    assert "mvp_ai_research_audit_not_valid" in policy.scope_reasons


def test_caller_cannot_rewrite_policy_and_source_manifest_as_a_new_trust_root(
    tmp_path,
):
    value = json.loads(Path("lesion_care_policy.proposed.json").read_text())
    value["labels"][2]["care_path"]["reason_codes"].append("forged_path")
    policy_path = tmp_path / "lesion_care_policy.proposed.json"
    policy_path.write_text(json.dumps(value))
    manifest = json.loads(Path("lesion_care_source_manifest.json").read_text())
    manifest["ai_research_audit"]["policy_sha256"] = hashlib.sha256(
        policy_path.read_bytes()
    ).hexdigest()
    (tmp_path / "lesion_care_source_manifest.json").write_text(
        json.dumps(manifest)
    )
    policy = load_lesion_care_policy(
        policy_path,
        report_path="LESION_CARE_EVIDENCE_REPORT.md",
        environment="development",
        input_types=("synthetic_profile", "fixture_image"),
    )
    assert policy.audit_approved is False
    assert policy.scope_authorized is False
    assert "mvp_ai_research_audit_not_valid" in policy.scope_reasons


def test_real_photo_input_is_rejected_by_the_synthetic_mvp_policy():
    policy = load_lesion_care_policy(
        "lesion_care_policy.proposed.json",
        report_path="LESION_CARE_EVIDENCE_REPORT.md",
        environment="development",
        input_types=("synthetic_profile", "real_photo"),
    )
    assert not policy.scope_authorized
    assert "input_type_not_authorized:real_photo" in policy.scope_reasons


def test_caller_created_fixture_manifest_is_not_authority(tmp_path):
    image = tmp_path / "claimed-synthetic.jpg"
    profile = tmp_path / "claimed-synthetic.json"
    image.write_bytes(b"arbitrary input")
    profile.write_text('{"age_years":25}')
    manifest = tmp_path / "forged-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "skinscan-mvp-fixtures-1",
        "authorized_environments": ["test"],
        "dataset_names": ["synthetic"],
        "split_proofs": ["fixture"],
        "image_sha256s": [hashlib.sha256(image.read_bytes()).hexdigest()],
        "profile_sha256s": [hashlib.sha256(profile.read_bytes()).hexdigest()],
    }))
    authorization = authorize_mvp_fixture_inputs(
        manifest,
        image_bytes=image.read_bytes(),
        profile_path=profile,
        environment="test",
        dataset_name="synthetic",
        split_proof="fixture",
        normalized_profile={"age_years": 25},
    )
    assert authorization.authorized is False
    assert authorization.reasons == ("fixture_manifest_not_trusted",)


@pytest.mark.parametrize(
    "label,expected",
    [
        ("closed_comedo", "retail_eligible"),
        ("open_comedo", "retail_eligible"),
        ("papule", "retail_eligible"),
        ("pustule", "retail_eligible"),
        ("nodule", "clinician_only"),
        ("atrophic_scar", "clinician_only"),
        ("hypertrophic_scar", "clinician_only"),
        ("melasma", "retail_eligible"),
        ("nevus", "monitoring_only"),
        ("other", "unsupported"),
    ],
)
def test_every_exact_label_has_an_independent_path(policy, label, expected):
    findings, pathways = _paths(label, _profile(), policy)
    pathway = next(row for row in pathways if row["lesion_type"] == label)
    assert pathway["status"] == expected
    decision = decide_exact_label_care(findings, pathways)
    evidence = next(row for row in decision["decision_evidence"]
                    if row["lesion_type"] == label)
    assert evidence["probability"] is None
    assert evidence["quality"] == "high"


@pytest.mark.parametrize(
    "label,missing",
    [
        ("melasma", "pregnancy_or_hormonal_medication_onset"),
        ("papule", "age_years"),
    ],
)
def test_unknown_conditional_intake_defers_only_affected_path(policy, label, missing):
    profile = _profile(**{missing: None})
    _, pathways = _paths(label, profile, policy)
    pathway = next(row for row in pathways if row["lesion_type"] == label)
    assert pathway["status"] == "deferred"
    assert missing in pathway["required_answers"]
    assert pathway["retail_target_actives"] == []


@pytest.mark.parametrize(
    "label,missing,expected",
    [
        ("nevus", "spot_new_or_changing", "monitoring_only"),
        ("other", "spot_new_or_changing", "unsupported"),
        ("atrophic_scar", "scar_duration_months", "clinician_only"),
    ],
)
def test_unknown_intake_does_not_reclassify_nonretail_path(
    policy, label, missing, expected,
):
    _, pathways = _paths(label, _profile(**{missing: None}), policy)
    pathway = next(row for row in pathways if row["lesion_type"] == label)
    assert pathway["status"] == expected
    assert missing in pathway["required_answers"]
    assert f"required_intake_unknown:{missing}" in pathway["reason_codes"]
    assert pathway["retail_target_actives"] == []


def test_missing_symptom_answers_are_not_a_favorable_default(policy):
    profile = _profile(
        spot_bleeding_itching_or_painful=None,
        spot_bleeding=None,
        spot_itching=None,
        spot_painful=None,
        spot_other_symptoms=None,
    )
    _, pathways = _paths("nevus", profile, policy)
    pathway = next(row for row in pathways if row["lesion_type"] == "nevus")
    assert pathway["status"] == "monitoring_only"
    assert set(pathway["required_answers"]) >= {
        "spot_bleeding", "spot_itching", "spot_painful", "spot_other_symptoms",
    }


def test_normalized_profile_round_trips_declared_unknown_fields():
    original = UserProfile.from_dict({
        "pregnancy_status": "not_pregnant",
        "unknown_fields": ["spot_new_or_changing"],
    })
    restored = UserProfile.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
