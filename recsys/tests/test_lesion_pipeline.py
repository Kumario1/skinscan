import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recsys.contracts import ContractViolation
from recsys.pipeline import run
from src.recommendation.lesion_care import (
    build_care_pathways,
    build_lesion_findings,
    decide_exact_label_care,
    exact_label_therapy_plan,
    load_lesion_care_policy,
)
from src.recommendation.schema import UserProfile


ROOT = Path(__file__).parents[2]
BASE = Path(__file__).parent / "fixtures" / "analysis_v3_sample.json"
PROFILE = Path(__file__).parent / "fixtures" / "profile_complete.json"
DATA = Path(__file__).parents[1] / "data"


def _profile(**overrides):
    value = json.loads(PROFILE.read_text())
    value.update({
        "spot_new_or_changing": False,
        "spot_bleeding_itching_or_painful": False,
        "active_acne_controlled": True,
        "scar_duration_months": 12,
        "pregnancy_or_hormonal_medication_onset": False,
        "abcde_change_present": False,
        "wound_closed": True,
        "scar_diagnosis_confirmed_by_clinician": True,
    })
    value.update(overrides)
    return UserProfile.from_dict(value)


def _analysis(tmp_path, labels, *, profile=None, mutate=None):
    profile = profile or _profile()
    normalized_profile_sha256 = hashlib.sha256(json.dumps(
        profile.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    fixture_profile_sha256 = {
        "196e888934136051a1beb499a6d7babd30e51bfcb4dba56dc82e7a7a9a1fe658": (
            "98148a01ad8da87339e01977d539813926d0d7b2b96f21d34ea1c1bbf72bbfc7"
        ),
        "ab380708a4267edc72df828d81330bea4d2122061013d98f16f7eea91a6c2e09": (
            "cf1ce100f13f85d89c75ffbfdfe048db7dbf9134270a72c25ac76ddce0ddae02"
        ),
    }[normalized_profile_sha256]
    policy = load_lesion_care_policy(
        ROOT / "lesion_care_policy.proposed.json",
        report_path=ROOT / "LESION_CARE_EVIDENCE_REPORT.md",
        environment="development",
        input_types=("synthetic_profile", "fixture_image"),
    )
    observations = [
        SimpleNamespace(label=label, score=0.9 - index * 0.01, region="left_cheek")
        for index, label in enumerate(labels)
    ]
    findings = build_lesion_findings(observations, evidence_source="fixture")
    pathways = build_care_pathways(findings, profile.to_dict(), policy)
    if mutate:
        mutate(pathways)
    decision = decide_exact_label_care(findings, pathways)
    decision["policy_version"] = policy.identity
    value = json.loads(BASE.read_text())
    value.update({
        "schema_version": "4",
        "source_image_sha256": (
            "15ac6670480316bb7f7ae83d3846ffcdc0a4c952a526186000283c378f32a7b0"
        ),
        "lesion_findings": findings,
        "care_pathways": pathways,
        # Deliberately invalid deprecated data proves it is never read.
        "concerns": [{"concern": "grouped_data_must_be_ignored", "severity": "bad"}],
        "decision": decision,
        "therapy_plan": exact_label_therapy_plan(pathways, policy),
        "input_profile": profile.to_dict(),
        "dataset": {
            "name": "fixture", "sample_id": "schema4", "split": "test",
            "split_proof": "synthetic-test-fixture",
        },
        "policies": {"lesion_care": {
            "identity": policy.identity,
            "sha256": policy.sha256,
            "report_sha256": policy.report_sha256,
            "source_manifest_sha256": policy.manifest_sha256,
            "fixture_manifest_sha256": (
                "10d018abc93ffb84a9ecee0b79cb16dbeaea92ff68108f9f3222f592fd001508"
            ),
            "fixture_image_sha256": (
                "15ac6670480316bb7f7ae83d3846ffcdc0a4c952a526186000283c378f32a7b0"
            ),
            "fixture_profile_sha256": fixture_profile_sha256,
            "fixture_normalized_profile_sha256": normalized_profile_sha256,
            "audit_approved": True,
            "scope_authorized": True,
            "input_scope": "synthetic_fixture",
        }},
    })
    path = tmp_path / ("analysis-" + "-".join(labels) + ".json")
    path.write_text(json.dumps(value))
    return path


def _document(tmp_path, labels, **kwargs):
    return run(
        _analysis(tmp_path, labels, **kwargs),
        data_root=DATA,
        generated_at="2026-07-16T00:00:00+00:00",
    )


@pytest.mark.parametrize(
    "label,expected",
    [
        ("closed_comedo", {"covered_by_product", "unfilled"}),
        ("open_comedo", {"covered_by_product", "unfilled"}),
        ("papule", {"covered_by_product"}),
        ("pustule", {"covered_by_product"}),
        ("nodule", {"clinician_only"}),
        ("atrophic_scar", {"clinician_only"}),
        ("hypertrophic_scar", {"clinician_only"}),
        ("melasma", {"covered_by_product"}),
        ("nevus", {"monitoring_only"}),
        ("other", {"unsupported"}),
    ],
)
def test_each_label_runs_end_to_end(tmp_path, label, expected):
    document = _document(tmp_path, [label])
    coverage = next(row for row in document["lesion_coverage"]
                    if row["lesion_type"] == label)
    assert coverage["status"] in expected
    assert all(target["lesion_type"] != "grouped_data_must_be_ignored"
               for target in document["target_lesions"])
    if label in {"nevus", "other", "nodule", "atrophic_scar", "hypertrophic_scar"}:
        assert "treatment" not in document["selected_products"]


def test_melasma_coverage_requires_deterministic_iron_oxide(tmp_path):
    document = _document(tmp_path, ["melasma"])
    coverage = document["lesion_coverage"][0]
    assert coverage["status"] == "covered_by_product"
    assert coverage["products"][0]["matched_actives"] == ["iron_oxides"]


def test_mixed_labels_dedupe_one_product_across_multiple_coverage_rows(tmp_path):
    document = _document(tmp_path, ["papule", "pustule"])
    selected = document["selected_products"]
    assert len(selected.values()) == len(set(selected.values()))
    assert len(selected) == len({step["slot"] for session in ("am", "pm", "per_label")
                                 for step in document["selected_regimen"][session]})
    treatment_ids = {
        row["products"][0]["product_id"] for row in document["lesion_coverage"]
        if row["lesion_type"] in {"papule", "pustule"}
    }
    assert len(treatment_ids) == 1


def test_product_without_relevant_active_cannot_claim_coverage(tmp_path):
    def mutate(pathways):
        papule = next(row for row in pathways if row["lesion_type"] == "papule")
        papule["retail_target_actives"] = [{
            "active_id": "nonexistent_active",
            "strength": "verified",
            "formulation": "synthetic fixture",
        }]

    document = _document(tmp_path, ["papule"], mutate=mutate)
    coverage = document["lesion_coverage"][0]
    assert coverage["status"] == "unfilled"
    assert coverage["products"] == []


def test_partial_verification_can_rank_when_required_facts_pass(tmp_path):
    document = _document(tmp_path, ["papule"])
    steps = [step for session in ("am", "pm", "per_label")
             for step in document["selected_regimen"][session]]
    assert steps
    assert any(step["verification_status"] == "partial" for step in steps)


def test_recsys_rejects_forged_schema4_scope_metadata(tmp_path):
    path = _analysis(tmp_path, ["papule"])
    data = json.loads(path.read_text())
    data["policies"]["lesion_care"]["sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="trusted MVP artifact"):
        run(path, data_root=DATA, generated_at="2026-07-16T00:00:00+00:00")


def test_recsys_rejects_unpinned_schema4_image_and_profile(tmp_path):
    path = _analysis(tmp_path, ["papule"])
    data = json.loads(path.read_text())
    data["source_image_sha256"] = "0" * 64
    data["policies"]["lesion_care"]["fixture_image_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="authorized fixture image"):
        run(path, data_root=DATA, generated_at="2026-07-16T00:00:00+00:00")

    path = _analysis(tmp_path, ["papule"])
    data = json.loads(path.read_text())
    data["input_profile"]["age_years"] = 26
    path.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="authorized pair"):
        run(path, data_root=DATA, generated_at="2026-07-16T00:00:00+00:00")


def test_recsys_rejects_individually_allowed_but_mismatched_profile_pair(tmp_path):
    path = _analysis(tmp_path, ["papule"])
    data = json.loads(path.read_text())
    data["policies"]["lesion_care"]["fixture_profile_sha256"] = (
        "cf1ce100f13f85d89c75ffbfdfe048db7dbf9134270a72c25ac76ddce0ddae02"
    )
    path.write_text(json.dumps(data))
    with pytest.raises(ContractViolation, match="not an authorized pair"):
        run(path, data_root=DATA, generated_at="2026-07-16T00:00:00+00:00")


def test_schema4_cannot_override_bound_profile_with_external_file(tmp_path):
    path = _analysis(tmp_path, ["papule"])
    with pytest.raises(ContractViolation, match="binds the resolved synthetic"):
        run(
            path,
            profile_path=PROFILE,
            data_root=DATA,
            generated_at="2026-07-16T00:00:00+00:00",
        )


def test_schema4_retail_spec_requires_audited_strength(tmp_path):
    def mutate(pathways):
        papule = next(row for row in pathways if row["lesion_type"] == "papule")
        papule["retail_target_actives"][0].pop("strength")

    with pytest.raises(ContractViolation, match="strength"):
        _document(tmp_path, ["papule"], mutate=mutate)


def test_underage_profile_cannot_target_adapalene(tmp_path):
    document = _document(tmp_path, ["papule"], profile=_profile(age_years=8))
    (target,) = document["target_lesions"]
    assert "adapalene" not in target["retail_target_actives"]
    assert target["retail_target_actives"] == ["benzoyl_peroxide"]


def test_deterministic_replay_for_exact_label_mix(tmp_path):
    path = _analysis(tmp_path, ["papule", "melasma"])
    kwargs = dict(data_root=DATA, generated_at="2026-07-16T00:00:00+00:00")
    assert json.dumps(run(path, **kwargs), sort_keys=True) == json.dumps(
        run(path, **kwargs), sort_keys=True
    )


def test_schema4_papule_golden_output(tmp_path):
    document = _document(tmp_path, ["papule"])

    def steps(session):
        return [
            {"slot": step["slot"], "product_id": step["product_id"]}
            for step in document["selected_regimen"][session]
        ]

    projection = {
        "status": document["status"],
        "target_lesions": [
            {
                "lesion_type": target["lesion_type"],
                "count": target["count"],
                "required_product_roles": target["required_product_roles"],
                "retail_target_actives": target["retail_target_actives"],
            }
            for target in document["target_lesions"]
        ],
        "selected_products": document["selected_products"],
        "lesion_coverage": [
            {
                "lesion_type": row["lesion_type"],
                "status": row["status"],
                "products": row["products"],
            }
            for row in document["lesion_coverage"]
        ],
        "selected_regimen": {
            "archetype": document["selected_regimen"]["archetype"],
            "am": steps("am"),
            "pm": steps("pm"),
            "per_label": steps("per_label"),
        },
    }
    expected = json.loads(
        (Path(__file__).parent / "fixtures" / "schema4_papule_golden.json").read_text()
    )
    assert projection == expected
