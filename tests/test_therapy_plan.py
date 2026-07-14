from pathlib import Path

import pytest

from src.recommendation.decision import TriagePolicy, decide_care
from src.recommendation.schema import Concern, ConcernReport, UserProfile
from src.recommendation.therapy import load_therapy_policy, plan_therapy


FIXTURE = Path(__file__).parent / "fixtures" / "therapy_policy_synthetic.json"


def _decision_report():
    report = ConcernReport("active", concerns=[
        Concern("acne_inflammatory", "forehead", 2, 0.9),
    ])
    decision = decide_care(report, TriagePolicy("test", "1", True))
    return decision, report


def _profile(**overrides):
    values = {
        "skin_type": "oily",
        "tone_bucket": "medium",
        "tone_source": "self_report",
        "age_years": 24,
        "pregnancy_status": "not_pregnant",
        "painful_or_deep_lesions": False,
        "prior_scarring": False,
    }
    values.update(overrides)
    return UserProfile(**values)


def test_missing_policy_defers_active_primary():
    decision, report = _decision_report()
    plan = plan_therapy(decision, report, _profile(), load_therapy_policy(None))
    assert plan.primary is None
    assert plan.deferred_reasons == ["clinician_reviewed_policy_missing"]
    assert plan.support_roles == ["cleanser", "moisturizer", "sunscreen"]


def test_synthetic_reviewed_policy_selects_deterministic_primary():
    decision, report = _decision_report()
    plan = plan_therapy(decision, report, _profile(), load_therapy_policy(FIXTURE))
    assert plan.primary.therapy == "azelaic_acid"
    assert plan.primary.cadence_source == "synthetic://azelaic-label"
    assert plan.course_weeks == plan.review_at_weeks == 12


@pytest.mark.parametrize("status", ["pregnant", "trying", "nursing", "unknown"])
def test_pregnancy_or_unknown_never_selects_a_retinoid_primary(status):
    decision, report = _decision_report()
    plan = plan_therapy(
        decision, report, _profile(pregnancy_status=status), load_therapy_policy(FIXTURE)
    )
    assert plan.primary is None or "adapalene" not in plan.primary.therapy
    assert all("adapalene" not in option.therapy for option in plan.alternatives)


def test_existing_active_creates_machine_readable_conflict_reason():
    decision, report = _decision_report()
    plan = plan_therapy(
        decision, report, _profile(current_actives=["benzoyl_peroxide"]),
        load_therapy_policy(FIXTURE),
    )
    assert "current_active_conflict:benzoyl_peroxide" in plan.deferred_reasons


def test_derm_first_has_no_active_treatment_path():
    report = ConcernReport("nodule", concerns=[
        Concern("acne_cystic", "chin_jaw", 4, 0.9),
    ])
    policy = TriagePolicy(
        "synthetic", "1", True, "identity", lambda value: value, 0.8, 0.6
    )
    decision = decide_care(report, policy)
    plan = plan_therapy(decision, report, _profile(), load_therapy_policy(FIXTURE))
    assert plan.primary is None
    assert plan.alternatives == []
    assert "treatment" not in plan.support_roles
    assert "avoid_self_start_or_stop_medicine_pending_professional_review" in (
        plan.deferred_reasons
    )


def test_policy_loader_rejects_direction_without_source(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"policy_id":"bad","version":"1","reviewed":true,"test_only":true,'
        '"paths":[{"therapy":"x","strength_band":"x","exposure":"leave_on",'
        '"cadence":"daily","role":"treatment"}]}'
    )
    with pytest.raises(ValueError, match="cadence_source"):
        load_therapy_policy(path)


def test_policy_loader_rejects_unknown_fields_and_non_boolean_test_only(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"policy_id":"bad","version":"1","reviewed":true,'
        '"test_only":"false","typo_ignored":1,"paths":[]}'
    )
    with pytest.raises(ValueError, match="unknown fields|test_only"):
        load_therapy_policy(path)


@pytest.mark.parametrize("status", ["pregnant", "trying", "nursing"])
def test_retinoid_gate_does_not_depend_on_optional_policy_exclusions(tmp_path, status):
    path = tmp_path / "retinoid.json"
    path.write_text(
        '{"policy_id":"test","version":"1","reviewed":true,"test_only":true,'
        '"paths":[{"therapy":"adapalene","strength_band":"0.1%",'
        '"exposure":"leave_on","cadence":"per_label",'
        '"cadence_source":"synthetic://label","role":"treatment",'
        '"concerns":["acne_inflammatory"]}]}'
    )
    decision, report = _decision_report()
    plan = plan_therapy(decision, report, _profile(pregnancy_status=status),
                        load_therapy_policy(path))
    assert plan.primary is None
    assert f"pregnancy_status_excludes:adapalene" in plan.deferred_reasons
