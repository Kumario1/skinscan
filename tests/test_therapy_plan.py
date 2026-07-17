import json
from pathlib import Path

import pytest

from src.recommendation.decision import TriagePolicy, decide_care
from src.recommendation.schema import CareDecision, Concern, ConcernReport, UserProfile
from src.recommendation.therapy import (
    RETINOID_THERAPIES, load_therapy_policy, plan_therapy,
)


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


# --- policy gating: every exclusion must surface a machine-readable reason ----

def _path_spec(**overrides):
    value = {
        "therapy": "azelaic_acid", "strength_band": "10%", "exposure": "leave_on",
        "cadence": "per_label", "cadence_source": "synthetic://label",
        "role": "treatment", "concerns": ["acne_inflammatory"],
    }
    value.update(overrides)
    return value


def _write_policy(tmp_path, paths, name="policy.json", **policy_overrides):
    value = {
        "policy_id": "gate-test", "version": "1", "reviewed": True,
        "test_only": True, "paths": paths,
    }
    value.update(policy_overrides)
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return load_therapy_policy(path)


def _plan(tmp_path, path_spec, **profile_overrides):
    decision, report = _decision_report()
    policy = _write_policy(tmp_path, [path_spec])
    return plan_therapy(decision, report, _profile(**profile_overrides), policy)


@pytest.mark.parametrize("spec, profile, reason", [
    ({"min_age_years": 18}, {"age_years": None}, "required_profile_unknown:age_years"),
    ({"min_age_years": 18}, {"age_years": 17}, "age_below_policy_minimum"),
    ({"max_age_years": 65}, {"age_years": 70}, "age_above_policy_maximum"),
    ({"excluded_sensitivity_conditions": ["rosacea"]},
     {"sensitivity_conditions": ["rosacea"]}, "sensitivity_condition_excludes:rosacea"),
    ({"conflicting_medications": ["isotretinoin"]},
     {"current_medications": ["isotretinoin"]}, "current_medication_conflict:isotretinoin"),
    ({"excluded_treatment_history": ["failed_azelaic"]},
     {"treatment_history": ["failed_azelaic"]}, "treatment_history_excludes:failed_azelaic"),
    ({"min_acne_duration_weeks": 8}, {"acne_duration_weeks": None},
     "required_profile_unknown:acne_duration_weeks"),
    ({"min_acne_duration_weeks": 8}, {"acne_duration_weeks": 4},
     "acne_duration_below_policy_minimum"),
    ({"max_acne_duration_weeks": 8}, {"acne_duration_weeks": 12},
     "acne_duration_above_policy_maximum"),
    ({"required_painful_or_deep_lesions": True}, {"painful_or_deep_lesions": None},
     "required_profile_unknown:painful_or_deep_lesions"),
    ({"required_painful_or_deep_lesions": True}, {"painful_or_deep_lesions": False},
     "profile_value_mismatch:painful_or_deep_lesions"),
    ({"required_prior_scarring": True}, {"prior_scarring": None},
     "required_profile_unknown:prior_scarring"),
    ({"required_prior_scarring": True}, {"prior_scarring": False},
     "profile_value_mismatch:prior_scarring"),
    ({"requires_known": ["skin_type"]}, {"skin_type": "unknown"},
     "required_profile_unknown:skin_type"),
], ids=lambda v: None)
def test_policy_gate_defers_with_reason(tmp_path, spec, profile, reason):
    plan = _plan(tmp_path, _path_spec(**spec), **profile)
    assert plan.primary is None
    assert reason in plan.deferred_reasons


@pytest.mark.parametrize("spec, profile", [
    ({"min_acne_duration_weeks": 8}, {"acne_duration_weeks": 8}),      # boundary: meets min
    ({"max_acne_duration_weeks": 8}, {"acne_duration_weeks": 8}),      # boundary: meets max
    ({"min_age_years": 18}, {"age_years": 18}),                        # boundary: meets min
    ({"required_painful_or_deep_lesions": True}, {"painful_or_deep_lesions": True}),
    ({"required_prior_scarring": False}, {"prior_scarring": False}),
], ids=lambda v: None)
def test_profile_that_satisfies_a_gate_is_eligible(tmp_path, spec, profile):
    """The inclusive side of each bound -- equal to the limit still qualifies."""
    plan = _plan(tmp_path, _path_spec(**spec), **profile)
    assert plan.primary is not None
    assert plan.deferred_reasons == []


@pytest.mark.parametrize("spec, profile", [
    ({"max_age_years": 65}, {"age_years": None}),
    ({"max_acne_duration_weeks": 8}, {"acne_duration_weeks": None}),
], ids=["age", "acne_duration"])
def test_a_max_bound_does_not_defer_on_an_unknown_value(tmp_path, spec, profile):
    """Deliberately asymmetric with the min bounds, which DO defer on unknown
    (confirmed 2026-07-15): a max bound is an appropriateness ceiling, not a
    safety floor. A policy that needs the value known says so via requires_known
    -- see the test below."""
    plan = _plan(tmp_path, _path_spec(**spec), **profile)
    assert plan.primary is not None
    assert plan.deferred_reasons == []


def test_requires_known_is_how_a_max_bound_policy_demands_the_value(tmp_path):
    plan = _plan(tmp_path, _path_spec(max_age_years=65, requires_known=["age_years"]),
                 age_years=None)
    assert plan.primary is None
    assert "required_profile_unknown:age_years" in plan.deferred_reasons


def test_non_retinoid_honours_policy_pregnancy_exclusions(tmp_path):
    """A non-retinoid is only excluded when the policy itself says so."""
    spec = _path_spec(excluded_pregnancy_statuses=["pregnant"])
    assert _plan(tmp_path, spec, pregnancy_status="pregnant").primary is None
    assert _plan(tmp_path, spec, pregnancy_status="not_pregnant").primary is not None


def test_unknown_pregnancy_defers_retinoid_but_not_azelaic(tmp_path):
    """Retinoids defer on unknown pregnancy; a non-retinoid has no such gate."""
    retinoid = _plan(tmp_path, _path_spec(therapy="adapalene", strength_band="0.1%"),
                     pregnancy_status="unknown")
    assert retinoid.primary is None
    assert "pregnancy_status_unknown_defers:adapalene" in retinoid.deferred_reasons
    assert _plan(tmp_path, _path_spec(), pregnancy_status="unknown").primary is not None


def test_path_whose_concerns_do_not_match_the_report_is_not_offered(tmp_path):
    plan = _plan(tmp_path, _path_spec(concerns=["acne_comedonal"]))
    assert plan.primary is None
    assert plan.deferred_reasons == ["no_policy_path_for_reported_concerns"]


def test_policy_with_no_paths_reports_no_path_rather_than_silence(tmp_path):
    decision, report = _decision_report()
    plan = plan_therapy(decision, report, _profile(), _write_policy(tmp_path, []))
    assert plan.primary is None
    assert plan.deferred_reasons == ["no_policy_path_for_reported_concerns"]


def test_eligible_paths_after_the_first_become_alternatives(tmp_path):
    decision, report = _decision_report()
    policy = _write_policy(tmp_path, [
        _path_spec(therapy="azelaic_acid"),
        _path_spec(therapy="benzoyl_peroxide", strength_band="2.5%"),
    ])
    plan = plan_therapy(decision, report, _profile(), policy)
    assert plan.primary.therapy == "azelaic_acid"
    assert [a.therapy for a in plan.alternatives] == ["benzoyl_peroxide"]
    assert plan.alternatives[0].reason == "eligible_policy_alternative"


def test_repeated_defer_reasons_are_reported_once(tmp_path):
    """Two paths blocked for the same reason must not duplicate it."""
    decision, report = _decision_report()
    policy = _write_policy(tmp_path, [
        _path_spec(therapy="azelaic_acid", min_age_years=18),
        _path_spec(therapy="benzoyl_peroxide", strength_band="2.5%", min_age_years=18),
    ])
    plan = plan_therapy(decision, report, _profile(age_years=16), policy)
    assert plan.deferred_reasons == ["age_below_policy_minimum"]


@pytest.mark.parametrize("disposition, guidance", [
    ("supportive_only", ["avoid_self_start_or_stop_medicine_pending_professional_review"]),
    ("maintenance", []),
])
def test_non_active_dispositions_carry_no_therapy(tmp_path, disposition, guidance):
    _, report = _decision_report()
    decision = CareDecision("routine", [], disposition, [], "1", True)
    plan = plan_therapy(decision, report, _profile(), _write_policy(tmp_path, [_path_spec()]))
    assert plan.primary is None
    assert plan.alternatives == []
    assert plan.deferred_reasons == guidance
    assert plan.support_roles == ["cleanser", "moisturizer", "sunscreen"]


# --- policy loader validation -------------------------------------------------

def test_absent_policy_file_is_an_unreviewed_policy_not_a_crash(tmp_path):
    policy = load_therapy_policy(tmp_path / "nope.json")
    assert policy.reviewed is False
    assert policy.source_path == str(tmp_path / "nope.json")
    assert policy.identifier == "missing-clinician-reviewed-policy:none"


def test_malformed_policy_json_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_therapy_policy(path)


@pytest.mark.parametrize("payload, match", [
    ('[]', "expected an object"),
    ('{"policy_id":"","version":"1","reviewed":true}', "policy_id"),
    ('{"policy_id":"p","version":"1","reviewed":"yes"}', "reviewed"),
    ('{"policy_id":"p","version":"1","reviewed":true,"reviewed_by":7}', "reviewed_by"),
    ('{"policy_id":"p","version":"1","reviewed":true}', "required for reviewed production"),
    ('{"policy_id":"p","version":"1","reviewed":true,"test_only":true,"paths":{}}',
     "paths: expected a list"),
    # the existing unknown-fields case never reaches this check
    ('{"policy_id":"p","version":"1","reviewed":true,"test_only":"false"}',
     "test_only: expected a boolean"),
    ('{"policy_id":"p","version":"1","reviewed":true,"reviewed_by":"derm","support_roles":"all"}',
     "support_roles: expected a list of strings"),
])
def test_policy_loader_rejects_malformed_documents(tmp_path, payload, match):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match=match):
        load_therapy_policy(path)


@pytest.mark.parametrize("spec, match", [
    ({"course_weeks": 0}, "course_weeks"),
    ({"review_at_weeks": -1}, "review_at_weeks"),
    ({"min_age_years": True}, "min_age_years"),
    ({"conflicting_actives": "benzoyl_peroxide"}, "conflicting_actives"),
    ({"required_prior_scarring": "yes"}, "required_prior_scarring"),
    ({"requires_known": ["not_a_profile_field"]}, "unknown profile fields"),
    ({"requires_known": ["pregnant_or_nursing"]}, "unknown profile fields"),
    ({"amount": 1}, "amount: expected string or null"),
    ({"amount": "pea_sized", "amount_source": 2}, "amount_source: expected string or null"),
    ({"amount": "pea_sized"}, "amount_source"),
    ({"reason": 5}, "reason"),
    ({"therapy": ""}, "therapy"),
    ({"cadence_source": None}, "cadence_source"),
    ({"typo": 1}, "unknown fields"),
])
def test_policy_loader_rejects_malformed_paths(tmp_path, spec, match):
    with pytest.raises(ValueError, match=match):
        _write_policy(tmp_path, [_path_spec(**spec)])


def test_policy_path_must_be_an_object(tmp_path):
    with pytest.raises(ValueError, match=r"paths\[0\]: expected an object"):
        _write_policy(tmp_path, ["azelaic_acid"])


def test_support_roles_fall_back_to_the_default_trio(tmp_path):
    policy = _write_policy(tmp_path, [], support_roles=[])
    assert policy.support_roles == ("cleanser", "moisturizer", "sunscreen")


# --- boundaries and defaults the mutation run found unpinned -------------------

def test_an_age_exactly_at_the_policy_maximum_is_still_eligible():
    """The bound is a ceiling, not an exclusion at the limit."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        plan = _plan(Path(d), _path_spec(max_age_years=65), age_years=65)
        assert plan.primary is not None
        assert plan.deferred_reasons == []


def test_an_acne_duration_exactly_at_the_policy_maximum_is_still_eligible(tmp_path):
    plan = _plan(tmp_path, _path_spec(max_acne_duration_weeks=8), acne_duration_weeks=8)
    assert plan.primary is not None


@pytest.mark.parametrize("therapy", sorted(RETINOID_THERAPIES))
@pytest.mark.parametrize("status", ["pregnant", "trying", "nursing"])
def test_every_retinoid_is_pregnancy_gated_without_the_policy_saying_so(tmp_path, therapy, status):
    """The hard-coded retinoid set is the safety net when a policy forgets its
    own excluded_pregnancy_statuses; every member must be covered by it."""
    plan = _plan(tmp_path, _path_spec(therapy=therapy), pregnancy_status=status)
    assert plan.primary is None
    assert f"pregnancy_status_excludes:{therapy}" in plan.deferred_reasons


def test_a_path_that_omits_concerns_defaults_to_both_acne_subtypes(tmp_path):
    """The default must be the documented pair, not "everything" or "nothing"."""
    spec = _path_spec()
    spec.pop("concerns")
    policy = _write_policy(tmp_path, [spec])
    assert policy.paths[0].concerns == ("acne_comedonal", "acne_inflammatory")


def test_an_explicitly_empty_concerns_list_falls_back_to_the_default(tmp_path):
    policy = _write_policy(tmp_path, [_path_spec(concerns=[])])
    assert policy.paths[0].concerns == ("acne_comedonal", "acne_inflammatory")


DEV_POLICY = Path(__file__).parent.parent / "configs" / "therapy_policy.dev.json"


def test_dev_default_policy_loads_reviewed_and_selects_bp_primary():
    policy = load_therapy_policy(DEV_POLICY)
    assert policy.reviewed is True
    assert policy.reviewed_by == "dev-default"
    assert [p.option.therapy for p in policy.paths] == ["benzoyl_peroxide"]
    decision, report = _decision_report()
    plan = plan_therapy(decision, report, _profile(), policy)
    assert plan.primary.therapy == "benzoyl_peroxide"
    assert plan.primary.strength_band == "2.5%"
    assert plan.deferred_reasons == []
