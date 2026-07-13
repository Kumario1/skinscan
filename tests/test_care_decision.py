import json
from pathlib import Path

import pytest

from src.recommendation.decision import TriagePolicy, conservative_unreviewed_policy, decide_care
from src.recommendation.schema import Concern, ConcernEvidence, ConcernReport


AUDIT_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "concern_correctness_cases.json").read_text()
)["cases"]


def _case(sample_id):
    return next(case for case in AUDIT_CASES if case["sample_id"] == sample_id)


def _report(case):
    concerns = []
    for item in case.get("concerns", []):
        labels = item.get("labels", {})
        concerns.append(Concern(
            item["concern"], item["region"], item["severity"], item["confidence"],
            lesion_count=item.get("lesion_count"),
            evidence=ConcernEvidence(labels, item["confidence"], 1),
        ))
    return ConcernReport(case["sample_id"], concerns=concerns, clear_skin=not concerns)


def approved_policy() -> TriagePolicy:
    # Synthetic fixture only; these values are not a release threshold.
    return TriagePolicy(
        "synthetic-test", "1", True, calibrator_id="identity-test-only",
        calibrator=lambda raw: raw, nodule_gate=0.8, abstain_lower=0.6,
    )


def test_clear_skin_is_routine_maintenance():
    decision = decide_care(ConcernReport("clear", clear_skin=True), approved_policy())
    assert (decision.triage_level, decision.therapy_disposition) == ("routine", "maintenance")


def test_random_120_high_count_without_real_nodule_does_not_suppress_treatment():
    report = ConcernReport("random-120", concerns=[
        Concern("acne_inflammatory", "forehead", 4, 0.9, lesion_count=67),
    ])
    decision = decide_care(report, approved_policy())
    assert decision.triage_level == "routine_plus_review"
    assert decision.therapy_disposition == "active_treatment"
    assert "high_count_or_severity_review" in decision.referral_reasons


def test_scarring_adds_review_without_suppressing_active_treatment():
    report = ConcernReport("scar", concerns=[
        Concern("acne_inflammatory", "forehead", 2, 0.8),
        Concern("acne_scarring", "left_cheek", 3, 0.8),
    ])
    decision = decide_care(report, approved_policy())
    assert (decision.triage_level, decision.therapy_disposition) == (
        "routine_plus_review", "active_treatment"
    )
    assert "scarring_risk" in decision.referral_reasons


def test_random_230_oracle_nodule_routes_derm_first():
    report = ConcernReport("random-230-oracle", concerns=[
        Concern("acne_cystic", "left_cheek", 4, 0.9, lesion_count=1,
                evidence=ConcernEvidence({"nodule": 1}, 0.9, 1)),
    ])
    decision = decide_care(report, approved_policy())
    assert (decision.triage_level, decision.therapy_disposition) == (
        "derm_first", "supportive_only"
    )
    assert decision.evidence[0].calibrated
    assert decision.evidence[0].probability == 0.9


def test_uncalibrated_nodule_signal_abstains_without_probability():
    report = ConcernReport("prediction", concerns=[
        Concern("acne_cystic", "chin_jaw", 4, 0.97, lesion_count=1,
                evidence=ConcernEvidence({"nodule": 1}, 0.97, 1)),
    ])
    decision = decide_care(report, conservative_unreviewed_policy())
    assert (decision.triage_level, decision.therapy_disposition) == (
        "abstain", "supportive_only"
    )
    assert decision.referral_reasons == ["unvalidated_nodule_evidence"]
    assert decision.evidence[0].probability is None
    assert not decision.evidence[0].calibrated


def test_approved_abstention_band_is_not_forced_to_a_binary_decision():
    report = ConcernReport("uncertain", concerns=[
        Concern("acne_cystic", "chin_jaw", 4, 0.7,
                evidence=ConcernEvidence({"nodule": 1}, 0.7, 1)),
    ])
    assert decide_care(report, approved_policy()).triage_level == "abstain"


def test_multiple_nodule_signals_are_order_invariant_and_conservative():
    high = Concern("acne_cystic", "left_cheek", 4, 0.95,
                   evidence=ConcernEvidence({"nodule": 1}, 0.95, 1))
    low = Concern("acne_cystic", "right_cheek", 1, 0.1,
                  evidence=ConcernEvidence({"nodule": 1}, 0.1, 1))
    forward = decide_care(ConcernReport("forward", concerns=[high, low]), approved_policy())
    reverse = decide_care(ConcernReport("reverse", concerns=[low, high]), approved_policy())
    assert forward.triage_level == reverse.triage_level == "derm_first"
    assert forward.therapy_disposition == reverse.therapy_disposition == "supportive_only"


def test_raw_non_high_risk_confidence_is_not_serialized_as_probability():
    report = ConcernReport("pigment", concerns=[
        Concern("hyperpigmentation", "forehead", 2, 0.43),
    ])
    decision = decide_care(report, approved_policy())
    assert decision.evidence[0].probability is None
    assert decision.evidence[0].quality == "low"
    words = str(decision.to_dict()).lower()
    assert "diagnos" not in words
    assert "malignan" not in words


@pytest.mark.parametrize("sample_id", ["random-120", "random-252", "random-274"])
def test_audit_count_only_predictions_do_not_suppress_treatment(sample_id):
    case = _case(sample_id)
    decision = decide_care(_report(case), approved_policy())
    assert decision.triage_level == case["expected"]["triage_level"]
    assert decision.therapy_disposition == case["expected"]["therapy_disposition"]
    assert case["held_out_release_evidence"] is False


@pytest.mark.parametrize("sample_id", ["random-252", "random-274"])
def test_missed_nodule_prediction_and_oracle_counterfactual_stay_distinct(sample_id):
    case = _case(sample_id)
    prediction = decide_care(_report(case), approved_policy())
    oracle_report = ConcernReport(f"{sample_id}-oracle", concerns=[
        Concern(
            "acne_cystic", "left_cheek", 4, 0.9, lesion_count=1,
            evidence=ConcernEvidence({"nodule": 1}, 0.9, 1),
        )
    ])
    oracle = decide_care(oracle_report, approved_policy())
    assert case["evidence_source"] == "prediction"
    assert case["oracle_counterfactual"]["nodule_present"] is True
    assert prediction.triage_level == "routine_plus_review"
    assert oracle.triage_level == "derm_first"


def test_random_147_raw_pigment_burden_is_quality_not_probability():
    case = _case("random-147")
    decision = decide_care(_report(case), approved_policy())
    assert decision.evidence[0].probability is case["expected"]["probability"]
    assert decision.evidence[0].quality == case["expected"]["quality"]
    assert case["held_out_release_evidence"] is False
