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


def test_calibrator_that_yields_no_probability_abstains_rather_than_treating():
    """RULES.md:247 -- "Uncalibrated nodule evidence produces abstain +
    supportive_only." A calibrator that returns no probability has calibrated
    nothing, so the safety-critical nodule signal must not fall through to
    active treatment merely because a calibrator was configured.
    """
    policy = TriagePolicy(
        "p", "1", approved=True, calibrator_id="cal-1",
        calibrator=lambda confidence: None, nodule_gate=0.6, abstain_lower=0.3,
    )
    report = ConcernReport("no-probability", concerns=[
        Concern("acne_cystic", "chin_jaw", 4, 0.97, lesion_count=1,
                evidence=ConcernEvidence({"nodule": 1}, 0.97, 1)),
        Concern("acne_inflammatory", "forehead", 2, 0.9,
                evidence=ConcernEvidence({"papule_pustule": 6}, 0.9, 1)),
    ])
    decision = decide_care(report, policy)
    assert (decision.triage_level, decision.therapy_disposition) == (
        "abstain", "supportive_only"
    )
    assert decision.referral_reasons == ["unvalidated_nodule_evidence"]


# --- TriagePolicy validation + escalation thresholds --------------------------
# Mutation testing showed these were unpinned: mutating the defaults or flipping
# the validation comparisons killed nothing.

def _approved(**overrides):
    values = {"policy_id": "p", "version": "1", "approved": True,
              "calibrator_id": "cal-1", "calibrator": lambda c: c,
              "nodule_gate": 0.6, "abstain_lower": 0.3}
    values.update(overrides)
    return TriagePolicy(**values)


@pytest.mark.parametrize("field", ["nodule_gate", "abstain_lower"])
@pytest.mark.parametrize("value", [-0.001, 1.001, 2.0, -1.0])
def test_gate_probabilities_must_be_a_probability(field, value):
    """A gate outside 0..1 can never fire (or always fires); it is a config
    error, not a policy."""
    with pytest.raises(ValueError, match=f"{field}: expected 0..1 or null"):
        _approved(**{field: value})


@pytest.mark.parametrize("field", ["nodule_gate", "abstain_lower"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_the_probability_bounds_are_inclusive(field, value):
    policy = _approved(nodule_gate=1.0, abstain_lower=0.0)
    assert getattr(policy, field) is not None
    assert _approved(**{field: value, "nodule_gate": 1.0, "abstain_lower": 0.0})


def test_an_approved_nodule_gate_requires_a_named_calibrator():
    """An approved gate with no calibrator would compare a raw detector score
    against a calibrated threshold."""
    with pytest.raises(ValueError, match="requires a named calibrator"):
        _approved(calibrator=None)
    with pytest.raises(ValueError, match="requires a named calibrator"):
        _approved(calibrator_id=None)
    with pytest.raises(ValueError, match="requires a named calibrator"):
        _approved(calibrator_id="")


def test_an_unapproved_policy_may_carry_a_gate_without_a_calibrator():
    policy = TriagePolicy("p", "1", approved=False, nodule_gate=0.6)
    assert policy.calibrator is None


def test_abstain_band_must_sit_below_the_referral_gate():
    """abstain_lower above nodule_gate would make the abstain band unreachable:
    anything that clears it already cleared the referral gate."""
    with pytest.raises(ValueError, match="abstain_lower must not exceed nodule_gate"):
        _approved(nodule_gate=0.3, abstain_lower=0.6)


def test_an_abstain_band_equal_to_the_gate_is_allowed():
    assert _approved(nodule_gate=0.6, abstain_lower=0.6).abstain_lower == 0.6


def test_policy_identifier_joins_id_and_version():
    assert TriagePolicy("skinscan", "3-experimental", approved=False).identifier == \
        "skinscan:3-experimental"


def test_repository_default_policy_is_unreviewed_and_threshold_free():
    policy = conservative_unreviewed_policy()
    assert policy.approved is False
    assert policy.nodule_gate is None and policy.abstain_lower is None
    assert policy.calibrator is None and policy.calibrator_id is None


@pytest.mark.parametrize("severity, referred", [(2, False), (3, False), (4, True)])
def test_overall_severity_escalates_to_review_at_the_policy_minimum(severity, referred):
    """Default severity_review_min is 4; 3 must NOT escalate."""
    report = ConcernReport("s", concerns=[
        Concern("acne_inflammatory", "forehead", severity, 0.9,
                evidence=ConcernEvidence({"papule_pustule": 30}, 0.9, 1)),
    ])
    decision = decide_care(report, conservative_unreviewed_policy())
    assert ("high_count_or_severity_review" in decision.referral_reasons) is referred


@pytest.mark.parametrize("concern, reason, minimum", [
    ("acne_scarring", "scarring_risk", 3),
    ("hyperpigmentation", "persistent_pigment_concern", 3),
])
def test_scarring_and_pigment_escalate_at_their_policy_minimum(concern, reason, minimum):
    def refers(severity):
        report = ConcernReport("s", concerns=[
            Concern(concern, "left_cheek", severity, 0.9,
                    evidence=ConcernEvidence({}, 0.9, 1)),
        ])
        return reason in decide_care(report, conservative_unreviewed_policy()).referral_reasons

    assert not refers(minimum - 1)
    assert refers(minimum)
    assert refers(minimum + 1)


def test_a_none_minimum_escalates_on_any_severity():
    """None means "no threshold": every occurrence is referred."""
    report = ConcernReport("s", concerns=[
        Concern("acne_scarring", "left_cheek", 1, 0.9, evidence=ConcernEvidence({}, 0.9, 1)),
    ])
    policy = TriagePolicy("p", "1", approved=False, scarring_review_min=None)
    assert "scarring_risk" in decide_care(report, policy).referral_reasons


def test_a_none_severity_minimum_never_escalates_on_overall_severity():
    report = ConcernReport("s", concerns=[
        Concern("acne_inflammatory", "forehead", 4, 0.9,
                evidence=ConcernEvidence({"papule_pustule": 40}, 0.9, 1)),
    ])
    policy = TriagePolicy("p", "1", approved=False, severity_review_min=None)
    assert "high_count_or_severity_review" not in decide_care(report, policy).referral_reasons


# --- evidence quality bands + provenance --------------------------------------
# Mutation testing showed the band boundaries and the probability clamp were
# unpinned: flipping >= to > or dropping the clamp killed nothing.

def _concern(confidence=0.9, max_confidence=None, severity=2, lesion_count=None,
             labels=None, source="raw_detector_signal", concern="acne_inflammatory"):
    return Concern(
        concern, "forehead", severity, confidence, lesion_count=lesion_count,
        evidence=ConcernEvidence(labels if labels is not None else {},
                                 max_confidence if max_confidence is not None else confidence,
                                 1, source=source),
    )


def _only(concern, policy=None):
    report = ConcernReport("q", concerns=[concern])
    return decide_care(report, policy or conservative_unreviewed_policy()).evidence[0]


@pytest.mark.parametrize("signal, quality", [
    (1.0, "high"), (0.8, "high"),        # inclusive lower bound
    (0.79, "medium"), (0.5, "medium"),   # inclusive lower bound
    (0.49, "low"), (0.01, "low"),
    (0.0, "unknown"),                    # no signal at all
])
def test_evidence_quality_bands_are_inclusive_at_each_boundary(signal, quality):
    assert _only(_concern(confidence=signal, max_confidence=signal)).quality == quality


def test_quality_prefers_the_detector_max_confidence_over_the_concern_confidence():
    assert _only(_concern(confidence=0.1, max_confidence=0.95)).quality == "high"


def test_quality_falls_back_to_concern_confidence_when_no_detector_signal():
    """max_confidence of 0 is falsy, so the concern's own confidence decides."""
    assert _only(_concern(confidence=0.9, max_confidence=0.0)).quality == "high"


def test_ordinary_evidence_reports_severity_and_never_a_probability():
    item = _only(_concern(severity=3))
    assert item.probability is None, "raw detector confidence is not a probability"
    assert item.calibrated is False
    assert "severity_3" in item.reasons


def test_ordinary_evidence_reports_lesion_count_only_when_counted():
    assert "lesion_count_12" in _only(_concern(lesion_count=12)).reasons
    assert not any(r.startswith("lesion_count_") for r in _only(_concern()).reasons)


def test_ordinary_evidence_flags_detector_labels_only_when_present():
    assert "detector_labels_present" in _only(_concern(labels={"papule_pustule": 3})).reasons
    assert "detector_labels_present" not in _only(_concern(labels={})).reasons


def test_ordinary_evidence_carries_the_concern_source_through():
    assert _only(_concern(source="annotation_oracle")).source == "annotation_oracle"


# --- nodule evidence: clamp, source, and gate boundaries ----------------------

def test_a_nodule_label_routes_through_the_nodule_gate_even_without_acne_cystic():
    item = _only(_concern(concern="acne_inflammatory", labels={"nodule": 1}))
    assert item.concern == "acne_nodular"
    assert item.reasons == ["safety_critical_nodule_signal"]


@pytest.mark.parametrize("raw, expected", [(1.4, 1.0), (-0.3, 0.0), (0.42, 0.42)])
def test_a_calibrator_returning_out_of_range_is_clamped_to_a_probability(raw, expected):
    """An uncontrolled calibrator must not be able to push a probability outside
    0..1 and skew the gate comparison."""
    policy = _approved(calibrator=lambda confidence: raw, nodule_gate=1.0, abstain_lower=0.0)
    item = _only(_concern(concern="acne_cystic"), policy)
    assert item.probability == expected


def test_uncalibrated_nodule_source_names_the_oracle_or_the_raw_detector():
    for source, expected in (("annotation_oracle", "annotation_oracle"),
                             ("raw_detector_signal", "raw_detector_signal")):
        item = _only(_concern(concern="acne_cystic", source=source))
        assert item.source == expected


def test_a_calibrated_nodule_names_the_calibrator_as_its_source():
    item = _only(_concern(concern="acne_cystic"), _approved())
    assert item.source == "cal-1"


@pytest.mark.parametrize("probability, triage", [
    (0.6, "derm_first"),   # inclusive at nodule_gate
    (0.59, "abstain"),     # falls to the abstain band
    (0.3, "abstain"),      # inclusive at abstain_lower
    (0.29, "routine"),     # below both gates
])
def test_the_nodule_gates_are_inclusive_at_their_thresholds(probability, triage):
    policy = _approved(nodule_gate=0.6, abstain_lower=0.3,
                       calibrator=lambda confidence: probability)
    report = ConcernReport("g", concerns=[_concern(concern="acne_cystic")])
    assert decide_care(report, policy).triage_level == triage


def test_a_zero_severity_concern_is_not_active_acne():
    """severity 0 means the concern was looked for and not found."""
    report = ConcernReport("z", concerns=[_concern(concern="acne_comedonal", severity=0)])
    assert decide_care(report, conservative_unreviewed_policy()).therapy_disposition == \
        "maintenance"
