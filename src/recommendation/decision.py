"""Concern evidence to independent triage and therapy disposition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .schema import CareDecision, Concern, ConcernReport, DecisionEvidence


@dataclass(frozen=True)
class TriagePolicy:
    policy_id: str
    version: str
    approved: bool
    calibrator_id: str | None = None
    calibrator: Callable[[float], float] | None = None
    nodule_gate: float | None = None
    abstain_lower: float | None = None
    severity_review_min: int | None = 4
    scarring_review_min: int | None = 3
    pigment_review_min: int | None = 3

    def __post_init__(self) -> None:
        for name in ("nodule_gate", "abstain_lower"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}: expected 0..1 or null")
        if self.approved and self.nodule_gate is not None:
            if self.calibrator is None or not self.calibrator_id:
                raise ValueError("approved nodule gate requires a named calibrator")
        if (self.abstain_lower is not None and self.nodule_gate is not None
                and self.abstain_lower > self.nodule_gate):
            raise ValueError("abstain_lower must not exceed nodule_gate")

    @property
    def identifier(self) -> str:
        return f"{self.policy_id}:{self.version}"


def conservative_unreviewed_policy() -> TriagePolicy:
    """Repository default: deliberately unreviewed and threshold-free."""
    return TriagePolicy(
        policy_id="skinscan-conservative-unreviewed",
        version="3-experimental",
        approved=False,
    )


def _quality(concern: Concern) -> str:
    signal = concern.evidence.max_confidence or concern.confidence
    if signal >= 0.8:
        return "high"
    if signal >= 0.5:
        return "medium"
    if signal > 0:
        return "low"
    return "unknown"


def _ordinary_evidence(concern: Concern) -> DecisionEvidence:
    reasons = [f"severity_{concern.severity}"]
    if concern.lesion_count is not None:
        reasons.append(f"lesion_count_{concern.lesion_count}")
    if concern.evidence.labels:
        reasons.append("detector_labels_present")
    # Raw detector confidence is intentionally not copied into probability.
    return DecisionEvidence(
        concern=concern.concern,
        probability=None,
        quality=_quality(concern),
        source=concern.evidence.source,
        calibrated=False,
        reasons=reasons,
    )


def _nodule_evidence(concern: Concern, policy: TriagePolicy) -> DecisionEvidence:
    calibrated = bool(policy.approved and policy.calibrator and policy.calibrator_id)
    probability = policy.calibrator(concern.confidence) if calibrated else None
    if probability is not None:
        probability = min(1.0, max(0.0, float(probability)))
    return DecisionEvidence(
        concern="acne_nodular",
        probability=probability,
        quality=_quality(concern),
        source=(policy.calibrator_id if calibrated else (
            "annotation_oracle" if concern.evidence.source == "annotation_oracle"
            else "raw_detector_signal"
        )),
        calibrated=calibrated,
        reasons=["safety_critical_nodule_signal"],
    )


def decide_care(report: ConcernReport, policy: TriagePolicy) -> CareDecision:
    """Apply a reviewed gate when available and abstain otherwise.

    Referral and therapy are separate outputs: count/scarring/pigment review
    language does not erase an otherwise eligible active-treatment intent.
    """
    evidence: list[DecisionEvidence] = []
    nodule_evidence: list[DecisionEvidence] = []
    for concern in report.concerns:
        if concern.concern == "acne_cystic" or concern.evidence.labels.get("nodule", 0):
            item = _nodule_evidence(concern, policy)
            nodule_evidence.append(item)
        else:
            item = _ordinary_evidence(concern)
        evidence.append(item)

    if nodule_evidence:
        reviewed_gate = policy.approved and policy.nodule_gate is not None
        # A calibrator that produced no probability has calibrated nothing: the
        # gates below would silently skip it and fall through to treatment.
        if (not all(item.calibrated and item.probability is not None
                    for item in nodule_evidence) or not reviewed_gate):
            return CareDecision(
                "abstain", ["unvalidated_nodule_evidence"], "supportive_only",
                evidence, policy.identifier, policy.approved,
            )
        probabilities = [
            item.probability for item in nodule_evidence if item.probability is not None
        ]
        if any(probability >= policy.nodule_gate for probability in probabilities):
            return CareDecision(
                "derm_first", ["suspected_nodule"], "supportive_only",
                evidence, policy.identifier, True,
            )
        if (policy.abstain_lower is not None
                and any(probability >= policy.abstain_lower for probability in probabilities)):
            return CareDecision(
                "abstain", ["uncertain_nodule_evidence"], "supportive_only",
                evidence, policy.identifier, True,
            )

    if report.clear_skin or not report.concerns:
        return CareDecision(
            "routine", [], "maintenance", evidence, policy.identifier, policy.approved,
        )

    referrals: list[str] = []
    active_acne = any(
        concern.concern in {"acne_comedonal", "acne_inflammatory"}
        and concern.severity > 0
        for concern in report.concerns
    )
    if (policy.severity_review_min is not None
            and report.overall_severity >= policy.severity_review_min):
        referrals.append("high_count_or_severity_review")
    if any(
        concern.concern == "acne_scarring"
        and (policy.scarring_review_min is None
             or concern.severity >= policy.scarring_review_min)
        for concern in report.concerns
    ):
        referrals.append("scarring_risk")
    if any(
        concern.concern == "hyperpigmentation"
        and (policy.pigment_review_min is None
             or concern.severity >= policy.pigment_review_min)
        for concern in report.concerns
    ):
        referrals.append("persistent_pigment_concern")

    return CareDecision(
        "routine_plus_review" if referrals else "routine",
        referrals,
        "active_treatment" if active_acne else "maintenance",
        evidence,
        policy.identifier,
        policy.approved,
    )
