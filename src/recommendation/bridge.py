"""Stage 2 -> Stage 3 bridge: per-lesion model output -> ConcernReport.

Pure function, no model imports — the numpy-only helpers from the classifier
(RAW_TO_CONCERN, concern_probs) are all it borrows from the CV side. This is the
join from the CV side to the rules side (CONTEXT.md "bridge", D-008).
"""
from __future__ import annotations
import bisect

from ..classification.classifier import RAW_TO_CONCERN, concern_probs
from ..config import load_config
from .schema import Concern, ConcernReport


def severity_from_count(count: int, thresholds: list[int]) -> int:
    """count -> ordinal severity 0-4 by upper-bound bisection of thresholds."""
    return bisect.bisect_right(thresholds, count)


def build_concern_report(image_id: str, lesion_probs: list[dict[str, float]],
                         regions: list[str], *, thresholds: list[int] | None = None,
                         low_light_flag: bool = False) -> ConcernReport:
    assert len(lesion_probs) == len(regions), "lesion_probs and regions must be parallel"
    if thresholds is None:
        thresholds = load_config()["concern_report"]["severity_count_thresholds"]

    groups: dict[tuple[str, str], list[float]] = {}  # (concern, region) -> confidences
    rejected = 0
    for probs, region in zip(lesion_probs, regions):
        top = max(probs, key=lambda k: probs[k])
        # A whole-detection false-positive rejection concern_probs can't make (it
        # only aggregates mass): if the top-1 class is Not_acne, the detector box
        # is not a lesion — drop it. Required by issue #4; forward-compatible with
        # the six-class retrain, inert against today's five-class model.
        if top == "Not_acne":
            rejected += 1
            continue
        concern = RAW_TO_CONCERN[top]
        groups.setdefault((concern, region), []).append(concern_probs(probs)[concern])

    concerns = [
        Concern(
            concern=concern,
            region=region,
            severity=severity_from_count(len(confs), thresholds),
            confidence=sum(confs) / len(confs),
            lesion_count=len(confs),
        )
        for (concern, region), confs in groups.items()
    ]
    return ConcernReport(
        image_id=image_id,
        concerns=concerns,
        clear_skin=not concerns,
        low_light_flag=low_light_flag,
        notes=f"dropped {rejected} detection(s) classified Not_acne" if rejected else "",
    )
