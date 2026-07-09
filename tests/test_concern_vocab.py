from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.classifier import (
    RAW_ACNE_CLASSES,
    RAW_TO_CONCERN,
    StubClassifier,
    concern_probs,
)
from src.recommendation.schema import CONCERNS


def test_mapping_targets_are_schema_concerns():
    assert set(RAW_TO_CONCERN.values()) <= CONCERNS


def test_mapping_is_proper_subset_of_model_classes():
    # Not_acne is intentionally in the class set but unmapped, so its mass drops
    # out of concern aggregation (STAGE2_NEGATIVES_DESIGN.md).
    assert set(RAW_TO_CONCERN) < set(RAW_ACNE_CLASSES)


def test_concern_probs_aggregates_and_preserves_mass():
    raw = {"Blackheads": 0.2, "Whiteheads": 0.1, "Cyst": 0.3, "Papules": 0.25, "Pustules": 0.15}
    out = concern_probs(raw)
    expected = {"acne_comedonal": 0.3, "acne_cystic": 0.3, "acne_inflammatory": 0.4}
    assert set(out) == set(expected)
    for k in expected:
        assert abs(out[k] - expected[k]) < 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_concern_probs_ignores_unknown_keys():
    raw = {"Blackheads": 0.5, "not_acne": 0.5}
    out = concern_probs(raw)
    assert "not_acne" not in out
    assert abs(out["acne_comedonal"] - 0.5) < 1e-9


def test_stub_classifier_round_trip():
    out = concern_probs(StubClassifier().predict(None))
    assert set(out) == {"acne_comedonal", "acne_cystic", "acne_inflammatory"}


if __name__ == "__main__":
    test_mapping_targets_are_schema_concerns()
    test_mapping_is_proper_subset_of_model_classes()
    test_concern_probs_aggregates_and_preserves_mass()
    test_concern_probs_ignores_unknown_keys()
    test_stub_classifier_round_trip()
    print("ok")
