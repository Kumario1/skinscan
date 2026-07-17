"""Stage-2 crop classifier helpers (src/classification/classifier.py).

Covers the parts that decide what the pipeline reports: the D-008 raw-class ->
concern aggregation (where Not_acne's softmax mass must DROP OUT rather than be
redistributed), the context crop, and label metadata.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.classifier import (
    RAW_ACNE_CLASSES, RAW_TO_CONCERN, concern_probs, crop_with_context,
    read_model_metadata,
)


# --- D-008 concern aggregation ------------------------------------------------

def test_concern_probs_always_reports_every_concern_key():
    out = concern_probs({})
    assert out == {"acne_comedonal": 0.0, "acne_cystic": 0.0, "acne_inflammatory": 0.0}


def test_comedonal_sums_blackheads_and_whiteheads():
    out = concern_probs({"Blackheads": 0.3, "Whiteheads": 0.2})
    assert out["acne_comedonal"] == pytest.approx(0.5)


def test_inflammatory_sums_papules_and_pustules():
    out = concern_probs({"Papules": 0.25, "Pustules": 0.15})
    assert out["acne_inflammatory"] == pytest.approx(0.4)


def test_not_acne_mass_drops_out_and_is_never_redistributed():
    """STAGE2_NEGATIVES_DESIGN.md: Not_acne is deliberately unmapped, so its
    softmax mass leaves the aggregate. Redistributing it would manufacture
    confidence the model never expressed."""
    probs = dict(zip(RAW_ACNE_CLASSES, [0.05, 0.05, 0.80, 0.04, 0.03, 0.03]))
    out = concern_probs(probs)

    assert sum(out.values()) == pytest.approx(0.20)
    assert "Not_acne" not in out
    assert all(value <= 0.2 for value in out.values())


def test_a_confident_not_acne_crop_reports_near_zero_for_every_concern():
    out = concern_probs({"Not_acne": 1.0})
    assert out == {"acne_comedonal": 0.0, "acne_cystic": 0.0, "acne_inflammatory": 0.0}


def test_unknown_raw_classes_are_ignored_not_crashed_on():
    out = concern_probs({"Rosacea": 0.9, "Cyst": 0.1})
    assert out["acne_cystic"] == pytest.approx(0.1)
    assert sum(out.values()) == pytest.approx(0.1)


def test_every_mapped_class_is_a_real_model_class():
    """A typo in RAW_TO_CONCERN would silently drop a whole lesion type."""
    assert set(RAW_TO_CONCERN) <= set(RAW_ACNE_CLASSES)
    assert set(RAW_ACNE_CLASSES) - set(RAW_TO_CONCERN) == {"Not_acne"}


# --- context crop -------------------------------------------------------------

def test_crop_is_square_and_centred_on_the_box():
    image = np.zeros((400, 400, 3), np.uint8)
    image[190:210, 190:210] = 255            # a bright 20x20 lesion at the centre
    crop = crop_with_context(image, (190, 190, 20, 20), pad=1.5, size=224)

    assert crop.shape == (224, 224, 3)
    assert crop[112, 112, 0] == 255, "the lesion sits at the crop centre"
    assert crop[0, 0, 0] == 0, "context padding is included around it"


def test_pad_controls_how_much_context_is_included():
    image = np.zeros((400, 400, 3), np.uint8)
    image[190:210, 190:210] = 255
    tight = crop_with_context(image, (190, 190, 20, 20), pad=1.0, size=224)
    loose = crop_with_context(image, (190, 190, 20, 20), pad=4.0, size=224)

    assert tight.mean() > loose.mean(), "more context = proportionally less lesion"


def test_a_box_at_the_edge_is_padded_by_edge_replication_not_black():
    """A corner lesion must not be surrounded by invented black pixels, which
    the classifier would read as real dark context."""
    image = np.full((400, 400, 3), 200, np.uint8)
    crop = crop_with_context(image, (0, 0, 20, 20), pad=3.0, size=64)

    assert crop.shape == (64, 64, 3)
    assert (crop == 200).all(), "edge replication keeps the surrounding tone"


def test_a_box_beyond_the_image_still_produces_a_full_crop():
    image = np.full((100, 100, 3), 128, np.uint8)
    crop = crop_with_context(image, (95, 95, 20, 20), pad=2.0, size=32)
    assert crop.shape == (32, 32, 3)


def test_a_degenerate_box_still_yields_a_usable_crop():
    """side is floored at 2px, so a zero-size detector box cannot produce an
    empty array that would explode inside PIL."""
    image = np.full((100, 100, 3), 128, np.uint8)
    crop = crop_with_context(image, (50, 50, 0, 0), pad=1.5, size=16)
    assert crop.shape == (16, 16, 3)


# --- label metadata -----------------------------------------------------------

def test_metadata_is_read_from_the_sidecar_next_to_the_weights(tmp_path):
    weights = tmp_path / "acne_model.keras"
    weights.write_bytes(b"not a real model")
    (tmp_path / "acne_model.keras.labels.json").write_text(
        json.dumps({"classes": ["A", "B"], "image_size": 128}))

    assert read_model_metadata(weights) == {"classes": ["A", "B"], "image_size": 128}


def test_absent_metadata_is_empty_rather_than_an_error(tmp_path):
    weights = tmp_path / "acne_model.keras"
    weights.write_bytes(b"not a real model")
    assert read_model_metadata(weights) == {}
