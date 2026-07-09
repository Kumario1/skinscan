"""Behavior tests for photo-side skin-tone triage (issue #6)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.tone import estimate_tone, sephora_tone_bucket, srgb_to_lab, tone_bucket


def test_srgb_to_lab_matches_reference_triples():
    rgb = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    lab = srgb_to_lab(rgb)

    assert np.allclose(lab[0], [0, 0, 0], atol=0.01)
    assert np.allclose(lab[1], [100, 0, 0], atol=0.02)
    assert np.allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)


def test_ita_bucket_cutoffs_are_inclusive():
    assert tone_bucket(41, light_min=41, medium_min=10) == "light"
    assert tone_bucket(40.99, light_min=41, medium_min=10) == "medium"
    assert tone_bucket(10, light_min=41, medium_min=10) == "medium"
    assert tone_bucket(9.99, light_min=41, medium_min=10) == "deep"
    assert tone_bucket(None, light_min=41, medium_min=10) == "unknown"


def test_too_few_skin_pixels_returns_unknown():
    image = np.full((10, 10, 3), [160, 120, 100], dtype=np.uint8)
    polygons = {"left_cheek": [(0, 0), (2, 0), (2, 2), (0, 2)]}

    result = estimate_tone(image, polygons, min_pixels=10)

    assert result.bucket == "unknown"
    assert result.ita is None
    assert result.sample_count < 10


def test_low_light_flag_uses_median_lab_lightness():
    image = np.full((20, 20, 3), [50, 35, 30], dtype=np.uint8)

    result = estimate_tone(image, min_pixels=10, low_light_l=35)

    assert result.sample_count >= 10
    assert result.median_l < 35
    assert result.low_light is True


def test_detector_boxes_are_excluded_with_crop_padding():
    image = np.full((10, 10, 3), [160, 120, 100], dtype=np.uint8)
    polygons = {"forehead": [(0, 0), (10, 0), (10, 10), (0, 10)]}

    result = estimate_tone(
        image,
        polygons,
        lesion_boxes=[(4, 4, 6, 6)],
        crop_pad=2,
        min_pixels=1,
    )

    assert result.sample_count == 84


def test_sephora_review_tones_map_to_three_buckets_and_keep_unknowns():
    assert sephora_tone_bucket("porcelain") == "light"
    assert sephora_tone_bucket("fair") == "light"
    assert sephora_tone_bucket("fairLight") == "light"
    assert sephora_tone_bucket("lightMedium") == "medium"
    assert sephora_tone_bucket("mediumTan") == "medium"
    assert sephora_tone_bucket("olive") == "medium"
    assert sephora_tone_bucket("ebony") == "deep"
    assert sephora_tone_bucket("rich") == "deep"
    assert sephora_tone_bucket("notSure") == "unknown"
    assert sephora_tone_bucket("unexpected future value") == "unknown"
    assert sephora_tone_bucket(None) == "unknown"


def test_profile_sampling_ignores_the_collapsed_far_cheek():
    image = np.full((20, 20, 3), [160, 120, 100], dtype=np.uint8)
    polygons = {
        "right_cheek": [(0, 0), (10, 0), (10, 10), (0, 10)],
        "left_cheek": [(15, 0), (17, 0), (17, 2), (15, 2)],
    }

    result = estimate_tone(image, polygons, min_pixels=1)

    assert result.sample_count == 100


def test_non_skin_colored_pixels_inside_face_polygons_are_excluded():
    image = np.full((10, 10, 3), [160, 120, 100], dtype=np.uint8)
    image[:2, :2] = [100, 150, 200]  # blue-gray eye/background-like pixels
    polygons = {"forehead": [(0, 0), (10, 0), (10, 10), (0, 10)]}

    result = estimate_tone(image, polygons, min_pixels=1)

    assert result.sample_count == 96


if __name__ == "__main__":
    test_srgb_to_lab_matches_reference_triples()
    test_ita_bucket_cutoffs_are_inclusive()
    test_too_few_skin_pixels_returns_unknown()
    test_low_light_flag_uses_median_lab_lightness()
    test_detector_boxes_are_excluded_with_crop_padding()
    test_sephora_review_tones_map_to_three_buckets_and_keep_unknowns()
    test_profile_sampling_ignores_the_collapsed_far_cheek()
    test_non_skin_colored_pixels_inside_face_polygons_are_excluded()
    print("ok")
