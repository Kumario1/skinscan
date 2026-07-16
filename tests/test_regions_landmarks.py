"""Landmark-path region location (src/pipeline/regions.py).

tests/test_regions.py covers the grid fallback and the no-model path. This
covers the landmark path itself: the lip hull, the six D-020 polygons, box
loading, and a real FaceLandmarker run against a committed AcneSCU photo.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.regions import (
    _convex_hull, grid_polygons, landmark_polygons, load_boxes, locate_regions,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "face_landmarker.task"
REAL_FACES = sorted((ROOT / "AcneSCU.v1-acnescu-original.voc" / "train").glob("*.jpg"))
D020_REGIONS = ["forehead", "nose", "right_cheek", "left_cheek", "perioral", "chin_jaw"]


# --- lip hull -----------------------------------------------------------------

def test_convex_hull_of_a_square_keeps_only_its_corners():
    points = np.array([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)])  # +1 interior point
    hull = _convex_hull(points)
    assert {tuple(p) for p in hull} == {(0, 0), (2, 0), (2, 2), (0, 2)}


def test_convex_hull_drops_points_on_an_edge():
    """Collinear points add no shape; keeping them would distort the perioral
    scaling, which multiplies the outline about its centroid."""
    points = np.array([(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)])
    hull = _convex_hull(points)
    assert (1, 0) not in {tuple(p) for p in hull}
    assert len(hull) == 4


def test_convex_hull_is_order_independent():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    forward = _convex_hull(np.array(square))
    reversed_ = _convex_hull(np.array(square[::-1]))
    assert {tuple(p) for p in forward} == {tuple(p) for p in reversed_}


# --- the six D-020 polygons ---------------------------------------------------

def _landmarks(count=468):
    """A deterministic stand-in face: a spiral guarantees distinct points."""
    angles = np.linspace(0, 8 * np.pi, count)
    radii = np.linspace(10, 500, count)
    return np.stack((500 + radii * np.cos(angles), 500 + radii * np.sin(angles)), axis=-1)


def test_landmark_polygons_returns_the_six_d020_regions_in_priority_order():
    polygons = landmark_polygons(_landmarks())
    assert list(polygons) == D020_REGIONS, "insertion order resolves polygon overlaps"


@pytest.mark.parametrize("count", [0, 1, 467])
def test_landmark_polygons_refuses_a_short_landmark_set(count):
    """Fewer than 468 points is not a face mesh; indexing it would silently
    build polygons out of whatever happened to be at those indices."""
    with pytest.raises(ValueError, match="at least 468"):
        landmark_polygons(_landmarks(count))


def test_perioral_scale_expands_the_lip_outline_about_its_centre():
    landmarks = _landmarks()
    small = landmark_polygons(landmarks, perioral_scale=1.0)["perioral"]
    large = landmark_polygons(landmarks, perioral_scale=2.0)["perioral"]

    assert np.allclose(small.mean(axis=0), large.mean(axis=0)), "centroid is fixed"
    def spread(polygon):
        return np.abs(polygon - polygon.mean(axis=0)).sum()
    assert spread(large) > spread(small)
    assert np.allclose(spread(large), 2 * spread(small))


# --- load_boxes ---------------------------------------------------------------

def test_load_boxes_of_nothing_is_empty():
    assert load_boxes(None) == []


@pytest.mark.parametrize("payload, expected", [
    ([[1, 2, 3, 4]], [(1, 2, 3, 4)]),
    ({"boxes": [[1, 2, 3, 4]]}, [(1, 2, 3, 4)]),
    ({"detections": [{"box": [1, 2, 3, 4]}, {"box": [5, 6, 7, 8]}]},
     [(1, 2, 3, 4), (5, 6, 7, 8)]),
    ([{"detections": [{"box": [1, 2, 3, 4]}]}], [(1, 2, 3, 4)]),  # predictions.json shape
    ({}, []),
], ids=["plain_list", "boxes_key", "detections", "wrapped_predictions", "empty_object"])
def test_load_boxes_reads_every_shipped_shape(tmp_path, payload, expected):
    path = tmp_path / "boxes.json"
    path.write_text(json.dumps(payload))
    assert load_boxes(path) == expected


# --- a real FaceLandmarker run ------------------------------------------------

@pytest.mark.real_models
@pytest.mark.skipif(not (MODEL.exists() and REAL_FACES),
                    reason="FaceLandmarker artifact or AcneSCU photos absent")
def test_a_real_photo_uses_landmarks_rather_than_the_grid():
    from src.classification.run_acne04_pipeline import load_rgb

    rgb = load_rgb(REAL_FACES[0])
    height, width = rgb.shape[:2]
    centre = [(width // 2 - 30, height // 2 - 30, width // 2 + 30, height // 2 + 30)]

    result = locate_regions(rgb, centre, model_path=MODEL)

    assert result.metadata["fallback"] is False
    assert result.metadata["method"] == "mediapipe_face_landmarker"
    assert result.metadata["face_detected"] is True
    assert list(result.polygons) == D020_REGIONS
    assert result.regions and result.regions[0] in D020_REGIONS

    # the reported face box must be inside the image and non-degenerate
    x0, y0, x1, y1 = result.metadata["face_box"]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


@pytest.mark.real_models
@pytest.mark.skipif(not (MODEL.exists() and REAL_FACES),
                    reason="FaceLandmarker artifact or AcneSCU photos absent")
def test_a_manual_face_box_forces_the_grid_even_when_a_face_is_detectable():
    """The manual override exists for profiles the detector misses; it must win
    over the landmarker rather than be quietly ignored."""
    from src.classification.run_acne04_pipeline import load_rgb

    rgb = load_rgb(REAL_FACES[0])
    height, width = rgb.shape[:2]
    face_box = (0, 0, width, height)

    result = locate_regions(rgb, [(10, 10, 20, 20)], face_box=face_box, model_path=MODEL)

    assert result.metadata["fallback"] is True
    assert "manual face box" in result.metadata["reason"]
    assert result.polygons == grid_polygons(rgb.shape, face_box)
