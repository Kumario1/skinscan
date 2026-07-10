"""Behavior tests for face-region assignment (issue #6)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.regions import assign_regions, fallback_regions, grid_polygons, locate_regions


def test_box_centroids_are_assigned_to_hand_built_polygons():
    polygons = {
        "left_cheek": [(0, 0), (40, 0), (40, 40), (0, 40)],
        "right_cheek": [(60, 0), (100, 0), (100, 40), (60, 40)],
    }

    assert assign_regions([(10, 10, 20, 20), (70, 10, 90, 30)], polygons) == (
        ["left_cheek", "right_cheek"],
        [],
    )


def test_centroid_outside_all_polygons_uses_nearest_polygon_and_says_so():
    polygons = {
        "nose": [(40, 20), (60, 20), (60, 40), (40, 40)],
        "chin_jaw": [(20, 70), (80, 70), (80, 90), (20, 90)],
    }

    assert assign_regions([(48, 48, 52, 52)], polygons) == (["nose"], [0])


def test_image_thirds_grid_fallback_is_deterministic():
    polygons = grid_polygons((90, 90, 3))
    boxes = [
        (40, 10, 50, 20),  # top
        (40, 40, 50, 50),  # middle center
        (10, 40, 20, 50),  # middle left
        (70, 40, 80, 50),  # middle right
        (40, 70, 50, 80),  # bottom center
        (10, 70, 20, 80),  # bottom side
    ]

    assert assign_regions(boxes, polygons) == (
        ["forehead", "nose", "right_cheek", "left_cheek", "perioral", "chin_jaw"],
        [],
    )


def test_fallback_result_loudly_reports_why_it_was_used():
    result = fallback_regions(
        (90, 90, 3),
        [(40, 40, 50, 50)],
        face_box=(0, 0, 90, 90),
        reason="MediaPipe FaceMesh unavailable",
    )

    assert result.regions == ["nose"]
    assert result.metadata == {
        "method": "grid_fallback",
        "fallback": True,
        "reason": "MediaPipe FaceMesh unavailable",
        "face_detected": True,
        "face_box": [0, 0, 90, 90],
        "forced_assignments": [],
    }


def test_import_does_not_load_mediapipe_or_opencv():
    import subprocess

    code = (
        "import sys; import src.pipeline.regions; "
        "heavy = [m for m in ('mediapipe', 'cv2') if m in sys.modules]; "
        "print(heavy); raise SystemExit(bool(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_landmarker_artifact_falls_back_without_hard_failure():
    image = np.zeros((90, 90, 3), dtype=np.uint8)

    result = locate_regions(
        image,
        [(40, 40, 50, 50)],
        model_path=Path("models/definitely-missing-face-landmarker.task"),
    )

    assert result.regions == ["nose"]
    assert result.metadata["fallback"] is True
    assert "missing" in result.metadata["reason"].lower()


if __name__ == "__main__":
    test_box_centroids_are_assigned_to_hand_built_polygons()
    test_centroid_outside_all_polygons_uses_nearest_polygon_and_says_so()
    test_image_thirds_grid_fallback_is_deterministic()
    test_fallback_result_loudly_reports_why_it_was_used()
    test_import_does_not_load_mediapipe_or_opencv()
    test_missing_landmarker_artifact_falls_back_without_hard_failure()
    print("ok")
