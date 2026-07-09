"""Tests for the Stage 2 -> Stage 3 bridge (src/recommendation/bridge.py).

Pure-function tests with dict literals only — no model imports. Standalone via
__main__ (pytest not required) but named test_* so `pytest tests/` also works.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendation.bridge import build_concern_report, severity_from_count

THRESHOLDS = [1, 5, 10, 20]


def _inflammatory(n):
    """n identical papule detections."""
    return [{"Papules": 0.9, "Pustules": 0.1}] * n


def test_severity_from_count_boundaries():
    cases = {0: 0, 1: 1, 4: 1, 5: 2, 9: 2, 10: 3, 19: 3, 20: 4, 25: 4}
    for count, sev in cases.items():
        assert severity_from_count(count, THRESHOLDS) == sev, count


def test_severity_from_lesion_count_end_to_end():
    for count, sev in {1: 1, 4: 1, 5: 2, 19: 3, 20: 4}.items():
        report = build_concern_report("img", _inflammatory(count),
                                      ["forehead"] * count, thresholds=THRESHOLDS)
        assert len(report.concerns) == 1
        assert report.concerns[0].lesion_count == count
        assert report.concerns[0].severity == sev, (count, report.concerns[0].severity)


def test_single_detection_becomes_one_concern():
    report = build_concern_report(
        "img",
        [{"Papules": 0.6, "Pustules": 0.2, "Blackheads": 0.1, "Whiteheads": 0.05, "Cyst": 0.05}],
        ["left_cheek"],
        thresholds=THRESHOLDS,
    )
    assert not report.clear_skin
    assert len(report.concerns) == 1
    c = report.concerns[0]
    assert c.concern == "acne_inflammatory"
    assert c.region == "left_cheek"
    assert c.lesion_count == 1
    assert c.severity == 1
    # confidence is the aggregated concern mass: Papules + Pustules
    assert abs(c.confidence - 0.8) < 1e-9


def test_not_acne_top_class_is_dropped():
    report = build_concern_report(
        "img",
        [
            {"Papules": 0.7, "Pustules": 0.1, "Blackheads": 0.1, "Whiteheads": 0.05, "Cyst": 0.05},
            {"Not_acne": 0.8, "Papules": 0.1, "Pustules": 0.05, "Blackheads": 0.03, "Whiteheads": 0.01, "Cyst": 0.01},
        ],
        ["nose", "nose"],
        thresholds=THRESHOLDS,
    )
    # only the first detection survives
    assert len(report.concerns) == 1
    assert report.concerns[0].concern == "acne_inflammatory"
    assert report.concerns[0].lesion_count == 1


def test_same_concern_two_regions_two_entries():
    report = build_concern_report(
        "img",
        _inflammatory(3) + _inflammatory(2),
        ["left_cheek", "left_cheek", "left_cheek", "right_cheek", "right_cheek"],
        thresholds=THRESHOLDS,
    )
    by_region = {c.region: c for c in report.concerns}
    assert set(by_region) == {"left_cheek", "right_cheek"}
    assert all(c.concern == "acne_inflammatory" for c in report.concerns)
    assert by_region["left_cheek"].lesion_count == 3
    assert by_region["right_cheek"].lesion_count == 2


def test_all_dropped_is_clear_skin_with_note():
    report = build_concern_report(
        "img",
        [
            {"Not_acne": 0.9, "Papules": 0.1},
            {"Not_acne": 0.7, "Blackheads": 0.3},
        ],
        ["forehead", "nose"],
        thresholds=THRESHOLDS,
    )
    assert report.clear_skin
    assert report.concerns == []
    assert "2" in report.notes and "Not_acne" in report.notes, report.notes


def test_no_detections_is_clear_skin_no_note():
    report = build_concern_report("img", [], [], thresholds=THRESHOLDS)
    assert report.clear_skin
    assert report.notes == ""


def test_low_light_flag_passes_through():
    assert build_concern_report("img", [], [], thresholds=THRESHOLDS,
                                low_light_flag=True).low_light_flag is True
    assert build_concern_report("img", [], [], thresholds=THRESHOLDS).low_light_flag is False


def test_confidence_is_mean_of_member_concern_mass():
    report = build_concern_report(
        "img",
        [{"Papules": 0.6, "Pustules": 0.0}, {"Papules": 0.4, "Pustules": 0.4}],
        ["chin_jaw", "chin_jaw"],
        thresholds=THRESHOLDS,
    )
    # per-detection concern mass: 0.6 and 0.8 -> mean 0.7
    assert len(report.concerns) == 1
    assert abs(report.concerns[0].confidence - 0.7) < 1e-9


def test_import_pulls_in_no_heavy_ml():
    """Acceptance: importing the bridge must not drag in TF/YOLO/mediapipe."""
    import subprocess

    code = (
        "import sys; import src.recommendation.bridge; "
        "heavy = [m for m in ('tensorflow', 'ultralytics', 'torch', 'mediapipe', 'cv2') "
        "if m in sys.modules]; "
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    test_single_detection_becomes_one_concern()
    test_severity_from_count_boundaries()
    test_severity_from_lesion_count_end_to_end()
    test_same_concern_two_regions_two_entries()
    test_low_light_flag_passes_through()
    test_confidence_is_mean_of_member_concern_mass()
    test_import_pulls_in_no_heavy_ml()
    test_not_acne_top_class_is_dropped()
    test_all_dropped_is_clear_skin_with_note()
    test_no_detections_is_clear_skin_no_note()
    print("ok")
