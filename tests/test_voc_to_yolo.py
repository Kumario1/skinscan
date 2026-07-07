"""Unit tests for the ACNE04 -> YOLO geometry (src/detection/voc_to_yolo.py).

The module docstring claims the geometry "is unit-tested" — this file makes
that true. Covers the center-vs-corner conversion, out-of-bounds clamping,
degenerate/invalid-size errors, label formatting, and a round-trip against the
inverse transform in visualize_labels. Standalone via __main__ (no pytest
needed) but named test_* so `pytest tests/` works too.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.voc_to_yolo import Box, voc_box_to_yolo, yolo_line
from src.detection.visualize_labels import yolo_to_corners


def assert_raises(exc, fn, *args):
    raised = False
    try:
        fn(*args)
    except exc:
        raised = True
    assert raised, f"expected {exc.__name__} from {fn.__name__}{args}"


def test_center_conversion():
    # Box(10,20,30,60) in a 100x200 image -> center (0.2,0.2), size (0.2,0.2)
    result = voc_box_to_yolo(Box(10, 20, 30, 60), 100, 200)
    assert result == (0.2, 0.2, 0.2, 0.2), result


def test_clamps_overspill():
    # negative corners clamp to 0 before conversion
    result = voc_box_to_yolo(Box(-10, -10, 50, 50), 100, 100)
    assert result == (0.25, 0.25, 0.5, 0.5), result


def test_degenerate_raises():
    # fully outside -> collapses to zero-area after clamp
    assert_raises(ValueError, voc_box_to_yolo, Box(150, 10, 190, 20), 100, 100)
    # bad image size
    assert_raises(ValueError, voc_box_to_yolo, Box(1, 1, 2, 2), 0, 100)


def test_yolo_line_format():
    line = yolo_line(0, (0.5, 0.5, 0.1, 0.1))
    assert line == "0 0.500000 0.500000 0.100000 0.100000", line


def test_round_trip_with_visualizer():
    # convert forward, then invert with the visualizer; corners must match
    xc, yc, w, h = voc_box_to_yolo(Box(40, 30, 80, 90), 200, 150)
    corners = yolo_to_corners(xc, yc, w, h, 200, 150)
    original = (40, 30, 80, 90)
    for got, want in zip(corners, original):
        assert abs(got - want) <= 1, (corners, original)


if __name__ == "__main__":
    test_center_conversion()
    test_clamps_overspill()
    test_degenerate_raises()
    test_yolo_line_format()
    test_round_trip_with_visualizer()
    print("ok")
