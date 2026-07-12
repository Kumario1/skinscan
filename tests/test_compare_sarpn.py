"""Geometry + mapping tests for the SA-RPN A/B harness (compare_sarpn.py).

No API, no model weights: detect functions are stubs, everything else is
coordinate math. Standalone via __main__ but named test_* for pytest.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.compare_sarpn import (
    compare, dedupe, report_from_detections, tile_origins, tile_pipeline,
    zoom_crops, zoom_pipeline,
)

THRESHOLDS = [1, 5, 10, 20]


def test_tile_origins_cover_and_clamp():
    starts = tile_origins(2500, 1024, 896)
    assert starts[0] == 0 and starts[-1] == 2500 - 1024  # flush to the edge
    assert all(b - a <= 896 for a, b in zip(starts, starts[1:]))
    assert tile_origins(800, 1024, 896) == [0]  # image smaller than a tile


def test_tile_pipeline_offsets_and_cross_tile_dedupe():
    rgb = np.zeros((1200, 2000, 3), np.uint8)
    calls = []

    def detect(tile):
        calls.append(tile.shape)
        # one fake lesion at (10, 10)-(40, 40) in every tile's own coords
        return [{"label": "papule", "score": 0.9, "bbox": [10, 10, 40, 40]}]

    dets, rects = tile_pipeline(rgb, detect, tile=1024, overlap=128)
    assert len(rects) == len(calls) == 6  # 2 rows x 3 cols
    assert (10, 10, 40, 40) in {tuple(d["bbox"]) for d in dets}
    assert (986, 186, 1016, 216) in {tuple(d["bbox"]) for d in dets}  # last tile
    # a duplicate from an overlap region collapses to one detection
    dup = dets + [{"label": "papule", "score": 0.8, "bbox": [12, 12, 40, 40]}]
    kept = dedupe(sorted(dup, key=lambda d: d["score"], reverse=True))
    assert len(kept) == len(dets)


def test_zoom_crops_cluster_and_clamp():
    shape = (2000, 2000, 3)
    near = [(100, 100, 130, 130), (150, 110, 180, 140)]  # padded squares overlap
    far = [(100, 100, 130, 130), (1500, 1500, 1530, 1530)]
    assert len(zoom_crops(near, 4.0, 192, shape)) == 1
    assert len(zoom_crops(far, 4.0, 192, shape)) == 2
    (left, top, right, bottom), = zoom_crops([(5, 5, 25, 25)], 4.0, 192, shape)
    assert left == 0 and top == 0 and right - left == bottom - top == 192


def test_zoom_pipeline_maps_back_to_original_coords():
    rgb = np.zeros((2000, 2000, 3), np.uint8)

    def detect(crop):
        assert crop.shape == (1024, 1024, 3)
        return [{"label": "pustule", "score": 0.7, "bbox": [462, 462, 562, 562]}]

    dets, rects = zoom_pipeline(rgb, [(400, 400, 440, 440)], detect,
                                pad=4.0, min_side=192)
    (left, top, right, _), = rects
    scale = (right - left) / 1024
    x0, y0, x1, y1 = dets[0]["bbox"]
    assert abs(x0 - (462 * scale + left)) < 1e-6
    assert abs(y1 - (562 * scale + top)) < 1e-6
    # crop center == box center, so the mapped det sits on the box center too
    assert abs((x0 + x1) / 2 - 420) < 1e-6 and abs((y0 + y1) / 2 - 420) < 1e-6


def test_report_maps_labels_and_drops_non_concerns():
    dets = ([{"label": "papule", "score": 0.9, "bbox": [0, 0, 1, 1]}] * 5
            + [{"label": "open_comedo", "score": 0.6, "bbox": [0, 0, 1, 1]}]
            + [{"label": "nevus", "score": 0.9, "bbox": [0, 0, 1, 1]}])
    report = report_from_detections("img", dets, ["forehead"] * 7, THRESHOLDS)
    by_concern = {c.concern: c for c in report.concerns}
    assert by_concern["acne_inflammatory"].lesion_count == 5
    assert by_concern["acne_inflammatory"].severity == 2  # 5 lesions -> sev 2
    assert by_concern["acne_comedonal"].lesion_count == 1
    assert "dropped 1" in report.notes
    assert report_from_detections("img", [], [], THRESHOLDS).clear_skin


def test_compare_matches_by_iou():
    a = [{"label": "papule", "score": 0.9, "bbox": [0, 0, 100, 100]},
         {"label": "nodule", "score": 0.8, "bbox": [500, 500, 600, 600]}]
    b = [{"label": "papule", "score": 0.7, "bbox": [10, 10, 110, 110]},
         {"label": "pustule", "score": 0.6, "bbox": [505, 505, 605, 605]},
         {"label": "melasma", "score": 0.5, "bbox": [900, 900, 950, 950]}]
    result = compare(a, b)
    assert result["matched_same_label"] == 1
    assert result["matched_different_label"] == 1
    assert result["zoom_only"] == 0 and result["tile_only"] == 1
    labels = result["label_disagreements"][0]
    assert labels["zoom"]["label"] == "nodule" and labels["tile"]["label"] == "pustule"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
