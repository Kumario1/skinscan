"""Geometry + mapping tests for the SA-RPN A/B harness (compare_sarpn.py).

No API, no model weights: zoom's detect function is a stub, tile delegates
to (stubbed) production code, everything else is coordinate math. Uses the
monkeypatch fixture, so pytest is required (no standalone __main__ runner).

Tile-path geometry (tile_origins/make_tiles), HTTP validation, coordinate
restoration, label mapping, and dedupe now live in src/pipeline/sarpn.py and
are tested in tests/test_sarpn.py — not duplicated here.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline import compare_sarpn
from src.pipeline.compare_sarpn import compare, zoom_crops, zoom_pipeline
from src.pipeline.sarpn import SarpnSettings


def _settings(**overrides):
    config = deepcopy(load_config())
    config["sa_rpn"].update(overrides)
    return SarpnSettings.from_config(config)


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


def test_tile_comparison_uses_production_inference(monkeypatch):
    rgb = np.zeros((50, 60, 3), np.uint8)
    called = {}

    def fake_infer(image, settings, **kwargs):
        called["image_shape"] = image.shape
        called["settings"] = settings
        return []

    monkeypatch.setattr(compare_sarpn, "infer_native_tiles", fake_infer)

    settings = _settings(tile_size=1024, tile_overlap=128)
    observations, rects = compare_sarpn.run_tile_comparison(rgb, settings)

    assert called["image_shape"] == rgb.shape
    assert called["settings"] is settings
    assert observations == []
    # tile rectangles come from production make_tiles, independent of the
    # (stubbed) HTTP call — the whole image fits in one 1024px tile.
    assert rects == [(0, 0, 60, 50)]
