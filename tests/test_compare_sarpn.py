"""Geometry + mapping tests for the SA-RPN A/B harness (compare_sarpn.py).

No API, no model weights: zoom's detect function is a stub, tile delegates
to (stubbed) production code, everything else is coordinate math. Uses the
monkeypatch fixture, so pytest is required (no standalone __main__ runner).

Tile-path geometry (tile_origins/make_tiles), HTTP validation, coordinate
restoration, label mapping, and dedupe now live in src/pipeline/sarpn.py and
are tested in tests/test_sarpn.py — not duplicated here.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.pipeline import compare_sarpn
from src.pipeline.compare_sarpn import compare, zoom_crops, zoom_pipeline
from src.pipeline.sarpn import LesionObservation, SarpnSettings


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


def test_observations_from_zoom_dicts_applies_min_score_and_bbox_validation():
    """Finding 14: the zoom path must apply the same client-side score/bbox
    validation the tile path gets for free from src.pipeline.sarpn, so the
    A/B comparison is symmetric — drop invalid entries rather than crash."""
    settings = _settings(min_score=0.5)
    dets = [
        {"label": "papule", "score": 0.9, "bbox": [0, 0, 10, 10]},
        {"label": "papule", "score": 0.1, "bbox": [0, 0, 10, 10]},  # below min_score
        {"label": "papule", "score": float("nan"), "bbox": [0, 0, 10, 10]},  # NaN score
        {"label": "papule", "score": 0.9, "bbox": [0, 0, 0, 10]},  # degenerate bbox
        {"label": "papule", "score": True, "bbox": [0, 0, 10, 10]},  # bool score
    ]

    observations = compare_sarpn._observations_from_zoom_dicts(dets, (100, 100, 3), settings)

    assert len(observations) == 1
    assert observations[0].score == 0.9
    assert observations[0].box == (0.0, 0.0, 10.0, 10.0)


def test_load_optional_catalog_merges_tier2_products(tmp_path):
    """Finding 1 (compare side): the harness must merge catalog_tier2.json
    the same way src.pipeline.e2e.load_optional_catalog does, but without
    importing e2e (forbidden by design — this harness must not depend on
    the user-facing e2e module)."""
    catalog_path = tmp_path / "catalog.json"
    tier2_path = tmp_path / "catalog_tier2.json"
    catalog_path.write_text(json.dumps([
        {"product_id": "p1", "name": "Cleanser", "brand": "b",
         "category": "cleanser", "actives": []},
    ]))
    tier2_path.write_text(json.dumps([
        {"product_id": "tier2-serum", "name": "Tier 2 Serum", "brand": "b",
         "category": "serum", "actives": [], "tier": 2, "no_outcome_data": True},
    ]))

    products = compare_sarpn._load_optional_catalog(catalog_path)

    assert products is not None
    assert {p.product_id for p in products} == {"p1", "tier2-serum"}


def test_load_optional_catalog_ignores_absent_tier2(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps([
        {"product_id": "p1", "name": "Cleanser", "brand": "b",
         "category": "cleanser", "actives": []},
    ]))

    products = compare_sarpn._load_optional_catalog(catalog_path)

    assert {p.product_id for p in products} == {"p1"}


def test_load_optional_catalog_degrades_on_corrupt_catalog(tmp_path, capsys):
    """Corrupt catalog.json must degrade to catalog=None with a printed
    note, matching the module docstring's promise ("same degradation as
    e2e") — not crash the harness."""
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("not-json")

    products = compare_sarpn._load_optional_catalog(catalog_path)

    assert products is None
    out = capsys.readouterr().out
    assert "catalog" in out.lower()


def test_load_optional_catalog_missing_path_returns_none(tmp_path, capsys):
    products = compare_sarpn._load_optional_catalog(tmp_path / "missing.json")

    assert products is None
    out = capsys.readouterr().out
    assert "catalog" in out.lower()


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
        called["kwargs"] = kwargs
        return []

    monkeypatch.setattr(compare_sarpn, "infer_native_tiles", fake_infer)

    settings = _settings(tile_size=1024, tile_overlap=128)
    observations, rects = compare_sarpn.run_tile_comparison(rgb, settings)

    assert called["image_shape"] == rgb.shape
    assert called["settings"] is settings
    # Finding 13: the harness must ask for genuine raw (un-deduped)
    # observations so run_downstream's own dedupe pass produces a real
    # raw_detections -> detections_after_dedupe delta, instead of tile
    # inference silently deduping twice.
    assert called["kwargs"] == {"dedupe": False}
    assert observations == []
    # tile rectangles come from production make_tiles, independent of the
    # (stubbed) HTTP call — the whole image fits in one 1024px tile.
    assert rects == [(0, 0, 60, 50)]


def test_tile_pipeline_reports_genuine_raw_to_deduped_delta(monkeypatch, tmp_path):
    """Harness-level Finding 13 regression: with cross-tile duplicate
    detections, raw_detections must exceed detections_after_dedupe — not
    equal it (which is what infer_native_tiles' own internal dedupe used to
    force onto the tile path only)."""
    rgb = np.zeros((6, 6, 3), np.uint8)
    duplicate_observations = [
        LesionObservation("papule", "papule", 0.9, (3, 0, 4, 1), 0, (0, 0, 4, 4)),
        LesionObservation("papule", "papule", 0.5, (2, 0, 4, 1), 1, (2, 0, 6, 4)),
    ]

    def fake_infer(image, settings, **kwargs):
        assert kwargs.get("dedupe") is False
        return duplicate_observations

    monkeypatch.setattr(compare_sarpn, "infer_native_tiles", fake_infer)

    settings = _settings(tile_size=4, tile_overlap=1, dedupe_threshold=0.5)
    observations, rects = compare_sarpn.run_tile_comparison(rgb, settings)
    result = compare_sarpn.run_downstream(
        "tile", rgb, observations, rects, "face.jpg", settings, tmp_path,
        catalog=None, ranker=None, skin_type="combination", pregnant=False,
        seconds=0.1, top=5,
    )

    assert result["raw_detections"] == 2
    assert result["detections_after_dedupe"] == 1
    assert result["raw_detections"] > result["detections_after_dedupe"]
