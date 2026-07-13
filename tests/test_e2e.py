from __future__ import annotations

import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.pipeline.e2e import main


ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, *, width: int = 1200, height: int = 700) -> None:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = x
    rgb[:, :, 1] = 150
    rgb[:, :, 2] = 110
    Image.fromarray(rgb, "RGB").save(path, format="JPEG", quality=95)


def _write_catalog(path: Path) -> None:
    path.write_text(json.dumps([
        {
            "product_id": "gentle-cleanser",
            "name": "Gentle Cleanser",
            "brand": "Fixture",
            "category": "cleanser",
            "actives": [],
        },
        {
            "product_id": "daily-spf",
            "name": "Daily SPF",
            "brand": "Fixture",
            "category": "spf",
            "actives": [],
        },
    ]))


@contextmanager
def _serve_sarpn(*, malformed_request: int | None = None):
    state = SimpleNamespace(request_count=0, paths=[])
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            with lock:
                state.request_count += 1
                request_number = state.request_count
                state.paths.append(self.path)
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            tile = Image.open(BytesIO(base64.b64decode(body["image"]))).convert("RGB")
            left_red = tile.getpixel((0, tile.height // 2))[0]

            if request_number == malformed_request:
                payload = b"not-json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if left_red < 20:
                detections = [
                    {"label": "Papule", "score": 0.91, "bbox": [300, 120, 500, 320]},
                    {"label": "mystery lesion", "score": 0.72, "bbox": [40, 400, 140, 500]},
                ]
            else:
                detections = [
                    {"label": "pustule", "score": 0.88, "bbox": [124, 120, 324, 320]},
                ]
            encoded = json.dumps({"count": len(detections), "detections": detections}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    state.url = f"http://fixture:secret@{host}:{port}/predict?token=hidden#fragment"
    try:
        yield state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture
def fake_sarpn_server():
    with _serve_sarpn() as server:
        yield server


def _args(image_path: Path, output_dir: Path, api: str, catalog_path: Path) -> list[str]:
    return [
        "--image", str(image_path),
        "--out", str(output_dir),
        "--api", api,
        "--catalog", str(catalog_path),
        "--face-landmarker", str(image_path.parent / "missing.task"),
        "--tile-size", "1024",
        "--overlap", "128",
        "--request-batch-size", "1",
    ]


def test_importing_default_e2e_loads_no_legacy_models():
    code = """
import sys
import src.pipeline.e2e
forbidden = {
    "tensorflow",
    "ultralytics",
    "src.classification.classifier",
    "src.classification.run_acne04_pipeline",
    "src.recommendation.bridge",
    "src.recommendation.ranker",
    "src.recommendation.concern_stats",
}
loaded = sorted(name for name in forbidden if name in sys.modules)
print(loaded)
raise SystemExit(bool(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fixture_e2e_writes_complete_v2_artifact_set(tmp_path, fake_sarpn_server):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) == 0

    assert {path.name for path in output_dir.iterdir()} == {
        "analysis.json", "routine.json", "detections.jpg",
        "region_overlay.jpg", "lesion_sheet.jpg",
    }
    for name in ("detections.jpg", "region_overlay.jpg", "lesion_sheet.jpg"):
        with Image.open(output_dir / name) as diagnostic:
            diagnostic.verify()
        with Image.open(output_dir / name) as diagnostic:
            assert diagnostic.width > 0 and diagnostic.height > 0

    analysis_text = (output_dir / "analysis.json").read_text()
    analysis = json.loads(analysis_text)
    assert analysis["schema_version"] == "2.0"
    assert analysis["pipeline"] == {
        "identifier": "sa-rpn-native-tiles",
        "endpoint": "http://" + fake_sarpn_server.url.split("@", 1)[1].split("?", 1)[0],
        "tile_size": 1024,
        "overlap": 128,
        "minimum_score": 0.3,
        "dedupe_threshold": 0.5,
    }
    assert analysis["region_mapping"]["method"] == "grid_fallback"
    assert "missing" in analysis["region_mapping"]["reason"].lower()
    assert analysis["recommendation_status"] == "complete"
    assert len(analysis["detections"]) == 2
    assert analysis["detections"][0].keys() >= {
        "normalized_label", "original_label", "confidence", "box", "region",
        "mapped_concern", "source_tile", "observation_status",
    }
    assert {item["normalized_label"] for item in analysis["detections"]} == {
        "papule", "mystery_lesion",
    }
    unsupported = next(
        item for item in analysis["detections"]
        if item["normalized_label"] == "mystery_lesion"
    )
    assert unsupported["observation_status"] == "unsupported"
    assert unsupported["mapped_concern"] is None
    assert "secret" not in analysis_text
    assert "hidden" not in analysis_text
    assert "fixture" not in analysis["pipeline"]["endpoint"]
    assert "probs" not in analysis_text
    assert "probabilities" not in analysis_text
    assert all("legacy" not in path for path in fake_sarpn_server.paths)


@pytest.mark.parametrize("catalog_case", ["missing", "invalid", "unreadable"])
def test_catalog_failure_keeps_analysis_and_diagnostics(
    tmp_path, fake_sarpn_server, catalog_case,
):
    image_path = tmp_path / "face.jpg"
    output_dir = tmp_path / "output"
    catalog_path = tmp_path / "catalog.json"
    _write_image(image_path, width=800)
    if catalog_case == "invalid":
        catalog_path.write_text("not-json")
    elif catalog_case == "unreadable":
        catalog_path.mkdir()

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) == 0

    analysis = json.loads((output_dir / "analysis.json").read_text())
    assert analysis["recommendation_status"] == "unavailable"
    expected = "missing" if catalog_case == "missing" else catalog_case
    assert expected in analysis["recommendation_reason"].lower()
    assert not (output_dir / "routine.json").exists()
    assert {path.name for path in output_dir.iterdir()} == {
        "analysis.json", "detections.jpg", "region_overlay.jpg", "lesion_sheet.jpg",
    }


def test_recommendation_exception_does_not_erase_analysis(
    tmp_path, fake_sarpn_server, monkeypatch,
):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path, width=800)
    _write_catalog(catalog_path)

    def fail_recommendation(*args, **kwargs):
        raise RuntimeError("recommendation exploded")

    monkeypatch.setattr("src.pipeline.e2e.recommend", fail_recommendation)
    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) == 0

    analysis = json.loads((output_dir / "analysis.json").read_text())
    assert fake_sarpn_server.request_count == 1
    assert analysis["recommendation_status"] == "unavailable"
    assert "recommendation exploded" in analysis["recommendation_reason"]
    assert not (output_dir / "routine.json").exists()
    assert all((output_dir / name).exists() for name in (
        "analysis.json", "detections.jpg", "region_overlay.jpg", "lesion_sheet.jpg",
    ))


def test_failed_identification_preserves_previous_output(tmp_path):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)
    output_dir.mkdir()
    marker = output_dir / "last-success.txt"
    marker.write_text("keep me")

    with _serve_sarpn(malformed_request=2) as server:
        assert main(_args(image_path, output_dir, server.url, catalog_path)) != 0
        assert server.request_count == 2
        assert all("legacy" not in path for path in server.paths)

    assert {path.name for path in output_dir.iterdir()} == {"last-success.txt"}
    assert marker.read_text() == "keep me"
    assert not list(tmp_path.glob(".output.*"))
