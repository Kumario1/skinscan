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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.e2e import load_optional_catalog, main


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


def test_failed_identification_preserves_previous_output(tmp_path, capsys):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)
    output_dir.mkdir()
    # Marking output_dir as a prior pipeline output (analysis.json present)
    # satisfies the Finding 2 guard so the run gets far enough to hit the
    # SA-RPN endpoint and fail there, before ever reaching the publish step.
    (output_dir / "analysis.json").write_text('{"schema_version": "2.0"}')
    marker = output_dir / "last-success.txt"
    marker.write_text("keep me")

    with _serve_sarpn(malformed_request=2) as server:
        assert main(_args(image_path, output_dir, server.url, catalog_path)) != 0
        assert server.request_count == 2
        assert all("legacy" not in path for path in server.paths)

    assert {path.name for path in output_dir.iterdir()} == {"analysis.json", "last-success.txt"}
    assert marker.read_text() == "keep me"
    assert not list(tmp_path.glob(".output.*"))

    stderr = capsys.readouterr().err
    assert "fixture" not in stderr
    assert "secret" not in stderr
    assert "hidden" not in stderr


def test_load_optional_catalog_merges_tier2_products(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    tier2_path = tmp_path / "catalog_tier2.json"
    _write_catalog(catalog_path)
    tier2_path.write_text(json.dumps([
        {
            "product_id": "tier2-serum",
            "name": "Tier 2 Serum",
            "brand": "Fixture",
            "category": "serum",
            "actives": [],
            "tier": 2,
            "no_outcome_data": True,
        },
    ]))

    products, reason = load_optional_catalog(catalog_path)

    assert reason is None
    assert {product.product_id for product in products} == {
        "gentle-cleanser", "daily-spf", "tier2-serum",
    }


def test_load_optional_catalog_ignores_absent_tier2(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)

    products, reason = load_optional_catalog(catalog_path)

    assert reason is None
    assert {product.product_id for product in products} == {"gentle-cleanser", "daily-spf"}


def test_load_optional_catalog_degrades_on_broken_tier2(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    tier2_path = tmp_path / "catalog_tier2.json"
    _write_catalog(catalog_path)
    tier2_path.write_text("not-json")

    products, reason = load_optional_catalog(catalog_path)

    assert products is None
    assert reason is not None
    assert "invalid" in reason.lower()


def test_empty_catalog_is_treated_as_unavailable(tmp_path, fake_sarpn_server):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path, width=800)
    catalog_path.write_text("[]")

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) == 0

    analysis = json.loads((output_dir / "analysis.json").read_text())
    assert analysis["recommendation_status"] == "unavailable"
    assert analysis["recommendation_reason"] == "catalog is empty"
    assert not (output_dir / "routine.json").exists()


# --- Finding 2: refuse to replace an unrelated pre-existing --out dir ------

def test_main_refuses_to_replace_unrelated_output_dir(tmp_path, fake_sarpn_server):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)
    output_dir.mkdir()
    unrelated = output_dir / "unrelated.txt"
    unrelated.write_text("do not touch me")

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) != 0

    assert unrelated.read_text() == "do not touch me"
    assert {path.name for path in output_dir.iterdir()} == {"unrelated.txt"}
    assert fake_sarpn_server.request_count == 0


def test_main_replaces_empty_output_dir(tmp_path, fake_sarpn_server):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)
    output_dir.mkdir()

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) == 0
    assert (output_dir / "analysis.json").exists()


def test_main_replaces_prior_pipeline_output_dir(tmp_path, fake_sarpn_server):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)
    output_dir.mkdir()
    (output_dir / "analysis.json").write_text('{"schema_version": "2.0"}')
    (output_dir / "stale.jpg").write_text("old diagnostic")

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) == 0
    assert not (output_dir / "stale.jpg").exists()
    analysis = json.loads((output_dir / "analysis.json").read_text())
    assert analysis["schema_version"] == "2.0"
    assert analysis["recommendation_status"] == "complete"


# --- Finding 11: regular file at --out is refused early (chosen behavior) --

def test_regular_file_at_output_path_is_refused_early(tmp_path, fake_sarpn_server):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)
    output_dir.write_text("not a directory")

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) != 0

    assert output_dir.is_file()
    assert output_dir.read_text() == "not a directory"
    assert fake_sarpn_server.request_count == 0


def test_remove_path_helper_handles_files_dirs_and_missing(tmp_path):
    from src.pipeline.e2e import _remove_path

    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    _remove_path(file_path)
    assert not file_path.exists()

    dir_path = tmp_path / "dir"
    dir_path.mkdir()
    (dir_path / "inner.txt").write_text("y")
    _remove_path(dir_path)
    assert not dir_path.exists()

    _remove_path(tmp_path / "does-not-exist")  # must not raise


# --- Finding 5: a concurrent-run race must not destroy the other run's ------
# --- fresh output; the failed run leaves its backup on disk instead. -------

def test_publish_conflict_preserves_concurrent_output(tmp_path, fake_sarpn_server, monkeypatch):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)
    output_dir.mkdir()
    (output_dir / "analysis.json").write_text('{"schema_version": "2.0", "marker": "prior-run"}')

    real_rename = Path.rename

    def flaky_rename(self, target):
        target_path = Path(target)
        if target_path == output_dir and ".staging-" in self.name:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "concurrent-marker.txt").write_text("fresh from concurrent run")
            raise OSError("simulated concurrent publish collision")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) != 0

    assert {path.name for path in output_dir.iterdir()} == {"concurrent-marker.txt"}
    assert (output_dir / "concurrent-marker.txt").read_text() == "fresh from concurrent run"

    backups = list(tmp_path.glob(".output.backup-*"))
    assert len(backups) == 1
    backup_analysis = json.loads((backups[0] / "analysis.json").read_text())
    assert backup_analysis["marker"] == "prior-run"


def test_publish_conflict_error_message_mentions_backup_path(
    tmp_path, fake_sarpn_server, monkeypatch, capsys,
):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)
    output_dir.mkdir()
    (output_dir / "analysis.json").write_text('{"schema_version": "2.0", "marker": "prior-run"}')

    real_rename = Path.rename

    def flaky_rename(self, target):
        target_path = Path(target)
        if target_path == output_dir and ".staging-" in self.name:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "concurrent-marker.txt").write_text("fresh from concurrent run")
            raise OSError("simulated concurrent publish collision")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) != 0

    backups = list(tmp_path.glob(".output.backup-*"))
    assert len(backups) == 1
    stderr = capsys.readouterr().err
    assert str(backups[0]) in stderr


# --- Finding 8: a stale backup from any pid is recoverable ------------------

def test_stale_any_pid_backup_is_restored_then_replaced_on_success(tmp_path, fake_sarpn_server):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)

    stale_backup = tmp_path / ".output.backup-99999"
    stale_backup.mkdir()
    (stale_backup / "marker.txt").write_text("stranded from a crashed run")

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) == 0

    assert output_dir.exists()
    assert not (output_dir / "marker.txt").exists()
    assert (output_dir / "analysis.json").exists()
    assert not stale_backup.exists()
    assert not list(tmp_path.glob(".output.backup-*"))


def test_stale_any_pid_backup_is_restored_when_publish_then_fails(
    tmp_path, fake_sarpn_server, monkeypatch,
):
    image_path = tmp_path / "face.jpg"
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "output"
    _write_image(image_path)
    _write_catalog(catalog_path)

    stale_backup = tmp_path / ".output.backup-99999"
    stale_backup.mkdir()
    (stale_backup / "marker.txt").write_text("stranded from a crashed run")

    real_rename = Path.rename

    def flaky_rename(self, target):
        target_path = Path(target)
        if target_path == output_dir and ".staging-" in self.name:
            raise OSError("simulated publish failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)

    assert main(_args(image_path, output_dir, fake_sarpn_server.url, catalog_path)) != 0

    assert output_dir.exists()
    assert (output_dir / "marker.txt").read_text() == "stranded from a crashed run"
