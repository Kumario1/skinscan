import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
from threading import Lock, Thread
import time

import numpy as np
from PIL import Image
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.recommendation.schema import Concern, ConcernEvidence, UserProfile
from src.pipeline.sarpn import (
    LesionObservation,
    SARPN_LABEL_TO_CONCERN,
    SARPN_NON_ACTIONABLE_LABELS,
    build_sarpn_concern_report,
    concern_to_dict,
    normalize_sarpn_label,
    SarpnResponseError,
    SarpnSettings,
    SarpnTransportError,
    Tile,
    dedupe_observations,
    infer_native_tiles,
    load_rgb,
    make_tiles,
)


def _config_with(**overrides):
    config = deepcopy(load_config())
    config["sa_rpn"].update(overrides)
    return config


def _settings(endpoint_url, **overrides):
    return SarpnSettings.from_config(
        _config_with(endpoint_url=endpoint_url, tile_size=4, tile_overlap=1, **overrides)
    )


@pytest.fixture
def fake_http_server():
    state = {"requests": [], "responses": [], "delays": {}, "next_index": 0}
    state_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            with state_lock:
                index = state["next_index"]
                state["next_index"] += 1
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.loads(body)
            assert set(payload) == {"image"}
            decoded = base64.b64decode(payload["image"], validate=True)
            image = Image.open(BytesIO(decoded))
            image.load()
            assert image.format == "JPEG"
            with state_lock:
                state["requests"].append({"index": index, "path": self.path, "payload": payload})
            time.sleep(state["delays"].get(index, 0))
            response_item = state.get("response_factory", lambda _image, i: state["responses"][i])(image, index)
            status, response = response_item
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except BrokenPipeError:
                pass

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/predict"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def _observation(label, score=0.9):
    return LesionObservation(label, label, score, (0, 0, 1, 1), 0, (0, 0, 4, 4))


def test_legacy_concern_constructor_populates_regions():
    concern = Concern("acne_inflammatory", "left_cheek", 2, 0.8, 4)
    assert concern.regions == ["left_cheek"]
    assert concern.evidence == ConcernEvidence()


def test_user_profile_accepts_unknown_tone():
    assert UserProfile("combination", "unknown", "photo").tone_bucket == "unknown"


@pytest.mark.parametrize(("server_label", "normalized"), [
    ("Closed comedo", "closed_comedo"), ("open comedo", "open_comedo"),
    ("Papule", "papule"), ("Pustule", "pustule"), ("Nodule", "nodule"),
    ("Atrophic scar", "atrophic_scar"), ("Hypertrophic scar", "hypertrophic_scar"),
    ("Melasma", "melasma"), ("Nevus", "nevus"), ("other", "other"),
    ("  New--Label_name ", "new_label_name"),
])
def test_normalize_exact_server_labels(server_label, normalized):
    assert normalize_sarpn_label(server_label) == normalized


def test_label_tables_are_exact():
    assert SARPN_LABEL_TO_CONCERN["papule"] == "acne_inflammatory"
    assert SARPN_LABEL_TO_CONCERN["atrophic_scar"] == "acne_scarring"
    assert SARPN_NON_ACTIONABLE_LABELS == {"nevus", "other"}


def test_bridge_aggregates_evidence_regions_and_confidence():
    observations = [_observation("Papule", 0.72), _observation("Pustule", 0.96)]
    report, updated, safety = build_sarpn_concern_report(
        "img", observations, ["right_cheek", "left_cheek"], load_config()["sa_rpn"]["severity"])
    concern = report.concerns[0]
    assert concern.concern == "acne_inflammatory"
    assert concern.region == "left_cheek"
    assert concern.regions == ["left_cheek", "right_cheek"]
    assert concern.confidence == pytest.approx(0.84)
    assert concern.evidence == ConcernEvidence({"papule": 1, "pustule": 1}, 0.96, 2)
    assert concern_to_dict(concern) == {
        "concern": "acne_inflammatory", "regions": ["left_cheek", "right_cheek"],
        "severity": 2, "confidence": 0.84, "lesion_count": 2,
        "evidence": {"labels": {"papule": 1, "pustule": 1}, "max_confidence": 0.96,
                     "affected_region_count": 2},
    }
    assert [item.normalized_label for item in updated] == ["papule", "pustule"]
    assert safety == []


@pytest.mark.parametrize(("concern", "counts", "expected"), [
    ("acne_comedonal", (0, 1, 7, 8, 19, 20, 39, 40), (0, 1, 1, 2, 2, 3, 3, 4)),
    ("acne_inflammatory", (0, 1, 5, 6, 14, 15, 29, 30), (0, 1, 1, 2, 2, 3, 3, 4)),
])
def test_count_threshold_boundaries(concern, counts, expected):
    reverse = {"acne_comedonal": "closed_comedo", "acne_inflammatory": "papule"}
    config = load_config()["sa_rpn"]["severity"]
    for count, severity in zip(counts, expected):
        report, _, _ = build_sarpn_concern_report("img", [_observation(reverse[concern])] * count,
                                                   ["forehead"] * count, config)
        assert (report.concerns[0].severity if count else 0) == severity


def test_severity_special_cases_region_escalation_and_low_confidence_cap():
    config = load_config()["sa_rpn"]["severity"]
    cases = [
        ([_observation("papule")] * 2, ["forehead", "nose"], 2),
        ([_observation("papule")] * 3, ["forehead", "nose", "chin_jaw"], 3),
        ([_observation("nodule", 0.31)], ["forehead"], 4),
        ([_observation("hypertrophic scar")], ["forehead"], 3),
        ([_observation("papule", 0.4)] * 30, ["forehead"] * 30, 1),
    ]
    for observations, regions, expected in cases:
        report, _, _ = build_sarpn_concern_report("img", observations, regions, config)
        assert report.concerns[0].severity == expected


def test_safety_observations_are_non_actionable_and_policy_driven():
    config = load_config()["sa_rpn"]["severity"]
    observations = [_observation("Nevus", 0.81), _observation("other", 0.5), _observation("Alien label", 0.9)]
    report, updated, safety = build_sarpn_concern_report(
        "img", observations, ["nose", "nose", "chin_jaw"], config)
    assert report.clear_skin and report.concerns == []
    assert [item.observation_status for item in updated] == ["non_actionable", "non_actionable", "unsupported"]
    assert [item.code for item in safety] == ["nevus_observation", "other_observation", "unsupported_label"]
    assert safety[0].professional_review is True
    assert safety[1].professional_review is False
    assert safety[2].professional_review is False


def test_bridge_rejects_parallel_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        build_sarpn_concern_report("img", [_observation("papule")], [], load_config()["sa_rpn"]["severity"])


def test_sarpn_settings_load_production_config():
    settings = SarpnSettings.from_config(load_config())

    assert settings.endpoint_url == "http://localhost:8000/predict"
    assert settings.tile_size == 1024
    assert settings.tile_overlap == 128
    assert settings.connect_timeout_seconds == 5
    assert settings.read_timeout_seconds == 120
    assert settings.request_batch_size == 4
    assert settings.min_score == 0.3
    assert settings.dedupe_threshold == 0.5


def test_sarpn_settings_do_not_alias_source_severity_config():
    config = load_config()
    settings = SarpnSettings.from_config(config)

    config["sa_rpn"]["severity"]["confidence_cutoff"] = 0.1
    config["sa_rpn"]["severity"]["count_thresholds"]["acne_comedonal"].append(100)

    assert settings.severity["confidence_cutoff"] == 0.5
    assert settings.severity["count_thresholds"]["acne_comedonal"] == (1, 8, 20, 40)


def test_sarpn_settings_severity_is_recursively_immutable():
    settings = SarpnSettings.from_config(load_config())

    with pytest.raises(TypeError):
        settings.severity["confidence_cutoff"] = 0.1
    with pytest.raises(TypeError):
        settings.severity["count_thresholds"]["acne_comedonal"] = (1,)
    with pytest.raises(TypeError):
        settings.severity["professional_review"]["nevus"]["min_count"] = 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tile_size": 0}, "tile_size"),
        ({"tile_overlap": -1}, "tile_overlap"),
        ({"tile_overlap": 1024}, "tile_overlap"),
        ({"connect_timeout_seconds": 0}, "connect_timeout_seconds"),
        ({"read_timeout_seconds": -1}, "read_timeout_seconds"),
        ({"request_batch_size": 0}, "request_batch_size"),
        ({"min_score": -0.01}, "min_score"),
        ({"min_score": 1.01}, "min_score"),
        ({"dedupe_threshold": -0.01}, "dedupe_threshold"),
        ({"dedupe_threshold": 1.01}, "dedupe_threshold"),
    ],
)
def test_sarpn_settings_reject_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        SarpnSettings.from_config(_config_with(**overrides))


def test_load_rgb_applies_exif_orientation(tmp_path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (3, 2))
    image.putdata([
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ])
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, quality=100, subsampling=0, exif=exif)

    rgb = load_rgb(path)

    assert rgb.shape == (3, 2, 3)
    assert np.linalg.norm(rgb[0, 0].astype(int) - np.array([255, 255, 0])) < 20
    assert np.linalg.norm(rgb[0, 1].astype(int) - np.array([255, 0, 0])) < 20


def test_make_tiles_covers_right_and_bottom_edges():
    tiles = make_tiles((1200, 2000, 3), tile_size=1024, overlap=128)

    assert len(tiles) == 6
    assert tiles[0] == Tile(index=0, x=0, y=0, width=1024, height=1024)
    assert tiles[-1].x + tiles[-1].width == 2000
    assert tiles[-1].y + tiles[-1].height == 1200


def test_client_posts_base64_jpeg_and_validates_response(fake_http_server):
    fake_http_server["responses"] = [(200, {"count": 1, "detections": [
        {"label": "papule", "score": 0.9, "bbox": [0, 0, 2, 2]}
    ]})]

    result = infer_native_tiles(np.zeros((4, 4, 3), dtype=np.uint8), _settings(fake_http_server["url"]))

    assert result == [LesionObservation("papule", "Papule", 0.9, (0, 0, 2, 2), 0, (0, 0, 4, 4))]
    assert fake_http_server["requests"][0]["path"] == "/predict"


def test_exact_timeout_tuple_is_passed():
    class RecordingSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.timeout = None

        def post(self, url, **kwargs):
            self.timeout = kwargs["timeout"]
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"count": 0, "detections": []}'
            return response

    session = RecordingSession()
    infer_native_tiles(np.zeros((4, 4, 3), dtype=np.uint8), _settings("http://localhost/predict"), session_factory=lambda: session)
    assert session.timeout == (5, 120)


def test_concurrent_tiles_use_distinct_sessions():
    sessions = []
    lock = Lock()

    class RecordingSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.was_closed = False

        def post(self, url, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"count": 0, "detections": []}'
            return response

        def close(self):
            self.was_closed = True
            super().close()

    def session_factory():
        session = RecordingSession()
        with lock:
            sessions.append(session)
        return session

    infer_native_tiles(
        np.zeros((6, 6, 3), dtype=np.uint8),
        _settings("http://localhost/predict", request_batch_size=4),
        session_factory=session_factory,
    )

    assert len(sessions) == 4
    assert len({id(session) for session in sessions}) == 4
    assert all(session.was_closed for session in sessions)


def test_all_tiles_are_requested_and_results_restore_in_tile_order(fake_http_server):
    fake_http_server["responses"] = [(200, {"count": 1, "detections": [
        {"label": "papule", "score": 0.9, "bbox": [0, 0, 1, 1]}
    ]}) for _ in range(4)]
    fake_http_server["delays"] = {0: 0.08, 1: 0.04}

    result = infer_native_tiles(
        np.zeros((6, 6, 3), dtype=np.uint8),
        _settings(fake_http_server["url"], request_batch_size=4),
    )

    assert len(fake_http_server["requests"]) == 4
    assert [item.tile_index for item in result] == [0, 1, 2, 3]
    assert [item.box for item in result] == [(0, 0, 1, 1), (2, 0, 3, 1), (0, 2, 1, 3), (2, 2, 3, 3)]


def test_restored_boxes_are_clipped_to_full_image_bounds(fake_http_server):
    fake_http_server["responses"] = [(200, {"count": 0, "detections": []}) for _ in range(3)] + [(200, {"count": 1, "detections": [
        {"label": "papule", "score": 0.9, "bbox": [1, 1, 9, 9]}
    ]})]

    result = infer_native_tiles(
        np.zeros((6, 6, 3), dtype=np.uint8),
        _settings(fake_http_server["url"], request_batch_size=1),
    )

    assert result[0].box == (3, 3, 6, 6)


@pytest.mark.parametrize("status", [500])
def test_one_http_500_fails_the_entire_analysis(fake_http_server, status):
    fake_http_server["responses"] = [(200, {"count": 0, "detections": []}), (status, {}), (200, {"count": 0, "detections": []}), (200, {"count": 0, "detections": []})]
    with pytest.raises(SarpnTransportError, match=r"tile 1.*127\.0\.0\.1"):
        infer_native_tiles(np.zeros((6, 6, 3), dtype=np.uint8), _settings(fake_http_server["url"], request_batch_size=1))


def test_one_timeout_fails_the_entire_analysis(fake_http_server):
    fake_http_server["responses"] = [(200, {"count": 0, "detections": []}) for _ in range(4)]
    fake_http_server["delays"] = {1: 0.1}
    with pytest.raises(SarpnTransportError, match=r"tile 1.*127\.0\.0\.1"):
        infer_native_tiles(np.zeros((6, 6, 3), dtype=np.uint8), _settings(fake_http_server["url"], request_batch_size=1, read_timeout_seconds=0.01))


@pytest.mark.parametrize(
    ("response", "field"),
    [
        ({}, "detections"),
        ({"count": 0, "detections": {}}, "detections"),
        ({"count": 1, "detections": [{"label": " ", "score": 0.9, "bbox": [0, 0, 1, 1]}]}, "label"),
        ({"count": 1, "detections": [{"label": "papule", "score": True, "bbox": [0, 0, 1, 1]}]}, "score"),
        ({"count": 1, "detections": [{"label": "papule", "score": 1.1, "bbox": [0, 0, 1, 1]}]}, "score"),
        ({"count": 1, "detections": [{"label": "papule", "score": 0.9, "bbox": [2, 0, 1, 1]}]}, "bbox"),
        ({"count": 1, "detections": [{"label": "papule", "score": 0.9, "bbox": [0, 0, 0, 1]}]}, "bbox"),
        ({"count": 1, "detections": [{"label": "papule", "score": 0.9, "bbox": [5, 5, 6, 6]}]}, "bbox"),
    ],
)
def test_invalid_response_fields_are_rejected(fake_http_server, response, field):
    fake_http_server["responses"] = [(200, response)]
    with pytest.raises(SarpnResponseError, match=rf"tile 0.*{field}"):
        infer_native_tiles(np.zeros((4, 4, 3), dtype=np.uint8), _settings(fake_http_server["url"]))


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf")])
def test_nan_or_infinite_box_coordinate_is_rejected(coordinate):
    class NonStandardJsonSession(requests.Session):
        def post(self, url, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response.json = lambda: {"count": 1, "detections": [
                {"label": "papule", "score": 0.9, "bbox": [0, 0, coordinate, 1]}
            ]}
            return response

    with pytest.raises(SarpnResponseError, match=r"tile 0.*bbox"):
        infer_native_tiles(
            np.zeros((4, 4, 3), dtype=np.uint8),
            _settings("http://localhost/predict"),
            session_factory=NonStandardJsonSession,
        )


def test_missing_count_is_rejected_over_real_http(fake_http_server):
    fake_http_server["responses"] = [(200, {"detections": []})]
    with pytest.raises(SarpnResponseError, match=r"tile 0.*count"):
        infer_native_tiles(np.zeros((4, 4, 3), dtype=np.uint8), _settings(fake_http_server["url"]))


@pytest.mark.parametrize(
    "response",
    [
        {"count": True, "detections": []},
        {"count": -1, "detections": []},
        {"count": 1, "detections": []},
    ],
)
def test_invalid_count_is_rejected(fake_http_server, response):
    fake_http_server["responses"] = [(200, response)]
    with pytest.raises(SarpnResponseError, match=r"tile 0.*count"):
        infer_native_tiles(np.zeros((4, 4, 3), dtype=np.uint8), _settings(fake_http_server["url"]))


def test_min_score_is_applied_client_side(fake_http_server):
    fake_http_server["responses"] = [(200, {"count": 1, "detections": [
        {"label": "papule", "score": 0.2, "bbox": [0, 0, 1, 1]}
    ]})]
    assert infer_native_tiles(np.zeros((4, 4, 3), dtype=np.uint8), _settings(fake_http_server["url"])) == []


def test_dedupe_is_class_agnostic_and_keeps_higher_confidence():
    observations = [
        LesionObservation("papule", "Papule", 0.91, (10, 10, 40, 40), 0, (0, 0, 1024, 1024)),
        LesionObservation("pustule", "Pustule", 0.80, (12, 12, 39, 39), 1, (0, 0, 1024, 1024)),
    ]
    assert dedupe_observations(observations, threshold=0.5) == [observations[0]]


def test_dedupe_preserves_overlap_equal_to_threshold():
    observations = [
        LesionObservation("papule", "Papule", 0.9, (0, 0, 10, 10), 0, (0, 0, 10, 10)),
        LesionObservation("pustule", "Pustule", 0.8, (5, 0, 15, 10), 1, (5, 0, 15, 10)),
    ]
    assert dedupe_observations(observations, threshold=0.5) == observations
