import json

import pytest

import src.pipeline.batch as batch_module
from src.pipeline.batch import (
    BatchInterrupted, BatchRequest, E2EStageRunner, PermanentBatchError, RetryPolicy,
    TransientBatchError, atomic_write_json, run_batch,
)
from src.pipeline.provenance import sha256_file


def request(tmp_path, sample="sample", **semantic_overrides):
    semantic = {
        "detector": "detector-a",
        "identification_config": {"threshold": 0.3},
        "region_config": {"method": "grid"},
        "profile": {"skin_type": "oily"},
        "catalog": "catalog-a",
        "policies": "policy-a",
        "ranker": None,
        "render_config": {"quality": 92},
    }
    semantic.update(semantic_overrides)
    return BatchRequest(sample, f"source-{sample}", tmp_path / sample, semantic)


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, req, stage, context):
        self.calls.append(stage)
        if stage == "identified":
            return {"observations": [{"label": "papule"}], "identification_count": 1}
        return {stage: True}


def policy(max_attempts=3):
    return RetryPolicy(max_attempts=max_attempts, base_delay_seconds=1,
                       max_delay_seconds=4, jitter_seconds=0)


def test_transient_identification_failure_retries_with_backoff_and_attempt_rows(tmp_path):
    calls = 0
    sleeps = []

    def runner(req, stage, context):
        nonlocal calls
        if stage == "identified":
            calls += 1
            if calls == 1:
                raise TransientBatchError("temporary 503")
            return {"observations": [{"label": "papule"}]}
        return {stage: True}

    manifest = tmp_path / "manifest.json"
    summary = run_batch([request(tmp_path)], manifest, runner, retry_policy=policy(),
                        sleeper=sleeps.append, random=lambda: 0, monotonic=lambda: 1)
    assert summary.exit_code == 0
    assert sleeps == [1]
    row = json.loads(manifest.read_text())["images"]["sample"]
    assert len(row["attempts"]) == 2
    assert row["attempts"][0]["failure_class"] == "transient_transport"
    assert row["attempts"][1]["failure_class"] is None
    assert summary.retried == 1


def test_permanent_malformed_response_is_not_retried(tmp_path):
    def runner(req, stage, context):
        raise PermanentBatchError("response schema invalid")

    manifest = tmp_path / "manifest.json"
    summary = run_batch([request(tmp_path)], manifest, runner, retry_policy=policy(),
                        sleeper=lambda _: None, monotonic=lambda: 1)
    row = json.loads(manifest.read_text())["images"]["sample"]
    assert row["state"] == "permanent_failed"
    assert len(row["attempts"]) == 1
    assert summary.exit_code == 1 and summary.failed == 1


def test_interrupt_after_identification_resumes_without_another_identification(tmp_path):
    first_calls = []

    def interrupted(req, stage, context):
        first_calls.append(stage)
        if stage == "identified":
            return {"observations": [{"label": "papule"}]}
        raise BatchInterrupted()

    manifest = tmp_path / "manifest.json"
    with pytest.raises(BatchInterrupted):
        run_batch([request(tmp_path)], manifest, interrupted, retry_policy=policy(),
                  sleeper=lambda _: None, monotonic=lambda: 1)
    assert (tmp_path / "sample/.checkpoints/identified.json").exists()
    resumed = RecordingRunner()
    summary = run_batch([request(tmp_path)], manifest, resumed, retry_policy=policy(),
                        sleeper=lambda _: None, monotonic=lambda: 1)
    assert summary.exit_code == 0
    assert "identified" not in resumed.calls
    assert resumed.calls[0] == "regions_and_concerns"


def test_complete_fresh_image_is_skipped_on_rerun(tmp_path):
    manifest = tmp_path / "manifest.json"
    runner = RecordingRunner()
    run_batch([request(tmp_path)], manifest, runner, sleeper=lambda _: None,
              monotonic=lambda: 1)
    rerun = RecordingRunner()
    summary = run_batch([request(tmp_path)], manifest, rerun, sleeper=lambda _: None,
                        monotonic=lambda: 1)
    assert rerun.calls == []
    assert summary.skipped == 1 and summary.completed == 1


@pytest.mark.parametrize("changed,expected", [
    ({"catalog": "catalog-b"}, "decision_and_recommendation"),
    ({"profile": {"skin_type": "dry"}}, "decision_and_recommendation"),
    ({"detector": "detector-b"}, "identified"),
])
def test_changed_inputs_resume_from_first_affected_stage(tmp_path, changed, expected):
    manifest = tmp_path / "manifest.json"
    run_batch([request(tmp_path)], manifest, RecordingRunner(), sleeper=lambda _: None,
              monotonic=lambda: 1)
    rerun = RecordingRunner()
    summary = run_batch([request(tmp_path, **changed)], manifest, rerun,
                        sleeper=lambda _: None, monotonic=lambda: 1)
    assert rerun.calls[0] == expected
    assert summary.stale == 1


def test_changed_source_invalidates_identification_but_keeps_stable_run_id(tmp_path):
    manifest = tmp_path / "manifest.json"
    original = request(tmp_path)
    run_batch([original], manifest, RecordingRunner(), sleeper=lambda _: None,
              monotonic=lambda: 1)
    old_run_id = json.loads(manifest.read_text())["run_id"]
    changed = BatchRequest(original.sample_id, "new-source", original.artifact_dir,
                           original.semantic_inputs)
    rerun = RecordingRunner()
    run_batch([changed], manifest, rerun, sleeper=lambda _: None, monotonic=lambda: 1)
    assert rerun.calls[0] == "identified"
    assert json.loads(manifest.read_text())["run_id"] == old_run_id


def test_changed_nested_production_detector_identity_invalidates_identification(tmp_path):
    manifest = tmp_path / "manifest.json"
    original = request(tmp_path, e2e={"detector_sha256": "detector-a"})
    run_batch([original], manifest, RecordingRunner(), sleeper=lambda _: None,
              monotonic=lambda: 1)
    changed = request(tmp_path, e2e={"detector_sha256": "detector-b"})
    rerun = RecordingRunner()
    run_batch([changed], manifest, rerun, sleeper=lambda _: None, monotonic=lambda: 1)
    assert rerun.calls[0] == "identified"


def test_atomic_write_failure_leaves_last_valid_manifest_readable(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"version": 1})
    real_replace = batch_module.os.replace

    def fail_replace(source, target):
        if target == path:
            raise OSError("injected replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(batch_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_write_json(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}


def test_summary_and_exit_code_count_every_requested_image_once(tmp_path):
    requests = [request(tmp_path, "ok"), request(tmp_path, "bad")]

    def runner(req, stage, context):
        if req.sample_id == "bad":
            raise PermanentBatchError("bad contract")
        return {stage: True}

    summary = run_batch(requests, tmp_path / "manifest.json", runner,
                        sleeper=lambda _: None, monotonic=lambda: 1)
    assert summary.to_dict() == {
        "requested": 2, "completed": 1, "failed": 1, "retried": 0,
        "skipped": 0, "stale": 0, "total_attempts": 2,
    }
    assert summary.exit_code == 1


def test_production_stage_runner_resume_makes_no_second_sarpn_request(tmp_path):
    from test_e2e import _serve_sarpn, _write_image, _write_verified_catalog, ROOT

    image = tmp_path / "face.jpg"
    catalog = tmp_path / "catalog.json"
    _write_image(image, width=800)
    _write_verified_catalog(catalog)
    with _serve_sarpn() as server:
        semantic = {
            "image_path": str(image),
            "detector": "synthetic-detector",
            "identification_config": {"tile_size": 1024},
            "region_config": {"method": "grid"},
            "profile": json.loads((ROOT / "tests/fixtures/profile_complete.json").read_text()),
            "catalog": sha256_file(catalog),
            "policies": "synthetic-policy",
            "ranker": None,
            "render_config": {"quality": 92},
            "e2e": {
                "endpoint_url": server.url,
                "request_batch_size": 1,
                "face_landmarker_path": str(tmp_path / "missing.task"),
                "catalog_path": str(catalog),
                "therapy_policy_path": str(ROOT / "tests/fixtures/therapy_policy_synthetic.json"),
                "profile": json.loads((ROOT / "tests/fixtures/profile_complete.json").read_text()),
                "dataset": {"name": "synthetic", "sample_id": "face", "split": "smoke",
                            "split_proof": "fixture"},
                "detector_sha256": "synthetic-detector",
                "eligibility_debug": True,
            },
        }
        req = BatchRequest("face", sha256_file(image), tmp_path / "artifacts", semantic)
        real = E2EStageRunner()

        def interrupt_after_identified(request, stage, context):
            if stage == "regions_and_concerns":
                raise BatchInterrupted()
            return real(request, stage, context)

        manifest = tmp_path / "manifest.json"
        with pytest.raises(BatchInterrupted):
            run_batch([req], manifest, interrupt_after_identified,
                      sleeper=lambda _: None, monotonic=lambda: 1)
        assert server.request_count == 1
        summary = run_batch([req], manifest, real, sleeper=lambda _: None,
                            monotonic=lambda: 1)
        assert summary.exit_code == 0
        assert server.request_count == 1
        routine = json.loads((req.artifact_dir / "routine.json").read_text())
        assert "eligibility_rejections" not in routine
        assert "validation_errors" not in routine
        assert (req.artifact_dir / "eligibility_rejections.json").exists()

        semantic["e2e"]["eligibility_debug"] = False
        summary = run_batch([req], manifest, real, sleeper=lambda _: None,
                            monotonic=lambda: 1)
        assert summary.exit_code == 0
        assert server.request_count == 1
        assert not (req.artifact_dir / "eligibility_rejections.json").exists()


def test_production_stage_runner_rejects_source_hash_mismatch_before_http(tmp_path):
    from test_e2e import _write_image

    image = tmp_path / "face.jpg"
    _write_image(image, width=800)
    req = BatchRequest(
        "face", "not-the-image-hash", tmp_path / "artifacts",
        {"image_path": str(image), "e2e": {}},
    )
    with pytest.raises(PermanentBatchError, match="does not match"):
        E2EStageRunner()(req, "identified", {})


@pytest.mark.parametrize("status,error", [
    (400, PermanentBatchError), (401, PermanentBatchError), (503, TransientBatchError),
])
def test_production_stage_runner_retries_only_5xx_http_statuses(
    tmp_path, monkeypatch, status, error,
):
    from test_e2e import _write_image
    from src.pipeline.sarpn import SarpnHTTPStatusError
    import src.pipeline.sarpn as sarpn_module

    image = tmp_path / "face.jpg"
    _write_image(image, width=800)

    def fail(*_args, **_kwargs):
        raise SarpnHTTPStatusError(f"HTTP {status}", status)

    monkeypatch.setattr(sarpn_module, "infer_native_tiles", fail)
    req = BatchRequest(
        "face", sha256_file(image), tmp_path / "artifacts",
        {"image_path": str(image), "e2e": {}},
    )
    with pytest.raises(error):
        E2EStageRunner()(req, "identified", {})
