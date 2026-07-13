"""Resumable, checkpointed per-image batch orchestration."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random as random_module
import tempfile
import time
from typing import Callable, Mapping, Sequence

from .provenance import (
    catalog_bundle_identity, compute_replay_key, file_identity, sha256_file,
)


STAGES = (
    "identified", "regions_and_concerns", "decision_and_recommendation",
    "rendered", "published",
)
TERMINAL_STATES = {"complete", "retryable_failed", "permanent_failed"}


class TransientBatchError(RuntimeError):
    pass


class PermanentBatchError(RuntimeError):
    pass


class BatchInterrupted(BaseException):
    """Injected interruption used to prove checkpoint durability."""


@dataclass(frozen=True)
class BatchRequest:
    sample_id: str
    source_image_sha256: str
    artifact_dir: Path
    semantic_inputs: Mapping[str, object]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if min(self.base_delay_seconds, self.max_delay_seconds, self.jitter_seconds) < 0:
            raise ValueError("retry delays must be non-negative")


@dataclass(frozen=True)
class BatchSummary:
    requested: int
    completed: int
    failed: int
    retried: int
    skipped: int
    stale: int
    total_attempts: int

    @property
    def exit_code(self) -> int:
        return 0 if self.failed == 0 and self.completed == self.requested else 1

    def to_dict(self) -> dict[str, int]:
        return {
            "requested": self.requested, "completed": self.completed,
            "failed": self.failed, "retried": self.retried, "skipped": self.skipped,
            "stale": self.stale, "total_attempts": self.total_attempts,
        }


def atomic_write_json(path: Path, value: object) -> None:
    """Durable same-directory replace followed by parse/read-back validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if parsed != value:
            raise OSError(f"atomic JSON read-back mismatch: {path}")
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _manifest(path: Path, run_id: str, requested: int) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": "1", "run_id": run_id,
                "requested": requested, "images": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("images"), dict):
        raise ValueError(f"batch manifest invalid: {path}")
    if value.get("run_id") != run_id:
        raise ValueError("batch manifest run_id does not match request set")
    value["requested"] = requested
    return value


def _run_id(requests: Sequence[BatchRequest]) -> str:
    return compute_replay_key([
        {"sample_id": request.sample_id}
        for request in sorted(requests, key=lambda item: item.sample_id)
    ])[:20]


def _fragment(request: BatchRequest, stage: str) -> dict[str, object]:
    from src.config import load_config

    semantic = request.semantic_inputs
    runtime_config = load_config()
    e2e = semantic.get("e2e", {})
    e2e = e2e if isinstance(e2e, Mapping) else {}
    identification_keys = (
        "endpoint_url", "tile_size", "tile_overlap", "connect_timeout_seconds",
        "read_timeout_seconds", "request_batch_size", "min_score",
        "dedupe_threshold", "detector_sha256",
    )
    common = {
        "source_image_sha256": request.source_image_sha256,
        "detector": semantic.get("detector"),
        "identification_config": semantic.get("identification_config"),
        "e2e_identification": {
            key: e2e.get(key) for key in identification_keys if key in e2e
        },
        "runtime_sa_rpn": runtime_config["sa_rpn"],
        "oracle_annotations": file_identity(e2e.get("oracle_annotations_path")),
    }
    if stage == "identified":
        return common
    common["region_config"] = semantic.get("region_config")
    common["face_landmarker"] = file_identity(e2e.get("face_landmarker_path"))
    common["runtime_regions"] = runtime_config["regions"]
    common["runtime_tone"] = runtime_config["tone"]
    common["classification_crop_pad"] = runtime_config["classification"]["crop_pad"]
    if stage == "regions_and_concerns":
        return common
    common.update({
        "profile": semantic.get("profile"),
        "catalog": semantic.get("catalog"),
        "policies": semantic.get("policies"),
        "ranker": semantic.get("ranker"),
        "e2e_profile": e2e.get("profile"),
        "e2e_dataset": e2e.get("dataset"),
        "evidence_source": "prediction",
        "catalog_file": catalog_bundle_identity(e2e.get("catalog_path")),
        "therapy_policy_file": file_identity(e2e.get("therapy_policy_path")),
    })
    if stage == "decision_and_recommendation":
        return common
    common["render_config"] = semantic.get("render_config")
    return common


def _checkpoint_path(request: BatchRequest, stage: str) -> Path:
    return request.artifact_dir / ".checkpoints" / f"{stage}.json"


def _load_resume_context(
    request: BatchRequest,
) -> tuple[dict[str, object], int, bool]:
    context: dict[str, object] = {}
    next_stage = 0
    stale = False
    for index, stage in enumerate(STAGES):
        path = _checkpoint_path(request, stage)
        if not path.exists():
            break
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stale = True
            break
        expected = compute_replay_key(_fragment(request, stage))
        if (not isinstance(value, dict) or value.get("schema_version") != "1"
                or value.get("stage") != stage or value.get("input_fragment") != expected
                or not isinstance(value.get("data"), dict)):
            stale = True
            break
        context.update(value["data"])
        next_stage = index + 1
    return context, next_stage, stale


def _failure_class(exc: BaseException) -> str:
    if isinstance(exc, (TransientBatchError, TimeoutError, ConnectionError)):
        return "transient_transport"
    if isinstance(exc, PermanentBatchError):
        return "permanent_contract"
    return "permanent_unclassified"


def run_batch(
    requests: Sequence[BatchRequest],
    manifest_path: Path,
    stage_runner: Callable[[BatchRequest, str, Mapping[str, object]], Mapping[str, object]],
    *,
    retry_policy: RetryPolicy = RetryPolicy(),
    sleeper: Callable[[float], None] = time.sleep,
    random: Callable[[], float] = random_module.random,
    monotonic: Callable[[], float] = time.monotonic,
    writer: Callable[[Path, object], None] = atomic_write_json,
) -> BatchSummary:
    ids = [request.sample_id for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("batch requests contain duplicate sample_id")
    run_id = _run_id(requests)
    manifest = _manifest(manifest_path, run_id, len(requests))
    images: dict[str, object] = manifest["images"]
    skipped_ids: set[str] = set()
    stale_ids: set[str] = set()
    retried_ids: set[str] = set()

    for request in requests:
        row = images.setdefault(request.sample_id, {
            "source_image_sha256": request.source_image_sha256,
            "state": "in_progress",
            "last_verified_stage": None,
            "attempts": [],
            "artifact_dir": str(request.artifact_dir),
            "replay_key": None,
        })
        row["source_image_sha256"] = request.source_image_sha256
        row["artifact_dir"] = str(request.artifact_dir)
        desired_replay_key = compute_replay_key(_fragment(request, "published"))
        context, next_index, stale = _load_resume_context(request)
        if stale:
            stale_ids.add(request.sample_id)
            row["state"] = "stale"
            row["last_verified_stage"] = STAGES[next_index - 1] if next_index else None
            writer(manifest_path, manifest)
        if next_index == len(STAGES) and row.get("state") == "complete":
            skipped_ids.add(request.sample_id)
            continue
        attempts_for_replay = sum(
            attempt.get("replay_key") == desired_replay_key
            for attempt in row["attempts"]
        )
        if row.get("state") == "permanent_failed" and attempts_for_replay:
            continue
        if (row.get("state") == "retryable_failed"
                and attempts_for_replay >= retry_policy.max_attempts):
            continue

        while attempts_for_replay < retry_policy.max_attempts:
            attempt_number = len(row["attempts"]) + 1
            attempts_for_replay += 1
            if attempts_for_replay > 1:
                retried_ids.add(request.sample_id)
            attempt = {
                "attempt_id": f"{run_id}:{request.sample_id}:{attempt_number}",
                "failure_class": None,
                "latency_ms": None,
                "replay_key": desired_replay_key,
            }
            row["attempts"].append(attempt)
            row["state"] = "in_progress"
            writer(manifest_path, manifest)
            started = monotonic()
            try:
                for index in range(next_index, len(STAGES)):
                    stage = STAGES[index]
                    data = stage_runner(request, stage, dict(context))
                    if not isinstance(data, Mapping):
                        raise PermanentBatchError(f"stage {stage} returned non-object data")
                    data = dict(data)
                    checkpoint = {
                        "schema_version": "1",
                        "stage": stage,
                        "input_fragment": compute_replay_key(_fragment(request, stage)),
                        "data": data,
                    }
                    writer(_checkpoint_path(request, stage), checkpoint)
                    context.update(data)
                    next_index = index + 1
                    row["last_verified_stage"] = stage
                    row["replay_key"] = desired_replay_key
                    writer(manifest_path, manifest)
                attempt["latency_ms"] = max(0, round((monotonic() - started) * 1000))
                row["state"] = "complete"
                writer(manifest_path, manifest)
                break
            except BatchInterrupted:
                attempt["latency_ms"] = max(0, round((monotonic() - started) * 1000))
                row["state"] = "in_progress"
                writer(manifest_path, manifest)
                raise
            except BaseException as exc:
                failure = _failure_class(exc)
                attempt["failure_class"] = failure
                attempt["latency_ms"] = max(0, round((monotonic() - started) * 1000))
                transient = failure == "transient_transport"
                row["state"] = "retryable_failed" if transient else "permanent_failed"
                writer(manifest_path, manifest)
                if not transient or attempts_for_replay >= retry_policy.max_attempts:
                    break
                delay = min(
                    retry_policy.max_delay_seconds,
                    retry_policy.base_delay_seconds * (2 ** (attempt_number - 1)),
                ) + retry_policy.jitter_seconds * random()
                sleeper(delay)

    requested_rows = [images[request.sample_id] for request in requests]
    completed = sum(row.get("state") == "complete" for row in requested_rows)
    summary = BatchSummary(
        requested=len(requests),
        completed=completed,
        failed=len(requests) - completed,
        retried=len(retried_ids),
        skipped=len(skipped_ids),
        stale=len(stale_ids),
        total_attempts=sum(len(row.get("attempts", [])) for row in requested_rows),
    )
    manifest["summary"] = summary.to_dict()
    writer(manifest_path, manifest)
    return summary


def load_requests(path: Path) -> list[BatchRequest]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("images") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("batch request file: expected a list or {images: [...]} object")
    requests: list[BatchRequest] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"batch request images[{index}]: expected an object")
        requests.append(BatchRequest(
            sample_id=str(row["sample_id"]),
            source_image_sha256=str(row["source_image_sha256"]),
            artifact_dir=Path(row["artifact_dir"]),
            semantic_inputs=row.get("semantic_inputs", {}),
        ))
    return requests


def _observation_payload(value) -> dict[str, object]:
    return {
        "label": value.label,
        "label_name": value.label_name,
        "score": value.score,
        "box": list(value.box),
        "tile_index": value.tile_index,
        "tile_box": list(value.tile_box),
        "region": value.region,
        "mapped_concern": value.mapped_concern,
        "observation_status": value.observation_status,
    }


def _observation_from(value: Mapping[str, object]):
    from .sarpn import LesionObservation
    return LesionObservation(
        str(value["label"]), str(value["label_name"]), float(value["score"]),
        tuple(value["box"]), int(value["tile_index"]), tuple(value["tile_box"]),
        value.get("region"), value.get("mapped_concern"),
        str(value.get("observation_status", "actionable")),
    )


def _report_from(value: Mapping[str, object]):
    from src.recommendation.schema import Concern, ConcernEvidence, ConcernReport
    concerns = []
    for item in value.get("concerns", []):
        evidence = item.get("evidence", {})
        regions = list(item.get("regions", []))
        concerns.append(Concern(
            item["concern"], regions[0], int(item["severity"]), float(item["confidence"]),
            item.get("lesion_count"), regions,
            ConcernEvidence(dict(evidence.get("labels", {})),
                            float(evidence.get("max_confidence", 0)),
                            int(evidence.get("affected_region_count", 0)),
                            str(evidence.get("source", "prediction"))),
        ))
    return ConcernReport(
        str(value["image_id"]), concerns, bool(value.get("clear_skin")),
        bool(value.get("low_light_flag")), str(value.get("notes", "")),
    )


class E2EStageRunner:
    """Production stage runner used by the batch CLI.

    Request semantic_inputs must include ``image_path`` and may contain an
    ``e2e`` object with the same explicit paths/identity fields as the single-
    image CLI. The identified checkpoint contains full normalized observations,
    so later-stage resume performs no detector HTTP work.
    """

    def _values(self, request: BatchRequest) -> tuple[Path, Mapping[str, object]]:
        image = request.semantic_inputs.get("image_path")
        if not isinstance(image, str):
            raise PermanentBatchError("semantic_inputs.image_path is required")
        raw = request.semantic_inputs.get("e2e", {})
        if not isinstance(raw, Mapping):
            raise PermanentBatchError("semantic_inputs.e2e must be an object")
        return Path(image), raw

    def _settings(self, values: Mapping[str, object]):
        from src.config import load_config
        from .sarpn import SarpnSettings
        config = load_config()
        settings = SarpnSettings.from_config(config)
        overrides = {}
        for key in (
            "endpoint_url", "tile_size", "tile_overlap", "connect_timeout_seconds",
            "read_timeout_seconds", "request_batch_size", "min_score",
            "dedupe_threshold",
        ):
            if key in values:
                overrides[key] = values[key]
        settings = replace(settings, **overrides)
        settings._validate()
        return settings

    def __call__(
        self, request: BatchRequest, stage: str, context: Mapping[str, object]
    ) -> Mapping[str, object]:
        image_path, values = self._values(request)
        settings = self._settings(values)
        if stage == "identified":
            from .sarpn import (
                SarpnHTTPStatusError, SarpnResponseError, SarpnTransportError,
                infer_native_tiles, load_rgb,
            )
            try:
                if sha256_file(image_path) != request.source_image_sha256:
                    raise PermanentBatchError(
                        "source_image_sha256 does not match image_path bytes"
                    )
                oracle_path = values.get("oracle_annotations_path")
                if isinstance(oracle_path, str):
                    from .oracle import load_voc_oracle_observations
                    observations = load_voc_oracle_observations(oracle_path)
                else:
                    observations = infer_native_tiles(load_rgb(image_path), settings)
            except SarpnHTTPStatusError as exc:
                if 500 <= exc.status_code < 600:
                    raise TransientBatchError(str(exc)) from exc
                raise PermanentBatchError(str(exc)) from exc
            except SarpnTransportError as exc:
                raise TransientBatchError(str(exc)) from exc
            except SarpnResponseError as exc:
                raise PermanentBatchError(str(exc)) from exc
            return {"identified_observations": [_observation_payload(item) for item in observations]}

        observations = [_observation_from(item) for item in context["identified_observations"]]
        if stage == "regions_and_concerns":
            from .regions import locate_regions
            from .sarpn import build_sarpn_concern_report, concern_to_dict, load_rgb
            from .tone import estimate_tone
            rgb = load_rgb(image_path)
            boxes = [item.box for item in observations]
            model = values.get("face_landmarker_path")
            region_result = locate_regions(
                rgb, boxes, model_path=Path(model) if isinstance(model, str) else None
            )
            tone = estimate_tone(rgb, region_result.polygons, boxes)
            report, updated, safety = build_sarpn_concern_report(
                image_path.name, observations, region_result.regions, settings.severity,
                low_light_flag=bool(tone.low_light),
                evidence_source=("annotation_oracle"
                                 if isinstance(values.get("oracle_annotations_path"), str)
                                 else "prediction"),
            )
            return {
                "observations": [_observation_payload(item) for item in updated],
                "report": {
                    "image_id": report.image_id,
                    "concerns": [concern_to_dict(item) for item in report.concerns],
                    "clear_skin": report.clear_skin,
                    "low_light_flag": report.low_light_flag,
                    "notes": report.notes,
                },
                "tone": asdict(tone),
                "region_mapping": dict(region_result.metadata),
                "safety_observations": [asdict(item) for item in safety],
            }

        if stage == "decision_and_recommendation":
            from src.recommendation.decision import conservative_unreviewed_policy, decide_care
            from src.recommendation.engine import recommend
            from src.recommendation.schema import UserProfile
            from src.recommendation.therapy import load_therapy_policy, plan_therapy
            from .e2e import _read_git_state, load_optional_catalog
            from .provenance import build_provenance
            from .sarpn import sanitize_endpoint

            report = _report_from(context["report"])
            profile_raw = values.get("profile", request.semantic_inputs.get("profile", {}))
            if not isinstance(profile_raw, Mapping):
                raise PermanentBatchError("e2e.profile must be an object")
            profile = UserProfile.from_dict(profile_raw)
            therapy_path_raw = values.get("therapy_policy_path")
            therapy_path = Path(therapy_path_raw) if isinstance(therapy_path_raw, str) else None
            therapy_policy = load_therapy_policy(therapy_path)
            triage_policy = conservative_unreviewed_policy()
            decision = decide_care(report, triage_policy)
            plan = plan_therapy(decision, report, profile, therapy_policy)
            if decision.therapy_disposition == "active_treatment" and plan.primary is None:
                decision = replace(decision, therapy_disposition="defer")
            catalog_path_raw = values.get("catalog_path")
            catalog_path = Path(catalog_path_raw) if isinstance(catalog_path_raw, str) else None
            catalog, catalog_reason = load_optional_catalog(catalog_path)
            if catalog_reason is None and not catalog:
                catalog_reason = "catalog is empty"
            from src.config import load_config
            runtime_config = load_config()
            provenance = build_provenance(
                {
                    "source_image_sha256": request.source_image_sha256,
                    "evidence_source": (
                        "oracle" if isinstance(values.get("oracle_annotations_path"), str)
                        else "prediction"
                    ),
                    "oracle_annotations": file_identity(
                        values.get("oracle_annotations_path")
                    ),
                    "dataset": values.get("dataset", {
                        "name": "unknown", "sample_id": request.sample_id,
                        "split": "unknown", "split_proof": None,
                    }),
                    "input_profile": profile.to_dict(),
                    "effective_config": {
                        "pipeline": (
                            "acnescu-voc-oracle"
                            if isinstance(values.get("oracle_annotations_path"), str)
                            else "sa-rpn-native-tiles"
                        ),
                        "endpoint": sanitize_endpoint(settings.endpoint_url),
                        "tile_size": settings.tile_size,
                        "overlap": settings.tile_overlap,
                        "minimum_score": settings.min_score,
                        "class_min_scores": dict(settings.class_min_scores),
                        "dedupe_threshold": settings.dedupe_threshold,
                        "severity": settings.severity,
                        "regions": runtime_config["regions"],
                        "tone": runtime_config["tone"],
                        "classification_crop_pad": runtime_config["classification"]["crop_pad"],
                        "face_landmarker": file_identity(values.get("face_landmarker_path")),
                    },
                    "models": {"detector": (
                        {"state": "not_applicable", "sha256": None,
                         "identity": "annotation_oracle"}
                        if isinstance(values.get("oracle_annotations_path"), str)
                        else {"sha256": values.get("detector_sha256")}
                    ),
                               "classifier": {"state": "not_applicable", "sha256": None}},
                    "catalog": catalog_bundle_identity(catalog_path),
                    "ranker": {"state": "none", "sha256": None},
                    "policies": {
                        "triage": {"identity": triage_policy.identifier,
                                   "reviewed": False, "sha256": None},
                        "therapy": {**file_identity(therapy_path),
                                    "identity": therapy_policy.identifier,
                                    "reviewed": therapy_policy.reviewed},
                    },
                },
                clock=lambda: datetime.now(timezone.utc), git_reader=_read_git_state,
            )
            analysis = {
                **provenance,
                "image_id": image_path.name,
                "pipeline": {"identifier": (
                                 "acnescu-voc-oracle"
                                 if isinstance(values.get("oracle_annotations_path"), str)
                                 else "sa-rpn-native-tiles"
                             ),
                             "endpoint": sanitize_endpoint(settings.endpoint_url)},
                "detections": list(context["observations"]),
                "concerns": context["report"]["concerns"],
                "clear_skin": report.clear_skin,
                "skin_tone": context["tone"],
                "region_mapping": context["region_mapping"],
                "safety_observations": context["safety_observations"],
                "decision": decision.to_dict(), "therapy_plan": plan.to_dict(),
                "recommendation_status": "unavailable",
            }
            routine = None
            if catalog_reason:
                analysis["recommendation_reason"] = catalog_reason
            else:
                recommendation = recommend(
                    report, catalog, profile, triage_policy=triage_policy,
                    therapy_policy=therapy_policy,
                )
                analysis["decision"] = recommendation.decision.to_dict()
                analysis["therapy_plan"] = recommendation.therapy_plan.to_dict()
                if recommendation.validation_errors:
                    analysis["recommendation_status"] = "invalid"
                    analysis["recommendation_errors"] = recommendation.validation_errors
                else:
                    from .e2e import v3_routine_payload
                    routine = v3_routine_payload(recommendation, provenance)
                    analysis["recommendation_status"] = (
                        "partial"
                        if (recommendation.decision.therapy_disposition == "defer"
                            and recommendation.therapy_plan.primary is not None)
                        else "complete"
                    )
            return {"analysis": analysis, "routine": routine}

        if stage == "rendered":
            from .regions import locate_regions
            from .sarpn import (
                draw_detection_overlay, draw_lesion_sheet, draw_region_overlay, load_rgb,
            )
            rgb = load_rgb(image_path)
            observations = [_observation_from(item) for item in context["observations"]]
            model = values.get("face_landmarker_path")
            region_result = locate_regions(
                rgb, [item.box for item in observations],
                model_path=Path(model) if isinstance(model, str) else None,
            )
            rendered = request.artifact_dir / ".rendered"
            rendered.mkdir(parents=True, exist_ok=True)
            draw_detection_overlay(rgb, observations, rendered / "detections.jpg")
            draw_region_overlay(rgb, observations, region_result, rendered / "region_overlay.jpg")
            draw_lesion_sheet(rgb, observations, rendered / "lesion_sheet.jpg")
            return {"rendered_files": ["detections.jpg", "region_overlay.jpg", "lesion_sheet.jpg"]}

        if stage == "published":
            try:
                atomic_write_json(request.artifact_dir / "analysis.json", context["analysis"])
                routine_path = request.artifact_dir / "routine.json"
                if context.get("routine") is not None:
                    atomic_write_json(routine_path, context["routine"])
                elif routine_path.exists():
                    routine_path.unlink()
                rendered = request.artifact_dir / ".rendered"
                for name in context["rendered_files"]:
                    os.replace(rendered / name, request.artifact_dir / name)
                try:
                    rendered.rmdir()
                except OSError:
                    pass
            except OSError as exc:
                raise TransientBatchError(f"artifact publication failed: {exc}") from exc
            return {"published": True}
        raise PermanentBatchError(f"unknown batch stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        requests = load_requests(args.requests)
        summary = run_batch(requests, args.manifest, E2EStageRunner())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"batch request invalid: {exc}")
        return 2
    print(json.dumps(summary.to_dict(), sort_keys=True))
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
