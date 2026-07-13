from copy import deepcopy
import json

import pytest

from src.evaluation.e2e_release import compare_counterfactuals, evaluate_release, wilson_interval
from src.pipeline.provenance import build_provenance


GATES = {
    "clinician_policy_approval": True,
    "adequate_calibration_cohort": True,
    "external_clinical_review_set": True,
    "verified_real_catalog_overlay": True,
    "remote_detector_identity": True,
}


def artifact(sample_id, *, split="valid", dirty=False, detector="det", triage="routine",
             disposition="active_treatment", source_hash=None,
             evidence_source="prediction"):
    inputs = {
            "source_image_sha256": source_hash or f"source-{sample_id}",
            "evidence_source": evidence_source,
            "dataset": {"name": "synthetic", "sample_id": sample_id,
                        "split": split, "split_proof": "synthetic-fixture"},
            "input_profile": {"skin_type": "oily", "pregnancy_status": "not_pregnant"},
            "effective_config": {"x": 1},
            "models": {"detector": {"sha256": detector}},
            "catalog": {"sha256": "catalog"},
            "ranker": {"state": "none", "sha256": None},
            "policies": {"triage": {"sha256": "triage"},
                         "therapy": {"sha256": "therapy"}},
        }
    if evidence_source == "oracle":
        inputs["oracle_annotations"] = {"state": "available", "sha256": "oracle-xml"}
    envelope = build_provenance(
        inputs,
        clock=lambda: "artifact-time",
        git_reader=lambda: {"git_commit": "abc", "dirty": dirty},
    )
    envelope.update({
        "decision": {
            "triage_level": triage, "referral_reasons": [],
            "therapy_disposition": disposition, "decision_evidence": [],
            "policy_version": "synthetic", "policy_reviewed": True,
        },
        "therapy_plan": {"primary": {"therapy": "azelaic_acid"}},
        "recommendation_status": "complete",
    })
    return envelope


def routine_from(analysis, *, violation=False):
    product = {
        "product_id": "aza", "routine_roles": ([] if violation else ["treatment"]),
        "actives": ["azelaic_acid"],
        "drug_actives": [{"name": "azelaic_acid", "strength": "10%"}],
    }
    return {
        **deepcopy(analysis),
        "selected_products": {"treatment": product},
        "selected_regimen": {"am": [], "pm": [{"role": "treatment"}]},
        "alternatives": {},
        "explanation": [{"role": "treatment", "product_id": "aza",
                         "delivered_active": "azelaic_acid"}],
        "validation_errors": ["injected"] if violation else [],
    }


def write_run(tmp_path, analysis, *, routine=True, violation=False):
    run = tmp_path / analysis["dataset"]["sample_id"]
    run.mkdir()
    (run / "analysis.json").write_text(json.dumps(analysis))
    if routine:
        (run / "routine.json").write_text(json.dumps(routine_from(analysis, violation=violation)))
    return run


def write_manifest(tmp_path, rows, *, gates=GATES, stratified=False, evidence_source=None):
    path = tmp_path / f"manifest-{len(list(tmp_path.glob('manifest-*')))}.json"
    normalized = []
    for row in rows:
        item = dict(row)
        sample_id = str(item["sample_id"])
        item.setdefault("split", "valid")
        item.setdefault("split_proof", "synthetic-fixture")
        item.setdefault("source_image_sha256", f"source-{sample_id}")
        item.setdefault("evidence_source", evidence_source or "prediction")
        normalized.append(item)
    payload = {"samples": normalized, "external_gates": gates, "stratified": stratified}
    if evidence_source is not None:
        payload["evidence_source"] = evidence_source
    path.write_text(json.dumps(payload))
    return path


def test_training_sample_causes_named_preflight_failure(tmp_path):
    analysis = artifact("train", split="train")
    run = write_run(tmp_path, analysis)
    manifest = write_manifest(tmp_path, [{"sample_id": "train"}])
    report = evaluate_release([run], manifest, generated_at="report-time")
    assert report["release_status"] == "failed_preflight"
    assert any("release_split_invalid:train" in reason
               for reason in report["preflight"]["failures"])


@pytest.mark.parametrize("case", ["unknown", "dirty", "stale", "mixed"])
def test_unknown_dirty_stale_and_mixed_samples_fail_by_name(tmp_path, case):
    first = artifact("one", split="unknown" if case == "unknown" else "valid",
                     dirty=case == "dirty")
    runs = [write_run(tmp_path, first)]
    rows = [{"sample_id": "one"}]
    if case == "stale":
        first["semantic_inputs"]["input_profile"]["skin_type"] = "dry"
        (runs[0] / "analysis.json").write_text(json.dumps(first))
    if case == "mixed":
        second = artifact("two", detector="different")
        runs.append(write_run(tmp_path, second))
        rows.append({"sample_id": "two"})
    report = evaluate_release(runs, write_manifest(tmp_path, rows), generated_at="time")
    failures = " ".join(report["preflight"]["failures"])
    expected = {"unknown": "release_split_invalid", "dirty": "dirty_or_unknown_code",
                "stale": "stale_replay_key", "mixed": "mixed_semantic_inputs"}[case]
    assert expected in failures


def test_tiny_known_confusion_fixture_has_exact_counts_and_intervals(tmp_path):
    positive = artifact("positive", triage="derm_first", disposition="supportive_only")
    negative = artifact("negative", triage="routine", disposition="active_treatment")
    runs = [write_run(tmp_path, positive), write_run(tmp_path, negative)]
    rows = [
        {"sample_id": "positive", "oracle": {"nodule_present": True,
                                               "triage_level": "derm_first"},
         "clinician": {"therapy_disposition": "supportive_only",
                       "referral_reasons": [], "primary_therapy": "azelaic_acid"},
         "detector_counts": {"nodule": {"tp": 1, "fp": 0, "fn": 1}},
         "attempts": [{"attempt_id": "1"}]},
        {"sample_id": "negative", "oracle": {"nodule_present": False,
                                               "triage_level": "routine"},
         "clinician": {"therapy_disposition": "active_treatment",
                       "referral_reasons": [], "primary_therapy": "azelaic_acid"},
         "detector_counts": {"nodule": {"tp": 0, "fp": 1, "fn": 0}},
         "attempts": [{"attempt_id": "2"}]},
    ]
    report = evaluate_release(runs, write_manifest(tmp_path, rows), generated_at="time")
    metric = report["metrics"]["nodule_triage"]
    assert (metric["tp"], metric["fp"], metric["tn"], metric["fn"]) == (1, 0, 1, 0)
    assert metric["sensitivity"] == wilson_interval(1, 1)
    assert report["metrics"]["detector_by_class"]["nodule"]["recall"] == wilson_interval(1, 2)
    batch = report["metrics"]["batch"]
    assert {key: batch[key] for key in ("requested", "completed", "failed", "total_attempts")} == {
        "requested": 2, "completed": 2, "failed": 0, "total_attempts": 2,
    }


def test_missing_clinician_labels_is_blocked_not_zero_agreement(tmp_path):
    analysis = artifact("sample")
    run = write_run(tmp_path, analysis)
    row = {"sample_id": "sample", "oracle": {"nodule_present": False}}
    report = evaluate_release([run], write_manifest(tmp_path, [row]), generated_at="time")
    assert report["metrics"]["clinician_disposition_agreement"] == {
        "status": "blocked", "reason": "labels_missing",
    }
    assert "clinician_disposition_labels_missing" in report["preflight"][
        "blocked_external_gates"
    ]


def test_product_role_and_validation_violations_are_counted(tmp_path):
    analysis = artifact("sample")
    run = write_run(tmp_path, analysis, violation=True)
    row = {"sample_id": "sample", "oracle": {"nodule_present": False}}
    report = evaluate_release([run], write_manifest(tmp_path, [row]), generated_at="time")
    assert report["metrics"]["product_role_violations"] == 1
    assert report["metrics"]["validation_violations"] == 1
    assert report["metrics"]["selected_product_count_per_role"]["treatment"]["max"] == 1


def test_report_is_deterministic_apart_from_injected_timestamp(tmp_path):
    analysis = artifact("sample")
    run = write_run(tmp_path, analysis)
    row = {"sample_id": "sample", "oracle": {"nodule_present": False}}
    manifest = write_manifest(tmp_path, [row])
    first = evaluate_release([run], manifest, generated_at="one")
    second = evaluate_release([run], manifest, generated_at="two")
    first["generated_at"] = second["generated_at"]
    assert first == second


def test_prediction_and_oracle_counterfactuals_are_compared_without_pooling(tmp_path):
    prediction = artifact("missed", triage="routine", disposition="active_treatment")
    oracle = artifact("missed", triage="derm_first", disposition="supportive_only",
                      evidence_source="oracle")
    (tmp_path / "prediction").mkdir()
    (tmp_path / "oracle").mkdir()
    prediction_run = write_run(tmp_path / "prediction", prediction)
    oracle_run = write_run(tmp_path / "oracle", oracle)
    row = {"sample_id": "missed", "oracle": {"nodule_present": True}}
    prediction_report = evaluate_release(
        [prediction_run],
        write_manifest(tmp_path, [row], evidence_source="prediction"),
        generated_at="time",
    )
    oracle_report = evaluate_release(
        [oracle_run],
        write_manifest(tmp_path, [row], evidence_source="oracle"),
        generated_at="time",
    )
    comparison = compare_counterfactuals(prediction_report, oracle_report)
    assert comparison["sample_count"] == 1
    assert comparison["disagreement_count"] == 1
    assert comparison["disagreements"][0]["sample_id"] == "missed"


def test_prediction_and_oracle_rows_cannot_be_pooled(tmp_path):
    first = artifact("prediction")
    second = artifact("oracle", evidence_source="oracle")
    runs = [write_run(tmp_path, first), write_run(tmp_path, second)]
    rows = [
        {"sample_id": "prediction", "evidence_source": "prediction"},
        {"sample_id": "oracle", "evidence_source": "oracle"},
    ]
    report = evaluate_release(runs, write_manifest(tmp_path, rows), generated_at="time")
    assert "cohort:mixed_prediction_oracle_evidence" in report["preflight"]["failures"]


def test_every_manifest_sample_and_source_hash_must_match_a_run(tmp_path):
    analysis = artifact("present")
    run = write_run(tmp_path, analysis)
    rows = [
        {"sample_id": "present", "source_image_sha256": "wrong"},
        {"sample_id": "missing"},
    ]
    report = evaluate_release([run], write_manifest(tmp_path, rows), generated_at="time")
    failures = report["preflight"]["failures"]
    assert "present:manifest_artifact_source_image_mismatch" in failures
    assert "missing:run_artifact_missing" in failures
