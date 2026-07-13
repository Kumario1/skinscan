"""Honest v3 cohort preflight and deterministic release metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from src.pipeline.provenance import validate_artifact_freshness


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, object]:
    if total == 0:
        return {"successes": successes, "total": total, "rate": None,
                "lower": None, "upper": None}
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    margin /= denominator
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object: {path}")
    return value


def _semantic_signature(artifact: Mapping[str, object]) -> tuple[object, ...]:
    models = artifact.get("models", {})
    detector = models.get("detector", {}) if isinstance(models, Mapping) else {}
    catalog = artifact.get("catalog", {})
    ranker = artifact.get("ranker", {})
    policies = artifact.get("policies", {})
    return (
        detector.get("sha256") if isinstance(detector, Mapping) else None,
        artifact.get("config_sha256"),
        catalog.get("sha256") if isinstance(catalog, Mapping) else None,
        ranker.get("sha256") if isinstance(ranker, Mapping) else None,
        json.dumps(policies, sort_keys=True, separators=(",", ":")),
    )


def _triage_positive(level: str | None) -> bool:
    return level == "derm_first"


def _decision(artifact: Mapping[str, object]) -> Mapping[str, object]:
    value = artifact.get("decision", {})
    return value if isinstance(value, Mapping) else {}


def _selected_counts(routine: Mapping[str, object] | None) -> dict[str, int]:
    if not routine:
        return {}
    selected = routine.get("selected_products", {})
    if not isinstance(selected, Mapping):
        return {"__invalid__": 1}
    return {
        str(role): len(value) if isinstance(value, list) else 1
        for role, value in sorted(selected.items())
    }


def _product_role_violations(routine: Mapping[str, object] | None) -> int:
    if not routine:
        return 0
    selected = routine.get("selected_products", {})
    if not isinstance(selected, Mapping):
        return 1
    violations = 0
    for role, product in selected.items():
        if not isinstance(product, Mapping):
            violations += 1
            continue
        roles = product.get("routine_roles", [])
        if not isinstance(roles, list) or role not in roles:
            violations += 1
    return violations


def _selected_product_total(routine: Mapping[str, object] | None) -> int:
    if not routine or not isinstance(routine.get("selected_products"), Mapping):
        return 0
    return sum(
        len(value) if isinstance(value, list) else 1
        for value in routine["selected_products"].values()
    )


def _therapy_target_delivered(
    analysis: Mapping[str, object], routine: Mapping[str, object] | None,
) -> bool | None:
    plan = analysis.get("therapy_plan", {})
    primary = plan.get("primary") if isinstance(plan, Mapping) else None
    if not isinstance(primary, Mapping) or not isinstance(primary.get("therapy"), str):
        return None
    if not routine or not isinstance(routine.get("selected_products"), Mapping):
        return False
    product = routine["selected_products"].get(primary.get("role", "treatment"))
    if not isinstance(product, Mapping):
        return False
    actives = product.get("actives", [])
    carried = set(actives) if isinstance(actives, list) else set()
    drug = product.get("drug_actives", [])
    if isinstance(drug, list):
        carried.update(
            item.get("name") for item in drug
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        )
    return primary["therapy"] in carried


def _explanation_violations(routine: Mapping[str, object] | None) -> int:
    if not routine:
        return 0
    selected = routine.get("selected_products", {})
    explanations = routine.get("explanation", [])
    if not isinstance(selected, Mapping) or not isinstance(explanations, list):
        return 1
    count = 0
    for item in explanations:
        if not isinstance(item, Mapping):
            count += 1
            continue
        product = selected.get(item.get("role"))
        if not isinstance(product, Mapping) or product.get("product_id") != item.get("product_id"):
            count += 1
            continue
        active = item.get("delivered_active")
        carried = set(product.get("actives", []))
        carried.update(
            value.get("name") for value in product.get("drug_actives", [])
            if isinstance(value, Mapping)
        )
        if active is not None and active not in carried:
            count += 1
    return count


def evaluate_release(
    run_dirs: Sequence[str | Path],
    cohort_manifest: str | Path,
    *,
    generated_at: str,
) -> dict[str, object]:
    manifest = _load_json(Path(cohort_manifest), "cohort manifest")
    rows = manifest.get("samples", [])
    if not isinstance(rows, list):
        raise ValueError("cohort manifest samples: expected a list")
    by_id = {
        str(row.get("sample_id")): row
        for row in rows if isinstance(row, Mapping) and row.get("sample_id") is not None
    }

    failures: list[str] = []
    blockers: list[str] = []
    samples: list[tuple[str, dict[str, object], dict[str, object] | None, Mapping[str, object]]] = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    signatures: set[tuple[object, ...]] = set()
    manifest_ids = [
        str(row.get("sample_id"))
        for row in rows if isinstance(row, Mapping) and row.get("sample_id") is not None
    ]
    if len(manifest_ids) != len(set(manifest_ids)):
        failures.append("cohort:duplicate_manifest_sample_id")
    evidence_sources: Counter[str] = Counter()

    for raw_dir in sorted(Path(path) for path in run_dirs):
        analysis = _load_json(raw_dir / "analysis.json", "analysis artifact")
        routine_path = raw_dir / "routine.json"
        routine = _load_json(routine_path, "routine artifact") if routine_path.exists() else None
        dataset = analysis.get("dataset", {})
        sample_id = str(dataset.get("sample_id") if isinstance(dataset, Mapping)
                        else raw_dir.name)
        manifest_row = by_id.get(sample_id, {})
        if sample_id not in by_id:
            failures.append(f"{sample_id}:cohort_manifest_row_missing")
        semantic_inputs = analysis.get("semantic_inputs", {})
        artifact_source = (
            semantic_inputs.get("evidence_source")
            if isinstance(semantic_inputs, Mapping) else None
        )
        declared_source = (
            manifest_row.get("evidence_source", manifest.get("evidence_source", artifact_source))
            if isinstance(manifest_row, Mapping) else artifact_source
        )
        if artifact_source not in {"prediction", "oracle"}:
            failures.append(f"{sample_id}:artifact_evidence_source_invalid")
        if declared_source not in {"prediction", "oracle"}:
            failures.append(f"{sample_id}:manifest_evidence_source_invalid")
        if declared_source != artifact_source:
            failures.append(f"{sample_id}:evidence_source_mismatch")
        evidence_sources[str(artifact_source)] += 1
        split = dataset.get("split") if isinstance(dataset, Mapping) else None
        if split not in {"valid", "test", "external"}:
            failures.append(f"{sample_id}:release_split_invalid:{split or 'missing'}")
        split_proof = dataset.get("split_proof") if isinstance(dataset, Mapping) else None
        if not split_proof:
            failures.append(f"{sample_id}:split_proof_missing")
        manifest_split = manifest_row.get("split") if isinstance(manifest_row, Mapping) else None
        manifest_proof = (
            manifest_row.get("split_proof") if isinstance(manifest_row, Mapping) else None
        )
        if manifest_split is None:
            failures.append(f"{sample_id}:manifest_split_missing")
        elif manifest_split != split:
            failures.append(f"{sample_id}:manifest_artifact_split_mismatch")
        if not manifest_proof:
            failures.append(f"{sample_id}:manifest_split_proof_missing")
        elif manifest_proof != split_proof:
            failures.append(f"{sample_id}:manifest_artifact_split_proof_mismatch")
        code = analysis.get("code", {})
        if not isinstance(code, Mapping) or code.get("dirty") is not False:
            failures.append(f"{sample_id}:dirty_or_unknown_code")
        detector = analysis.get("models", {})
        detector = detector.get("detector", {}) if isinstance(detector, Mapping) else {}
        if artifact_source == "prediction":
            if not isinstance(detector, Mapping) or not detector.get("sha256"):
                failures.append(f"{sample_id}:detector_identity_unknown")
        else:
            oracle_identity = (
                semantic_inputs.get("oracle_annotations", {})
                if isinstance(semantic_inputs, Mapping) else {}
            )
            if (not isinstance(oracle_identity, Mapping)
                    or not oracle_identity.get("sha256")):
                failures.append(f"{sample_id}:oracle_annotation_identity_unknown")
        for reason in validate_artifact_freshness(analysis):
            failures.append(f"{sample_id}:{reason}")
        if routine is not None:
            for reason in validate_artifact_freshness(routine):
                failures.append(f"{sample_id}:routine_{reason}")
            if routine.get("replay_key") != analysis.get("replay_key"):
                failures.append(f"{sample_id}:analysis_routine_replay_mismatch")
        source_hash = analysis.get("source_image_sha256")
        manifest_source_hash = (
            manifest_row.get("source_image_sha256")
            if isinstance(manifest_row, Mapping) else None
        )
        if not manifest_source_hash:
            failures.append(f"{sample_id}:manifest_source_image_sha256_missing")
        elif manifest_source_hash != source_hash:
            failures.append(f"{sample_id}:manifest_artifact_source_image_mismatch")
        if sample_id in seen_ids:
            failures.append(f"{sample_id}:duplicate_sample_id")
        seen_ids.add(sample_id)
        if isinstance(source_hash, str) and source_hash in seen_sources:
            failures.append(f"{sample_id}:duplicate_source_image_sha256")
        if isinstance(source_hash, str):
            seen_sources.add(source_hash)
        signatures.add(_semantic_signature(analysis))
        samples.append((sample_id, analysis, routine, manifest_row))

    for missing_id in sorted(set(by_id) - seen_ids):
        failures.append(f"{missing_id}:run_artifact_missing")

    if len(signatures) > 1 and not manifest.get("stratified", False):
        failures.append("cohort:mixed_semantic_inputs_without_stratification")
    elif len(signatures) > 1:
        # This command emits aggregate metrics, so named strata must be
        # evaluated separately before they are compared.
        failures.append("cohort:mixed_semantic_inputs_aggregate_not_supported")
    if evidence_sources["prediction"] and evidence_sources["oracle"]:
        failures.append("cohort:mixed_prediction_oracle_evidence")

    gates = manifest.get("external_gates", {})
    if not isinstance(gates, Mapping):
        gates = {}
    for gate in (
        "clinician_policy_approval", "adequate_calibration_cohort",
        "external_clinical_review_set", "verified_real_catalog_overlay",
        "remote_detector_identity",
    ):
        if gates.get(gate) is not True:
            blockers.append(gate)

    tp = fp = tn = fn = abstained = 0
    triage_confusion: Counter[str] = Counter()
    disposition_total = disposition_agree = 0
    referral_tp = referral_fp = referral_fn = 0
    therapy_total = therapy_agree = 0
    detector_counts: dict[str, Counter[str]] = {}
    role_violations = validation_violations = explanation_violations = 0
    selected_product_total = 0
    therapy_target_total = therapy_target_delivered = 0
    contraindication_conflict_violations = 0
    freshness_checks = freshness_failures = 0
    selected_counts: dict[str, list[int]] = {}
    attempt_total = completed = failed = 0
    per_image_attempts: list[dict[str, object]] = []
    counterfactual_samples: list[dict[str, object]] = []

    for sample_id, analysis, routine, row in samples:
        prediction = _decision(analysis)
        predicted_level = prediction.get("triage_level")
        semantic_inputs = analysis.get("semantic_inputs", {})
        source = (semantic_inputs.get("evidence_source")
                  if isinstance(semantic_inputs, Mapping) else None)
        counterfactual_samples.append({
            "sample_id": sample_id,
            "evidence_source": source,
            "triage_level": predicted_level,
            "therapy_disposition": prediction.get("therapy_disposition"),
        })
        oracle = row.get("oracle", {}) if isinstance(row, Mapping) else {}
        oracle = oracle if isinstance(oracle, Mapping) else {}
        oracle_nodule = oracle.get("nodule_present")
        if isinstance(oracle_nodule, bool):
            if predicted_level == "abstain":
                abstained += 1
            elif _triage_positive(predicted_level):
                if oracle_nodule: tp += 1
                else: fp += 1
            elif oracle_nodule:
                fn += 1
            else:
                tn += 1
        oracle_level = oracle.get("triage_level")
        if isinstance(oracle_level, str):
            triage_confusion[f"{oracle_level}->{predicted_level}"] += 1

        clinician = row.get("clinician", {}) if isinstance(row, Mapping) else {}
        clinician = clinician if isinstance(clinician, Mapping) else {}
        clinician_disposition = clinician.get("therapy_disposition")
        if isinstance(clinician_disposition, str):
            disposition_total += 1
            disposition_agree += clinician_disposition == prediction.get("therapy_disposition")
        predicted_referrals = set(prediction.get("referral_reasons", []))
        clinician_referrals = clinician.get("referral_reasons")
        if isinstance(clinician_referrals, list):
            truth = set(clinician_referrals)
            referral_tp += len(predicted_referrals & truth)
            referral_fp += len(predicted_referrals - truth)
            referral_fn += len(truth - predicted_referrals)
        clinician_therapy = clinician.get("primary_therapy")
        plan = analysis.get("therapy_plan", {})
        primary = plan.get("primary") if isinstance(plan, Mapping) else None
        if isinstance(clinician_therapy, str):
            therapy_total += 1
            therapy_agree += isinstance(primary, Mapping) and primary.get("therapy") == clinician_therapy

        raw_detector = row.get("detector_counts", {}) if isinstance(row, Mapping) else {}
        if isinstance(raw_detector, Mapping):
            for label, counts in raw_detector.items():
                if not isinstance(counts, Mapping):
                    continue
                counter = detector_counts.setdefault(str(label), Counter())
                for key in ("tp", "fp", "fn"):
                    counter[key] += int(counts.get(key, 0))

        role_violations += _product_role_violations(routine)
        selected_product_total += _selected_product_total(routine)
        explanation_violations += _explanation_violations(routine)
        if routine:
            validation = routine.get("validation_errors", [])
            validation_violations += len(validation) if isinstance(validation, list) else 1
            if isinstance(validation, list):
                contraindication_conflict_violations += sum(
                    "contraindicat" in str(item).lower()
                    or "conflict" in str(item).lower()
                    or "duplicate_active" in str(item).lower()
                    for item in validation
                )
        analysis_errors = analysis.get("recommendation_errors", [])
        if isinstance(analysis_errors, list):
            validation_violations += len(analysis_errors)
            contraindication_conflict_violations += sum(
                "contraindicat" in str(item).lower()
                or "conflict" in str(item).lower()
                or "duplicate_active" in str(item).lower()
                for item in analysis_errors
            )
        elif analysis_errors:
            validation_violations += 1
        delivered = _therapy_target_delivered(analysis, routine)
        if delivered is not None:
            therapy_target_total += 1
            therapy_target_delivered += delivered
        for role, count in _selected_counts(routine).items():
            selected_counts.setdefault(role, []).append(count)
        attempts = row.get("attempts", []) if isinstance(row, Mapping) else []
        attempt_total += len(attempts) if isinstance(attempts, list) else 0
        normalized_attempts = []
        if isinstance(attempts, list):
            for attempt in attempts:
                if isinstance(attempt, Mapping):
                    normalized_attempts.append({
                        "attempt_id": attempt.get("attempt_id"),
                        "failure_class": attempt.get("failure_class"),
                        "latency_ms": attempt.get("latency_ms"),
                    })
        per_image_attempts.append({"sample_id": sample_id, "attempts": normalized_attempts})
        freshness_checks += 1 + (routine is not None)
        freshness_failures += len(validate_artifact_freshness(analysis))
        if routine is not None:
            freshness_failures += len(validate_artifact_freshness(routine))
        status = analysis.get("recommendation_status")
        if status in {"complete", "partial", "invalid", "unavailable"}:
            completed += 1
        else:
            failed += 1
        if status != "complete":
            failures.append(
                f"{sample_id}:recommendation_not_complete:"
                f"{status or 'missing'}"
            )
        if status in {"complete", "partial"} and routine is None:
            failures.append(f"{sample_id}:routine_artifact_missing")
        if status in {"invalid", "unavailable"} and routine is not None:
            failures.append(f"{sample_id}:routine_present_for_{status}_recommendation")

    nodule_total = tp + fp + tn + fn + abstained
    if disposition_total == 0:
        blockers.append("clinician_disposition_labels_missing")
    if not any(isinstance(row.get("oracle", {}), Mapping)
               and isinstance(row.get("oracle", {}).get("nodule_present"), bool)
               for row in rows if isinstance(row, Mapping)):
        blockers.append("nodule_oracle_labels_missing")
    if role_violations:
        failures.append("cohort:product_role_violations")
    if validation_violations:
        failures.append("cohort:regimen_validation_violations")
    if explanation_violations:
        failures.append("cohort:explanation_product_violations")
    if freshness_failures:
        failures.append("cohort:artifact_freshness_violations")

    detector_metrics: dict[str, object] = {}
    for label, counts in sorted(detector_counts.items()):
        detector_metrics[label] = {
            "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
            "precision": wilson_interval(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": wilson_interval(counts["tp"], counts["tp"] + counts["fn"]),
        }

    report = {
        "schema_version": "1",
        "generated_at": generated_at,
        "release_status": "failed_preflight" if failures else (
            "blocked" if blockers else "eligible"
        ),
        "preflight": {
            "status": "failed" if failures else "passed",
            "failures": sorted(set(failures)),
            "blocked_external_gates": sorted(set(blockers)),
        },
        "cohort": {
            "sample_count": len(samples),
            "stratified": bool(manifest.get("stratified")),
            "evidence_sources": dict(sorted(evidence_sources.items())),
        },
        "counterfactual_samples": sorted(
            counterfactual_samples, key=lambda item: item["sample_id"]
        ),
        "metrics": {
            "detector_by_class": detector_metrics,
            "nodule_triage": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn, "abstained": abstained,
                "sensitivity": wilson_interval(tp, tp + fn),
                "specificity": wilson_interval(tn, tn + fp),
                "ppv": wilson_interval(tp, tp + fp),
                "abstention_rate": wilson_interval(abstained, nodule_total),
            },
            "triage_confusion": dict(sorted(triage_confusion.items())),
            "clinician_disposition_agreement": (
                wilson_interval(disposition_agree, disposition_total)
                if disposition_total else {"status": "blocked", "reason": "labels_missing"}
            ),
            "referral_reasons": {
                "precision": wilson_interval(referral_tp, referral_tp + referral_fp),
                "recall": wilson_interval(referral_tp, referral_tp + referral_fn),
            },
            "therapy_plan_agreement": (
                wilson_interval(therapy_agree, therapy_total)
                if therapy_total else {"status": "blocked", "reason": "labels_missing"}
            ),
            "therapy_target_coverage": wilson_interval(
                therapy_target_delivered, therapy_target_total
            ),
            "product_role_violations": role_violations,
            "product_role_precision": wilson_interval(
                max(0, selected_product_total - role_violations), selected_product_total
            ),
            "contraindication_conflict_violations": contraindication_conflict_violations,
            "validation_violations": validation_violations,
            "explanation_product_violations": explanation_violations,
            "artifact_freshness": {
                "checked_artifacts": freshness_checks,
                "violations": freshness_failures,
            },
            "selected_product_count_per_role": {
                role: {"sample_counts": counts, "max": max(counts)}
                for role, counts in sorted(selected_counts.items())
            },
            "batch": {
                "requested": len(samples), "completed": completed, "failed": failed,
                "total_attempts": attempt_total,
                "per_image": sorted(per_image_attempts, key=lambda item: item["sample_id"]),
            },
        },
    }
    return report


def compare_counterfactuals(
    prediction_report: Mapping[str, object], oracle_report: Mapping[str, object],
) -> dict[str, object]:
    """Compare separately evaluated prediction/oracle cohorts without pooling."""
    def _rows(report: Mapping[str, object], expected: str) -> dict[str, Mapping[str, object]]:
        cohort = report.get("cohort", {})
        sources = cohort.get("evidence_sources", {}) if isinstance(cohort, Mapping) else {}
        if not isinstance(sources, Mapping) or set(sources) != {expected}:
            raise ValueError(f"counterfactual {expected} report has wrong evidence source")
        values = report.get("counterfactual_samples", [])
        if not isinstance(values, list):
            raise ValueError("counterfactual_samples: expected a list")
        return {
            str(item["sample_id"]): item
            for item in values if isinstance(item, Mapping) and "sample_id" in item
        }

    predicted = _rows(prediction_report, "prediction")
    oracle = _rows(oracle_report, "oracle")
    if set(predicted) != set(oracle):
        raise ValueError("counterfactual reports must contain the same sample IDs")
    differences = []
    for sample_id in sorted(predicted):
        left, right = predicted[sample_id], oracle[sample_id]
        if (left.get("triage_level"), left.get("therapy_disposition")) != (
            right.get("triage_level"), right.get("therapy_disposition")
        ):
            differences.append({
                "sample_id": sample_id,
                "prediction": {
                    "triage_level": left.get("triage_level"),
                    "therapy_disposition": left.get("therapy_disposition"),
                },
                "oracle": {
                    "triage_level": right.get("triage_level"),
                    "therapy_disposition": right.get("therapy_disposition"),
                },
            })
    return {
        "sample_count": len(predicted),
        "disagreement_count": len(differences),
        "disagreements": differences,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()
    try:
        report = evaluate_release(args.runs, args.manifest, generated_at=generated_at)
    except ValueError as exc:
        print(f"evaluation failed: {exc}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["preflight"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
