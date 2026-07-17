"""Schema-validated, evidence-backed catalog verification overlay."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .catalog import CatalogProduct, _label_stated_actives
from .contracts import ContractViolation, sha256_file

SCHEMA_VERSION = "recsys-verification-1"
FRESHNESS_DAYS = {"regulatory_label": 180, "default": 90}
_APPROVED_PROVENANCE_FIELDS = (
    "reviewer_id", "reviewer_type", "approved_at", "source_url",
    "retrieved_at", "source_sha256",
)
_ALLOWED_FACT_KEYS = frozenset({
    "intended_areas", "routine_roles", "format", "exposure", "drug_actives",
    "otc_drug", "label_source", "label_verified_at", "broad_spectrum", "spf",
    "comedogenic_claim", "irritant_features", "contraindications", "evidence_roles",
    "evidence_grade", "cadence", "cadence_source", "amount", "amount_source",
    "source_set_id", "ndc_product_code", "label_version", "label_effective_date",
    "source_hash",
})
_ASSERTION_KEYS = frozenset({
    "status", "reviewer_id", "reviewer_type", "approved_at", "source_url",
    "retrieved_at", "source_sha256", "facts",
})
_FACT_ENUMS = {
    "intended_areas": frozenset({"face", "neck", "body", "eye", "lip", "unknown"}),
    # Must cover every role gates.py asks a slot for: contracts.SLOTS names five
    # slots and profile_gate_reasons maps each to a role (spf -> "sunscreen",
    # otherwise the slot's own name). The four-role vocabulary was inherited from
    # src.recommendation.schema, which has no serum slot; recsys does. A role the
    # enum cannot express is a slot no overlay can ever fill, so "serum" was not
    # thin evidence -- it was unreachable, and role_not_verified:serum vetoed the
    # slot at 100% verification just as surely as at 1%.
    "routine_roles": frozenset({
        "cleanser", "treatment", "serum", "moisturizer", "sunscreen",
    }),
    "exposure": frozenset({
        "unknown", "rinse_off", "short_contact", "leave_on", "mask", "scrub", "peel",
    }),
    "comedogenic_claim": frozenset({
        "unknown", "claimed_noncomedogenic", "not_claimed",
    }),
    "cadence": frozenset({
        "am", "pm", "am_pm", "daily", "once_daily", "twice_daily", "per_label",
    }),
}
_FACT_LIST_FIELDS = frozenset({
    "intended_areas", "routine_roles", "irritant_features", "contraindications",
    "evidence_roles",
})
_FACT_STRING_FIELDS = frozenset({
    "format", "evidence_grade",
})
_FACT_NULLABLE_STRING_FIELDS = frozenset({
    "label_source", "label_verified_at", "cadence_source", "amount", "amount_source",
    "source_set_id", "ndc_product_code", "label_version", "label_effective_date",
})
_FACT_BOOL_FIELDS = frozenset({"otc_drug", "broad_spectrum"})
_FACT_SEQUENCE_FIELDS = _FACT_LIST_FIELDS | {"drug_actives"}


def _timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _validate_timestamp(value: object, field: str, product_id: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractViolation(field, f"expected a non-empty ISO-8601 timestamp for {product_id}")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractViolation(field, f"invalid ISO-8601 timestamp for {product_id}") from exc
    if parsed.tzinfo is None:
        raise ContractViolation(field, f"timestamp must include a timezone for {product_id}")


def _validate_fact(key: str, value: object, product_id: str) -> None:
    field = f"verification.facts.{key}"
    if key in _FACT_LIST_FIELDS:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ContractViolation(field, f"expected a list of strings for {product_id}")
        if key in {"intended_areas", "routine_roles"}:
            allowed = _FACT_ENUMS[key]
            invalid = set(value) - allowed
            if invalid:
                raise ContractViolation(field, f"unknown values {sorted(invalid)} for {product_id}")
        return
    if key in _FACT_ENUMS:
        if key == "cadence" and value is None:
            return
        if not isinstance(value, str) or value not in _FACT_ENUMS[key]:
            raise ContractViolation(field, f"expected one of {sorted(_FACT_ENUMS[key])} for {product_id}")
        return
    if key in _FACT_STRING_FIELDS:
        if not isinstance(value, str) or not value.strip():
            raise ContractViolation(field, f"expected a non-empty string for {product_id}")
        return
    if key in _FACT_NULLABLE_STRING_FIELDS:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ContractViolation(field, f"expected a string or null for {product_id}")
        return
    if key in _FACT_BOOL_FIELDS:
        if value is None:
            return
        if not isinstance(value, bool):
            raise ContractViolation(field, f"expected a boolean for {product_id}")
        return
    if key == "spf":
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractViolation(field, f"expected a non-negative integer for {product_id}")
        return
    if key == "source_hash":
        if value is None:
            return
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ContractViolation(field, f"expected a SHA-256 hex digest for {product_id}")
        return
    if key == "drug_actives":
        if not isinstance(value, list):
            raise ContractViolation(field, f"expected a list of objects for {product_id}")
        for index, active in enumerate(value):
            active_field = f"{field}[{index}]"
            if (
                not isinstance(active, dict)
                or "name" not in active
                or set(active) - {"name", "strength", "source"}
            ):
                raise ContractViolation(
                    active_field,
                    "expected exactly name, strength, and source",
                )
            if not isinstance(active.get("name"), str) or not active["name"].strip():
                raise ContractViolation(
                    active_field,
                    "name must be a non-empty string",
                )
            for item in ("strength", "source"):
                value = active.get(item)
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                ):
                    raise ContractViolation(
                        active_field,
                        f"{item} must be a string or null",
                    )
        return
    raise ContractViolation(field, f"unsupported fact for {product_id}")


def _validate_approved_assertion(assertion: object, product_id: str) -> dict:
    if not isinstance(assertion, dict):
        raise ContractViolation("verification.assertion", f"expected an object for {product_id}")
    extra = set(assertion) - _ASSERTION_KEYS
    if extra:
        raise ContractViolation("verification.assertion", f"unknown fields {sorted(extra)} for {product_id}")
    for field in _APPROVED_PROVENANCE_FIELDS:
        if not isinstance(assertion.get(field), str) or not assertion[field].strip():
            raise ContractViolation(
                f"verification.{field}",
                f"required for approved assertion on {product_id}",
            )
    if assertion["reviewer_type"] not in {"human", "agent"}:
        raise ContractViolation("verification.reviewer_type", f"expected human or agent for {product_id}")
    _validate_timestamp(assertion["approved_at"], "verification.approved_at", product_id)
    _validate_timestamp(assertion["retrieved_at"], "verification.retrieved_at", product_id)
    digest = assertion["source_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ContractViolation("verification.source_sha256", f"expected a SHA-256 hex digest for {product_id}")
    facts = assertion.get("facts")
    if not isinstance(facts, dict) or not facts:
        raise ContractViolation("verification.facts", f"expected a non-empty object for {product_id}")
    unknown_facts = set(facts) - _ALLOWED_FACT_KEYS
    if unknown_facts:
        raise ContractViolation("verification.facts", f"unknown fields {sorted(unknown_facts)} for {product_id}")
    for key, value in facts.items():
        _validate_fact(key, value, product_id)
    return facts


def load_verification_overlay(
    root: str | Path,
    *,
    now: dt.datetime | None = None,
) -> tuple[dict[str, dict], list[str], dict | None]:
    root = Path(root)
    approved_path = root / "approved.json"
    if not approved_path.exists():
        return {}, ["no verification overlay"], None
    value = json.loads(approved_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ContractViolation("verification.schema_version", "expected recsys-verification-1")
    now = now or dt.datetime.now(dt.timezone.utc)
    overlay: dict[str, dict] = {}
    warnings: list[str] = []
    # Group across rows first: one product_id may legitimately appear in more
    # than one row (separate review batches), and approved_at -- not row order --
    # decides which facts win.
    approved: dict[str, list] = {}
    for product in value.get("products") or []:
        product_id = product.get("product_id")
        if not product_id:
            raise ContractViolation("verification.product_id", "missing")
        for assertion in product.get("assertions") or []:
            if not isinstance(assertion, dict):
                raise ContractViolation("verification.assertion", f"expected an object for {product_id}")
            if assertion.get("status") == "approved":
                approved.setdefault(product_id, []).append(
                    (assertion, _validate_approved_assertion(assertion, product_id)))
    for product_id, assertions in approved.items():
        assertions.sort(key=lambda item: _timestamp(item[0]["approved_at"]))
        for assertion, facts in assertions:
            source = assertion.get("source_url")
            digest = assertion.get("source_sha256")
            evidence_path = root / "evidence" / str(digest)
            if not digest or not evidence_path.exists():
                raise ContractViolation("verification.evidence", f"missing snapshot for {product_id}")
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != digest:
                raise ContractViolation("verification.evidence", f"hash mismatch for {product_id}")
            grade = facts.get("evidence_grade", "default")
            max_age = FRESHNESS_DAYS.get(grade, FRESHNESS_DAYS["default"])
            if now - _timestamp(assertion["retrieved_at"]) > dt.timedelta(days=max_age):
                warnings.append(f"verification_stale:{product_id}:{source}")
                continue
            patch = overlay.setdefault(product_id, {"_sources": []})
            patch.update(facts)
            patch["_sources"].append({"url": source, "sha256": digest, "grade": grade})
    return overlay, warnings, {
        "path": str(approved_path),
        "sha256": sha256_file(approved_path),
        "products": len(overlay),
    }


def apply_verification(
    products: list[CatalogProduct], overlay: dict[str, dict]
) -> list[CatalogProduct]:
    verified = []
    for product in products:
        facts = overlay.get(product.product_id)
        if not facts:
            verified.append(product)
            continue
        # A drug row's otc_drug came from its own FDA label; an overlay fact
        # flipping it to True would pull the product out of prescription_options
        # and into the ranked pool -- "listed, never placed" undone by one
        # boolean an agent misread off a marketing category. Refuse it outright.
        if product.drug_actives and "otc_drug" in facts:
            raise ContractViolation(
                "verification.otc_drug",
                f"refusing to override the label's own OTC status for {product.product_id}",
            )
        # The same bypass by deletion: an empty drug_actives list is valid by
        # shape (a list is a list) and skips the label-contract check below
        # because there is nothing to check -- but merged onto a drug row it
        # clears drug_actives while `actives` keeps the drug names, and a row
        # with no drug_actives is not a prescription. Un-prescriptioning a drug
        # takes fresh label evidence, not an eraser.
        if product.drug_actives and facts.get("drug_actives") == []:
            raise ContractViolation(
                "verification.drug_actives",
                f"refusing to clear the label's own drug_actives for {product.product_id}",
            )
        # Actives are minted through the same door the catalog enforces: each
        # active named, dosed, and cited to the label, or nothing merges. The
        # old bare-name union let {"name": "tretinoin"} with no strength and no
        # source add tretinoin to a cosmetic -- true of no committed assertion,
        # but a door that only the catalog file locked.
        verified_actives = _label_stated_actives({
            "drug_actives": facts.get("drug_actives", []),
            "label_source": facts.get("label_source", product.label_source),
        })
        if facts.get("drug_actives") and verified_actives is None:
            raise ContractViolation(
                "verification.drug_actives",
                f"drug_actives for {product.product_id} do not clear the label"
                " contract (name, strength, and a DailyMed citation per active)",
            )
        fields = set(product.__dataclass_fields__)
        updates = {
            key: tuple(value) if key in _FACT_SEQUENCE_FIELDS else value
            for key, value in facts.items()
            if key in fields
        }
        if "contraindications" in facts:
            updates["contraindications_verified"] = True
        if "daily_support" in (facts.get("evidence_roles") or []):
            updates["daily_support_verified"] = True
        updates["actives"] = tuple(sorted(set(product.actives) | set(verified_actives or ())))
        if "spf" in facts:
            updates["spf_source"] = "verified"
        verified.append(replace(product, **updates))
    return verified
