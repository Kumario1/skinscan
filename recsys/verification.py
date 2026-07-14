"""Schema-validated, evidence-backed catalog verification overlay."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .catalog import CatalogProduct
from .contracts import ContractViolation, sha256_file

SCHEMA_VERSION = "recsys-verification-1"
FRESHNESS_DAYS = {"regulatory_label": 180, "default": 90}


def _timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


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
    for product in value.get("products") or []:
        product_id = product.get("product_id")
        if not product_id:
            raise ContractViolation("verification.product_id", "missing")
        assertions = sorted(
            (a for a in product.get("assertions") or [] if a.get("status") == "approved"),
            key=lambda a: a.get("approved_at") or a.get("retrieved_at") or "",
        )
        for assertion in assertions:
            source = assertion.get("source_url")
            digest = assertion.get("source_sha256")
            evidence_path = root / "evidence" / str(digest)
            if not digest or not evidence_path.exists():
                raise ContractViolation("verification.evidence", f"missing snapshot for {product_id}")
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != digest:
                raise ContractViolation("verification.evidence", f"hash mismatch for {product_id}")
            facts = assertion.get("facts") or {}
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
        spf = facts.get("spf", product.spf)
        verified.append(replace(
            product,
            spf=int(spf) if spf is not None else None,
            spf_source="verified" if "spf" in facts else product.spf_source,
            broad_spectrum=facts.get("broad_spectrum", product.broad_spectrum),
            cadence=facts.get("cadence", product.cadence),
            contraindications=tuple(facts.get("contraindications") or product.contraindications),
            discontinued=bool(facts.get("discontinued", product.discontinued)),
        ))
    return verified
