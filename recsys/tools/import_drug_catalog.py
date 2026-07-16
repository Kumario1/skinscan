"""data/processed/catalog_drug.json (DailyMed rows) -> a recsys drug catalog.

Kept as a catalog of its own rather than merged into catalog_full.json: the
signal stores are keyed by the cosmetics catalog's sha256, and folding drug rows
into that file would change the hash and silently strand every store.

    python -m recsys.tools.import_drug_catalog \
        --source data/processed/catalog_drug.json \
        --out recsys/data/derived/catalog_drug.json

This command never approves facts. It copies what the label already stated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..catalog import CATALOG_SCHEMA_VERSION, load_catalog
from .common import write_json

BUILDER_VERSION = "recsys.tools.import_drug_catalog@1"
EMPTY_INCI_SHA256 = hashlib.sha256(json.dumps([], ensure_ascii=False).encode()).hexdigest()


def to_row(product: dict) -> dict:
    """One src-side Product dict -> one recsys catalog row."""
    actives = sorted({item["name"] for item in product.get("drug_actives") or []})
    return {
        "product_id": product["product_id"],
        "name": product.get("name") or "",
        "brand": product.get("brand") or "",
        "category": product.get("category") or "treatment",
        # A prescription has no shelf price, and the pharmacy price depends on
        # insurance. Leaving it null keeps it out of every price-capped routine.
        "price_usd": None,
        "size": product.get("size"),
        "format": product.get("format"),
        "spf": None,
        "spf_source": None,
        "inci": [],                       # a drug label publishes no INCI list
        "inci_sha256": EMPTY_INCI_SHA256,
        "actives": actives,
        "drug_actives": product.get("drug_actives") or [],
        "otc_drug": product.get("otc_drug"),
        "label_source": product.get("label_source"),
        "label_verified_at": product.get("label_verified_at"),
        "cadence": product.get("cadence"),
        "cadence_source": product.get("cadence_source"),
        "intended_areas": product.get("intended_areas") or [],
        "exposure": product.get("exposure"),
        "routine_roles": product.get("routine_roles") or [],
        "evidence_grade": product.get("evidence_grade"),
    }


def build(source: Path, out: Path) -> dict:
    rows = json.loads(source.read_text(encoding="utf-8"))
    products = sorted((to_row(row) for row in rows), key=lambda row: row["product_id"])
    document = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "source": {"path": str(source), "sha256": hashlib.sha256(
            source.read_bytes()).hexdigest()},
        "products": products,
    }
    write_json(out, document)
    loaded, _ = load_catalog(out)  # fail loudly rather than write an invalid catalog
    return {"rows": len(rows), "kept": len(loaded),
            "prescription": sum(1 for row in products if row["otc_drug"] is False)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/processed/catalog_drug.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("recsys/data/derived/catalog_drug.json"))
    args = parser.parse_args(argv)
    if not args.source.exists():
        raise SystemExit(f"no drug catalog at {args.source} — run "
                         "`python -m src.recommendation.verification_loop rebuild` first")
    print(build(args.source, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
