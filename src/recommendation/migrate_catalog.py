"""Storage-only migration of legacy catalog JSON to schema v2 unknowns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import Product


def migrate_catalog_v2(source: Path, destination: Path) -> dict[str, int]:
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("catalog: expected a JSON list")
    products = [Product.from_dict(row) for row in value]
    migrated = 0
    for row, product in zip(value, products):
        version = str(row.get("catalog_schema_version", "legacy")).lower()
        if version in {"1", "legacy"}:
            product.catalog_schema_version = "2"
            # Schema migration is storage normalization, never verification.
            product.intended_areas = []
            product.routine_roles = []
            product.format = "unknown"
            product.exposure = "unknown"
            product.drug_actives = []
            product.otc_drug = None
            product.label_source = None
            product.label_verified_at = None
            product.broad_spectrum = None
            product.spf = None
            product.comedogenic_claim = "unknown"
            product.evidence_roles = []
            product.evidence_grade = "unknown"
            product.cadence = None
            product.cadence_source = None
            product.amount = None
            product.amount_source = None
            migrated += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([product.to_dict() for product in products], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"products": len(products), "migrated": migrated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(migrate_catalog_v2(args.source, args.destination))


if __name__ == "__main__":
    main()
