"""product_info.csv -> popularity signal store. Deterministic, no LLM.

Per catalog product: loves_count and its percentile within the SAME category
across the FULL dump (not just the catalog subset), so "top 20% most-loved"
stays honest at any catalog scale. Marked snapshot-2023 so explanations can be
honest about staleness.

    python -m recsys.tools.build_popularity --raw-dir <dir> \
        --catalog recsys/data/catalog/seed_catalog.json \
        --out recsys/data/signals/popularity.v1.json --data-root recsys/data
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ..contracts import sha256_file
from .build_catalog import SEPHORA_CATEGORY_MAP
from .common import DEFAULT_RAW_DIR, STORE_SCHEMA_VERSION, register_store, write_json

BUILDER = "recsys.tools.build_popularity@1"
SNAPSHOT = "snapshot-2023"


def build(raw_dir: Path, catalog_path: Path, out_path: Path, data_root: Path) -> dict:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    wanted = {p["product_id"] for p in catalog["products"]}
    csv_path = raw_dir / "product_info.csv"

    loves_by_category: dict[str, list[int]] = defaultdict(list)
    loves_by_pid: dict[str, tuple[str, int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if (raw.get("primary_category") or "").strip() != "Skincare":
                continue
            key = ((raw.get("secondary_category") or "").strip(),
                   (raw.get("tertiary_category") or "").strip())
            category = SEPHORA_CATEGORY_MAP.get(key)
            if category is None:
                continue
            try:
                loves = int(raw.get("loves_count") or 0)
            except ValueError:
                loves = 0
            loves_by_category[category].append(loves)
            pid = (raw.get("product_id") or "").strip()
            if pid in wanted:
                loves_by_pid[pid] = (category, loves)

    products: dict[str, dict] = {}
    for pid, (category, loves) in loves_by_pid.items():
        peers = loves_by_category[category]
        below = sum(1 for value in peers if value < loves)
        percentile = round(below / (len(peers) - 1), 4) if len(peers) > 1 else 0.5
        products[pid] = {"loves": loves, "category_percentile": min(percentile, 1.0)}

    payload = {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": "popularity",
        "version": "v1",
        "signal_age": SNAPSHOT,
        "percentile_basis": "full-dump same-category loves_count",
        "products": products,
    }
    write_json(out_path, payload)
    register_store(
        data_root, name="popularity", kind="popularity", version="v1",
        store_path=out_path, builder=BUILDER,
        source={
            "catalog_sha256": sha256_file(catalog_path),
            "product_info_sha256": sha256_file(csv_path),
        },
        coverage={"products": len(products), "catalog_products": len(wanted)},
    )
    log = {"products_covered": len(products)}
    print(log)
    return log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.raw_dir, args.catalog, args.out, args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
