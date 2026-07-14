"""Kaggle reviews CSVs -> review_stats signal store. Deterministic, no LLM.

Per catalog product: review count, mean rating, Bayesian-smoothed rating
(smoothed toward the global mean with pseudo-count m), plus per-skin-type
sub-cells when a cell has at least --min-cell reviews. The dump is a frozen
snapshot, so this store is static until the code or dump changes.

    python -m recsys.tools.build_review_stats --raw-dir <dir> \
        --catalog recsys/data/catalog/seed_catalog.json \
        --out recsys/data/signals/review_stats.v1.json --data-root recsys/data
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ..contracts import SKIN_TYPES, sha256_file
from .common import DEFAULT_RAW_DIR, STORE_SCHEMA_VERSION, register_store, write_json

BUILDER = "recsys.tools.build_review_stats@1"
SNAPSHOT = "snapshot-2023"


def build(raw_dir: Path, catalog_path: Path, out_path: Path, data_root: Path,
          m: int = 20, min_cell: int = 20) -> dict:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    wanted = {p["product_id"] for p in catalog["products"]}
    review_csvs = sorted(raw_dir.glob("reviews_*.csv"))
    if not review_csvs:
        raise SystemExit(f"no reviews_*.csv under {raw_dir}")

    global_n = 0
    global_sum = 0.0
    counts: dict[str, list] = defaultdict(lambda: [0, 0.0])          # pid -> [n, sum]
    cells: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0.0])  # (pid, skin) -> [n, sum]
    for path in review_csvs:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rating = float(row.get("rating") or "")
                except ValueError:
                    continue
                global_n += 1
                global_sum += rating
                pid = row.get("product_id")
                if pid not in wanted:
                    continue
                counts[pid][0] += 1
                counts[pid][1] += rating
                skin = (row.get("skin_type") or "").strip().lower()
                if skin in SKIN_TYPES and skin != "unknown":
                    cells[(pid, skin)][0] += 1
                    cells[(pid, skin)][1] += rating

    global_mean = round(global_sum / global_n, 4) if global_n else 0.0

    def smoothed(n: int, total: float) -> float:
        return round((total + m * global_mean) / (n + m), 4)

    products: dict[str, dict] = {}
    for pid, (n, total) in counts.items():
        entry = {"n": n, "mean": round(total / n, 4), "smoothed": smoothed(n, total)}
        by_skin = {}
        for skin in SKIN_TYPES:
            cell_n, cell_sum = cells.get((pid, skin), (0, 0.0))
            if cell_n >= min_cell:
                by_skin[skin] = {
                    "n": cell_n,
                    "mean": round(cell_sum / cell_n, 4),
                    "smoothed": smoothed(cell_n, cell_sum),
                }
        if by_skin:
            entry["by_skin_type"] = by_skin
        products[pid] = entry

    payload = {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": "review_stats",
        "version": "v1",
        "signal_age": SNAPSHOT,
        "smoothing_m": m,
        "min_cell": min_cell,
        "global_mean": global_mean,
        "products": products,
    }
    write_json(out_path, payload)
    register_store(
        data_root, name="review_stats", kind="review_stats", version="v1",
        store_path=out_path, builder=BUILDER,
        source={"reviews_csv_sha256s": [sha256_file(p) for p in review_csvs]},
        coverage={"products": len(products), "catalog_products": len(wanted)},
    )
    log = {"reviews": global_n, "products_covered": len(products), "global_mean": global_mean}
    print(log)
    return log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--m", type=int, default=20, help="Bayesian pseudo-count")
    parser.add_argument("--min-cell", type=int, default=20)
    args = parser.parse_args(argv)
    build(args.raw_dir, args.catalog, args.out, args.data_root, args.m, args.min_cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
