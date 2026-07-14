"""Cached review labels -> concern-efficacy signal store.

The model-labeling pass is deliberately separate from aggregation: this command
consumes the append-only JSONL contract produced by the D-023 labeler. That
keeps deterministic store rebuilds free, resumable, and networkless.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..contracts import sha256_file
from .common import STORE_SCHEMA_VERSION, register_store, write_json

PROMPT_VERSION = "p7"
BUILDER = "recsys.tools.build_concern_efficacy@1"
ACNE_CONCERNS = frozenset((
    "acne_comedonal", "acne_inflammatory", "acne_cystic", "acne_general",
))


def _cell(counts: dict[str, int], prior: float, smoothing_m: float) -> dict:
    helped = counts.get("helped", 0)
    worsened = counts.get("worsened", 0)
    unclear = counts.get("unclear", 0)
    n = helped + worsened
    return {
        "n": n,
        "helped": helped,
        "worsened": worsened,
        "n_unclear": unclear,
        "help_rate": round(helped / n, 6) if n else None,
        "smoothed": round((helped + smoothing_m * prior) / (n + smoothing_m), 6),
    }


def build(
    labels_path: Path,
    out_path: Path,
    data_root: Path,
    *,
    catalog_products: int,
    catalog_product_ids: frozenset[str] | None = None,
    smoothing_m: float = 20,
    sub_cell_min_n: int = 5,
) -> dict:
    records = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    global_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    product_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    skin_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record.get("status") != "ok" or record.get("prompt_version") != PROMPT_VERSION:
            continue
        product_id = str(record["product_id"])
        if catalog_product_ids is not None and product_id not in catalog_product_ids:
            continue
        skin_type = str(record.get("skin_type") or "unknown")
        for label in record.get("labels") or []:
            concern = label.get("concern")
            outcome = label.get("outcome")
            if not concern or outcome not in {"helped", "worsened", "unclear"}:
                continue
            global_counts[concern][outcome] += 1
            product_counts[(product_id, concern)][outcome] += 1
            skin_counts[(product_id, concern, skin_type)][outcome] += 1

    priors = {
        concern: counts["helped"] / (counts["helped"] + counts["worsened"])
        for concern, counts in global_counts.items()
        if counts["helped"] + counts["worsened"]
    }
    products: dict[str, dict] = {}
    for (product_id, concern), counts in sorted(product_counts.items()):
        if concern not in priors:
            continue
        all_cell = _cell(counts, priors[concern], smoothing_m)
        by_skin_type = {}
        for (pid, cell_concern, skin_type), skin in sorted(skin_counts.items()):
            if pid != product_id or cell_concern != concern:
                continue
            cell = _cell(skin, priors[concern], smoothing_m)
            if cell["n"] >= sub_cell_min_n:
                by_skin_type[skin_type] = cell
        products.setdefault(product_id, {})[concern] = {
            "all": all_cell,
            "by_skin_type": by_skin_type,
        }

    acne_n15 = sum(
        any(concern in ACNE_CONCERNS and entry["all"]["n"] >= 15
            for concern, entry in concerns.items())
        for concerns in products.values()
    )
    payload = {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": "concern_efficacy",
        "version": "v1",
        "prompt_version": PROMPT_VERSION,
        "smoothing_m": smoothing_m,
        "sub_cell_min_n": sub_cell_min_n,
        "confidence_n": 20,
        "priors": dict(sorted((key, round(value, 6)) for key, value in priors.items())),
        "products": products,
    }
    write_json(out_path, payload)
    coverage = {
        "products": len(products),
        "catalog_products": catalog_products,
        "products_with_acne_cell_n15": acne_n15,
    }
    register_store(
        data_root, name="concern_efficacy", kind="concern_efficacy", version="v1",
        store_path=out_path, builder=BUILDER,
        source={"labels_sha256": sha256_file(labels_path), "prompt_version": PROMPT_VERSION},
        coverage=coverage,
    )
    return coverage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smoothing-m", type=float, default=20)
    parser.add_argument("--sub-cell-min-n", type=int, default=5)
    args = parser.parse_args(argv)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_product_ids = frozenset(
        str(product["product_id"])
        for product in catalog.get("products") or []
        if product.get("product_id") is not None
    )
    coverage = build(
        args.labels, args.out, args.data_root,
        catalog_products=len(catalog_product_ids),
        catalog_product_ids=catalog_product_ids,
        smoothing_m=args.smoothing_m,
        sub_cell_min_n=args.sub_cell_min_n,
    )
    print(coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
