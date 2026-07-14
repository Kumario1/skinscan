"""Kaggle Sephora product_info.csv -> catalog JSON (full or seed).

Deterministic and timestamp-free: same dump + same code -> byte-identical
output (tested). The catalog is product identity + details ONLY — review
stats, popularity, and every other signal live in their own stores.

    python -m recsys.tools.build_catalog --raw-dir <dir> --out recsys/data/derived/catalog_full.json
    python -m recsys.tools.build_catalog --raw-dir <dir> \
        --only-ids recsys/data/catalog/seed_ids.txt --out recsys/data/catalog/seed_catalog.json

SEPHORA_CATEGORY_MAP copied from src/recommendation/import_catalog.py —
transcribed from the actual CSV taxonomy, not from memory (toners are LEAVE-ON
and therefore treatments, per the 2026-07-13 e2e finding).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from ..catalog import CATALOG_SCHEMA_VERSION
from ..contracts import sha256_file
from ..inci import parse_ingredients
from .common import DEFAULT_RAW_DIR, write_json

BUILDER_VERSION = "recsys.tools.build_catalog@1"

SEPHORA_CATEGORY_MAP: dict[tuple[str, str], str] = {
    ("Cleansers", "Face Wash & Cleansers"): "cleanser",
    ("Cleansers", "Toners"): "treatment",
    ("Cleansers", "Makeup Removers"): "cleanser",
    ("Cleansers", "Face Wipes"): "cleanser",
    ("Cleansers", ""): "cleanser",
    ("Cleansers", "Exfoliators"): "treatment",
    ("Treatments", "Face Serums"): "serum",
    ("Treatments", "Facial Peels"): "treatment",
    ("Treatments", "Blemish & Acne Treatments"): "treatment",
    ("Masks", "Face Masks"): "treatment",
    ("Masks", "Sheet Masks"): "treatment",
    ("Moisturizers", "Moisturizers"): "moisturizer",
    ("Moisturizers", "Mists & Essences"): "moisturizer",
    ("Moisturizers", "Face Oils"): "moisturizer",
    ("Moisturizers", "Night Creams"): "moisturizer",
    ("Moisturizers", "Decollete & Neck Creams"): "moisturizer",
    ("Moisturizers", ""): "moisturizer",
    ("Sunscreen", "Face Sunscreen"): "spf",
    ("Sunscreen", ""): "spf",
}

_SPF_RE = re.compile(r"\bspf\s*(\d{1,3})", re.IGNORECASE)

_FORMAT_WORDS = (
    "gel", "cream", "foam", "lotion", "serum", "oil", "balm", "stick", "mist",
    "fluid", "wash", "bar", "toner", "essence", "peel", "mask",
)


def _inci_tokens(raw: str) -> list[str]:
    """The dump's ingredients column is a stringified Python list. Strip the
    list syntax, split on commas, keep non-empty tokens."""
    cleaned = re.sub(r"[\[\]'\"]", " ", raw or "")
    return [t.strip() for t in cleaned.split(",") if t.strip()]


def _parse_spf(name: str) -> int | None:
    match = _SPF_RE.search(name or "")
    return int(match.group(1)) if match else None


def _parse_format(name: str) -> str | None:
    low = (name or "").lower()
    for word in _FORMAT_WORDS:
        if re.search(rf"\b{word}\b", low):
            return word
    return None


def _parse_price(raw) -> float | None:
    try:
        return float(str(raw).strip()) if raw not in (None, "") else None
    except ValueError:
        return None


def row_to_product(raw: dict) -> dict | None:
    if (raw.get("primary_category") or "").strip() != "Skincare":
        return None
    key = ((raw.get("secondary_category") or "").strip(),
           (raw.get("tertiary_category") or "").strip())
    category = SEPHORA_CATEGORY_MAP.get(key)
    if category is None:
        return None
    product_id = (raw.get("product_id") or "").strip()
    if not product_id:
        return None
    name = (raw.get("product_name") or "").strip()
    inci = _inci_tokens(raw.get("ingredients") or "")
    actives, _comedogenic = parse_ingredients(raw.get("ingredients") or "")
    spf = _parse_spf(name)
    return {
        "product_id": product_id,
        "name": name,
        "brand": (raw.get("brand_name") or "").strip(),
        "category": category,
        "price_usd": _parse_price(raw.get("price_usd")),
        "size": (raw.get("size") or "").strip() or None,
        "format": _parse_format(name),
        "spf": spf,
        "spf_source": "name_parse" if spf is not None else None,
        "inci": inci,
        "inci_sha256": hashlib.sha256(
            json.dumps(inci, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "actives": actives,
    }


def build(raw_dir: Path, out_path: Path, only_ids: Path | None = None) -> dict:
    csv_path = raw_dir / "product_info.csv"
    wanted: set[str] | None = None
    if only_ids is not None:
        wanted = {
            line.split("#", 1)[0].strip()
            for line in only_ids.read_text(encoding="utf-8").splitlines()
        }
        wanted.discard("")

    products = []
    rows = 0
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows += 1
            product = row_to_product(raw)
            if product is None:
                continue
            if wanted is not None and product["product_id"] not in wanted:
                continue
            products.append(product)
    products.sort(key=lambda p: p["product_id"])

    if wanted is not None:
        missing = wanted - {p["product_id"] for p in products}
        if missing:
            raise SystemExit(f"seed ids not found in dump: {sorted(missing)}")

    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source": {
            "dataset": "kaggle-sephora",
            "product_info_sha256": sha256_file(csv_path),
        },
        "builder_version": BUILDER_VERSION,
        "products": products,
    }
    write_json(out_path, payload)
    log = {"rows": rows, "kept": len(products),
           "by_category": {c: sum(1 for p in products if p["category"] == c)
                           for c in ("cleanser", "treatment", "serum", "moisturizer", "spf")}}
    print(log)
    return log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only-ids", type=Path, default=None,
                        help="seed id list (one id per line, # comments allowed)")
    args = parser.parse_args(argv)
    build(args.raw_dir, args.out, args.only_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
