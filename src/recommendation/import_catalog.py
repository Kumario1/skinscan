"""Catalog importer — raw CSV -> normalized catalog.json (D-009).

Turns messy product rows into the shape CATALOG_SCHEMA.md defines: the free-text
ingredient string is parsed ONCE, here, into a canonical `actives` list plus a
comedogenic flag list, so the recommender never parses ingredients at query time
(D-006). Unmappable categories are dropped; zero-active products are kept (valid
carriers, e.g. plain moisturizers). Stdlib only (csv, not pandas) per repo
convention.

Vocabularies below are transcribed from CATALOG_SCHEMA.md, not from memory.

ponytail: matching is exact-after-normalization plus this synonym table — no
fuzzy/edit-distance. If real-world INCI misses matter, the upgrade path is to
grow the synonym table first, and add fuzzy matching only as a last resort.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .schema import CATEGORIES, Product

# --- vocabularies (from CATALOG_SCHEMA.md) ---------------------------------
# normalized ingredient string -> canonical active ID. Keys are what
# normalize_token() produces (lowercase, single-spaced). Every canonical active
# from the doc's "Canonical actives" section appears here via its plain spelling,
# plus the two doc-named synonyms and a few obvious INCI variants.
CANONICAL_ACTIVES: dict[str, str] = {
    # acne / exfoliation
    "salicylic acid": "salicylic_acid",
    "benzoyl peroxide": "benzoyl_peroxide",
    "adapalene": "adapalene",
    "azelaic acid": "azelaic_acid",
    "glycolic acid": "glycolic_acid",
    "lactic acid": "lactic_acid",
    "mandelic acid": "mandelic_acid",
    # pigmentation
    "niacinamide": "niacinamide",
    "vitamin c": "vitamin_c",
    "ascorbic acid": "vitamin_c",          # doc-named synonym
    "alpha arbutin": "alpha_arbutin",
    "arbutin": "alpha_arbutin",
    "tranexamic acid": "tranexamic_acid",
    "kojic acid": "kojic_acid",
    "retinol": "retinol",
    # barrier / hydration
    "ceramides": "ceramides",
    "ceramide": "ceramides",
    "hyaluronic acid": "hyaluronic_acid",
    "sodium hyaluronate": "hyaluronic_acid",  # doc-named synonym
    "glycerin": "glycerin",
    "glycerine": "glycerin",
    "glycerol": "glycerin",
    "squalane": "squalane",
    "panthenol": "panthenol",
    "centella": "centella",
    "centella asiatica": "centella",
    # soothing
    "allantoin": "allantoin",
    "madecassoside": "madecassoside",
    "zinc": "zinc",
}
CANONICAL_IDS = set(CANONICAL_ACTIVES.values())

# From CATALOG_SCHEMA.md "Comedogenic flag list". The doc's final line —
# "certain cocoa/wheat-germ derivatives" — is intentionally omitted: it names no
# exact INCI string to match on, and we parse only what we can pin down.
COMEDOGENIC: dict[str, str] = {
    "coconut oil": "coconut_oil",
    "isopropyl myristate": "isopropyl_myristate",
    "isopropyl palmitate": "isopropyl_palmitate",
    "algae extract": "algae_extract",
}
COMEDOGENIC_IDS = set(COMEDOGENIC.values())


# --- normalization ---------------------------------------------------------
def normalize_token(s: str) -> list[str]:
    """Lowercase, punctuation/number-tolerant candidate strings for one token.

    Parenthetical aliases yield extra candidates: "Ascorbic Acid (Vitamin C)"
    -> ["ascorbic acid", "vitamin c"]. Everything non-alphabetic collapses to a
    single space (drops "2.5%"-style noise); order preserved, deduped.
    """
    s = s.lower()
    parts = re.findall(r"\(([^)]*)\)", s)         # inner text of each paren group
    parts.insert(0, re.sub(r"\([^)]*\)", " ", s))  # outer text, parens removed
    candidates: list[str] = []
    for part in parts:
        cleaned = re.sub(r"[^a-z]+", " ", part).strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    return candidates


def _lookup(cand: str, table: dict[str, str], ids: set[str]) -> Optional[str]:
    """Match a normalized candidate against a vocabulary table, then fall back to
    the snake_case ID form ("vitamin c" -> "vitamin_c")."""
    hit = table.get(cand)
    if hit is None:
        snake = cand.replace(" ", "_")
        if snake in ids:
            hit = snake
    return hit


def parse_ingredients(raw: str) -> tuple[list[str], list[str]]:
    """Split an INCI string on commas and pull out the actives and comedogenic
    flags we recognize. Unrecognized tokens are silently dropped (parse only
    what we use). Returns (sorted unique actives, sorted unique flags)."""
    actives: set[str] = set()
    comedogenic: set[str] = set()
    for token in raw.split(","):
        for cand in normalize_token(token):
            active = _lookup(cand, CANONICAL_ACTIVES, CANONICAL_IDS)
            if active:
                actives.add(active)
            flag = _lookup(cand, COMEDOGENIC, COMEDOGENIC_IDS)
            if flag:
                comedogenic.add(flag)
    return sorted(actives), sorted(comedogenic)


def _parse_price(raw) -> Optional[float]:
    """Prices are decorative (D-001): a float if it parses cleanly, else None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# --- row -> product --------------------------------------------------------
def product_from_row(row: dict, idx: int) -> Optional[Product]:
    """Build a Product from a fixture row (columns:
    name,brand,category,ingredients,price). Returns None if the category is not
    in the closed vocabulary (dropped at import).

    Follow-up (D-015): the real Kaggle Sephora dump uses different column names
    and a broader category taxonomy. A `row_adapter` that renames its columns to
    the five above and maps its categories onto CATEGORIES belongs right here,
    at this function's input — the rest of the importer stays untouched.
    """
    category = (row.get("category") or "").strip().lower()
    if category not in CATEGORIES:
        return None
    actives, comedogenic = parse_ingredients(row.get("ingredients") or "")
    return Product(
        product_id=f"p{idx:05d}",
        name=(row.get("name") or "").strip(),
        brand=(row.get("brand") or "").strip(),
        category=category,
        actives=actives,
        comedogenic_flags=comedogenic,
        price_usd=_parse_price(row.get("price")),
        price_is_stale=True,
    )


def import_csv(csv_path, out_path) -> dict:
    """Read a catalog CSV, normalize each row, write out_path as a JSON list of
    products, and return/print a log dict. Deterministic -> idempotent."""
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    rows = 0
    dropped_category = 0
    products: list[Product] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            rows += 1
            product = product_from_row(row, idx)
            if product is None:
                dropped_category += 1
                continue
            products.append(product)

    with_actives = sum(1 for p in products if p.actives)
    log = {
        "rows": rows,
        "kept": len(products),
        "dropped_category": dropped_category,
        "with_actives": with_actives,
        "zero_actives": len(products) - with_actives,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in products], f, indent=2, sort_keys=True)
        f.write("\n")

    print(log)
    return log


def load_catalog(path) -> list[Product]:
    """Read a catalog.json back into Product objects — what the engine consumes."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Product(**d) for d in data]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import a product CSV into a normalized catalog.json.",
    )
    parser.add_argument("--csv", required=True, help="input CSV path")
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: paths.catalog_processed from config)",
    )
    args = parser.parse_args(argv)

    out = args.out
    if out is None:
        from src.config import load_config  # lazy: avoids importing yaml unless needed
        out = load_config()["paths"]["catalog_processed"]

    import_csv(args.csv, out)


if __name__ == "__main__":
    main()
