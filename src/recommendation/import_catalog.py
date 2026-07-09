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
from collections import Counter
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
    """Build a Product from a simple row (columns: name, brand, category,
    ingredients, price, plus an optional product_id). Returns None if the
    category is not in the closed vocabulary (dropped at import).

    This is the importer's per-row seam (D-015). The real Kaggle Sephora dump
    has different column names and a broader taxonomy, so `sephora_row_to_simple`
    below renames its columns and maps its categories onto CATEGORIES *before*
    this point — every line here stays format-agnostic. `product_id` is passed
    through when present (the Sephora id is load-bearing for joining reviews);
    synthesized from the row index otherwise.
    """
    category = (row.get("category") or "").strip().lower()
    if category not in CATEGORIES:
        return None
    actives, comedogenic = parse_ingredients(row.get("ingredients") or "")
    # Honor a source-supplied id (the Sephora adapter passes one through —
    # load-bearing for joining reviews); simple rows have none, so synthesize.
    product_id = (row.get("product_id") or "").strip() or f"p{idx:05d}"
    return Product(
        product_id=product_id,
        name=(row.get("name") or "").strip(),
        brand=(row.get("brand") or "").strip(),
        category=category,
        actives=actives,
        comedogenic_flags=comedogenic,
        price_usd=_parse_price(row.get("price")),
        price_is_stale=True,
    )


# --- Sephora adapter (the real Kaggle product_info.csv; D-015) -------------
# Feeds product_from_row(): rename the Sephora columns and map its taxonomy onto
# CATEGORIES, so everything downstream stays format-agnostic (D-009 unchanged).
#
# Keep only primary_category == "Skincare", then this exact-string table on
# (secondary, tertiary). Transcribed from the actual CSV, not from memory; the
# table, the non-obvious calls, and the drop policy live in CATALOG_SCHEMA.md.
SEPHORA_CATEGORY_MAP: dict[tuple[str, str], str] = {
    ("Cleansers", "Face Wash & Cleansers"): "cleanser",
    ("Cleansers", "Toners"): "cleanser",
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


def sephora_row_to_simple(raw: dict) -> Optional[dict]:
    """Map a raw Sephora product_info.csv row to the importer's simple row shape,
    or return None if it isn't a mappable face-routine skincare product (wrong
    primary category, or a (secondary, tertiary) pair not in the table)."""
    if (raw.get("primary_category") or "").strip() != "Skincare":
        return None
    key = ((raw.get("secondary_category") or "").strip(),
           (raw.get("tertiary_category") or "").strip())
    category = SEPHORA_CATEGORY_MAP.get(key)
    if category is None:
        return None
    return {
        "product_id": (raw.get("product_id") or "").strip(),
        "name": raw.get("product_name") or "",
        "brand": raw.get("brand_name") or "",
        "category": category,
        "ingredients": raw.get("ingredients") or "",
        "price": raw.get("price_usd"),
    }


def _sephora_drop_label(raw: dict) -> str:
    """A glanceable reason a Sephora row was dropped: the primary category for
    non-skincare, else the full "Skincare / secondary / tertiary" pair."""
    prim = (raw.get("primary_category") or "").strip()
    if prim != "Skincare":
        return prim or "(uncategorized)"
    sec = (raw.get("secondary_category") or "").strip()
    ter = (raw.get("tertiary_category") or "").strip()
    return f"Skincare / {sec} / {ter}"


def import_csv(csv_path, out_path, fmt: str = "simple") -> dict:
    """Read a catalog CSV, normalize each row, write out_path as a JSON list of
    products, and return/print a log dict. Deterministic -> idempotent.

    fmt="simple" reads the importer's own five-column shape (unchanged).
    fmt="sephora" runs each row through the Sephora adapter first and adds a
    dropped-by-category breakdown + kept-by-category tally to the log."""
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    rows = 0
    dropped_category = 0
    dropped_by_category: Counter[str] = Counter()
    products: list[Product] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for idx, raw in enumerate(csv.DictReader(f)):
            rows += 1
            if fmt == "sephora":
                row = sephora_row_to_simple(raw)
                if row is None:
                    dropped_category += 1
                    dropped_by_category[_sephora_drop_label(raw)] += 1
                    continue
            else:
                row = raw
            product = product_from_row(row, idx)
            if product is None:
                dropped_category += 1
                continue
            products.append(product)

    with_actives = sum(1 for p in products if p.actives)
    log: dict[str, object] = {
        "rows": rows,
        "kept": len(products),
        "dropped_category": dropped_category,
        "with_actives": with_actives,
        "zero_actives": len(products) - with_actives,
    }
    if fmt == "sephora":
        # both breakdowns get a stable, glanceable order: drops by size,
        # keeps in canonical routine order.
        kept = Counter(p.category for p in products)
        log["dropped_by_category"] = dict(dropped_by_category.most_common())
        log["kept_by_category"] = {c: kept[c] for c in CATEGORIES if kept[c]}

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
        "--format",
        choices=("simple", "sephora"),
        default="simple",
        help="input row format (default: simple; sephora = Kaggle product_info.csv)",
    )
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

    import_csv(args.csv, out, fmt=args.format)


if __name__ == "__main__":
    main()
