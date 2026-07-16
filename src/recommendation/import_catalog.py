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
from pathlib import Path
from typing import Optional

from .schema import CATEGORIES, Product, excludes_face

# --- vocabularies (from CATALOG_SCHEMA.md) ---------------------------------
# normalized ingredient string -> canonical active ID. Keys are what
# normalize_token() produces (lowercase, single-spaced). Every canonical active
# from the doc's "Canonical actives" section appears here via its plain spelling,
# plus the two doc-named synonyms and a few obvious INCI variants.
CANONICAL_ACTIVES: dict[str, str] = {
    # acne / exfoliation
    "salicylic acid": "salicylic_acid",
    "betaine salicylate": "salicylic_acid",  # ester BHA (K-beauty), same class
    "gluconolactone": "gluconolactone",      # PHA exfoliant
    "willow": "willow_bark",                 # Salix (Willow) Bark Extract — botanical BHA source
    "willow bark extract": "willow_bark",
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
    # suffixed INCI codes (e2e finding 2026-07-13: "Ceramide NP" x250 in the
    # Sephora dump vs plain "Ceramides" x11 — without these, most ceramide
    # barrier products went untagged)
    "ceramide np": "ceramides",
    "ceramide ap": "ceramides",
    "ceramide eop": "ceramides",
    "ceramide ng": "ceramides",
    "ceramide ns": "ceramides",
    "ceramide eos": "ceramides",
    "hyaluronic acid": "hyaluronic_acid",
    "sodium hyaluronate": "hyaluronic_acid",  # doc-named synonym
    "glycerin": "glycerin",
    "glycerine": "glycerin",
    "glycerol": "glycerin",
    "squalane": "squalane",
    "panthenol": "panthenol",
    "centella": "centella",
    "centella asiatica": "centella",
    # real Sephora INCI forms (e2e finding 2026-07-13: bare "centella" never
    # appears in the dump — without these the soothe active matched 0 products)
    "centella asiatica extract": "centella",
    "centella asiatica leaf extract": "centella",
    "centella asiatica leaf water": "centella",
    "centella asiatica leaf cell extract": "centella",
    "centella asiatica meristem cell culture": "centella",
    "centella asiatica flower leaf stem extract": "centella",
    "hydrocotyl": "centella",   # paren alias in "Centella Asiatica (Hydrocotyl) Extract"
    "gotu kola": "centella",    # paren alias in "Centella asiatica (Gotu Kola) Extract"
    "cica": "centella",
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


# --- ingredient-KB enrichment (spec 2026-07-10-ingredient-kb) --------------
# Optional pass: when a KB is supplied, comedogenic flags become a superset of
# the hand-list above (KB-derived flags added) and each product gets a
# per-concern ingredient_match. ingredient_kb imports normalize_token from this
# module, so the import is lazy here to avoid a circular import at module load.
def enrich_product(product: Product, raw_ingredients: str, kb: dict) -> None:
    """Fold KB signal into a product in place: union the hand-list comedogenic
    flags with KB-derived ones, and attach ingredient_match {concern: float}."""
    from .ingredient_kb import kb_comedogenic_flags, product_matches
    flags = set(product.comedogenic_flags) | kb_comedogenic_flags(raw_ingredients, kb)
    product.comedogenic_flags = sorted(flags)
    product.ingredient_match = product_matches(raw_ingredients, kb)


def product_dict(product: Product) -> dict:
    """Explicit versioned serializer with legacy KB omissions preserved."""
    d = product.to_dict()
    if not d.get("ingredient_match"):
        d.pop("ingredient_match", None)
    if d.get("tier", 1) == 1:
        d.pop("tier", None)
    if not d.get("no_outcome_data"):
        d.pop("no_outcome_data", None)
    return d


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
    contract = _source_contract(row)
    return Product(
        product_id=product_id,
        name=(row.get("name") or "").strip(),
        brand=(row.get("brand") or "").strip(),
        category=category,
        actives=actives,
        comedogenic_flags=comedogenic,
        price_usd=_parse_price(row.get("price")),
        price_is_stale=True,
        catalog_schema_version="2",
        **contract,
    )


def _source_contract(row: dict) -> dict[str, object]:
    """Preserve source taxonomy facts without upgrading them to label proof."""
    secondary = row.get("source_secondary") or ""
    tertiary = row.get("source_tertiary") or ""
    if not secondary and not tertiary:
        return {}
    area = ["neck"] if tertiary == "Decollete & Neck Creams" else ["face"]
    key = (secondary, tertiary)
    metadata: dict[tuple[str, str], tuple[list[str], str, str]] = {
        ("Cleansers", "Face Wash & Cleansers"): ([], "cleanser", "rinse_off"),
        ("Cleansers", "Makeup Removers"): ([], "makeup_remover", "rinse_off"),
        ("Cleansers", "Face Wipes"): ([], "wipe", "rinse_off"),
        ("Cleansers", "Toners"): ([], "toner", "leave_on"),
        ("Cleansers", "Exfoliators"): ([], "scrub", "scrub"),
        ("Treatments", "Face Serums"): ([], "serum", "leave_on"),
        ("Treatments", "Facial Peels"): ([], "peel", "peel"),
        ("Treatments", "Blemish & Acne Treatments"): ([], "unknown", "unknown"),
        ("Masks", "Face Masks"): ([], "mask", "mask"),
        ("Masks", "Sheet Masks"): ([], "mask", "mask"),
        ("Moisturizers", "Moisturizers"): ([], "cream", "leave_on"),
        ("Moisturizers", "Mists & Essences"): ([], "mist", "leave_on"),
        ("Moisturizers", "Face Oils"): ([], "oil", "leave_on"),
        ("Moisturizers", "Night Creams"): ([], "cream", "leave_on"),
        ("Moisturizers", "Decollete & Neck Creams"): ([], "cream", "leave_on"),
        ("Sunscreen", "Face Sunscreen"): ([], "sunscreen", "leave_on"),
    }
    roles, fmt, exposure = metadata.get(key, ([], "unknown", "unknown"))
    return {
        "intended_areas": area,
        "routine_roles": roles,
        "format": fmt,
        "exposure": exposure,
    }


# --- Sephora adapter (the real Kaggle product_info.csv; D-015) -------------
# Feeds product_from_row(): rename the Sephora columns and map its taxonomy onto
# CATEGORIES, so everything downstream stays format-agnostic (D-009 unchanged).
#
# Keep only primary_category == "Skincare", then this exact-string table on
# (secondary, tertiary). Transcribed from the actual CSV, not from memory; the
# table, the non-obvious calls, and the drop policy live in CATALOG_SCHEMA.md.
SEPHORA_CATEGORY_MAP: dict[tuple[str, str], str] = {
    ("Cleansers", "Face Wash & Cleansers"): "cleanser",
    # Toners are LEAVE-ON (e2e 2026-07-13): an actives-bearing toner in the
    # rinse-off cleanser step misstates delivery — it is a treatment step.
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
        "source_secondary": key[0],
        "source_tertiary": key[1],
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


# --- beautyapi tier-2 adapter (thebeautyapi/beautyproducts JSONL) ----------
# The beautyapi `category` field is coarse (skincare/suncare/...), so the
# five-way catalog category is inferred from the product NAME (suncare short-
# circuits to spf). Rules are ordered: the first hit wins. Products whose
# category can't be inferred are dropped (spec deliverable 4). Heuristic and
# auditable; grow the keyword list rather than reaching for fuzzy matching.
_NAME_CATEGORY_RULES: list[tuple[str, str]] = [
    (r"sunscreen|\bspf\b|sun protection|\buv\b", "spf"),
    (r"cleanser|cleansing|face wash|micellar|makeup remover|foaming", "cleanser"),
    (r"\btoner\b", "treatment"),  # leave-on, same reasoning as the Sephora map
    (r"peel|exfoliat|\bmask\b|\bacne\b|blemish|clarifying|spot treatment"
     r"|\btreatment\b", "treatment"),
    (r"serum|ampoule|ampule|essence|elixir|\bdrops?\b|booster|concentrate", "serum"),
    (r"moisturiz|moisturis|\bcream\b|lotion|\bgel\b|balm|\bmist\b|emulsion"
     r"|\bbutter\b|hydrat|\boil\b", "moisturizer"),
]


def infer_beautyapi_category(name: str, category) -> Optional[str]:
    """Map a beautyapi product to one of CATEGORIES, or None to drop it."""
    if (category or "").strip().lower() == "suncare":
        return "spf"
    low = (name or "").lower()
    for pattern, cat in _NAME_CATEGORY_RULES:
        if re.search(pattern, low):
            return cat
    return None


def beautyapi_row_to_simple(raw: dict) -> Optional[dict]:
    """Map a beautyapi JSONL product to the importer's simple row shape, or None
    if its category can't be inferred. The INCI string is reconstructed from the
    structured ingredient entries in position order (so parse_ingredients and
    the KB pass work exactly as they do for the CSV formats)."""
    category = infer_beautyapi_category(raw.get("name"), raw.get("category"))
    if category is None:
        return None
    entries = sorted(raw.get("ingredients") or [],
                     key=lambda e: e.get("position") if e.get("position") is not None else 0)
    names = [(e.get("label_name") or e.get("name") or "").strip() for e in entries]
    ingredients = ", ".join(n for n in names if n)
    return {
        "product_id": f"b{raw.get('id')}",   # 'b' prefix keeps tier-2 ids disjoint
        "name": raw.get("name") or "",
        "brand": raw.get("brand") or "",
        "category": category,
        "ingredients": ingredients,
        "price": None,
    }


_VERIFICATION_FIELDS = {
    "intended_areas", "routine_roles", "format", "exposure", "drug_actives",
    "otc_drug", "label_source", "label_verified_at", "broad_spectrum", "spf",
    "comedogenic_claim", "irritant_features", "contraindications",
    "evidence_roles", "evidence_grade", "cadence", "cadence_source", "amount",
    "amount_source", "source_set_id", "ndc_product_code", "label_version",
    "label_effective_date", "source_hash",
}

_EVIDENCE_FIELDS = {
    "status", "source_url", "retrieved_at", "source_sha256", "reviewer_id",
    "reviewer_type", "approved_at", "facts",
}


def load_verification_overlay(path: str | Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"verification overlay {path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or str(value.get("schema_version")) != "2":
        raise ValueError("verification overlay: schema_version must be '2'")
    rows = value.get("products")
    if not isinstance(rows, list):
        raise ValueError("verification overlay: expected a list or {products: [...]} object")
    result: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        field_path = f"verification.products[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{field_path}: expected an object")
        product_id = row.get("product_id")
        if not isinstance(product_id, str) or not product_id:
            raise ValueError(f"{field_path}.product_id: expected a non-empty string")
        if product_id in result:
            raise ValueError(f"{field_path}.product_id: duplicate {product_id!r}")
        unknown = set(row) - {"product_id", "assertions"}
        if unknown:
            raise ValueError(f"verification product {product_id}: unknown fields {sorted(unknown)}")
        assertions = row.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            raise ValueError(f"verification product {product_id}.assertions: expected a list")
        patch: dict[str, object] = {}
        for assertion_index, assertion in enumerate(assertions):
            assertion_path = f"verification product {product_id}.assertions[{assertion_index}]"
            if not isinstance(assertion, dict):
                raise ValueError(f"{assertion_path}: expected an object")
            extra = set(assertion) - _EVIDENCE_FIELDS
            if extra:
                raise ValueError(f"{assertion_path}: unknown fields {sorted(extra)}")
            status = assertion.get("status")
            if status not in {"approved", "proposed", "stale"}:
                raise ValueError(f"{assertion_path}.status: expected approved, proposed, or stale")
            facts = assertion.get("facts")
            if not isinstance(facts, dict) or not facts:
                raise ValueError(f"{assertion_path}.facts: expected a non-empty object")
            fact_extra = set(facts) - _VERIFICATION_FIELDS
            if fact_extra:
                raise ValueError(f"{assertion_path}.facts: unknown fields {sorted(fact_extra)}")
            if status != "approved":
                continue
            for key in (
                "source_url", "retrieved_at", "source_sha256", "reviewer_id",
                "reviewer_type", "approved_at"
            ):
                if not isinstance(assertion.get(key), str) or not assertion[key]:
                    raise ValueError(f"{assertion_path}.{key}: expected a non-empty string")
            if assertion["reviewer_type"] not in {"human", "agent"}:
                raise ValueError(
                    f"{assertion_path}.reviewer_type: expected human or agent"
                )
            source_hash = assertion["source_sha256"]
            if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash.lower()):
                raise ValueError(f"{assertion_path}.source_sha256: expected a SHA-256 hex digest")
            overlap = set(patch) & set(facts)
            if overlap:
                raise ValueError(f"{assertion_path}.facts: duplicate approved fields {sorted(overlap)}")
            patch.update(facts)
        if patch:
            result[product_id] = patch
    return result


def apply_verification_overlay(
    products: list[Product],
    overlay: dict[str, dict[str, object]],
) -> tuple[list[Product], list[str]]:
    applied: list[Product] = []
    seen: set[str] = set()
    for product in products:
        patch = overlay.get(product.product_id)
        if patch is None:
            applied.append(product)
            continue
        merged = product.to_dict()
        merged.update(patch)
        # A verified drug active is also carried product content. Preserve the
        # complete normalized carried-active vocabulary used by safety checks.
        raw_drug = merged.get("drug_actives", [])
        carried = set(merged.get("actives", []))
        if isinstance(raw_drug, list):
            for active in raw_drug:
                if isinstance(active, dict) and isinstance(active.get("name"), str):
                    carried.add(active["name"])
        merged["actives"] = sorted(carried)
        try:
            applied.append(Product.from_dict(merged))
        except ValueError as exc:
            raise ValueError(f"verification product {product.product_id}: {exc}") from exc
        seen.add(product.product_id)
    return applied, sorted(set(overlay) - seen)


def _quarantine_reasons(product: Product, role: str) -> list[str]:
    reasons: list[str] = []
    if excludes_face(product.intended_areas):
        reasons.append("intended_area_not_face")
    if role not in product.routine_roles:
        reasons.append("routine_role_not_verified")
    if product.format == "unknown":
        reasons.append("format_unknown")
    if product.exposure == "unknown":
        reasons.append("exposure_unknown")
    if product.exposure in {"mask", "scrub", "peel"}:
        reasons.append("non_daily_format")
    if not product.cadence:
        reasons.append("instruction_cadence_unknown")
    if not product.cadence_source:
        reasons.append("instruction_cadence_source_missing")
    if role == "treatment":
        # D-033: OTC status no longer gates the treatment role
        if not product.drug_actives:
            reasons.append("drug_active_not_verified")
        if any(active.strength is None for active in product.drug_actives):
            reasons.append("drug_active_strength_missing")
        if any(active.source is None for active in product.drug_actives):
            reasons.append("drug_active_source_missing")
        if not product.label_source:
            reasons.append("label_source_missing")
        if not product.label_verified_at:
            reasons.append("label_verification_timestamp_missing")
    if role == "sunscreen":
        if product.broad_spectrum is not True:
            reasons.append("broad_spectrum_not_verified")
        if product.spf is None or product.spf < 30:
            reasons.append("spf_below_30_or_unknown")
        if not product.label_source:
            reasons.append("label_source_missing")
        if not product.label_verified_at:
            reasons.append("label_verification_timestamp_missing")
    if role in {"moisturizer", "sunscreen"} and (
        product.comedogenic_claim != "claimed_noncomedogenic"
    ):
        reasons.append("noncomedogenic_claim_not_verified")
    return reasons


def build_quarantine_report(
    products: list[Product],
    unmatched_verification_ids: list[str] | None = None,
) -> dict[str, object]:
    category_role = {
        "cleanser": "cleanser", "treatment": "treatment", "serum": "treatment",
        "moisturizer": "moisturizer", "spf": "sunscreen",
    }
    rows: dict[str, object] = {}
    for product in sorted(products, key=lambda item: item.product_id):
        roles = sorted(set(product.routine_roles) | {category_role[product.category]})
        quarantined = {
            role: reasons
            for role in roles
            if (reasons := _quarantine_reasons(product, role))
        }
        rows[product.product_id] = {
            "quarantined_roles": quarantined,
        }
    return {
        "schema_version": "1",
        "products": rows,
        "unmatched_verification_ids": sorted(unmatched_verification_ids or []),
    }


def build_completeness_report(
    products: list[Product], *, support_minimum: int = 25
) -> dict[str, object]:
    """Report verified support-role inventory without weakening eligibility."""
    roles = ("cleanser", "moisturizer", "sunscreen")
    counts = {
        role: sum(not _quarantine_reasons(product, role) for product in products)
        for role in roles
    }
    modeled = {
        "azelaic_acid_10": (("azelaic_acid", "10%"),),
        "benzoyl_peroxide_2_5": (("benzoyl_peroxide", "2.5%"),),
        "adapalene_0_1_benzoyl_peroxide_2_5": (
            ("adapalene", "0.1%"), ("benzoyl_peroxide", "2.5%")
        ),
    }
    treatment_counts = {key: 0 for key in modeled}
    for product in products:
        exact = tuple(sorted((active.name, active.strength) for active in product.drug_actives))
        for key, expected in modeled.items():
            if exact == tuple(sorted(expected)) and not _quarantine_reasons(product, "treatment"):
                treatment_counts[key] += 1
    support_complete = all(count >= support_minimum for count in counts.values())
    treatment_complete = all(treatment_counts.values())
    return {
        "schema_version": "1",
        "support_minimum": support_minimum,
        "eligible_by_role": counts,
        "shortfalls": {
            role: support_minimum - count
            for role, count in counts.items() if count < support_minimum
        },
        "treatment_paths": treatment_counts,
        "missing_treatment_paths": [
            key for key, count in treatment_counts.items() if not count
        ],
        "complete": support_complete and treatment_complete,
    }


def _write_quarantine(path: str | Path | None, report: dict[str, object]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def import_beautyapi(
    jsonl_path, out_path, kb: dict | None = None, *,
    verification: str | Path | None = None,
    quarantine_out: str | Path | None = None,
) -> dict:
    """Import the beautyproducts JSONL into a tier-2 catalog.json (same Product
    schema, plus tier=2 and no_outcome_data=True). Products that don't map to
    one of the five categories are dropped. Deterministic -> idempotent."""
    jsonl_path = Path(jsonl_path)
    out_path = Path(out_path)

    rows = 0
    dropped_category = 0
    dropped_by_category: Counter[str] = Counter()
    products: list[Product] = []
    for idx, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        rows += 1
        raw = json.loads(line)
        row = beautyapi_row_to_simple(raw)
        if row is None:
            dropped_category += 1
            dropped_by_category[(raw.get("category") or "(none)")] += 1
            continue
        product = product_from_row(row, idx)
        if product is None:
            dropped_category += 1
            continue
        product.tier = 2
        product.no_outcome_data = True
        if kb is not None:
            enrich_product(product, row["ingredients"], kb)
        products.append(product)

    products, unmatched = apply_verification_overlay(
        products, load_verification_overlay(verification)
    )
    kept = Counter(p.category for p in products)
    log: dict[str, object] = {
        "rows": rows,
        "kept": len(products),
        "dropped_category": dropped_category,
        "with_actives": sum(1 for p in products if p.actives),
        "dropped_by_category": dict(dropped_by_category.most_common()),
        "kept_by_category": {c: kept[c] for c in CATEGORIES if kept[c]},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([product_dict(p) for p in products], f, indent=2, sort_keys=True)
        f.write("\n")
    _write_quarantine(quarantine_out, build_quarantine_report(products, unmatched))
    print(log)
    return log


def import_csv(
    csv_path, out_path, fmt: str = "simple", kb: dict | None = None, *,
    verification: str | Path | None = None,
    quarantine_out: str | Path | None = None,
) -> dict:
    """Read a catalog CSV, normalize each row, write out_path as a JSON list of
    products, and return/print a log dict. Deterministic -> idempotent.

    fmt="simple" reads the importer's own five-column shape (unchanged).
    fmt="sephora" runs each row through the Sephora adapter first and adds a
    dropped-by-category breakdown + kept-by-category tally to the log.

    kb (optional): an ingredient KB from ingredient_kb.load_kb. When present,
    each product is enriched with KB-derived comedogenic flags + ingredient
    match scores; when absent the output is byte-identical to before (D-006)."""
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
            if kb is not None:
                enrich_product(product, row.get("ingredients") or "", kb)
            products.append(product)

    products, unmatched = apply_verification_overlay(
        products, load_verification_overlay(verification)
    )
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
        json.dump([product_dict(p) for p in products], f, indent=2, sort_keys=True)
        f.write("\n")
    _write_quarantine(quarantine_out, build_quarantine_report(products, unmatched))

    print(log)
    return log


def load_catalog(path) -> list[Product]:
    """Read a catalog.json back into Product objects — what the engine consumes."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("catalog: expected a JSON list")
    return [Product.from_dict(d) for d in data]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import a product CSV into a normalized catalog.json.",
    )
    parser.add_argument("--csv", required=True,
                        help="input CSV path (or beautyproducts JSONL for --format beautyapi)")
    parser.add_argument(
        "--format",
        choices=("simple", "sephora", "beautyapi"),
        default="simple",
        help="input row format (simple; sephora = Kaggle product_info.csv; "
             "beautyapi = tier-2 beautyproducts JSONL)",
    )
    parser.add_argument(
        "--verification", default=None,
        help="optional schema-validated product verification overlay",
    )
    parser.add_argument(
        "--quarantine-out", default=None,
        help="optional deterministic per-product role quarantine report",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: paths.catalog_processed from config)",
    )
    parser.add_argument(
        "--kb",
        default=None,
        help="optional ingredient_kb.json: enriches comedogenic flags + "
             "ingredient_match (spec 2026-07-10-ingredient-kb)",
    )
    args = parser.parse_args(argv)

    out = args.out
    if out is None:
        from src.config import load_config  # lazy: avoids importing yaml unless needed
        out = load_config()["paths"]["catalog_processed"]

    kb = None
    if args.kb:
        from .ingredient_kb import load_kb
        kb = load_kb(args.kb)

    if args.format == "beautyapi":
        import_beautyapi(
            args.csv, out, kb=kb, verification=args.verification,
            quarantine_out=args.quarantine_out,
        )
    else:
        import_csv(
            args.csv, out, fmt=args.format, kb=kb, verification=args.verification,
            quarantine_out=args.quarantine_out,
        )


if __name__ == "__main__":
    main()
