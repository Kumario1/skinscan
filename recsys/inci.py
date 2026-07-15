"""INCI ingredient parsing: canonical actives + comedogenic flags.

COPIED from src/recommendation/import_catalog.py (the CANONICAL_ACTIVES synonym
table encodes hard-won fixes against the real 8,494-row Sephora dump — e.g.
"Ceramide NP" x250 vs plain "Ceramides" x11, and six centella INCI variants).
Do not edit from memory; grow the table from observed dump strings.

ponytail: matching is exact-after-normalization plus this synonym table — no
fuzzy/edit-distance. If real-world INCI misses matter, grow the synonym table
first; fuzzy matching only as a last resort.
"""
from __future__ import annotations

import re
from typing import Optional

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
    "ascorbic acid": "vitamin_c",
    "alpha arbutin": "alpha_arbutin",
    "arbutin": "alpha_arbutin",
    "tranexamic acid": "tranexamic_acid",
    "kojic acid": "kojic_acid",
    "retinol": "retinol",
    # barrier / hydration
    "ceramides": "ceramides",
    "ceramide": "ceramides",
    "ceramide np": "ceramides",
    "ceramide ap": "ceramides",
    "ceramide eop": "ceramides",
    "ceramide ng": "ceramides",
    "ceramide ns": "ceramides",
    "ceramide eos": "ceramides",
    "hyaluronic acid": "hyaluronic_acid",
    "sodium hyaluronate": "hyaluronic_acid",
    "glycerin": "glycerin",
    "glycerine": "glycerin",
    "glycerol": "glycerin",
    "squalane": "squalane",
    "panthenol": "panthenol",
    "centella": "centella",
    "centella asiatica": "centella",
    "centella asiatica extract": "centella",
    "centella asiatica leaf extract": "centella",
    "centella asiatica leaf water": "centella",
    "centella asiatica leaf cell extract": "centella",
    "centella asiatica meristem cell culture": "centella",
    "centella asiatica flower leaf stem extract": "centella",
    "hydrocotyl": "centella",
    "gotu kola": "centella",
    "cica": "centella",
    # soothing
    "allantoin": "allantoin",
    "madecassoside": "madecassoside",
    "zinc": "zinc",
}
CANONICAL_IDS = set(CANONICAL_ACTIVES.values())

COMEDOGENIC: dict[str, str] = {
    "coconut oil": "coconut_oil",
    "isopropyl myristate": "isopropyl_myristate",
    "isopropyl palmitate": "isopropyl_palmitate",
    "algae extract": "algae_extract",
}
COMEDOGENIC_IDS = set(COMEDOGENIC.values())


def normalize_token(s: str) -> list[str]:
    """Lowercase, punctuation/number-tolerant candidate strings for one token.

    Parenthetical aliases yield extra candidates: "Ascorbic Acid (Vitamin C)"
    -> ["ascorbic acid", "vitamin c"]. Everything non-alphabetic collapses to a
    single space (drops "2.5%"-style noise); order preserved, deduped.
    """
    s = s.lower()
    parts = re.findall(r"\(([^)]*)\)", s)
    parts.insert(0, re.sub(r"\([^)]*\)", " ", s))
    candidates: list[str] = []
    for part in parts:
        cleaned = re.sub(r"[^a-z]+", " ", part).strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    return candidates


def _lookup(cand: str, table: dict[str, str], ids: set[str]) -> Optional[str]:
    hit = table.get(cand)
    if hit is None:
        snake = cand.replace(" ", "_")
        if snake in ids:
            hit = snake
    return hit


# Allergen aliases that must NOT go in CANONICAL_ACTIVES (that table drives
# catalog INCI parsing, where bare "BHA" usually means the antioxidant
# butylated hydroxyanisole, not beta hydroxy acid). In a user's declared-allergy
# context "BHA" unambiguously means salicylic acid, so resolve it here only.
ALLERGY_SYNONYMS: dict[str, str] = {
    "bha": "salicylic_acid",
}

# Retinoid stems for a fail-closed pregnancy scan over raw INCI. Cosmetic esters
# ("Retinyl Palmitate", "Hydroxypinacolone Retinoate", "Retinyl Retinoate") never
# resolve to a canonical active, so the actives set can't be trusted here.
RETINOID_MARKERS: tuple[str, ...] = (
    "retinol", "retinal", "retinaldehyde", "retinoate", "retinyl", "retinoic",
    "tretinoin", "isotretinoin", "adapalene", "tazarotene",
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(s: str) -> list[str]:
    """Lowercase word/number tokens, e.g. 'Salicylic Acid 0.5%' -> ['salicylic',
    'acid', '0', '5']."""
    return _WORD_RE.findall(s.lower())


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    """True if `needle` appears as a contiguous run of tokens in `haystack`.
    For a single-token needle this is plain membership; multi-token needles
    ('salicylic acid') must appear adjacent, avoiding stray substring hits."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def resolve_allergy_actives(allergy: str) -> set[str]:
    """Canonical active ids a declared allergy maps onto, via the INCI synonym
    table and the allergy-only alias table (so 'salicylic acid'/'BHA'/'ascorbic
    acid' land on salicylic_acid/salicylic_acid/vitamin_c)."""
    ids: set[str] = set()
    for cand in normalize_token(allergy):
        hit = _lookup(cand, CANONICAL_ACTIVES, CANONICAL_IDS)
        if hit:
            ids.add(hit)
        alias = ALLERGY_SYNONYMS.get(cand)
        if alias:
            ids.add(alias)
    return ids


def allergy_matches(allergy: str, inci: tuple[str, ...], actives: tuple[str, ...]) -> bool:
    """Fail-closed allergen check for one declared allergy. Vetoes when the
    allergy (1) resolves through the synonym tables onto a parsed active, (2)
    appears as a token run inside any raw INCI ingredient (catches fragrance,
    parfum, limonene, lanolin, preservatives, extracts — none of which are
    canonical actives), or (3) appears as a token run inside a parsed active id
    (so 'salicylic acid' matches 'salicylic_acid' even with no INCI)."""
    needle = _words(allergy)
    if not needle:
        return False
    active_set = set(actives)
    if resolve_allergy_actives(allergy) & active_set:
        return True
    for ingredient in inci:
        if _contains_subsequence(_words(ingredient), needle):
            return True
    for active in active_set:
        if _contains_subsequence(_words(active), needle):
            return True
    return False


def contains_retinoid(inci: tuple[str, ...]) -> bool:
    """Fail-closed retinoid scan over raw INCI: any ingredient carrying a known
    retinoid stem (including cosmetic esters that never parse to a canonical
    active). Substring match on purpose — 'retinal' inside 'retinaldehyde' and
    'retinoate' inside 'hydroxypinacolone retinoate' must both fire."""
    for ingredient in inci:
        low = ingredient.lower()
        if any(marker in low for marker in RETINOID_MARKERS):
            return True
    return False


def parse_ingredients(raw: str) -> tuple[list[str], list[str]]:
    """Split an INCI string on commas and pull out the actives and comedogenic
    flags we recognize. Unrecognized tokens are silently dropped. Returns
    (sorted unique actives, sorted unique comedogenic flags)."""
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
