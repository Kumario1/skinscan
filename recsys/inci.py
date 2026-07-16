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

import json
import re
from pathlib import Path
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
    "salicylate": "salicylic_acid",
}

# Allergen families. A person declares a CATEGORY -- "parabens", "sulfates",
# "vitamin E" -- but a label only ever names members: "Methylparaben", "Sodium
# Lauryl Sulfate", "Tocopherol". Whole-token matching on the declared word alone
# therefore made the precise declaration WEAKER than the vague one ("aloe"
# vetoed, "aloe vera" did not), and made a paraben allergy -- among the most
# commonly declared -- a gate that could never fire, since no label on earth
# spells it "paraben".
#
# Members are matched with the same whole-token rule as everything else, never as
# bare substrings. Substring matching is what token equality was chosen to avoid:
# "nut" inside Coconut and Nutmeg would veto half the catalog for someone who is
# only allergic to tree nuts. The table buys the recall without paying that cost.
# Keys are singular; _allergy_phrases de-pluralizes before lookup. Grow this from
# observed label strings, per the module note above -- not from memory.
ALLERGY_FAMILIES: dict[str, frozenset[str]] = {
    "paraben": frozenset({
        "methylparaben", "ethylparaben", "propylparaben", "butylparaben",
        "isobutylparaben", "isopropylparaben", "benzylparaben",
        "sodium methylparaben", "sodium ethylparaben", "sodium propylparaben",
        "sodium butylparaben",
    }),
    "sulfate": frozenset({
        "sodium lauryl sulfate", "sodium laureth sulfate",
        "ammonium lauryl sulfate", "ammonium laureth sulfate",
        "sodium coco sulfate", "sodium myreth sulfate", "tea lauryl sulfate",
    }),
    "salicylate": frozenset({
        "salicylic acid", "capryloyl salicylic acid", "betaine salicylate",
        "sodium salicylate", "methyl salicylate", "butyloctyl salicylate",
        "ethylhexyl salicylate", "hexyldodecyl salicylate",
        "isoamyl salicylate", "benzyl salicylate", "tridecyl salicylate",
        "salix alba", "salix nigra", "willow bark",
    }),
    "benzoate": frozenset({
        "sodium benzoate", "benzoic acid", "benzyl benzoate",
        "potassium benzoate", "calcium benzoate", "ammonium benzoate",
    }),
    "vitamin e": frozenset({
        "tocopherol", "tocopheryl acetate", "tocopheryl linoleate",
        "tocopheryl succinate", "tocopheryl nicotinate", "tocophersolan",
        "tocotrienol",
    }),
    "aloe": frozenset({"aloe barbadensis", "aloe ferox", "aloe arborescens"}),
    "aloe vera": frozenset({
        "aloe", "aloe barbadensis", "aloe ferox", "aloe arborescens",
    }),
    "lanolin": frozenset({
        "lanolin", "lanolin alcohol", "lanolin oil", "lanolin wax",
        "hydrogenated lanolin", "acetylated lanolin", "isopropyl lanolate",
        "adeps lanae", "wool wax", "wool fat", "wool alcohols",
    }),
    "coconut": frozenset({
        "cocos nucifera", "coconut oil", "coconut alkanes", "coconut acid",
        "cocamidopropyl betaine", "cocamide mea", "cocamide dea",
        "coco glucoside", "sodium cocoyl isethionate", "sodium coco sulfate",
    }),
    "shea": frozenset({
        "butyrospermum parkii", "shea butter", "vitellaria paradoxa",
    }),
    # Tree nuts only. Coconut is a drupe and Nutmeg a seed -- neither belongs
    # here, and both are why members stay whole-token.
    "nut": frozenset({
        "prunus amygdalus", "prunus dulcis", "sweet almond", "almond",
        "corylus avellana", "hazelnut", "juglans regia", "walnut",
        "anacardium occidentale", "cashew", "macadamia ternifolia",
        "macadamia integrifolia", "pistacia vera", "pistachio",
        "carya illinoinensis", "pecan", "bertholletia excelsa", "brazil nut",
        "castanea sativa", "chestnut", "pinus pinea", "pine nut",
    }),
    "fragrance": frozenset({"parfum", "aroma"}),
    "essential oil": frozenset({
        "limonene", "linalool", "citral", "geraniol", "eugenol", "citronellol",
        "farnesol", "melaleuca alternifolia", "tea tree",
        "lavandula angustifolia", "lavandula hybrida", "lavender",
        "mentha piperita", "peppermint", "eucalyptus globulus",
        "rosmarinus officinalis", "rosemary", "citrus aurantium",
        "citrus limon", "citrus sinensis", "cananga odorata", "ylang ylang",
        "pelargonium graveolens", "cymbopogon", "citronella",
        "thymus vulgaris", "syzygium aromaticum", "clove",
    }),
    "tea tree": frozenset({"melaleuca alternifolia"}),
    "lavender": frozenset({"lavandula angustifolia", "lavandula hybrida"}),
}
ALLERGY_FAMILIES["tree nut"] = ALLERGY_FAMILIES["nut"]

# Retinoid stems for a fail-closed pregnancy scan over raw INCI. Cosmetic esters
# ("Retinyl Palmitate", "Hydroxypinacolone Retinoate", "Retinyl Retinoate") never
# resolve to a canonical active, so the actives set can't be trusted here.
#
# Derived from safety_rules.json, never written out here: the pregnancy gate is
# `actives & knowledge.retinoids or contains_retinoid(inci)`, and a name held by
# only one of those two arms is a hole. Drug rows carry inci: [], so this scan
# cannot see them at all and knowledge.retinoids alone decides -- tazarotene
# (pregnancy category X) sat in this tuple and NOT in safety_rules.json, leaving
# Tazorac/Arazlo unvetoed for a pregnant user, reachable the moment anything
# listed them as a candidate. Reading the same file both arms read makes that
# divergence unrepresentable rather than merely unlikely.
_SAFETY_RULES_PATH = Path(__file__).parent / "data" / "knowledge" / "safety_rules.json"


def _load_retinoid_markers(path: Path = _SAFETY_RULES_PATH) -> tuple[str, ...]:
    """Every retinoid name from safety_rules.json, as raw-INCI substrings:
    canonical active ids (`retinoids`) plus the label-only spellings that never
    parse to one (`retinoid_inci_stems`). Underscored ids become spaces so a
    future "retinyl_palmitate" still matches the label text "Retinyl Palmitate".
    Raises rather than degrading to an empty scan: a silent no-op here is a
    teratogen shipped to a pregnant user."""
    rules = json.loads(path.read_text(encoding="utf-8"))
    names = set(rules.get("retinoids") or []) | set(rules.get("retinoid_inci_stems") or [])
    markers = {name.lower().replace("_", " ").strip() for name in names}
    markers.discard("")
    if not markers:
        raise ValueError(f"no retinoid names in {path}; the pregnancy scan would pass everything")
    return tuple(sorted(markers))


RETINOID_MARKERS: tuple[str, ...] = _load_retinoid_markers()

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


def _singular_variants(phrase: str) -> list[str]:
    """`phrase` plus its de-pluralized forms. People declare allergies in the
    plural ("I'm allergic to sulfates") while labels list one substance in the
    singular, so without this the plural spelling silently matches nothing.
    Both -s and -es strips are offered as candidates and the caller tries each;
    a wrong strip ("sulfat") simply matches nothing, so guessing is free."""
    out = [phrase]
    if phrase.endswith("es") and len(phrase) > 4:
        out.append(phrase[:-2])
    if phrase.endswith("s") and not phrase.endswith("ss") and len(phrase) > 3:
        out.append(phrase[:-1])
    return [p for i, p in enumerate(out) if p and p not in out[:i]]


def _allergy_phrases(allergy: str) -> list[str]:
    """Candidate spellings of one declared allergy, most literal first: the
    verbatim string (digits intact, so "C12-15 Alkyl Benzoate" still matches),
    the normalize_token candidates (parenthetical aliases split out, "0.5%"
    noise dropped), and a singular of each. Purely additive — every phrase the
    old code matched on is still in here."""
    phrases = [allergy.lower().strip(), *normalize_token(allergy)]
    out: list[str] = []
    for phrase in phrases:
        for variant in _singular_variants(phrase):
            if variant not in out:
                out.append(variant)
    return out


def resolve_allergy_actives(allergy: str) -> set[str]:
    """Canonical active ids a declared allergy maps onto, via the INCI synonym
    table and the allergy-only alias table (so 'salicylic acid'/'BHA'/'ascorbic
    acid' land on salicylic_acid/salicylic_acid/vitamin_c). Drug rows carry no
    INCI, so for them this is the only way an allergy can be seen at all."""
    ids: set[str] = set()
    for cand in _allergy_phrases(allergy):
        hit = _lookup(cand, CANONICAL_ACTIVES, CANONICAL_IDS)
        if hit:
            ids.add(hit)
        alias = ALLERGY_SYNONYMS.get(cand)
        if alias:
            ids.add(alias)
    return ids


def allergy_matches(allergy: str, inci: tuple[str, ...], actives: tuple[str, ...]) -> bool:
    """Table-driven allergen check for one declared allergy, biased toward
    vetoing. Matches when the allergy (1) resolves through the synonym tables
    onto a parsed active, (2) appears — itself, de-pluralized, or as any member
    of its ALLERGY_FAMILIES entry — as a token run inside a raw INCI ingredient
    (catches fragrance, parfum, limonene, lanolin, preservatives and extracts,
    none of which are canonical actives), or (3) appears as a token run inside a
    parsed active id (so 'salicylic acid' matches 'salicylic_acid' with no INCI).

    Not fail-closed, and the old docstring's claim that it was invited exactly
    the gap it papered over: an allergen this module has never been told about
    still passes silently. It is only as good as the tables, which is why they
    grow from observed label strings and why every entry has a regression test.
    """
    phrases = _allergy_phrases(allergy)
    active_set = set(actives)
    if resolve_allergy_actives(allergy) & active_set:
        return True
    haystacks = [_words(ingredient) for ingredient in inci]
    haystacks.extend(_words(active) for active in active_set)
    for phrase in phrases:
        for candidate in (phrase, *sorted(ALLERGY_FAMILIES.get(phrase, ()))):
            needle = _words(candidate)
            if needle and any(_contains_subsequence(h, needle) for h in haystacks):
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
