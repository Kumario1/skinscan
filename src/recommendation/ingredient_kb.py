"""Ingredient knowledge base + per-concern ingredient-match score.

Spec: docs/superpowers/specs/2026-07-10-ingredient-kb-design.md. Complements the
concern-efficacy recommender (plan 015 / D-023) — it does NOT touch that data
layer. Two artifacts live here:

1. `build_kb(rows)` walks the `thebeautyapi/beautyproducts` product rows and
   aggregates per normalized ingredient name: comedogenicity/irritancy (max on
   conflict — conservative), the union of functions, the strongest actives
   `rating` ("direct actives" beats "supporting actives"), and every alias
   encountered (`label_name`, `other_names`, `ph_eur_name`) so catalog lookups
   hit more. The CLI (`python -m src.recommendation.ingredient_kb`) reads the
   raw JSONL and writes a deterministic, sorted `ingredient_kb.json`.

2. `match_score(raw_ingredients, concern, kb)` scores a product's raw INCI
   string for one concern against a hand-curated `CONCERN_ACTIVES` table
   (extending the D-006 actives map): +1 per beneficial active discounted by
   INCI position (concentration order), -1 per comedogenicity >= 3 ingredient
   for acne concerns only, squashed to [0, 1]. Pure function, no I/O, no ML.

The match score is only ever a RANKING TIEBREAKER — review-backed concern-stats
(D-023) dominate whenever they exist; this orders products with equal/absent
stats (spec "Ranking integration").
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .import_catalog import normalize_token

# --- hand-curated concern -> beneficial actives (extends engine.CONCERN_ACTIVES
# / RULES.md §1 / D-006). Entries are normalized ingredient NAMES and a few
# function keywords (matched against a KB entry's `functions`). Auditable, no ML.
CONCERN_ACTIVES: dict[str, set[str]] = {
    "acne_comedonal": {
        "salicylic acid", "adapalene", "retinol", "retinal", "retinaldehyde",
        "azelaic acid", "mandelic acid", "glycolic acid", "niacinamide",
    },
    "acne_inflammatory": {
        "benzoyl peroxide", "azelaic acid", "niacinamide", "adapalene",
        "salicylic acid", "zinc",
    },
    "acne_cystic": {  # soothing only — cystic routes to a professional (RULES.md)
        "azelaic acid", "niacinamide", "centella asiatica", "centella",
        "allantoin", "zinc",
    },
    "acne_general": {
        "salicylic acid", "benzoyl peroxide", "adapalene", "retinol",
        "azelaic acid", "niacinamide", "glycolic acid", "mandelic acid", "zinc",
    },
    "hyperpigmentation": {
        "niacinamide", "ascorbic acid", "vitamin c", "azelaic acid",
        "alpha arbutin", "arbutin", "tranexamic acid", "kojic acid", "retinol",
    },
    "dryness": {
        "glycerin", "glycerol", "hyaluronic acid", "sodium hyaluronate",
        "ceramide", "ceramides", "squalane", "panthenol", "allantoin",
        "humectant",  # function keyword (glycerin/HA/etc. carry it in the KB)
    },
}
CONCERNS = list(CONCERN_ACTIVES)

# actives rating strength, highest wins on conflict ("direct actives" beats
# "supporting actives"); "sensitizers" is an irritation tag, kept lowest.
_RATING_RANK = {"sensitizers": 1, "supporting actives": 2, "direct actives": 3}


def normalize_name(s) -> str | None:
    """Primary normalized form of one ingredient name, via the importer's own
    normalize_token (lowercase, punctuation/number-stripped). None if empty."""
    if not s:
        return None
    cands = normalize_token(s)
    return cands[0] if cands else None


def _parse_grade(s) -> int | None:
    """comedogenicity/irritancy are strings, sometimes ranges ("0-3", "3-5").
    Collapse to the MAX digit — same conservative rule used across conflicts."""
    if s is None:
        return None
    nums = [int(n) for n in re.findall(r"\d+", str(s))]
    return max(nums) if nums else None


def _max_opt(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _better_rating(cur: str | None, new: str | None) -> str | None:
    if _RATING_RANK.get(new or "", 0) > _RATING_RANK.get(cur or "", 0):
        return new
    return cur


def _split_other_names(s) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in str(s).split(",") if p.strip()]


def build_kb(rows) -> dict:
    """Aggregate beautyproducts rows into a normalized-name -> metadata KB.

    rows: iterable of product dicts, each with an `ingredients` list of entries
    carrying name/label_name/comedogenicity/irritancy/functions/rating and the
    alias fields. Deterministic: keys and every nested list come back sorted.
    """
    acc: dict[str, dict] = {}
    for row in rows:
        for ing in row.get("ingredients") or []:
            key = normalize_name(ing.get("name") or ing.get("label_name"))
            if not key:
                continue
            e = acc.setdefault(key, {"comedogenicity": None, "irritancy": None,
                                     "functions": set(), "rating": None,
                                     "aliases": set()})
            e["comedogenicity"] = _max_opt(e["comedogenicity"],
                                           _parse_grade(ing.get("comedogenicity")))
            e["irritancy"] = _max_opt(e["irritancy"],
                                      _parse_grade(ing.get("irritancy")))
            e["functions"].update(f for f in (ing.get("functions") or []) if f)
            e["rating"] = _better_rating(e["rating"], ing.get("rating"))
            aliases = [ing.get("label_name"), ing.get("ph_eur_name")]
            aliases += _split_other_names(ing.get("other_names"))
            for al in aliases:
                a = normalize_name(al)
                if a and a != key:
                    e["aliases"].add(a)
    return {
        key: {
            "comedogenicity": e["comedogenicity"],
            "irritancy": e["irritancy"],
            "functions": sorted(e["functions"]),
            "rating": e["rating"],
            "aliases": sorted(e["aliases"]),
        }
        for key, e in sorted(acc.items())
    }


def load_kb(path) -> dict:
    """Read an ingredient_kb.json back into memory."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_alias_index(kb: dict) -> dict[str, str]:
    """normalized name/alias -> canonical KB key. Canonical keys win over
    aliases; among aliases the first (sorted) key wins — deterministic."""
    idx: dict[str, str] = {}
    for key in kb:
        idx[key] = key
    for key in kb:
        for alias in kb[key]["aliases"]:
            idx.setdefault(alias, key)
    return idx


def _resolve(cands: list[str], idx: dict[str, str]):
    """First normalized candidate that hits the KB (directly or via alias)."""
    for c in cands:
        key = idx.get(c)
        if key is not None:
            return key
    return None


def match_score(raw_ingredients: str, concern: str, kb: dict) -> float:
    """Ingredient-plausibility score in [0, 1] for one product x one concern.

    +weight per beneficial active (name in CONCERN_ACTIVES, or one of its KB
    functions is), -weight per comedogenicity >= 3 ingredient for ACNE concerns
    only, where weight = 1/(position+1) discounts by INCI position (lists are
    concentration-ordered). Sigmoid-squashed. Pure — takes the KB as an arg.
    """
    actives = CONCERN_ACTIVES.get(concern, set())
    is_acne = concern.startswith("acne")
    idx = build_alias_index(kb)
    raw = 0.0
    for position, token in enumerate(raw_ingredients.split(",")):
        cands = normalize_token(token)
        if not cands:
            continue
        weight = 1.0 / (position + 1)
        key = _resolve(cands, idx)
        entry = kb.get(key) if key else None
        fns = set(entry["functions"]) if entry else set()
        names = set(cands)
        if key:
            names.add(key)     # alias -> canonical name also counts as beneficial
        if (names & actives) or (fns & actives):
            raw += weight
        if is_acne and entry is not None:
            com = entry.get("comedogenicity")
            if com is not None and com >= 3:
                raw -= weight
    return 1.0 / (1.0 + math.exp(-raw))


def product_matches(raw_ingredients: str, kb: dict) -> dict[str, float]:
    """ingredient_match dict {concern: score} over every CONCERN_ACTIVES key."""
    return {c: match_score(raw_ingredients, c, kb) for c in CONCERNS}


def kb_comedogenic_flags(raw_ingredients: str, kb: dict) -> set[str]:
    """Snake_cased names of raw ingredients the KB grades comedogenicity >= 3.
    A superset signal for the catalog's hand-list (spec deliverable 3)."""
    idx = build_alias_index(kb)
    flags: set[str] = set()
    for token in raw_ingredients.split(","):
        cands = normalize_token(token)
        key = _resolve(cands, idx)
        if key is None:
            continue
        com = kb[key].get("comedogenicity")
        if com is not None and com >= 3:
            flags.add(key.replace(" ", "_"))
    return flags


def build_kb_from_jsonl(jsonl_path) -> dict:
    rows = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines()
            if line.strip()]
    return build_kb(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the ingredient KB JSON.")
    ap.add_argument("--jsonl", default="data/raw/beautyapi/beauty_data.jsonl",
                    help="raw beautyproducts JSONL (manual download; see README)")
    ap.add_argument("--out", default="data/processed/ingredient_kb.json")
    args = ap.parse_args(argv)
    kb = build_kb_from_jsonl(args.jsonl)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, sort_keys=True)
        f.write("\n")
    print({"ingredients": len(kb),
           "with_comedogenicity": sum(1 for e in kb.values()
                                      if e["comedogenicity"] is not None),
           "direct_actives": sum(1 for e in kb.values()
                                 if e["rating"] == "direct actives"),
           "out": str(out)})


if __name__ == "__main__":
    main()
