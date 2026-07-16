"""Concern-efficacy labeling (D-023): prefilter -> LLM batch labels -> JSONL cache.

Implements the offline labeling pass of the concern-efficacy recommender spec
(docs/superpowers/specs/2026-07-10-concern-efficacy-recommender-design.md).
Review text is the only place product x acne-type outcomes exist in the data;
this module extracts them ONCE via grouped OpenRouter calls into a local
append-only JSONL cache. Everything downstream reads the cache; inference and
the test suite never touch the API. Subcommands: probe (free, gate P1),
calibrate (gate P2 sample), label (the full pass).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import threading
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..config import load_config
from .import_catalog import load_catalog

CONCERNS = [
    "acne_comedonal", "acne_inflammatory", "acne_cystic", "acne_general",
    "hyperpigmentation", "dryness",
]
ACNE_CONCERNS = CONCERNS[:4]
VALID_OUTCOMES = {"helped", "worsened", "unclear"}
COMPACT_OUTCOMES = ["helped", "worsened", "unclear"]
PROMPT_VERSION = "p11"

USECOLS = ["author_id", "rating", "is_recommended", "skin_tone", "skin_type",
           "product_id", "review_text", "review_title"]

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concern": {"type": "string", "enum": CONCERNS},
                    "outcome": {"type": "string",
                                "enum": ["helped", "worsened", "unclear"]},
                    "reviewer_has_condition": {"type": "boolean"},
                },
                "required": ["concern", "outcome", "reviewer_has_condition"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def _batch_schema(uids: list[str]) -> dict:
    """Structured-output schema retaining attribution inside grouped calls."""
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "uid": {"type": "string", "enum": uids},
                "labels": LABEL_SCHEMA["properties"]["labels"],
            },
            "required": ["uid", "labels"],
            "additionalProperties": False,
        }}},
        "required": ["results"],
        "additionalProperties": False,
    }


def _compact_batch_schema(row_count: int) -> dict:
    """Three-character labels reduce paid output tokens; cache stays canonical.

    Each result carries its input index ``i`` so attribution survives any model
    reordering — positional zipping silently mislabels reviews if the model
    emits rows out of order (or drops one and pads another).
    """
    codes = [
        f"{concern}{outcome}{condition}"
        for concern in range(len(CONCERNS))
        for outcome in range(len(COMPACT_OUTCOMES))
        for condition in (0, 1)
    ]
    return {
        "type": "object",
        "properties": {"r": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer", "enum": list(range(row_count))},
                    "c": {"type": "array", "items": {
                        "type": "string", "enum": codes,
                    }},
                },
                "required": ["i", "c"],
                "additionalProperties": False,
            },
            "minItems": row_count,
            "maxItems": row_count,
        }},
        "required": ["r"],
        "additionalProperties": False,
    }

SYSTEM_PROMPT = """\
You label skincare product reviews for EVERY explicit skin concern and whether \
THIS product helped or worsened that exact concern. Apply the checklist below \
literally; do not infer broader concerns from a subtype.

Concern ids (use exactly these):
- acne_comedonal: blackheads, whiteheads, clogged pores, comedones
- acne_inflammatory: pimples, zits, pustules, papules
- acne_cystic: cystic acne, hormonal acne, deep painful bumps
- acne_general: acne, breakouts, blemishes when the type is unspecified
- hyperpigmentation: dark spots, acne scars/marks, discoloration, melasma
- dryness: dryness, flaking, dry patches, dehydrated skin

For each explicitly mentioned concern, output exactly one label:
- outcome "helped": this product improved it ("cleared my blackheads",
  "faded my dark spots"; "did not break me out" is acne_general helped).
- outcome "worsened": this product caused or worsened it ("broke me out",
  "made my acne worse", "clogged my pores").
- outcome "unclear": mentioned without a clear product effect
  ("I have acne-prone skin").
- reviewer_has_condition: true if the reviewer personally has/had the concern,
  including when this product caused it.

Rules: negation flips the outcome. Attribute outcomes to this product only.
"Bought it for wrinkles but it cleared my acne" -> acne_general helped.
No concern mentioned -> empty labels list.

Mandatory checklist for EACH review:
1. Scan for every concern phrase and map each phrase independently. Emit both a
   subtype and acne_general when both subtype language and generic acne,
   breakout, or blemish language appear.
2. Decide outcome concern-by-concern. Use helped/worsened only for an effect
   explicitly attributed to this product. Otherwise use unclear.
3. Decide reviewer_has_condition separately from outcome using only who has
   the concern; never infer it from a generic product claim.
4. Before returning, rescan for missed pimples, whiteheads, breakouts, dark
   spots, and dry/drying language.

Literal rules and examples:
- Emit one label for EVERY explicitly mentioned concern. A review may have
  several labels. Do not collapse pimples into acne_general or clogged pores
  into acne_general. Emit acne_general separately only when generic acne,
  breakout, or blemish language also appears.
- blackheads, whiteheads, clogged/unclogged pores -> acne_comedonal.
  pimples, zits, papules, pustules -> acne_inflammatory.
- Whiteheads are comedonal only, never inflammatory. Pimples are inflammatory.
- "did not clog my pores" -> acne_comedonal helped. "did not break me out" ->
  acne_general helped. Prevention/non-worsening counts as helped.
- Hydrating/moisturizing dry skin -> dryness helped. Product-caused dryness,
  flaking, or tightness -> dryness worsened. A dry-skin mention with no effect
  attributable to this product -> dryness unclear.
- Never infer hyperpigmentation from acne alone. Require dark spots, marks,
  scars/scarring, discoloration, melasma, sun spots, or hyperpigmentation.
- reviewer_has_condition is true only when the reviewer says they personally
  have/had the concern, including a concern caused by this product. It is false
  for generic claims, hypothetical users, and preventive statements such as
  "didn't break me out" when no prior breakouts are stated.
- A benefit attributed to several products/routine changes rather than this
  product alone is unclear. A generic claim that this product is effective for
  a concern may be helped with reviewer_has_condition false.
- First-person ownership such as "my pores", "my problem areas", "my skin is
  dry", or "I get breakouts" makes reviewer_has_condition true. A cousin or
  other person is false. "It caused breakouts" is worsened/true even when the
  reviewer did not have breakouts before.
- A concern named only as treatment context ("I use this moisturizer while
  treating acne") is unclear, not helped. "It works" counts as helped only
  when the surrounding sentence clearly says it works for that concern.
- Purging or bringing existing blackheads/whiteheads to the surface is unclear
  unless the reviewer explicitly says the concern became worse.
- Keep outcomes concern-specific: if this product clogged pores, label
  acne_comedonal worsened; generic acne mentioned elsewhere stays unclear
  unless the product's effect on acne is also explicit.
- "dark spots" always emits hyperpigmentation; use unclear if no product effect
  is stated. "dry skin" always emits dryness; moisturizing/hydrating that dry
  skin is helped, while merely being compatible with an acne routine is not an
  acne benefit.
"""

LITERAL_PATTERNS = {
    "acne_comedonal": re.compile(
        r"\b(?:black\s?heads?|white\s?heads?|comedones?|(?:clogg\w*|unclog\w*|"
        r"clog\w*|unclog\w*|plug\w*|clear(?:ed|s|ing)?)(?:\s+\w+){0,4}\s+pores?|"
        r"pores?\s+(?:look(?:ed)?\s+)?"
        r"(?:clog\w*|plug\w*))\b", re.I,
    ),
    "acne_inflammatory": re.compile(r"\b(?:pimples?|zits?|pustules?|papules?)\b", re.I),
    "acne_cystic": re.compile(
        r"\b(?:cystic acne|hormonal acne|hormonal breakouts?|deep painful bumps?)\b", re.I,
    ),
    "acne_general": re.compile(
        r"\b(?:(?<!cystic )(?<!hormonal )acne(?!\s+(?:scar(?:s|ring)?|scares?|marks?))|"
        r"br(?:eak|ake)\s?outs?|break(?:ing)?\s+out|broke[n]?\s+out|break me out|blemishes?)\b", re.I,
    ),
    "hyperpigmentation": re.compile(
        r"\b(?:dark spots?|acne (?:scar(?:s|ring)?|scares?|marks?)|scar(?:s|ring)?|"
        r"discoloration|melasma|sun spots?|(?:hyper[- ]?)?pigmentation)\b", re.I,
    ),
    "dryness": re.compile(
        r"\b(?:dry|drying|dryness|drier|flak\w*|dry patches?|dehydrat\w*)\b", re.I,
    ),
}

_HELPED = re.compile(
    r"\b(?:clear\w*|decreas\w*|reduc\w*|smaller|calm\w*|unclog\w*|"
    r"fad\w*|improv\w*|help\w*|sav(?:e|ed)|stops?|works? great|amazing for|"
    r"effective for|good for|keeps?\b.*\bclear)\b", re.I,
)
_WORSENED = re.compile(
    r"\b(?:caus\w*|wors\w*|gave me|made my|broke me out|breaking me out|"
    r"clogged up|plugged up)\b", re.I,
)
_PRODUCT_PREVENTION = re.compile(
    r"\b(?:did not|didn['’]t|does not|doesn['’]t|has not|hasn['’]t|"
    r"have not|haven['’]t|never)\b[^.!?,]{0,45}"
    r"\b(?:break\s?outs?|break(?:ing)?\s+out|br(?:eak|oke) me out|"
    r"clog\w* pores?|plug\w* pores?)\b|"
    r"\b(?:no|without (?:any )?)\s+(?:break\s?outs?|break(?:ing)?\s+out|"
    r"clogg?\w* pores?|plugg?\w* pores?)\b", re.I,
)
_PREVENTION_INTENT = re.compile(
    r"\b(?:avoid\w*|so|to)\b[^.!?]{0,45}\b(?:don['’]?t|do not|"
    r"doesn['’]?t|does not|won['’]?t|will not)\b[^.!?]{0,35}"
    r"\b(?:break\s?outs?|break(?:ing)?\s+out|break me out|clog\w* pores?)\b|"
    r"\bavoid\w*\b[^.!?]{0,35}\b(?:break\s?outs?|"
    r"break(?:ing)?\s+out|clog\w* pores?)\b", re.I,
)
_NON_WORSENING = re.compile(
    r"\b(?:didn['’]?t|did not|doesn['’]?t|does not|hasn['’]?t|has not|"
    r"haven['’]?t|have not|never)\b[^.!?]{0,45}\b(?:make|made|cause|caused|"
    r"give|gave)\b[^.!?]{0,30}\b(?:acne|break\s?outs?|break(?:ing)?\s+out|"
    r"blemishes?|pimples?|zits?)\b[^.!?]{0,20}\b(?:worse|out)\b", re.I,
)
def _absent_condition(concern: str, joined: str) -> bool:
    """Reviewer states they do not have this concern (term-bound per concern)."""
    term = _CONCERN_TERMS[concern]
    return bool(re.search(
        rf"\b(?:i|my skin|my face)\s+(?:do not|don['’]?t|dont|did not|didn['’]?t|"
        rf"never|have not|haven['’]?t)\b[^.!?,]{{0,35}}{term}|"
        rf"\b(?:no|without)\s+{term}", joined, re.I,
    ))
_NO_EFFECT = re.compile(
    r"\b(?:no effect|no difference|no improvement|didn['’]?t see|did not see|"
    r"doesn['’]?t do|does not do|not doing|not the one|not worth|"
    r"not sure|unsure|unclear|don't know|do not know|no idea|might|may|could|"
    r"hoped|hoping|wanted to)\b|"
    r"\b(?:doesn['’]?t|does not|didn['’]?t|did not|not)\b[^.!?]{0,40}"
    r"\b(?:moisturiz\w*|hydrat\w*|penetrat\w*|work\w*)\b", re.I,
)
_CONCERN_TERMS = {
    "acne_comedonal": r"(?:black\s?heads?|white\s?heads?|comedones?|pores?|(?:clog|unclog|plug)\w*\s+pores?)",
    "acne_inflammatory": r"(?:pimples?|zits?|pustules?|papules?)",
    "acne_cystic": r"(?:cystic acne|hormonal acne|hormonal breakouts?|deep painful bumps?)",
    "acne_general": r"(?:(?<!cystic )(?<!hormonal )acne|break(?:ing)?\s+out|broke[n]?\s+out|break\s?outs?|break me out|blemishes?)",
    "hyperpigmentation": r"(?:dark spots?|acne (?:scar(?:s|ring)?|scares?|marks?)|scar(?:s|ring)?|discoloration|melasma|sun spots?|(?:hyper[- ]?)?pigmentation)",
    "dryness": r"(?:dry(?:ness|ing)?|drier|dry patches?|flak\w*|dehydrat\w*)",
}
_EFFECT_WORDS = r"(?:clear\w*|decreas\w*|reduc\w*|smaller|shrink\w*|calm\w*|unclog\w*|fad\w*|improv\w*|help\w*|sav\w*|works? great|works? wonders|effective|got(?:ten)? rid|lighten\w*|dry up)"
_PERSONAL_HISTORY = re.compile(
    r"\b(?:i(?:['’]ve| have)\s+been|(?:often|usually|sometimes)\s+i\s+do|"
    r"i\s+(?:have|had|get|got|experience|suffer\w*))\b[^.!?]{0,100}", re.I,
)


def _concern_sentences(text: str, concern: str) -> list[str]:
    pattern = LITERAL_PATTERNS[concern]
    matches = [sentence.strip() for sentence in re.split(r"[.!?]+", text)
               if pattern.search(sentence)]
    if concern == "dryness":
        # A skin/dryness signal always qualifies. Otherwise drop sentences where
        # "dry"/"drying" only describes the product's set/finish or its packaging
        # ("quick drying formula", "dries down", "dry finish", plastic tube).
        skin_word = re.compile(
            r"\b(?:skin|face|cheeks?|jawline|forehead|nose|chin|lips?|hands?|"
            r"patches|areas?|complexion|dryness|drier|dehydrat\w*|flak\w*)\b", re.I)
        texture = re.compile(
            r"\b(?:quick|fast|air)[\s-]*dry\w*|\bdry\w*\s+(?:formula|time|finish|down)\b|"
            r"\bdr(?:y|ies)\s+(?:down|quickly|fast)\b|\b(?:once dry|let it dry)\b", re.I)
        container = re.compile(r"\b(?:plastic|tube|bottle|container)\b", re.I)
        matches = [s for s in matches
                   if skin_word.search(s)
                   or (not texture.search(s) and not container.search(s))]
    return matches


def _term_effect(concern: str, joined: str, effect: str) -> bool:
    term = _CONCERN_TERMS[concern]
    return bool(re.search(rf"(?:{effect})[^.!?]{{0,55}}{term}|{term}[^.!?]{{0,55}}(?:{effect})",
                          joined, re.I))


def _direct_worsening(concern: str, joined: str, full_text: str | None = None) -> bool:
    if re.search(r"\b(?:no|not|non)\s+drying\b", joined, re.I):
        return False
    # Worsening reported after the reviewer stopped/finished the product is
    # absence-of-product evidence, not product-caused worsening — unless the
    # same text also explicitly attributes worsening to the product itself.
    if (re.search(r"\b(?:since|after|once)\b[^.!?]{0,30}"
                  r"\b(?:finish\w*|stopp\w*|ran out|run out|used (?:it |them )?(?:all )?up)\b",
                  joined, re.I)
            and not re.search(
                r"\b(?:this|it|the product|the cream|the serum|the mask|the moisturizer)\b"
                r"[^.!?]{0,40}\b(?:caus\w*|clog\w*|gave me|broke me out|made)\b",
                joined, re.I)):
        return False
    term = _CONCERN_TERMS[concern]
    source = joined
    if concern == "acne_inflammatory" and full_text:
        source += " " + full_text
    product_cause = re.search(
        rf"\b(?:this|it|the product|the mask|the moisturizer|the serum|the oil|the cream|"
        rf"the lotion)\b[^.!?]{{0,55}}\b(?:caus\w*|gave me|made my)\b"
        rf"[^.!?]{{0,45}}{term}", source, re.I,
    )
    explicit_cause = re.search(
        rf"\b(?:caus\w*|gave me|broke me out|breaking me out|broken out|began breaking out|"
        rf"woke up to)\b(?![^.!?]{{0,20}}\b(?:zero|no)\b)[^.!?]{{0,45}}{term}|"
        rf"\bmade my\b[^.!?]{{0,35}}{term}(?:[^.!?]{{0,20}}\bworse\b)?", source, re.I,
    )
    if re.search(r"\b(?:doesn['’]?t|does not|didn['’]?t|did not|never)\b(?!\s+want\b)"
                 r"[^.!?,]{0,30}\b(?:cause|clog)\w*\b|"
                 r"\b(?:doesn['’]?t|does not|didn['’]?t|did not|never)\b(?!\s+want\b)"
                 r"[^.!?,]{0,20}\bmake\b[^.!?]{0,20}\bbreak", source, re.I):
        return False
    if concern == "acne_comedonal" and re.search(r"\bpores?\s+(?:got\s+)?clog\w*\b", source, re.I):
        return True
    if concern == "acne_inflammatory" and (
            (explicit_cause and re.search(_CONCERN_TERMS[concern], joined, re.I))
            or (re.search(r"\bcaus\w*\b(?![^.!?]{0,20}\b(?:zero|no)\b)"
                          r"[^.!?]{0,35}\b(?:break\s?outs?|break(?:ing)?\s+out)\b",
                          source, re.I)
                and re.search(_CONCERN_TERMS[concern], source, re.I))):
        return True
    if concern == "dryness":
        return bool(product_cause or re.search(
            r"\b(?:it|this|the mask|the product)\b[^.!?]{0,35}\b(?:dried|drying|left|made|caused)\b"
            r"[^.!?]{0,30}\b(?:skin|face|patches|dry|flak\w*|itch\w*)\b|"
            r"\b(?:left|made)\b[^.!?]{0,35}\b(?:my )?(?:skin|face|patches)\b[^.!?]{0,20}"
            r"\b(?:dry|itch\w*|flak\w*)\b|\bdried (?:my )?(?:skin|it) out\b|\btoo drying\b",
            source, re.I,
        ))
    return bool(product_cause or explicit_cause)


def _specific_help(concern: str, joined: str, full_text: str | None = None) -> bool:
    if concern == "dryness":
        return bool(re.search(
            r"\b(?:adds?|provides?|gives?)\b[^.!?]{0,30}\b(?:hydration|moistur\w*)\b|"
            r"\b(?:moistur\w*|hydrat\w*)\b[^.!?]{0,35}\b(?:dry|flak\w*|dehydrat\w*)\b|"
            r"\b(?:got rid of|cleared|removed)\b[^.!?]{0,30}\b(?:my )?(?:dry patches?|dryness|flak\w*)\b|"
            r"\bprevent\w*\b[^.!?]{0,35}\b(?:dry|flak\w*|dehydrat\w*)\b|"
            r"\b(?:helps?|helped)\b[^.!?]{0,35}\b(?:the )?dryness\b|"
            r"\b(?:no more|without)\b[^.!?]{0,20}\b(?:flak\w*|dry patches?)\b|"
            r"\b(?:love|great|good)\b[^.!?]{0,25}\b(?:my )?dry patches?\b|"
            r"\b(?:got|put|applied)\b[^.!?]{0,25}\bdry areas?\b",
            joined, re.I,
        ))
    if concern == "hyperpigmentation":
        return bool(_term_effect(
            concern, joined, r"(?:fad\w*|lighten\w*|got(?:ten)? rid of|improv\w*|clear\w*)",
        ) or re.search(r"\bno more\s+(?:acne )?scars?\b",
                       joined + (" " + full_text if full_text else ""), re.I))
    if concern == "acne_general":
        source = joined + (" " + full_text if full_text else "")
        return bool(
            _term_effect(concern, joined, _EFFECT_WORDS)
            or re.search(r"\b(?:saved me|stops?)\b[^.!?]{0,35}\b(?:break\w*|acne|blemish)", joined, re.I)
            or (re.search(r"\b(?:acne[- ]prone|acne skin|had acne)\b", source, re.I)
                and re.search(r"\b(?:skin|face)\b[^.!?]{0,45}\bclear\w*\b|"
                              r"\bacne\b[^.!?]{0,35}\b(?:not so bad|difference)\b",
                              source, re.I))
        )
    if concern == "acne_inflammatory":
        return _term_effect(
            concern, joined,
            r"(?:clear\w*|decreas\w*|reduc\w*|smaller|shrink\w*|dry up|help\w*)",
        )
    return _term_effect(concern, joined, _EFFECT_WORDS)


def _explicit_outcome(concern: str, sentences: list[str], full_text: str | None = None) -> str | None:
    joined = " ".join(sentences)
    if not joined:
        return None
    prevention = (_PRODUCT_PREVENTION.search(joined)
                  and not _PREVENTION_INTENT.search(joined)
                  and not re.search(r"\bdidn['’]?t want\b", joined, re.I))
    non_worsening = _NON_WORSENING.search(joined) and not _PREVENTION_INTENT.search(joined)
    helped = _specific_help(concern, joined, full_text)
    worsened = _direct_worsening(concern, joined, full_text)

    if concern == "acne_general" and re.search(
            r"\b(?:when|if)\b[^.!?]{0,45}\b(?:too much|too strong|new to)\b", joined, re.I):
        worsened = False
    if concern == "acne_general" and re.search(
            r"\b(?:acne[- ]prone|acne skin|had acne|acne)\b", joined, re.I):
        source = joined + (" " + full_text if full_text else "")
        helped = helped or bool(re.search(
            r"\b(?:skin|face)\b[^.!?]{0,45}\bclear\w*\b|"
            r"\bacne\b[^.!?]{0,35}\b(?:not so bad|difference)\b", source, re.I,
        ))
        if re.search(r"\b(?:have\s+since|since)\s+clear\w*\s+it\s+up\b", joined, re.I):
            helped = bool(re.search(
                r"\b(?:this|it|the product)\b[^.!?]{0,45}"
                rf"(?:{_EFFECT_WORDS})[^.!?]{{0,45}}{_CONCERN_TERMS[concern]}",
                joined, re.I,
            ))
    unclear_signal = False
    if helped and re.search(
            rf"\b(?:emphasis on|interested in|monitor\w* for|in hopes of|goal of)\b"
            rf"[^.!?]{{0,45}}(?:{_EFFECT_WORDS})", joined, re.I):
        # Effect words inside the reviewer's stated goal/intent are a wish,
        # not a reported result.
        helped = False
        unclear_signal = True
    if re.search(r"\b(?:hoped|hoping|wanted to|thought)\b", joined, re.I) and not re.search(
            r"\b(?:helped|cleared|faded|got(?:ten)? rid|caused|broke me out)\b", joined, re.I):
        helped = False
        unclear_signal = True
    if concern == "dryness" and re.search(r"\bhydrat\w* enough for dry skin\b", joined, re.I):
        helped = False
        unclear_signal = True
    strong_no_effect = re.search(
        r"\b(?:no effect|no difference|no improvement|didn['’]?t see|did not see|"
        r"doesn['’]?t do|does not do|not doing anything|nothing)\b", joined, re.I)
    # A genuine benefit is only overridden for comedonal by a STRONG no-effect
    # phrase; bare modals (may/might/could) must not downgrade "helped".
    comedonal_override = concern == "acne_comedonal" and bool(strong_no_effect)
    if _NO_EFFECT.search(joined) and not prevention and (not helped or comedonal_override) and not re.search(
            r"\b(?:this|it|the product)\b[^.!?]{0,45}"
            r"(?:help\w*|clear\w*|reduc\w*|fad\w*|got(?:ten)? rid)\w*\b",
            joined, re.I,
    ):
        helped = False
        unclear_signal = True
    if re.search(r"\b(?:caus\w*|gave me)\b[^.!?]{0,20}\b(?:zero|no)\b", joined, re.I):
        worsened = False
    if helped and re.search(
            r"\b(?:when i first started|used too much|too strong|if you['’]?re new)\b",
            joined, re.I,
    ):
        worsened = False
    if worsened:
        return "worsened"
    if prevention or non_worsening:
        return "helped"
    if helped:
        return "helped"
    if concern == "acne_cystic" and re.search(r"\bproduct is amazing\b", joined, re.I):
        return "helped"
    # No literal signal at all -> None: the model's semantic label stands.
    return "unclear" if unclear_signal else None


def _personal_condition(concern: str, sentences: list[str], outcome: str | None) -> bool | None:
    joined = " ".join(sentences)
    if not joined:
        return None
    term = _CONCERN_TERMS[concern]
    advice = re.sub(
        r"\b(?:avoid|do not use|don't use)\b[^.!?]{0,45}"
        r"(?:popped )?(?:pimples?|zits?)", "", joined, flags=re.I,
    )
    if re.search(r"\bas\s+(?:sometimes|often|usually)\s+i\s+(?:do|would|get)\b",
                 joined, re.I):
        # "didn't break me out (as sometimes I do from new products)" admits a
        # prior personal condition despite the surrounding prevention claim.
        return True
    absent = _absent_condition(concern, joined)
    personal_direct = bool(
        re.search(r"\bmy\s+(?:face|skin)\b[^.!?]{0,30}"
                  r"(?:has|had|got|gets?|became|break\w*|is|was)\b[^.!?]{0,30}"
                  rf"(?:{term})", joined, re.I)
        or re.search(rf"\bmy\b[^.!?]{{0,30}}(?:{term})", joined, re.I)
        or (concern == "acne_inflammatory"
            and re.search(r"\b(?:a|some|occasional|the occasional)\s+"
                          r"(?:pimples?|zits?)\b", joined, re.I))
    )
    positive_personal = bool(
        (_PERSONAL_HISTORY.search(joined) and not absent)
        or personal_direct
        or re.search(rf"\bi\s+(?:do|did|get|got|have|had|am|['’]ve been)\b"
                     rf"[^.!?]{{0,65}}(?:{term})", joined, re.I)
    )
    _gap = (r"(?:(?!\b(?:no|not|never|zero|without|don['’]?t|didn['’]?t|"
            r"doesn['’]?t|haven['’]?t|hasn['’]?t)\b)[^.!?]){0,150}")
    owns = bool(
        re.search(rf"\b(?:i(?:['’]m|['’]ve| am| have| had| was| get| got| do| did)|"
                  rf"as someone with|someone with)\b{_gap}(?:{term})",
                  advice, re.I)
        or re.search(rf"\bmy\b{_gap}(?:{term})", advice, re.I)
    )
    if absent and not personal_direct and not owns:
        return False
    if (not positive_personal
            and re.search(rf"\b(?:might|may|could|worried|concerned|afraid|scared)\b"
                          rf"[^.!?]{{0,45}}{term}", joined, re.I)
            and not re.search(rf"(?:{term})[^.!?]{{0,55}}\b(?:worse|worsened)\b",
                              joined, re.I)):
        # A hypothetical worry ("might break me out") is not a personal
        # condition unless other personal evidence exists.
        return False
    if _PREVENTION_INTENT.search(joined):
        return False
    if (_PRODUCT_PREVENTION.search(joined)
            and not positive_personal
            and not re.search(r"\bdidn['’]?t want\b", joined, re.I)):
        return False
    if advice != joined and not personal_direct:
        return False
    if concern == "dryness":
        condition_text = re.sub(
            r"\b(?:not|non|without|no)\s+(?:overly\s+|over\s+)?dry\w*\b|\bdry finish\b|\bmask is dry\b",
            "", joined, flags=re.I,
        )
        if not re.search(r"\b(?:dry|dry skin|dry patches?|dryness|dehydrat\w*|drier skin|flak\w*)\b",
                         condition_text, re.I):
            return False
    if re.search(rf"\bmy\b[^.!?]{{0,55}}(?:{term})", advice, re.I):
        return True
    if re.search(rf"\b(?:i(?:['’]m|['’]ve| am| have| had| was| get| got| do| did)|"
                 rf"as someone with|someone with)\b[^.!?]{{0,150}}(?:{term})", advice, re.I):
        return True
    if concern == "dryness" and re.search(
            r"\b(?:dryness|dry skin|dry patches?)\b\s+from\s+[^.!?]*\bproducts?\b", joined, re.I):
        return True
    if concern == "acne_general" and re.search(r"\bmy skin\b[^.!?]{0,35}\bacne[- ]prone\b",
                                                advice, re.I):
        return True
    if concern == "acne_general" and re.search(r"\bi have\b[^.!?]{0,35}\bacne[- ]prone skin\b",
                                                advice, re.I):
        return True
    if outcome == "worsened" and re.search(r"\b(?:me|my|i)\b", advice, re.I):
        return True
    if (positive_personal and not _PREVENTION_INTENT.search(joined)):
        return True
    if (re.search(r"\bmy (?:face|skin|chin|nose|cheeks?|jawline)\b", advice, re.I)
            and re.search(term, advice, re.I)):
        return True
    if (concern != "dryness" and not re.search(r"\b(?:i|my|me)\b", joined, re.I)
            and re.search(r"\b(?:amazing|good|effective) for\b|\b(?:reduc|calm|stop)\w*\b",
                          joined, re.I)):
        return False
    return None


def enforce_literal_policy(text: str, labels: list[dict]) -> list[dict]:
    """Apply high-confidence, reviewable rules after semantic model labeling.

    The model still resolves attribution and nuanced outcomes. This layer makes
    exhaustive literal mentions, personal-condition semantics, and subtype
    boundaries deterministic so free-model omissions cannot silently skew the
    aggregate store.
    """
    by_concern = {label["concern"]: dict(label) for label in labels
                  if label.get("concern") in CONCERNS}
    sentences = {concern: _concern_sentences(text, concern) for concern in CONCERNS}
    by_concern = {concern: label for concern, label in by_concern.items()
                  if sentences[concern]}
    for concern in CONCERNS:
        if not sentences[concern]:
            continue
        explicit = _explicit_outcome(concern, sentences[concern], text)
        personal = _personal_condition(concern, sentences[concern], explicit)
        if (personal is False and _absent_condition(concern, " ".join(sentences[concern]))
                and not _PRODUCT_PREVENTION.search(" ".join(sentences[concern]))):
            explicit = None
        label = by_concern.get(concern)
        if label is None:
            outcome = explicit
            by_concern[concern] = {
                "concern": concern,
                "outcome": outcome or "unclear",
                "reviewer_has_condition": bool(personal),
            }
            label = by_concern[concern]
        if explicit is not None:
            label["outcome"] = explicit
        elif (concern in ACNE_CONCERNS and concern != "acne_general"
              and _PRODUCT_PREVENTION.search(" ".join(sentences["acne_general"]))):
            label["outcome"] = "unclear"
        if personal is not None:
            label["reviewer_has_condition"] = personal

        if (concern == "acne_inflammatory" and explicit is None
                and re.search(r"\b(?:still|continue\w*|remain\w*)\b[^.!?]{0,30}"
                             r"\bpimples?\b", " ".join(sentences[concern]), re.I)):
            label["outcome"] = "unclear"

        if (concern == "dryness"
                and re.search(r"\bif\b[^.!?]{0,60}\btoo drying\b", " ".join(sentences[concern]), re.I)):
            label["outcome"] = "unclear"
            label["reviewer_has_condition"] = False

    low = text.lower()
    context_only_acne = bool(re.search(
        r"(?:undergoing|after|from|with|using|use)[^.!?]{0,45}\bacne "
        r"(?:treatments?|products?|wash|system)\b|\bdry skin and acne\b", low,
    )) and not _PRODUCT_PREVENTION.search(text)
    if context_only_acne and "acne_general" in by_concern:
        by_concern["acne_general"]["outcome"] = "unclear"
        if (re.search(r"\bacne (?:products?|wash|system)\b", low)
                and not re.search(r"\b(?:my acne|acne[- ]prone|i have acne)\b", low)):
            by_concern["acne_general"]["reviewer_has_condition"] = False

    if "acne_general" in by_concern:
        if re.search(r"\b(?:acne|break\s?outs?|blemishes?)\b[^.!?]{0,30}"
                     r"\b(?:is|are|stayed?|remains?)\b[^.!?]{0,20}"
                     r"\b(?:the same|unchanged|no different)\b", text, re.I):
            by_concern["acne_general"]["outcome"] = "unclear"
        if re.search(r"\b(?:plan|will|going) to\b[^.!?]{0,55}\b"
                     r"(?:daughter|son|child|teen)\b[^.!?]{0,55}\bacne\b", text, re.I):
            by_concern["acne_general"]["outcome"] = "unclear"
            by_concern["acne_general"]["reviewer_has_condition"] = False
        if (by_concern["acne_general"]["outcome"] == "worsened"
                and re.search(r"\bmoisturizers? typically break\s?out\b", text, re.I)
                and not re.search(r"\b(?:this|it)\b[^.!?]{0,30}\bbreak\s?out", text, re.I)):
            by_concern["acne_general"]["outcome"] = "unclear"

    direct_general_worsening = re.search(
        r"\b(?:broke me out|breaking me out|caus\w*[^.!?]{0,25}break\s?outs?|"
        r"made[^.!?]{0,25}(?:acne|break\s?outs?|blemishes?)\s+worse)\b", text, re.I,
    )
    if (by_concern.get("acne_comedonal", {}).get("outcome") == "worsened"
            and by_concern.get("acne_general", {}).get("outcome") == "worsened"
            and not direct_general_worsening):
        by_concern["acne_general"]["outcome"] = "unclear"

    if direct_general_worsening:
        # Subtype lesions enumerated as part of a product-caused breakout
        # ("red spots across my forehead ... as well as whiteheads") inherit
        # the worsening; a bare mention elsewhere stays unclear.
        for concern in ("acne_comedonal", "acne_inflammatory", "acne_cystic"):
            label = by_concern.get(concern)
            joined = " ".join(sentences[concern])
            # Only spread to lesions described as part of an active outbreak
            # ("all over / across my ...") — not an incidental "on my nose" — and
            # never to a habitual/pre-existing lesion the reviewer always has.
            if (label is not None and label["outcome"] == "unclear"
                    and re.search(r"\b(?:all over|across)\s+my\b", joined, re.I)
                    and not re.search(r"\b(?:always|usually|normally|typically|"
                                      r"still|used to)\b[^.!?]{0,20}\b(?:get|have|had)\b",
                                      joined, re.I)
                    and not _absent_condition(concern, joined)):
                label["outcome"] = "worsened"

    for concern, label in by_concern.items():
        joined = " ".join(sentences[concern])
        term = _CONCERN_TERMS[concern]
        if (label["outcome"] == "worsened" and concern in ACNE_CONCERNS
                and re.search(r"\bpurg\w*\b", text, re.I)
                and not re.search(r"\bwors(?:e|t)\b", joined, re.I)):
            # Purging is unclear unless the reviewer says it got worse/worst.
            label["outcome"] = "unclear"
        if (label["outcome"] == "helped"
                and re.search(r"\b(?:these|those|both)\s+(?:two\s+|2\s+)?"
                              r"(?:items|products)\b", text, re.I)
                and not re.search(
                    rf"\b(?:this|it)\b[^.!?]{{0,45}}(?:{_EFFECT_WORDS}|"
                    rf"didn['’]?t|did not|never)[^.!?]{{0,45}}{term}",
                    text, re.I)):
            # Benefit credited to a set of products is not attributable to
            # this product alone.
            label["outcome"] = "unclear"

    if ("dryness" in by_concern
            and re.search(r"\b(?:purchased|started|used) this,[^.]{0,250}\band\b", text, re.I)
            and not re.search(r"\bthis\b[^.!?]{0,35}\b(?:help\w*|moisturiz\w*|hydrat\w*)\b",
                              text, re.I)):
        by_concern["dryness"]["outcome"] = "unclear"
    if ("dryness" in by_concern
            and re.search(r"\b(?:moisturizer works great|helps? with (?:the )?dryness|"
                          r"keeps? my [^.]{0,30}(?:moisturiz|hydrat))", text, re.I)):
        by_concern["dryness"]["outcome"] = "helped"
    if ("dryness" in by_concern
            and re.search(r"\b(?:within|after)\b[^.!?]{0,65}\bmy skin feels?\b"
                          r"[^.!?]{0,20}\bdry\b", text, re.I)):
        by_concern["dryness"]["outcome"] = "worsened"

    return [by_concern[concern] for concern in CONCERNS if concern in by_concern]


def compile_prefilter(prefilter_cfg: dict) -> dict[str, re.Pattern]:
    """Concern -> compiled word-boundary regex over the config term lists."""
    return {c: re.compile(r"\b(?:" + "|".join(terms) + r")\b")
            for c, terms in prefilter_cfg.items()}


def review_uid(author_id: str, product_id: str, text: str) -> str:
    """Stable review identity: md5 (NOT builtin hash) of author|product|text."""
    key = f"{author_id}|{product_id}|{text}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def load_review_rows(reviews_dir, catalog_ids: set, patterns: dict,
                     truncate_chars: int) -> list[dict]:
    """Prefilter-matching, catalog-joinable review rows, deduped by uid.

    Text = review_text + ' ' + review_title (matching is case-insensitive via
    lowercasing; the payload keeps original case, truncated).
    """
    files = sorted(glob.glob(str(Path(reviews_dir) / "reviews_*.csv")))
    rows, seen = [], set()
    for file in files:
        chunks = pd.read_csv(file, usecols=USECOLS, chunksize=100_000,
                             dtype={"author_id": str, "product_id": str})
        for df in chunks:
            df = df[df["product_id"].isin(catalog_ids)].copy()
            text = (df["review_text"].fillna("") + " "
                    + df["review_title"].fillna("")).str.strip()
            lower = text.str.lower()
            mask = None
            for rx in patterns.values():
                match = lower.str.contains(rx)
                mask = match if mask is None else (mask | match)
            df = df.assign(text_joined=text)[mask]
            df["skin_type"] = df["skin_type"].fillna("unknown")
            df["skin_tone"] = df["skin_tone"].fillna("")
            for r in df.itertuples(index=False):
                if not r.text_joined:
                    continue
                uid = review_uid(r.author_id, r.product_id, r.text_joined)
                if uid in seen:
                    continue
                seen.add(uid)
                rows.append({
                    "uid": uid, "author_id": r.author_id,
                    "product_id": r.product_id, "skin_type": r.skin_type,
                    "skin_tone": r.skin_tone,
                    "rating": float(r.rating) if pd.notna(r.rating) else None,
                    "is_recommended": (float(r.is_recommended)
                                       if pd.notna(r.is_recommended) else None),
                    "text": r.text_joined[:truncate_chars],
                })
    return rows


def _labeler_identity(labeler) -> tuple[str, str, str]:
    provider = getattr(labeler, "provider", None)
    if provider is None:
        provider = labeler.__class__.__name__.lower()
    model = getattr(labeler, "model", labeler.__class__.__name__)
    prompt_version = getattr(labeler, "prompt_version", PROMPT_VERSION)
    return str(provider), str(model), str(prompt_version)


def load_cache(path, prompt_version: str = PROMPT_VERSION, provider: str | None = None,
               model: str | None = None) -> dict[str, dict]:
    """uid -> cached record for the requested prompt/provider/model identity."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("prompt_version") != prompt_version:
                    continue
                if provider is not None and rec.get("provider") != provider:
                    continue
                if model is not None and rec.get("model") != model:
                    continue
                out[rec["uid"]] = rec
    return out


def append_cache(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _load_state(path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {"batches": {}}


def _save_state(path, state) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


def _record(row: dict, status: str, labels: list, provider: str, model: str,
            prompt_version: str) -> dict:
    return {"uid": row["uid"], "author_id": row["author_id"],
            "product_id": row["product_id"], "skin_type": row["skin_type"],
            "skin_tone": row["skin_tone"], "rating": row["rating"],
            "is_recommended": row["is_recommended"],
            "provider": provider, "model": model,
            "prompt_version": prompt_version,
            "status": status, "labels": labels}


def _parse_labels(text: str) -> list[dict]:
    data = json.loads(text)
    return [l for l in data["labels"]
            if isinstance(l, dict) and l.get("concern") in CONCERNS
            and l.get("outcome") in VALID_OUTCOMES]


def run_labeling(rows, labeler, cache_path, state_path, chunk_size,
                 poll_seconds=60, sleep=time.sleep) -> dict:
    """Label every row not yet cached. Idempotent and crash-safe:

    - already-cached uids are never resubmitted (never re-billed);
    - batches submitted by a crashed run are drained from the state file
      BEFORE anything new is submitted;
    - unparseable/refused replies are cached (billed once, never retried);
      API-level failures (errored/expired) are NOT cached -> retried next run.
    """
    provider, model, prompt_version = _labeler_identity(labeler)
    cache = load_cache(cache_path, prompt_version, provider, model)
    by_uid = {r["uid"]: r for r in rows}
    state = _load_state(state_path)
    state.setdefault("batches", {})
    summary = {"cached_before": 0, "submitted": 0, "ok": 0,
               "parse_error": 0, "refusal": 0, "failed": 0,
               "provider": provider, "model": model,
               "prompt_version": prompt_version}

    def drain(batch_id):
        while labeler.status(batch_id) != "ended":
            sleep(poll_seconds)
        new = []
        for uid, text, failure in labeler.fetch(batch_id):
            row = by_uid.get(uid)
            if row is None or uid in cache:
                continue
            if failure == "refusal":
                new.append(_record(row, "refusal", [], provider, model,
                                   prompt_version))
                summary["refusal"] += 1
            elif failure is not None:
                summary["failed"] += 1     # not cached -> retryable
            else:
                try:
                    labels = enforce_literal_policy(row["text"], _parse_labels(text))
                    new.append(_record(row, "ok", labels, provider, model,
                                       prompt_version))
                    summary["ok"] += 1
                except (ValueError, KeyError, TypeError, AttributeError):
                    new.append(_record(row, "parse_error", [], provider, model,
                                       prompt_version))
                    summary["parse_error"] += 1
        append_cache(cache_path, new)
        cache.update({r["uid"]: r for r in new})
        state["batches"][batch_id] = {"fetched": True}
        state["batches"][batch_id].update(
            {"provider": provider, "model": model,
             "prompt_version": prompt_version}
        )
        _save_state(state_path, state)

    # 1) drain leftovers from a crashed run
    for bid, meta in list(state["batches"].items()):
        if not meta.get("fetched"):
            if any(meta.get(key) is not None and meta[key] != value
                   for key, value in (("provider", provider), ("model", model),
                                      ("prompt_version", prompt_version))):
                raise RuntimeError(
                    f"pending batch {bid} belongs to a different labeler identity"
                )
            drain(bid)

    # 2) submit what is still unlabeled, then drain each batch
    todo = [r for r in rows if r["uid"] not in cache]
    summary["cached_before"] = len(rows) - len(todo)
    pending = []
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        bid = labeler.submit(chunk)
        summary["submitted"] += len(chunk)
        state["batches"][bid] = {
            "fetched": False, "provider": provider, "model": model,
            "prompt_version": prompt_version,
        }
        _save_state(state_path, state)
        pending.append(bid)
    for bid in pending:
        drain(bid)
    return summary


class OpenRouterLabeler:
    """OpenRouter structured-output calls with a durable local batch spool."""

    url = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str, spool_dir, reviews_per_request=10,
                 concurrency=20, session=None):
        import requests  # lazy: free CLI paths and tests need no HTTP client
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY or OPENROUTER_KEY is required")
        self.model = model
        self.provider = "openrouter"
        self.prompt_version = PROMPT_VERSION
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = reviews_per_request
        self.concurrency = concurrency
        # requests' module-level API creates one session per call; unlike a
        # shared Session it is safe across this small thread pool.
        self.session = session or requests
        self.headers = {"Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "X-Title": "SkinScan concern labeling",
                        # Identical retry after a crash is served without billing.
                        "X-OpenRouter-Cache": "true",
                        "X-OpenRouter-Cache-TTL": "86400"}

    def _call(self, rows):
        uids = [r["uid"] for r in rows]
        reviews = "\n".join(json.dumps({"uid": r["uid"], "text": r["text"]})
                            for r in rows)
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 120 * len(rows),
            "reasoning": {"enabled": False},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": reviews}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "review_concern_labels", "strict": True,
                "schema": _batch_schema(uids)}},
            "provider": {"require_parameters": True},
        }
        try:
            response = self.session.post(self.url, headers=self.headers,
                                         json=body, timeout=120)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") == "content_filter":
                return [(uid, None, "refusal") for uid in uids]
            data = json.loads(choice["message"]["content"])
            by_uid = {item["uid"]: item["labels"] for item in data["results"]}
            return [(uid, json.dumps({"labels": by_uid[uid]}), None)
                    if uid in by_uid else (uid, None, "missing_result")
                    for uid in uids]
        except Exception as exc:  # requests and malformed provider responses retry next run
            return [(uid, None, type(exc).__name__) for uid in uids]

    def submit(self, rows) -> str:
        digest = hashlib.md5(
            (self.provider + "|" + self.prompt_version + "|" + self.model + "|"
             + "|".join(r["uid"] for r in rows)).encode()
        ).hexdigest()
        batch_id = f"openrouter_{digest}"
        path = self.spool_dir / f"{batch_id}.jsonl"
        existing = {}
        if path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                existing[rec[0]] = rec
        todo = [r for r in rows if existing.get(r["uid"], [None, None, "retry"])[2]
                is not None]
        groups = [todo[i:i + self.group_size]
                  for i in range(0, len(todo), self.group_size)]
        with path.open("a") as spool, ThreadPoolExecutor(
                max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._call, group) for group in groups]
            for future in as_completed(futures):
                for result in future.result():
                    spool.write(json.dumps(result) + "\n")
                spool.flush()
        return batch_id

    def status(self, batch_id: str) -> str:
        return "ended"

    def fetch(self, batch_id: str):
        latest = {}
        for line in (self.spool_dir / f"{batch_id}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            latest[rec[0]] = tuple(rec)
        return list(latest.values())


class AzureResponsesLabeler(OpenRouterLabeler):
    """Azure Responses API transport reusing the durable local spool."""

    def __init__(self, deployment: str, spool_dir, reviews_per_request=250,
                 concurrency=10, session=None, usage_path=None,
                 max_budget_usd=None, input_price_per_million=None,
                 output_price_per_million=None, max_requests=None,
                 reasoning_effort=None, timeout=None):
        import requests
        key = os.environ.get("AZURE_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        url = os.environ.get("TARGET_URL") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not key or not url or not deployment:
            raise RuntimeError(
                "Azure labeling requires TARGET_URL/AZURE_OPENAI_ENDPOINT, "
                "AZURE_KEY/AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT"
            )
        self.model = deployment
        self.provider = "azure"
        self.prompt_version = PROMPT_VERSION
        self.url = url
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = reviews_per_request
        self.concurrency = concurrency
        self.session = session or requests
        self.headers = {"api-key": key, "Content-Type": "application/json"}
        self.usage_path = Path(usage_path or self.spool_dir.parent / "azure_usage.jsonl")
        self.max_budget_usd = (float(max_budget_usd)
                               if max_budget_usd is not None else None)
        self.input_price_per_million = (
            _validated_price(input_price_per_million, "input_price_per_million")
            if input_price_per_million is not None else None
        )
        self.output_price_per_million = (
            _validated_price(output_price_per_million, "output_price_per_million")
            if output_price_per_million is not None else None
        )
        # A budget with no prices is not a budget: every request would cost $0
        # and the ceiling could never fire. Either both prices are present and
        # positive, or there is no cumulative dollar ceiling to enforce.
        if self.max_budget_usd is not None and (
                self.input_price_per_million is None
                or self.output_price_per_million is None):
            raise RuntimeError(
                "max_budget_usd cannot be enforced without both "
                "input_price_per_million and output_price_per_million"
            )
        self.max_requests = int(max_requests) if max_requests is not None else None
        self.reasoning_effort = (reasoning_effort
                                 or os.environ.get("AZURE_REASONING_EFFORT")
                                 or "medium")
        # Larger batches with reasoning can run for minutes; 180s truncated them.
        self.timeout = int(timeout or os.environ.get("AZURE_TIMEOUT_SECONDS") or 600)
        self._usage_lock = threading.RLock()
        self._reservations: dict[str, float] = {}
        self._ceilings: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _output_text(data: dict) -> str:
        if data.get("output_text"):
            return data["output_text"]
        for item in data.get("output") or []:
            for content in item.get("content") or []:
                if content.get("type") == "output_text" and content.get("text"):
                    return content["text"]
        raise ValueError("Azure response contained no output text")

    def _call(self, rows):
        uids = [row["uid"] for row in rows]
        reviews = "\n".join(json.dumps({"i": index, "text": row["text"]})
                            for index, row in enumerate(rows))
        compact_instructions = (
            SYSTEM_PROMPT
            + "\n\nFor this request, encode each label as a three-digit COH string: "
            "C is concern, O is outcome, and H is reviewer_has_condition "
            "(0=false, 1=true). Concern codes: "
            + ", ".join(f"{i}={value}" for i, value in enumerate(CONCERNS))
            + ". Outcome codes: "
            + ", ".join(f"{i}={value}" for i, value in enumerate(COMPACT_OUTCOMES))
            + ". Return r as an array of objects, exactly one per input line: "
            "each is {\"i\": that input line's i, \"c\": its label-code array}. "
            "Include every input index exactly once. Use \"c\": [] when no "
            "concern is mentioned."
        )
        max_output_tokens = 120 * len(rows)
        if self.reasoning_effort != "minimal":
            # Responses API reasoning tokens count against max_output_tokens;
            # without headroom the answer is truncated and the batch fails.
            max_output_tokens += 16_000
        body = {
            "model": self.model,
            "instructions": compact_instructions,
            "input": reviews,
            "max_output_tokens": max_output_tokens,
            "store": False,
            "text": {"format": {
                "type": "json_schema",
                "name": "review_concern_labels",
                "strict": True,
                "schema": _compact_batch_schema(len(rows)),
            }},
        }
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            body["reasoning"] = {"effort": self.reasoning_effort}
        response = None
        data = {}
        # The reservation is keyed by a stable local id; the ledger row later
        # adopts Azure's response id, so keep the reservation key separate or the
        # in-flight hold leaks and eventually trips the budget ceiling falsely.
        reservation_key = uuid4().hex
        request_id = reservation_key
        reservation_error = self._reserve_request(body, reservation_key)
        if reservation_error is not None:
            return [(uid, None, reservation_error) for uid in uids]
        status = "failed"
        try:
            response = self.session.post(
                self.url, headers=self.headers, json=body, timeout=self.timeout,
            )
            try:
                data = response.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            request_id = self._request_id(data, response, request_id)
            response.raise_for_status()
            parsed = json.loads(self._output_text(data))
            encoded_rows = parsed["r"]
            if len(encoded_rows) != len(uids):
                raise ValueError("Azure response row count did not match input")
            by_index: dict[int, list] = {}
            for item in encoded_rows:
                index = item["i"]
                if index in by_index:
                    raise ValueError("Azure response repeated a row index")
                by_index[index] = item["c"]
            if set(by_index) != set(range(len(uids))):
                raise ValueError("Azure response index set did not match input")
            decoded_rows = [[{
                "concern": CONCERNS[int(code[0])],
                "outcome": COMPACT_OUTCOMES[int(code[1])],
                "reviewer_has_condition": code[2] == "1",
            } for code in by_index[index]] for index in range(len(uids))]
            status = "succeeded"
            return [(uid, json.dumps({"labels": decoded_rows[index]}), None)
                    for index, uid in enumerate(uids)]
        except Exception as exc:
            return [(uid, None, type(exc).__name__) for uid in uids]
        finally:
            self._append_usage(rows, data, request_id, status, reservation_key,
                               response)

    def _reserve_request(self, body: dict, request_id: str) -> str | None:
        """Reserve a conservative per-request ceiling before HTTP submission.

        UTF-8 byte length is a safe upper bound on input token count, while
        max_output_tokens is the provider-enforced output ceiling. In-flight
        reservations prevent concurrent calls from collectively crossing the
        configured cumulative limits.
        """
        if self.max_budget_usd is None and self.max_requests is None:
            return None
        input_ceiling = len(json.dumps(
            body, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8"))
        output_ceiling = int(body.get("max_output_tokens") or 0)
        # No `or 0.0` fallback: __init__ refuses a max_budget_usd without both
        # positive prices, so this arithmetic can never quietly price a paid
        # request at $0 and hand back an unenforceable ceiling.
        reserved_cost = (
            (input_ceiling * self.input_price_per_million
             + output_ceiling * self.output_price_per_million) / 1e6
            if self.max_budget_usd is not None else 0.0
        )
        with self._usage_lock:
            usage = azure_usage_summary(self.usage_path, self.model, None)
            if (self.max_requests is not None
                    and usage["requests"] + len(self._reservations) + 1
                    > self.max_requests):
                return "request_ceiling"
            if self.max_budget_usd is not None:
                actual_cost = (
                    usage["input_tokens"] * self.input_price_per_million
                    + usage["output_tokens"] * self.output_price_per_million
                ) / 1e6
                if (actual_cost + sum(self._reservations.values())
                        + reserved_cost > self.max_budget_usd):
                    return "budget_ceiling"
            self._reservations[request_id] = reserved_cost
            self._ceilings[request_id] = (input_ceiling, output_ceiling)
        return None

    @staticmethod
    def _request_id(data: dict, response, fallback: str) -> str:
        headers = getattr(response, "headers", {}) or {}
        return str(data.get("id") or headers.get("x-request-id")
                   or headers.get("request-id") or fallback)

    def _append_usage(self, rows, data: dict, request_id: str, status: str,
                      reservation_key: str | None = None,
                      response=None) -> None:
        if not isinstance(data, dict):
            data = {}
        usage = data.get("usage") or {}
        if usage:
            input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        elif response is not None and getattr(response, "status_code", 0) >= 400:
            # An HTTP ERROR came back (e.g. 429 rate limit or 5xx): Azure rejected
            # the request before generating, so it is genuinely unbilled — record
            # $0, not the ceiling (otherwise rate-limit retries inflate the ledger
            # and can trip the budget guard mid-pass). Only the status code proves
            # this: a usage-less 2xx was generated and BILLED (a proxy's HTML
            # error page, a body truncated by a reset), and charging it $0 would
            # let an unbounded number of billed requests report as free.
            input_tokens = output_tokens = 0
        else:
            # No response, or a usage-less success: Azure may have generated and
            # billed it, so record the conservative reserved ceiling rather than
            # $0 — never under-count real spend. An unknown status code lands
            # here too, which is the safe direction.
            input_tokens, output_tokens = self._ceilings.get(
                reservation_key, (0, 0))
        record = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "request_id": request_id,
            "status": status,
            "rows": len(rows),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        }
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._usage_lock, self.usage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._reservations.pop(reservation_key, None)
            self._ceilings.pop(reservation_key, None)


def _azure_settings() -> tuple[str, str, str] | None:
    key = os.environ.get("AZURE_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    url = os.environ.get("TARGET_URL") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not any((key, url, deployment)):
        return None
    if not all((key, url, deployment)):
        raise RuntimeError(
            "Azure configuration is incomplete; set TARGET_URL/AZURE_OPENAI_ENDPOINT, "
            "AZURE_KEY/AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT"
        )
    return key, url, deployment


def _validated_price(value, source: str) -> float:
    """A price that can actually enforce a budget: finite and strictly positive.

    A price of 0 does not mean "free", it means "every ceiling computes to $0
    and can never fire". configs/default.yaml ships input_price_per_million:
    0.0 / output_price_per_million: 0.0 for the OpenRouter FREE model, directly
    adjacent to the Azure knobs, so copying them across is a one-character way
    to disable the budget guard on a paid endpoint. Refuse instead.
    """
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{source} must be a positive number of dollars per million "
            f"tokens; got {value!r}"
        ) from exc
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError(
            f"{source} must be > 0 to enforce the Azure budget ceiling; got "
            f"{value!r}. The 0.0 prices in configs/default.yaml belong to the "
            "free OpenRouter model and are not valid Azure prices."
        )
    return price


def _azure_env_price(name: str) -> float | None:
    """Validated $/million from the environment; None only when unset."""
    raw = os.environ.get(name)
    return None if raw is None else _validated_price(raw, name)


def _calibrated_output_tokens_per_row(ccfg: dict, deployment: str) -> float:
    path = Path(ccfg.get("azure_usage_path") or "")
    if not path.is_file():
        return 120.0
    rows = output_tokens = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "failed":
            continue
        if (record.get("provider", "azure") != "azure"
                or record.get("model") != deployment
                or record.get("prompt_version") != PROMPT_VERSION):
            continue
        rows += int(record.get("rows") or 0)
        output_tokens += int(record.get("output_tokens") or 0)
    if rows < 50:
        return 120.0
    return output_tokens / rows * 1.25


def azure_usage_summary(path, deployment: str,
                        prompt_version: str | None = PROMPT_VERSION) -> dict:
    """Summarize Azure attempts for a deployment, optionally across prompts."""
    summary = {"provider": "azure", "model": deployment,
               "prompt_version": prompt_version, "requests": 0,
               "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    path = Path(path or "")
    if not path.is_file():
        return summary
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (record.get("provider", "azure") != "azure"
                or record.get("model") != deployment
                or (prompt_version is not None
                    and record.get("prompt_version") != prompt_version)):
            continue
        summary["requests"] += 1
        summary["input_tokens"] += int(record.get("input_tokens") or 0)
        summary["output_tokens"] += int(record.get("output_tokens") or 0)
        summary["total_tokens"] += int(record.get("total_tokens") or 0)
    return summary


def _azure_max_requests(ccfg: dict) -> int:
    configured = os.environ.get("AZURE_MAX_REQUESTS") or os.environ.get(
        "AZURE_MAX_REQUEST_COUNT")
    if configured is None:
        configured = ccfg.get(
            "azure_max_requests",
            ccfg.get("max_request_count", ccfg.get("max_requests", 900)),
        )
    return int(configured)


def azure_preflight(rows, ccfg: dict) -> dict | None:
    azure = _azure_settings()
    if azure is None:
        return None
    _key, _url, deployment = azure
    estimated = estimate_cost(rows, ccfg)
    usage = azure_usage_summary(ccfg.get("azure_usage_path"), deployment, None)
    input_price = _validated_price(os.environ.get("AZURE_INPUT_PRICE_PER_MILLION"),
                                   "AZURE_INPUT_PRICE_PER_MILLION")
    output_price = _validated_price(os.environ.get("AZURE_OUTPUT_PRICE_PER_MILLION"),
                                    "AZURE_OUTPUT_PRICE_PER_MILLION")
    actual = (usage["input_tokens"] / 1e6 * input_price
              + usage["output_tokens"] / 1e6 * output_price)
    planned_requests = ((len(rows) + ccfg["reviews_per_request"] - 1)
                        // ccfg["reviews_per_request"])
    max_requests = _azure_max_requests(ccfg)
    result = {
        "provider": "azure", "deployment": deployment,
        "prompt_version": PROMPT_VERSION,
        "historical_requests": usage["requests"],
        "planned_requests": planned_requests,
        "request_count": usage["requests"] + planned_requests,
        "max_request_count": max_requests,
        "historical_input_tokens": usage["input_tokens"],
        "historical_output_tokens": usage["output_tokens"],
        "historical_cost_usd": actual,
        "estimated_cost_usd": estimated,
        "projected_cost_usd": actual + estimated,
        "max_budget_usd": float(ccfg["max_budget_usd"]),
    }
    if result["projected_cost_usd"] > result["max_budget_usd"]:
        raise RuntimeError(
            "Azure cumulative budget preflight failed: "
            f"${result['historical_cost_usd']:.4f} actual + "
            f"${result['estimated_cost_usd']:.4f} planned = "
            f"${result['projected_cost_usd']:.4f} > "
            f"${result['max_budget_usd']:.4f} budget"
        )
    if result["request_count"] > max_requests:
        raise RuntimeError(
            "Azure cumulative request preflight failed: "
            f"{result['request_count']} requests > {max_requests} max requests"
        )
    return result


def estimate_cost(rows, ccfg) -> float:
    """Conservative provider cost estimate using the maximum output allowance."""
    groups = (len(rows) + ccfg["reviews_per_request"] - 1) // ccfg["reviews_per_request"]
    input_tokens = sum(len(r["text"]) for r in rows) / 4 + 450 * groups
    azure = _azure_settings()
    output_tokens = 120 * len(rows)
    if azure is not None:
        _key, _url, deployment = azure
        output_tokens = _calibrated_output_tokens_per_row(ccfg, deployment) * len(rows)
        input_price = _azure_env_price("AZURE_INPUT_PRICE_PER_MILLION")
        output_price = _azure_env_price("AZURE_OUTPUT_PRICE_PER_MILLION")
        if input_price is None or output_price is None:
            raise RuntimeError(
                "Azure full-pass preflight requires AZURE_INPUT_PRICE_PER_MILLION "
                "and AZURE_OUTPUT_PRICE_PER_MILLION"
            )
        return (input_tokens / 1e6 * input_price
                + output_tokens / 1e6 * output_price)
    return (input_tokens / 1e6 * ccfg["input_price_per_million"]
            + output_tokens / 1e6 * ccfg["output_price_per_million"])


def _labeler(ccfg):
    azure = _azure_settings()
    if azure is not None:
        _key, _url, deployment = azure
        input_price = _azure_env_price("AZURE_INPUT_PRICE_PER_MILLION")
        output_price = _azure_env_price("AZURE_OUTPUT_PRICE_PER_MILLION")
        if input_price is None or output_price is None:
            raise RuntimeError(
                "Azure labeling requires AZURE_INPUT_PRICE_PER_MILLION and "
                "AZURE_OUTPUT_PRICE_PER_MILLION for runtime budget enforcement"
            )
        return AzureResponsesLabeler(
            deployment, ccfg["batch_spool_dir"], ccfg["reviews_per_request"],
            ccfg["request_concurrency"],
            usage_path=ccfg.get("azure_usage_path"),
            max_budget_usd=ccfg["max_budget_usd"],
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            max_requests=_azure_max_requests(ccfg),
            reasoning_effort=ccfg.get("azure_reasoning_effort"),
            timeout=ccfg.get("azure_timeout_seconds"),
        )
    return OpenRouterLabeler(
        ccfg["labeling_model"], ccfg["batch_spool_dir"],
        ccfg["reviews_per_request"], ccfg["request_concurrency"])


def _configured_labeler_identity(ccfg) -> tuple[str, str, str]:
    azure = _azure_settings()
    if azure is not None:
        return "azure", azure[2], PROMPT_VERSION
    return "openrouter", ccfg["labeling_model"], PROMPT_VERSION


def _match_counts(rows, patterns):
    """Per-concern joinable match counts + per-product cell sizes."""
    counts = {c: 0 for c in patterns}
    cells = {c: {} for c in patterns}
    for row in rows:
        low = row["text"].lower()
        for concern, rx in patterns.items():
            if rx.search(low):
                counts[concern] += 1
                cells[concern][row["product_id"]] = (
                    cells[concern].get(row["product_id"], 0) + 1)
    return counts, cells


def cmd_probe(rows, patterns) -> bool:
    counts, cells = _match_counts(rows, patterns)
    gate_products = set()
    print(f"joinable prefiltered rows: {len(rows)}")
    for concern in CONCERNS:
        n15 = [p for p, n in cells[concern].items() if n >= 15]
        print(f"{concern}: rows {counts[concern]}, products n>=15: {len(n15)}")
        if concern in ACNE_CONCERNS:
            gate_products.update(n15)
    passed = len(gate_products) >= 300
    print(f"gate_p1: {'PASS' if passed else 'FAIL'} "
          f"({len(gate_products)} >= 300 acne-concern products with n>=15)")
    return passed


def _validated_calibration_audit(audit_path, sample_path,
                                 expected_uids: list[str]) -> dict:
    audit_path = Path(audit_path)
    sample_path = Path(sample_path)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError("calibration audit artifact is missing or invalid") from exc
    if audit.get("schema_version") != "concern-calibration-audit-1":
        raise RuntimeError("calibration audit has an unsupported schema_version")
    if audit.get("policy_prompt_version") != PROMPT_VERSION:
        raise RuntimeError("calibration audit prompt version does not match")
    sample_sha256 = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    if audit.get("sample_sha256") != sample_sha256:
        raise RuntimeError("calibration audit sample_sha256 does not match")
    entries = audit.get("audits")
    if not isinstance(entries, list):
        raise RuntimeError("calibration audit must contain an audits list")
    uids = [entry.get("uid") for entry in entries if isinstance(entry, dict)]
    if (len(uids) != len(entries) or len(set(uids)) != len(uids)
            or set(uids) != set(expected_uids)):
        raise RuntimeError("calibration audit UIDs do not match the sample")
    if any(not isinstance(entry.get("exact_match"), bool) for entry in entries):
        raise RuntimeError("calibration audit exact_match values must be booleans")
    audited_rows = len(entries)
    exact_matches = sum(entry["exact_match"] for entry in entries)
    measured_agreement = exact_matches / audited_rows if audited_rows else 0.0
    try:
        declared_rows = int(audit.get("audited_rows"))
        declared_matches = int(audit.get("exact_matches"))
        declared_agreement = float(audit.get("measured_agreement"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("calibration audit summary is incomplete") from exc
    if declared_rows != audited_rows or declared_matches != exact_matches:
        raise RuntimeError("calibration audit summary counts do not match its rows")
    if abs(declared_agreement - measured_agreement) > 1e-9:
        raise RuntimeError("calibration audit measured agreement is inconsistent")
    reviewer_model = audit.get("reviewer_model")
    reasoning_effort = audit.get("reasoning_effort")
    if not reviewer_model or not reasoning_effort:
        raise RuntimeError("calibration audit reviewer identity is missing")
    return {
        "path": str(audit_path),
        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "sample_sha256": sample_sha256,
        "reviewer_model": reviewer_model,
        "reasoning_effort": reasoning_effort,
        "audited_rows": audited_rows,
        "exact_matches": exact_matches,
        "measured_agreement": measured_agreement,
    }


def cmd_calibrate(rows, ccfg, n, audited_rows=0, agreement=None,
                  audit_path=None) -> dict:
    if audit_path is None and (audited_rows or agreement is not None):
        raise RuntimeError(
            "raw audit counts cannot approve P2; pass --audit-file with a "
            "sample-bound audit artifact"
        )
    sample = sorted(rows, key=lambda r: r["uid"])[:n]   # deterministic
    labeler = _labeler(ccfg)
    summary = run_labeling(sample, labeler, ccfg["labels_path"],
                           ccfg["batch_state_path"], ccfg["batch_chunk_size"])
    provider, model, prompt_version = _labeler_identity(labeler)
    cache = load_cache(ccfg["labels_path"], prompt_version, provider, model)
    sample_recs = [cache[r["uid"]] for r in sample if r["uid"] in cache]
    ok = [r for r in sample_recs if r["status"] == "ok"]
    outcome_bearing = [r for r in ok if any(
        l["outcome"] in ("helped", "worsened") for l in r["labels"])]
    yield_rate = len(outcome_bearing) / max(len(sample), 1)
    yield_pass = yield_rate >= 0.30
    report_path = _calibration_report_path(ccfg)
    out_dir = report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    by_uid = {r["uid"]: r for r in sample}
    audited_records = sample_recs[:50]
    hand = pd.DataFrame([{"uid": r["uid"], "text": by_uid[r["uid"]]["text"],
                          "labels": json.dumps(r["labels"])}
                         for r in audited_records])
    sample_path = out_dir / "calibration_sample.csv"
    hand.to_csv(sample_path, index=False)
    audit = (_validated_calibration_audit(
        audit_path, sample_path, [record["uid"] for record in audited_records],
    ) if audit_path is not None else None)
    if audit is not None:
        audited_rows = audit["audited_rows"]
        agreement = audit["measured_agreement"]
    agreement_pass = agreement is not None and agreement >= 0.85
    audited_rows_pass = audited_rows >= 50
    report = {"sample_size": len(sample), "labeled_ok": len(ok),
              "outcome_bearing": len(outcome_bearing),
              "yield": round(yield_rate, 4), "run_summary": summary,
              "provider": provider, "model": model,
              "prompt_version": prompt_version,
              "yield_pass": yield_pass,
              "gate_p2_yield": "PASS" if yield_pass else "FAIL",
              "audited_rows": audited_rows,
              "audited_rows_pass": audited_rows_pass,
              "measured_agreement": agreement,
              "agreement_pass": agreement_pass,
              "gate_p2": "PASS" if yield_pass and agreement_pass
              and audited_rows_pass else "FAIL"}
    if audit is not None:
        report["audit"] = audit
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("audit runs/concern/calibration_sample.csv independently, then rerun "
          "calibrate with --audit-file")
    return report


def _calibration_report_path(ccfg) -> Path:
    return Path(ccfg.get("calibration_report_path")
                or Path(ccfg["batch_state_path"]).parent / "calibration_report.json")


def _require_calibration_report(ccfg) -> dict:
    path = _calibration_report_path(ccfg)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "full labeling requires a persisted P2 calibration report/sign-off"
        ) from exc
    try:
        measured_yield = float(report.get("yield", 0))
    except (TypeError, ValueError):
        measured_yield = 0
    if "yield" in report:
        yield_pass = measured_yield >= 0.30 and report.get(
            "yield_pass", report.get("gate_p2_yield", report.get("yield_gate", "PASS"))
        ) not in (False, "FAIL")
    else:
        yield_pass = (
            report.get("yield_pass") is True
            or report.get("yield_pass") == "PASS"
            or report.get("gate_p2_yield") == "PASS"
            or report.get("yield_gate") == "PASS"
        )
    agreement = report.get("measured_agreement", report.get("agreement"))
    audited_rows = report.get(
        "audited_rows", report.get("audited_row_count", report.get("audit_rows", 0))
    )
    try:
        agreement_pass = float(agreement) >= 0.85
        audited_rows_pass = int(audited_rows) >= 50
    except (TypeError, ValueError):
        agreement_pass = audited_rows_pass = False
    # The sign-off must certify the CURRENT policy/prompt version; a stale report
    # from an earlier version cannot approve a changed labeler (cf. the P3 gate).
    version_pass = report.get("prompt_version") == PROMPT_VERSION
    if not (yield_pass and agreement_pass and audited_rows_pass and version_pass):
        raise RuntimeError(
            "P2 sign-off calibration report failed: requires yield PASS, "
            "measured agreement >=0.85, audited rows >=50, and a report whose "
            f"prompt_version matches {PROMPT_VERSION!r}"
        )
    return report


def cmd_label(rows, ccfg, yes: bool) -> dict | None:
    provider, model, prompt_version = _configured_labeler_identity(ccfg)
    cache = load_cache(ccfg["labels_path"], prompt_version, provider, model)
    todo = [r for r in rows if r["uid"] not in cache]
    est_usd = estimate_cost(todo, ccfg)
    preflight = azure_preflight(todo, ccfg)
    print(f"to label: {len(todo)} of {len(rows)} "
          f"(est cost ${est_usd:.2f} on {ccfg['labeling_model']})")
    if preflight is not None:
        print(json.dumps({"azure_preflight": preflight}, indent=2))
    if not yes:
        print("dry run — pass --yes to submit")
        return None
    _require_calibration_report(ccfg)
    if est_usd > ccfg["max_budget_usd"]:
        raise RuntimeError(f"estimated ${est_usd:.2f} exceeds "
                           f"${ccfg['max_budget_usd']:.2f} budget ceiling")
    labeler = _labeler(ccfg)
    summary = run_labeling(rows, labeler, ccfg["labels_path"],
                           ccfg["batch_state_path"], ccfg["batch_chunk_size"])
    if preflight is not None:
        summary.update({
            "azure_historical_cost_usd": preflight["historical_cost_usd"],
            "azure_estimated_cost_usd": preflight["estimated_cost_usd"],
            "azure_projected_cost_usd": preflight["projected_cost_usd"],
            "azure_request_count": preflight["request_count"],
            "azure_max_request_count": preflight["max_request_count"],
        })
    print(json.dumps(summary, indent=2))
    print("next: python -m recsys.tools.build_concern_efficacy "
          "--labels data/processed/review_concern_labels.jsonl (see recsys/README.md)")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "calibrate", "label"):
        sp = sub.add_parser(name)
        sp.add_argument("--reviews-dir")
        sp.add_argument("--catalog")
        if name == "calibrate":
            sp.add_argument("--n", type=int)
            sp.add_argument("--audited-rows", type=int, default=0)
            sp.add_argument("--agreement", type=float)
            sp.add_argument("--audit-file", type=Path)
        if name == "label":
            sp.add_argument("--yes", action="store_true")
            # No --p2-approved flag: P2 is approved by a persisted, sample-bound
            # calibration report (_require_calibration_report), never by an
            # operator asserting it on the command line.
    args = ap.parse_args(argv)
    cfg = load_config()
    ccfg = cfg["concern"]
    patterns = compile_prefilter(ccfg["prefilter"])
    catalog = load_catalog(args.catalog or cfg["paths"]["catalog_processed"])
    catalog_ids = {p.product_id for p in catalog}
    rows = load_review_rows(args.reviews_dir or cfg["paths"]["reviews_raw"],
                            catalog_ids, patterns, ccfg["text_truncate_chars"])
    if args.cmd == "probe":
        cmd_probe(rows, patterns)
    elif args.cmd == "calibrate":
        cmd_calibrate(rows, ccfg, args.n or ccfg["calibration_sample_size"],
                      args.audited_rows, args.agreement, args.audit_file)
    else:
        cmd_label(rows, ccfg, args.yes)


if __name__ == "__main__":
    main()
