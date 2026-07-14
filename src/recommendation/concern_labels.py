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
import os
import re
import threading
import time
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
PROMPT_VERSION = "p7"

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
        r"\b(?:blackheads?|whiteheads?|comedones?|(?:clogg\w*|unclog\w*|"
        r"clear(?:ed|s|ing)?)(?:\s+\w+){0,4}\s+pores?|"
        r"pores?\s+(?:look(?:ed)?\s+)?"
        r"(?:clogg\w*|plugg\w*))\b", re.I,
    ),
    "acne_inflammatory": re.compile(r"\b(?:pimples?|zits?|pustules?|papules?)\b", re.I),
    "acne_cystic": re.compile(
        r"\b(?:cystic acne|hormonal acne|hormonal breakouts?|deep painful bumps?)\b", re.I,
    ),
    "acne_general": re.compile(
        r"\b(?:(?<!cystic )(?<!hormonal )acne(?!\s+(?:scars?|scarring|marks?))|"
        r"br(?:eak|ake)\s?outs?|blemishes?)\b", re.I,
    ),
    "hyperpigmentation": re.compile(
        r"\b(?:dark spots?|acne (?:scars?|marks?)|discoloration|melasma|"
        r"sun spots?|hyper[- ]?pigmentation)\b", re.I,
    ),
    "dryness": re.compile(
        r"\b(?:dry|drying|dryness|flak\w*|dry patches?|dehydrat\w*)\b", re.I,
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
    r"have not|haven['’]t|never)\b[^.!?]{0,45}"
    r"\b(?:break\s?outs?|br(?:eak|oke) me out|clog\w* pores?)\b|"
    r"\bwithout (?:any )?(?:break\s?outs?|clogg\w* pores?)\b", re.I,
)
_ABSENT_CONDITION = re.compile(
    r"\bi (?:do not|don't|dont|did not|didn't|never) (?:really )?"
    r"(?:get|have)[^.!?]{0,25}\b", re.I,
)


def _concern_sentences(text: str, concern: str) -> list[str]:
    pattern = LITERAL_PATTERNS[concern]
    matches = [sentence.strip() for sentence in re.split(r"[.!?]+", text)
               if pattern.search(sentence)]
    if concern == "dryness":
        matches = [sentence for sentence in matches
                   if not (re.search(r"\b(?:once dry|let it dry|dr(?:y|ies) down|dry finish)\b",
                                     sentence, re.I)
                           and not re.search(r"\b(?:skin|face|cheeks?|jawline|patches|dryness|drying)\b",
                                             sentence, re.I))]
    return matches


def _explicit_outcome(concern: str, sentences: list[str]) -> str | None:
    joined = " ".join(sentences)
    if not joined:
        return None
    if re.search(r"\b(?:this(?: product)?|it|the product)\s+"
                 r"(?:caused?|gave me|made my)\b", joined, re.I):
        return "worsened"
    if _PRODUCT_PREVENTION.search(joined):
        return "helped"
    if re.search(r"\bstops?\b[^.!?]{0,30}\b(?:acne|break\s?outs?|blemishes?)\b",
                 joined, re.I):
        return "helped"
    if _WORSENED.search(joined):
        return "worsened"
    if _HELPED.search(joined):
        return "helped"
    if concern == "acne_cystic" and re.search(r"\bproduct is amazing\b", joined, re.I):
        return "helped"
    return None


def _personal_condition(concern: str, sentences: list[str], outcome: str | None) -> bool | None:
    joined = " ".join(sentences)
    if not joined:
        return None
    term = LITERAL_PATTERNS[concern].pattern
    if re.search(rf"\bmy\b[^.!?]{{0,45}}(?:{term})", joined, re.I):
        return True
    if re.search(rf"\bi\s+(?:have|had|get|got|do|am|was|suffer\w*|experience\w*|"
                 rf"undergo\w*)\b[^.!?]{{0,150}}(?:{term})", joined, re.I):
        return True
    if _ABSENT_CONDITION.search(joined):
        return False
    if concern == "dryness" and re.search(r"\bmy skin (?:feels?|is|was)\b[^.!?]{0,25}\bdry\b",
                                           joined, re.I):
        return True
    if concern == "acne_general" and re.search(r"\bmy skin\b[^.!?]{0,25}\bacne[- ]prone\b",
                                                joined, re.I):
        return True
    if concern == "acne_general" and re.search(r"\bi have\b[^.!?]{0,35}\bacne[- ]prone skin\b",
                                                joined, re.I):
        return True
    if outcome == "worsened" and re.search(r"\b(?:me|my|i)\b", joined, re.I):
        return True
    if _PRODUCT_PREVENTION.search(joined):
        return False
    if (re.search(r"\bmy (?:face|skin|chin|nose|cheeks?|jawline)\b", joined, re.I)
            and LITERAL_PATTERNS[concern].search(joined)):
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
        explicit = _explicit_outcome(concern, sentences[concern])
        personal = _personal_condition(concern, sentences[concern], explicit)
        if (personal is False and _ABSENT_CONDITION.search(" ".join(sentences[concern]))
                and not _PRODUCT_PREVENTION.search(" ".join(sentences[concern]))):
            explicit = None
        label = by_concern.get(concern)
        if label is None:
            outcome = explicit
            if outcome is None and concern in ACNE_CONCERNS:
                sibling_outcomes = {
                    item["outcome"] for key, item in by_concern.items()
                    if key in ACNE_CONCERNS and item["outcome"] != "unclear"
                }
                if len(sibling_outcomes) == 1 and personal is not False:
                    outcome = sibling_outcomes.pop()
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
    key = f"{author_id}|{product_id}|{text[:300]}"
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


def load_cache(path, prompt_version: str = PROMPT_VERSION) -> dict[str, dict]:
    """uid -> cached record. Missing file -> empty (first run)."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("prompt_version") == prompt_version:
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


def _record(row: dict, status: str, labels: list) -> dict:
    return {"uid": row["uid"], "author_id": row["author_id"],
            "product_id": row["product_id"], "skin_type": row["skin_type"],
            "skin_tone": row["skin_tone"], "rating": row["rating"],
            "is_recommended": row["is_recommended"],
            "prompt_version": PROMPT_VERSION,
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
    cache = load_cache(cache_path)
    by_uid = {r["uid"]: r for r in rows}
    state = _load_state(state_path)
    summary = {"cached_before": 0, "submitted": 0, "ok": 0,
               "parse_error": 0, "refusal": 0, "failed": 0}

    def drain(batch_id):
        while labeler.status(batch_id) != "ended":
            sleep(poll_seconds)
        new = []
        for uid, text, failure in labeler.fetch(batch_id):
            row = by_uid.get(uid)
            if row is None or uid in cache:
                continue
            if failure == "refusal":
                new.append(_record(row, "refusal", []))
                summary["refusal"] += 1
            elif failure is not None:
                summary["failed"] += 1     # not cached -> retryable
            else:
                try:
                    labels = enforce_literal_policy(row["text"], _parse_labels(text))
                    new.append(_record(row, "ok", labels))
                    summary["ok"] += 1
                except (ValueError, KeyError, TypeError, AttributeError):
                    new.append(_record(row, "parse_error", []))
                    summary["parse_error"] += 1
        append_cache(cache_path, new)
        cache.update({r["uid"]: r for r in new})
        state["batches"][batch_id] = {"fetched": True}
        _save_state(state_path, state)

    # 1) drain leftovers from a crashed run
    for bid, meta in list(state["batches"].items()):
        if not meta.get("fetched"):
            drain(bid)

    # 2) submit what is still unlabeled, then drain each batch
    todo = [r for r in rows if r["uid"] not in cache]
    summary["cached_before"] = len(rows) - len(todo)
    pending = []
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        bid = labeler.submit(chunk)
        summary["submitted"] += len(chunk)
        state["batches"][bid] = {"fetched": False}
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
            (PROMPT_VERSION + "|" + self.model + "|"
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
                 concurrency=10, session=None, usage_path=None):
        import requests
        key = os.environ.get("AZURE_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        url = os.environ.get("TARGET_URL") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not key or not url or not deployment:
            raise RuntimeError(
                "Azure labeling requires TARGET_URL/AZURE_OPENAI_ENDPOINT, "
                "AZURE_KEY/AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT"
            )
        self.model = deployment
        self.url = url
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.group_size = reviews_per_request
        self.concurrency = concurrency
        self.session = session or requests
        self.headers = {"api-key": key, "Content-Type": "application/json"}
        self.usage_path = Path(usage_path or self.spool_dir.parent / "azure_usage.jsonl")
        self._usage_lock = threading.Lock()

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
        reviews = "\n".join(json.dumps({"uid": row["uid"], "text": row["text"]})
                            for row in rows)
        body = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": reviews,
            "max_output_tokens": 120 * len(rows),
            "store": False,
            "text": {"format": {
                "type": "json_schema",
                "name": "review_concern_labels",
                "strict": True,
                "schema": _batch_schema(uids),
            }},
        }
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            body["reasoning"] = {"effort": "minimal"}
        try:
            response = self.session.post(
                self.url, headers=self.headers, json=body, timeout=180,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            if usage:
                record = {
                    "model": self.model,
                    "prompt_version": PROMPT_VERSION,
                    "rows": len(rows),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                self.usage_path.parent.mkdir(parents=True, exist_ok=True)
                with self._usage_lock, self.usage_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            parsed = json.loads(self._output_text(data))
            by_uid = {item["uid"]: item["labels"] for item in parsed["results"]}
            return [(uid, json.dumps({"labels": by_uid[uid]}), None)
                    if uid in by_uid else (uid, None, "missing_result")
                    for uid in uids]
        except Exception as exc:
            return [(uid, None, type(exc).__name__) for uid in uids]


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


def estimate_cost(rows, ccfg) -> float:
    """Conservative provider cost estimate using the maximum output allowance."""
    groups = (len(rows) + ccfg["reviews_per_request"] - 1) // ccfg["reviews_per_request"]
    input_tokens = sum(len(r["text"]) for r in rows) / 4 + 450 * groups
    output_tokens = 120 * len(rows)
    if _azure_settings() is not None:
        input_price = os.environ.get("AZURE_INPUT_PRICE_PER_MILLION")
        output_price = os.environ.get("AZURE_OUTPUT_PRICE_PER_MILLION")
        if input_price is None or output_price is None:
            raise RuntimeError(
                "Azure full-pass preflight requires AZURE_INPUT_PRICE_PER_MILLION "
                "and AZURE_OUTPUT_PRICE_PER_MILLION"
            )
        return (input_tokens / 1e6 * float(input_price)
                + output_tokens / 1e6 * float(output_price))
    return (input_tokens / 1e6 * ccfg["input_price_per_million"]
            + output_tokens / 1e6 * ccfg["output_price_per_million"])


def _labeler(ccfg):
    azure = _azure_settings()
    if azure is not None:
        _key, _url, deployment = azure
        return AzureResponsesLabeler(
            deployment, ccfg["batch_spool_dir"], ccfg["reviews_per_request"],
            ccfg["request_concurrency"],
            usage_path=ccfg.get("azure_usage_path"),
        )
    return OpenRouterLabeler(
        ccfg["labeling_model"], ccfg["batch_spool_dir"],
        ccfg["reviews_per_request"], ccfg["request_concurrency"])


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


def cmd_calibrate(rows, ccfg, n) -> dict:
    sample = sorted(rows, key=lambda r: r["uid"])[:n]   # deterministic
    labeler = _labeler(ccfg)
    summary = run_labeling(sample, labeler, ccfg["labels_path"],
                           ccfg["batch_state_path"], ccfg["batch_chunk_size"])
    cache = load_cache(ccfg["labels_path"])
    sample_recs = [cache[r["uid"]] for r in sample if r["uid"] in cache]
    ok = [r for r in sample_recs if r["status"] == "ok"]
    outcome_bearing = [r for r in ok if any(
        l["outcome"] in ("helped", "worsened") for l in r["labels"])]
    yield_rate = len(outcome_bearing) / max(len(sample), 1)
    report = {"sample_size": len(sample), "labeled_ok": len(ok),
              "outcome_bearing": len(outcome_bearing),
              "yield": round(yield_rate, 4), "run_summary": summary,
              "gate_p2_yield": "PASS" if yield_rate >= 0.30 else "FAIL"}
    out_dir = Path(ccfg["batch_state_path"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    by_uid = {r["uid"]: r for r in sample}
    hand = pd.DataFrame([{"uid": r["uid"], "text": by_uid[r["uid"]]["text"],
                          "labels": json.dumps(r["labels"])}
                         for r in sample_recs[:50]])
    hand.to_csv(out_dir / "calibration_sample.csv", index=False)
    print(json.dumps(report, indent=2))
    print("hand-check 50 rows in runs/concern/calibration_sample.csv "
          "(gate P2 agreement >= 85% is the maintainer's call)")
    return report


def cmd_label(rows, ccfg, yes: bool, p2_approved=False) -> dict | None:
    cache = load_cache(ccfg["labels_path"])
    todo = [r for r in rows if r["uid"] not in cache]
    est_usd = estimate_cost(todo, ccfg)
    print(f"to label: {len(todo)} of {len(rows)} "
          f"(est cost ${est_usd:.2f} on {ccfg['labeling_model']})")
    if not yes:
        print("dry run — pass --yes to submit")
        return None
    if not p2_approved:
        raise RuntimeError("full labeling requires maintainer P2 sign-off; "
                           "rerun with --p2-approved after the 50-row check")
    if est_usd > ccfg["max_budget_usd"]:
        raise RuntimeError(f"estimated ${est_usd:.2f} exceeds "
                           f"${ccfg['max_budget_usd']:.2f} budget ceiling")
    labeler = _labeler(ccfg)
    summary = run_labeling(rows, labeler, ccfg["labels_path"],
                           ccfg["batch_state_path"], ccfg["batch_chunk_size"])
    print(json.dumps(summary, indent=2))
    print("next: .venv/bin/python -m src.recommendation.concern_stats")
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
        if name == "label":
            sp.add_argument("--yes", action="store_true")
            sp.add_argument("--p2-approved", action="store_true")
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
        cmd_calibrate(rows, ccfg, args.n or ccfg["calibration_sample_size"])
    else:
        cmd_label(rows, ccfg, args.yes, args.p2_approved)


if __name__ == "__main__":
    main()
