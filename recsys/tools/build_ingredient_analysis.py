"""LLM batch: per-product INCI analysis -> ingredient_analysis signal store.

The network call lives only in this offline build tool. Inference reads the
versioned store and never imports an HTTP client.

Store (signals/ingredient_analysis.v1.json), keyed by product_id:
    {"actives_beyond_table": [...],      # actives the exact-match INCI table missed
     "comedogenic_ingredients": [...],   # INCI names with comedogenicity >= 3
     "irritancy_tier": "low|medium|high",
     "fragrance_or_essential_oils": bool,
     "concern_fit_notes": {concern: str},
     "prompt_version": "...", "model_id": "...", "inci_sha256": "..."}

Cache (data/cache/ingredient_analysis_cache.jsonl): append-only JSONL keyed by
(product_id, inci_sha256, prompt_version, model_id) — a product is re-labeled
when its INCI, prompt, or provider model changes. Crash-safe: rerun resumes.

This output is a SCORING signal only, always labeled model-derived in
explanations; safety gates key off the deterministic INCI parser and
knowledge/safety_rules.json, never off LLM output.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path

from ..catalog import load_catalog
from ..contracts import CONCERNS, sha256_file
from .common import STORE_SCHEMA_VERSION, register_store, write_json

PROMPT_VERSION = "p1"
BUILDER = "recsys.tools.build_ingredient_analysis@1"
# The model that produced every entry in the committed store. A different model
# answers the same prompt differently, so it is part of the cache key and the
# store's provenance -- see _refuse_provenance_drift.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
URL = "https://openrouter.ai/api/v1/chat/completions"
IRRITANCY_TIERS = ("low", "medium", "high")

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "actives_beyond_table": {"type": "array", "items": {"type": "string"}},
        "comedogenic_ingredients": {"type": "array", "items": {"type": "string"}},
        "irritancy_tier": {"type": "string", "enum": list(IRRITANCY_TIERS)},
        "fragrance_or_essential_oils": {"type": "boolean"},
        "concern_fit_notes": {
            "type": "object",
            "properties": {c: {"type": "string"} for c in CONCERNS},
            "additionalProperties": False,
        },
    },
    "required": [
        "actives_beyond_table", "comedogenic_ingredients", "irritancy_tier",
        "fragrance_or_essential_oils", "concern_fit_notes",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
Analyze a cosmetic skincare product from its INCI list only. Return cautious,
evidence-based ingredient observations, not medical advice. List useful actives
that are absent from already_recognized_actives, ingredients commonly rated at
least 3/5 for comedogenicity, overall irritancy tier, whether fragrance or
essential oils appear, and short concern-fit notes only where the INCI supports
one of the allowed concern ids. Do not infer concentrations or make treatment
claims. Use normalized lowercase ingredient names in arrays.
"""

# PROMPT_VERSION is a cache-invalidation token: it must move whenever the text
# the model actually sees moves, or labels written against an older prompt are
# silently reused under the new one. Nothing links the two by construction, so
# pin the fingerprint of the exact prompt each generation was built from and
# check it on the money path. Editing SYSTEM_PROMPT or OUTPUT_SCHEMA without
# bumping PROMPT_VERSION now fails loudly instead of quietly reusing the cache.
PROMPT_FINGERPRINT = "ed0db6fa"  # fingerprint of the "p1" prompt


def prompt_fingerprint() -> str:
    """Fingerprint the exact bytes that determine the model's answer."""
    payload = SYSTEM_PROMPT + json.dumps(OUTPUT_SCHEMA, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()[:8]


def check_prompt_fingerprint() -> None:
    """Refuse to run a prompt whose text has drifted from its version token."""
    current = prompt_fingerprint()
    if current != PROMPT_FINGERPRINT:
        raise SystemExit(
            f"SYSTEM_PROMPT/OUTPUT_SCHEMA changed (fingerprint {PROMPT_FINGERPRINT} "
            f"-> {current}) but PROMPT_VERSION is still {PROMPT_VERSION!r}: every "
            f"label cached under the old prompt would be silently reused. Bump "
            f"PROMPT_VERSION (e.g. 'p2') and set PROMPT_FINGERPRINT = {current!r}."
        )


def read_cache(cache_path: Path) -> dict[tuple[str, str, str, str | None], dict]:
    cache: dict[tuple[str, str, str, str | None], dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            cache[(entry["product_id"], entry["inci_sha256"],
                   entry["prompt_version"], entry.get("model_id"))] = entry
    return cache


def append_cache(cache_path: Path, entry: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _entry(product: dict, model: str, data: dict) -> dict:
    if data.get("irritancy_tier") not in IRRITANCY_TIERS:
        raise ValueError("ingredient analysis returned an invalid irritancy_tier")
    notes = data.get("concern_fit_notes")
    if (not isinstance(notes, dict) or any(c not in CONCERNS for c in notes)
            or any(not isinstance(note, str) for note in notes.values())):
        raise ValueError("ingredient analysis returned invalid concern_fit_notes")
    arrays = ("actives_beyond_table", "comedogenic_ingredients")
    if any(not isinstance(data.get(k), list) or
           any(not isinstance(v, str) for v in data[k]) for k in arrays):
        raise ValueError("ingredient analysis returned invalid ingredient arrays")
    if not isinstance(data.get("fragrance_or_essential_oils"), bool):
        raise ValueError("ingredient analysis returned an invalid fragrance flag")
    return {
        "product_id": product["product_id"],
        "inci_sha256": product["inci_sha256"],
        "prompt_version": PROMPT_VERSION,
        "model_id": model,
        "actives_beyond_table": sorted({v.strip().lower()
                                         for v in data["actives_beyond_table"] if v.strip()}),
        "comedogenic_ingredients": sorted({v.strip().lower()
                                           for v in data["comedogenic_ingredients"] if v.strip()}),
        "irritancy_tier": data["irritancy_tier"],
        "fragrance_or_essential_oils": data["fragrance_or_essential_oils"],
        "concern_fit_notes": dict(sorted((k, v.strip()) for k, v in notes.items()
                                          if v.strip())),
    }


def _batch_output_schema(product_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "enum": product_ids},
                        "analysis": OUTPUT_SCHEMA,
                    },
                    "required": ["product_id", "analysis"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def label_products(products: list[dict], model: str, session=None,
                   sleep=time.sleep, retries: int = 4) -> list[dict]:
    """Label a product batch through one OpenRouter structured-output call."""
    import requests  # lazy: tests and inference need no network client

    if not products:
        return []
    product_ids = [product["product_id"] for product in products]
    if len(set(product_ids)) != len(product_ids):
        raise ValueError("ingredient-analysis batch product ids must be unique")
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY or OPENROUTER_KEY is required")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max(2000, 600 * len(products)),
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps([{
                "product_id": product["product_id"],
                "name": product.get("name"),
                "category": product.get("category"),
                "inci": product.get("inci") or [],
                "already_recognized_actives": product.get("actives") or [],
            } for product in products], sort_keys=True)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "ingredient_analysis_batch", "strict": True,
            "schema": _batch_output_schema(product_ids),
        }},
        "provider": {"require_parameters": True},
    }
    last_error = None
    for attempt in range(retries):
        try:
            response = (session or requests).post(
                URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                         "X-Title": "SkinScan ingredient analysis"},
                json=body,
                timeout=120,
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") == "content_filter":
                raise RuntimeError("ingredient analysis batch was refused")
            data = json.loads(choice["message"]["content"])
            by_id = {result["product_id"]: result["analysis"]
                     for result in data["results"]}
            if set(by_id) != set(product_ids) or len(data["results"]) != len(products):
                raise ValueError("ingredient analysis returned incomplete batch")
            return [_entry(product, model, by_id[product["product_id"]])
                    for product in products]
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                sleep(2 ** attempt)
    raise RuntimeError(
        f"ingredient analysis failed for batch after {retries} attempts"
    ) from last_error


def label_product(product: dict, model: str, session=None, sleep=time.sleep,
                  retries: int = 4) -> dict:
    """Compatibility wrapper for callers labeling one product."""
    return label_products([product], model, session, sleep, retries)[0]


def _head(items: list[tuple[str, str]], limit: int = 10) -> str:
    """Name the first few affected products, for a log line or an error."""
    if not items:
        return ""
    return ", e.g. " + ", ".join(product_id for product_id, _error in items[:limit])


def _refuse_provenance_drift(out_path: Path, model: str,
                             allow_model_change: bool) -> None:
    """Refuse to replace a store whose labels came from a different prompt/model.

    Unlike every other artifact here, this store cannot be rederived from the
    raw dump: it is model output, and the provider is not deterministic even at
    temperature 0 (the cache holds keys whose two answers disagree). Its
    reproducibility contract is therefore narrower -- byte-identical from the
    label cache, for the model and prompt it was built with. A run under a
    different model silently rewrites all 60 labels with unrelated ones and
    updates the registry sha to match, so the engine loads them without a
    murmur. Overwriting is a deliberate relabel, not a rebuild; make it be typed.
    """
    if allow_model_change or not out_path.exists():
        return
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return  # unreadable/absent store is a fresh build, not a drift
    stored_models = sorted({entry["model_id"] for entry in existing.get("products", {}).values()
                            if "model_id" in entry})
    stored_prompt = existing.get("prompt_version")
    drift = []
    if stored_models and stored_models != [model]:
        drift.append(f"model_ids {stored_models} -> [{model!r}]")
    if stored_prompt is not None and stored_prompt != PROMPT_VERSION:
        drift.append(f"prompt_version {stored_prompt!r} -> {PROMPT_VERSION!r}")
    if not drift:
        return
    raise SystemExit(
        f"refusing to overwrite {out_path.name} ({'; '.join(drift)}): its labels "
        f"are model output and are not reproducible across models, so replacing "
        f"them relabels the catalog rather than rebuilding it. Pass "
        f"--model {stored_models[0] if stored_models else model} to rebuild the "
        f"committed store from cache, or --allow-model-change to relabel on purpose."
    )


def build(catalog_path: Path, out_path: Path, data_root: Path, cache_path: Path,
          model: str, max_new_labels: int = 100, concurrency: int = 1,
          products_per_request: int = 10, min_coverage: float = 0.95,
          allow_model_change: bool = False) -> dict:
    check_prompt_fingerprint()
    products, _header = load_catalog(catalog_path)
    product_rows = [product.to_dict() for product in products]
    # Before the cost guard: a drifted model is the reason the labels would be
    # bought at all, and it is a provenance error, not a budgeting one.
    _refuse_provenance_drift(out_path, model, allow_model_change)
    cache = read_cache(cache_path)
    uncached = sum(
        (product["product_id"], product["inci_sha256"], PROMPT_VERSION, model) not in cache
        for product in product_rows
    )
    if uncached > max_new_labels:
        raise SystemExit(
            f"refusing {uncached} paid labels; rerun with "
            f"--max-new-labels {uncached} after checking cost"
        )
    pending = [product for product in product_rows
               if (product["product_id"], product["inci_sha256"],
                   PROMPT_VERSION, model) not in cache]
    batch_size = max(1, products_per_request)
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(label_products, batch, model): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                batch_entries = future.result()
            except Exception as exc:
                failures.extend((product["product_id"], str(exc)) for product in batch)
                continue
            for product, entry in zip(batch, batch_entries, strict=True):
                append_cache(cache_path, entry)
                cache[(product["product_id"], product["inci_sha256"],
                       PROMPT_VERSION, model)] = entry
    entries = []
    dropped = []
    for product in product_rows:
        key = (product["product_id"], product["inci_sha256"], PROMPT_VERSION, model)
        entry = cache.get(key)
        if entry is None:
            continue
        try:
            # Lenient on the resume/read path: a cached entry that no longer
            # validates (e.g. CONCERNS changed between runs) is dropped rather
            # than aborting an otherwise no-op rebuild. Fresh labels stay strict
            # via _entry on the WRITE path above. A drop still costs coverage --
            # it is a product missing from the store, however it went missing.
            entries.append(_entry(product, entry.get("model_id", model), entry))
        except ValueError as exc:
            dropped.append((product["product_id"], str(exc)))

    # Coverage is measured on what reaches the STORE, not on what sits in the
    # cache: an entry the read path drops is as absent from the store as one
    # that was never labeled, and the engine scores both products at a neutral
    # 0.5 with no warning. Free-tier rate limits fail whole batches transiently
    # and the same products succeed on a later resume, so write the store from
    # whatever survives once coverage clears the floor, and abort only when
    # coverage is genuinely low. The floor is checked unconditionally -- a run
    # with zero request failures can still be missing most of the catalog.
    coverage = len(entries) / max(1, len(product_rows))
    if coverage < min_coverage:
        raise RuntimeError(
            f"ingredient analysis coverage {coverage:.1%} below {min_coverage:.0%} "
            f"floor ({len(failures)} unlabeled{_head(failures)}, {len(dropped)} "
            f"dropped as invalid{_head(dropped)}); rerun resumes cache"
        )
    if failures or dropped:
        print(f"warning: {len(entries)} of {len(product_rows)} products in store at "
              f"{coverage:.1%} coverage ({len(failures)} unlabeled{_head(failures)}, "
              f"{len(dropped)} dropped as invalid{_head(dropped)}); rerun resumes cache")

    payload = {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": "ingredient_analysis",
        "version": "v1",
        "prompt_version": PROMPT_VERSION,
        "products": {e["product_id"]: {
            k: v for k, v in e.items() if k != "product_id"
        } for e in entries},
    }
    write_json(out_path, payload)
    register_store(
        data_root, name="ingredient_analysis", kind="ingredient_analysis", version="v1",
        store_path=out_path, builder=BUILDER,
        source={"catalog_sha256": sha256_file(catalog_path),
                "model_ids": sorted({e["model_id"] for e in entries}),
                "prompt_version": PROMPT_VERSION},
        coverage={"products": len(entries), "catalog_products": len(product_rows)},
    )
    log = {"products_covered": len(entries), "cache_entries": len(cache),
           "coverage": round(coverage, 4), "unlabeled": len(failures),
           "dropped_invalid": len(dropped)}
    print(log)
    return log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-labels", type=int, default=100,
                        help="paid-call guard; raise explicitly for larger runs")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--products-per-request", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.95,
                        help="write the store once this fraction of the catalog "
                             "reaches it, even if some products remain rate-limited")
    parser.add_argument("--allow-model-change", action="store_true",
                        help="relabel an existing store with a different model or "
                             "prompt; its labels are not reproducible across models")
    args = parser.parse_args(argv)
    cache_path = args.cache or args.data_root / "cache" / "ingredient_analysis_cache.jsonl"

    build(args.catalog, args.out, args.data_root, cache_path, args.model,
          args.max_new_labels, args.concurrency, args.products_per_request,
          args.min_coverage, args.allow_model_change)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
