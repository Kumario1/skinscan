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
(product_id, inci_sha256, prompt_version) — a product is re-labeled only when
its INCI or the prompt changes. Crash-safe: rerun resumes from the cache.

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
from pathlib import Path

from ..catalog import load_catalog
from ..contracts import CONCERNS, sha256_file
from .common import STORE_SCHEMA_VERSION, register_store, write_json

PROMPT_VERSION = "p1"
BUILDER = "recsys.tools.build_ingredient_analysis@1"
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


def read_cache(cache_path: Path) -> dict[tuple[str, str, str], dict]:
    cache: dict[tuple[str, str, str], dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            cache[(entry["product_id"], entry["inci_sha256"], entry["prompt_version"])] = entry
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


def label_product(product: dict, model: str, session=None, sleep=time.sleep, retries: int = 4) -> dict:
    """Label one product through OpenRouter structured output."""
    import requests  # lazy: tests and inference need no network client

    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY or OPENROUTER_KEY is required")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "product_id": product["product_id"],
                "name": product.get("name"),
                "category": product.get("category"),
                "inci": product.get("inci") or [],
                "already_recognized_actives": product.get("actives") or [],
            }, sort_keys=True)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "ingredient_analysis", "strict": True, "schema": OUTPUT_SCHEMA,
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
                raise RuntimeError(f"ingredient analysis refused product {product['product_id']}")
            return _entry(product, model, json.loads(choice["message"]["content"]))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                sleep(2 ** attempt)
    raise RuntimeError(
        f"ingredient analysis failed for {product['product_id']} after {retries} attempts"
    ) from last_error


def build(catalog_path: Path, out_path: Path, data_root: Path, cache_path: Path,
          model: str, max_new_labels: int = 100, concurrency: int = 1) -> dict:
    products, _header = load_catalog(catalog_path)
    product_rows = [product.to_dict() for product in products]
    cache = read_cache(cache_path)
    uncached = sum(
        (product["product_id"], product["inci_sha256"], PROMPT_VERSION) not in cache
        for product in product_rows
    )
    if uncached > max_new_labels:
        raise SystemExit(
            f"refusing {uncached} paid labels; rerun with "
            f"--max-new-labels {uncached} after checking cost"
        )
    pending = [product for product in product_rows
               if (product["product_id"], product["inci_sha256"], PROMPT_VERSION) not in cache]
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(label_product, product, model): product for product in pending}
        for future in as_completed(futures):
            product = futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                failures.append((product["product_id"], str(exc)))
                continue
            append_cache(cache_path, entry)
            cache[(product["product_id"], product["inci_sha256"], PROMPT_VERSION)] = entry
    if failures:
        raise RuntimeError(
            f"ingredient analysis failed for {len(failures)} products; rerun resumes cache: "
            + ", ".join(product_id for product_id, _error in failures[:10])
        )

    entries = []
    for product in product_rows:
        key = (product["product_id"], product["inci_sha256"], PROMPT_VERSION)
        entry = cache[key]
        entries.append(_entry(product, entry.get("model_id", model), entry))

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
    log = {"products_covered": len(entries), "cache_entries": len(cache)}
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
    args = parser.parse_args(argv)
    cache_path = args.cache or args.data_root / "cache" / "ingredient_analysis_cache.jsonl"

    build(args.catalog, args.out, args.data_root, cache_path, args.model,
          args.max_new_labels, args.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
