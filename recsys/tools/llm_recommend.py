"""Optional Azure selector over products already approved by recsys gates."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

from recsys.contracts import ContractViolation
from recsys.pipeline import (
    CATALOG_SELECTOR_SLOTS,
    DEFAULT_DATA_ROOT,
    CandidateSelection,
    SelectionUnavailable,
    emit,
    run,
    skipped_signal_warnings,
)

SELECTABLE_SLOTS = CATALOG_SELECTOR_SLOTS

PROMPT_VERSION = "azure-catalog-selector-v1"
INSTRUCTIONS = """\
Choose one cosmetic catalog product ID for every required routine slot using only
the supplied candidates. Treatment may be selected only when it is required.
Serum is optional. Return null for every unused optional slot. Do not diagnose,
invent products or ingredients, change therapy intent, or return any prose.
"""


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(_json(value) + "\n")


def _without_paths(value):
    if isinstance(value, dict):
        return {
            key: _without_paths(item)
            for key, item in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def _output_text(data: dict) -> str:
    if data.get("output_text"):
        return data["output_text"]
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    raise SelectionUnavailable("azure_response_missing_output")


def _selection_schema() -> dict:
    nullable_id = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {slot: nullable_id for slot in SELECTABLE_SLOTS},
        "required": list(SELECTABLE_SLOTS),
        "additionalProperties": False,
    }


class AzureCatalogSelector:
    """One structured Azure Responses call plus a guarded similarity cache."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        model_identity: str | None = None,
        cache_secret: str | None = None,
        cache_path: str | Path,
        usage_path: str | Path,
        session=None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
        reasoning_effort: str | None = None,
        timeout: int = 120,
    ):
        self.endpoint = endpoint or os.environ.get("TARGET_URL") or os.environ.get(
            "AZURE_OPENAI_ENDPOINT"
        )
        self.api_key = api_key or os.environ.get("AZURE_KEY") or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.model_identity = (
            model_identity
            or os.environ.get("AZURE_OPENAI_MODEL_IDENTITY")
        )
        self.cache_secret = cache_secret or os.environ.get("SKINSCAN_CACHE_SECRET")
        self.cache_path = Path(cache_path)
        self.usage_path = Path(usage_path)
        self.input_price = self._price(
            input_price_per_million, "AZURE_INPUT_PRICE_PER_MILLION"
        )
        self.output_price = self._price(
            output_price_per_million, "AZURE_OUTPUT_PRICE_PER_MILLION"
        )
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "AZURE_REASONING_EFFORT", "minimal"
        )
        self.timeout = timeout
        if session is None:
            import requests
            session = requests
        self.session = session
        self._pending: dict[str, dict] = {}

    @staticmethod
    def _price(explicit: float | None, environment_name: str) -> float | None:
        raw = explicit if explicit is not None else os.environ.get(environment_name)
        if raw in (None, ""):
            return None
        value = float(raw)
        if value < 0:
            raise ValueError(f"{environment_name} must be non-negative")
        return value

    def _require_configuration(self) -> None:
        if not all((
            self.endpoint, self.api_key, self.deployment, self.model_identity,
            self.cache_secret,
        )):
            raise SelectionUnavailable("configuration_missing")

    @staticmethod
    def _candidate_rows(candidates: dict) -> list[dict]:
        rows = []
        for slot in SELECTABLE_SLOTS:
            for item in candidates.get(slot, []):
                product = item.product
                rows.append({
                    "role": slot,
                    "product_id": product.product_id,
                    "name": product.name,
                    "brand": product.brand,
                    "price_usd": product.price_usd,
                    "actives": list(product.actives),
                    "verification": item.verification_status,
                    "score": item.final,
                    "signals": {signal.name: signal.value for signal in item.signals},
                    "uncertainty": list(item.uncertainty),
                })
        return rows

    def _cache_key(self, context: dict, candidates: dict, versions: dict) -> str:
        profile = context["profile"]
        safety = {
            field: sorted(value) if isinstance(value, list) else value
            for field, value in profile.items()
            if field not in {"skin_type", "tone_bucket"}
        }
        statuses = {
            item["lesion_type"]: item["status"]
            for item in context["care_pathways"]
        }
        condition_similarity = {
            "lesion_findings": sorted(
                (item["lesion_type"], item["count"], statuses.get(item["lesion_type"]))
                for item in context["lesion_findings"]
            ),
        }
        material = {
            "provider": "azure",
            "model": self.model_identity,
            "deployment": self.deployment,
            "reasoning_effort": self.reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "similarity": {
                **condition_similarity,
                "skin_type": profile.get("skin_type"),
                "tone_bucket": profile.get("tone_bucket"),
            },
            "safety": safety,
            "decision": context["decision"],
            "therapy_plan": context["therapy_plan"],
            "care_pathways": context["care_pathways"],
            "required_slots": context["required_slots"],
            "versions": _without_paths(versions),
            "candidate_pool": {
                slot: sorted(item.product.product_id for item in candidates.get(slot, []))
                for slot in SELECTABLE_SLOTS
            },
        }
        return hmac.new(
            self.cache_secret.encode("utf-8"), _json(material).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _cached(self, cache_key: str) -> dict | None:
        if not self.cache_path.exists():
            return None
        found = None
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("cache_key") == cache_key:
                found = item
        return found

    @staticmethod
    def _validate_product_ids(product_ids: object, context: dict, candidates: dict) -> dict:
        if not isinstance(product_ids, dict) or set(product_ids) != set(SELECTABLE_SLOTS):
            raise SelectionUnavailable("invalid_slot_set")
        allowed = {
            slot: {item.product.product_id for item in candidates.get(slot, [])}
            for slot in SELECTABLE_SLOTS
        }
        for slot in context["required_slots"]:
            if not isinstance(product_ids.get(slot), str) or not product_ids[slot]:
                raise SelectionUnavailable(f"required_slot_missing:{slot}")
        if "treatment" not in context["required_slots"] and product_ids["treatment"] is not None:
            raise SelectionUnavailable("treatment_not_allowed")
        selected = []
        for slot, product_id in product_ids.items():
            if product_id is None:
                continue
            if not isinstance(product_id, str) or product_id not in allowed[slot]:
                raise SelectionUnavailable(f"product_not_safe_for_slot:{slot}")
            selected.append(product_id)
        if len(selected) != len(set(selected)):
            raise SelectionUnavailable("duplicate_product")
        return dict(product_ids)

    def _usage(self, data: dict) -> dict:
        raw = data.get("usage") or {}
        input_tokens = int(raw.get("input_tokens") or 0)
        output_tokens = int(raw.get("output_tokens") or 0)
        estimated = None
        if self.input_price is not None and self.output_price is not None:
            estimated = round(
                input_tokens * self.input_price / 1_000_000
                + output_tokens * self.output_price / 1_000_000,
                6,
            )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated,
        }

    def _record_usage(self, **entry) -> None:
        _append_jsonl(self.usage_path, {
            "provider": "azure",
            "model": self.model_identity,
            "model_identity": self.model_identity,
            "deployment": self.deployment,
            "prompt_version": PROMPT_VERSION,
            **entry,
        })

    def __call__(self, context: dict, candidates: dict, versions: dict) -> CandidateSelection:
        rows = self._candidate_rows(candidates)
        counts = {
            slot: len(candidates.get(slot, [])) for slot in SELECTABLE_SLOTS
        }
        try:
            self._require_configuration()
        except SelectionUnavailable as exc:
            raise SelectionUnavailable(str(exc), {
                "source": "azure_catalog_selector",
                "provider": "azure",
                "model": self.model_identity,
                "model_identity": self.model_identity,
                "deployment": self.deployment,
                "prompt_version": PROMPT_VERSION,
                "cache_hit": None,
                "cache_status": "unavailable",
                "candidate_count": len(rows),
                "candidate_counts": counts,
                "latency_ms": 0,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": None,
                },
            }) from None
        cache_key = self._cache_key(context, candidates, versions)
        cached = self._cached(cache_key)
        cache_status = "miss"
        if cached is not None:
            try:
                product_ids = self._validate_product_ids(
                    cached.get("product_ids"), context, candidates
                )
            except SelectionUnavailable:
                cache_status = "invalidated"
            else:
                usage = {
                    "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0
                }
                self._record_usage(
                    cache_key=cache_key, cache_hit=True, cache_status="hit",
                    status="cached", candidate_count=len(rows),
                    candidate_counts=counts, latency_ms=0, usage=usage,
                )
                return CandidateSelection(product_ids, {
                    "source": "azure_catalog_selector",
                    "provider": "azure",
                    "model": cached.get("model") or self.deployment,
                    "model_identity": self.model_identity,
                    "deployment": self.deployment,
                    "prompt_version": PROMPT_VERSION,
                    "cache_key": cache_key,
                    "cache_hit": True,
                    "cache_status": "hit",
                    "candidate_count": len(rows),
                    "candidate_counts": counts,
                    "latency_ms": 0,
                    "usage": usage,
                })

        body = {
            "model": self.deployment,
            "instructions": INSTRUCTIONS,
            "input": _json({"user": context, "candidates": rows}),
            # A deployment alias need not reveal its model family. Keep enough
            # ceiling for hidden reasoning even when capability detection below
            # cannot safely attach an explicit reasoning setting.
            "max_output_tokens": 4_000,
            "store": False,
            "text": {"format": {
                "type": "json_schema",
                "name": "skinscan_catalog_selection",
                "strict": True,
                "schema": _selection_schema(),
            }},
        }
        if self.deployment.startswith(("gpt-5", "o1", "o3", "o4")):
            body["reasoning"] = {"effort": self.reasoning_effort}

        started = time.monotonic()
        data = {}
        status = "failed"
        error = None
        try:
            response = self.session.post(
                self.endpoint,
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            if not isinstance(response_data, dict):
                raise SelectionUnavailable("malformed_response")
            data = response_data
            if data.get("status") not in (None, "completed"):
                raise SelectionUnavailable(
                    f"azure_response_{data.get('status') or 'unknown'}"
                )
            product_ids = self._validate_product_ids(
                json.loads(_output_text(data)), context, candidates
            )
            status = "succeeded"
        except SelectionUnavailable as exc:
            error = str(exc)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            error = "malformed_response"
        except Exception as exc:
            error = f"azure_error:{type(exc).__name__}"
        latency_ms = round((time.monotonic() - started) * 1000)
        usage = self._usage(data)
        response_model = data.get("model") or self.deployment
        self._record_usage(
            model=response_model, cache_key=cache_key, cache_hit=False,
            cache_status=cache_status, status=status, candidate_count=len(rows),
            candidate_counts=counts, latency_ms=latency_ms, usage=usage,
        )

        metadata = {
            "source": "azure_catalog_selector",
            "provider": "azure",
            "model": response_model,
            "model_identity": self.model_identity,
            "deployment": self.deployment,
            "prompt_version": PROMPT_VERSION,
            "cache_key": cache_key,
            "cache_hit": False,
            "cache_status": cache_status,
            "candidate_count": len(rows),
            "candidate_counts": counts,
            "latency_ms": latency_ms,
            "usage": usage,
        }
        if error is not None:
            raise SelectionUnavailable(error, metadata)
        result = CandidateSelection(product_ids, metadata)
        self._pending[cache_key] = {
            "cache_key": cache_key,
            "product_ids": product_ids,
            "provider": "azure",
            "model": response_model,
            "prompt_version": PROMPT_VERSION,
        }
        return result

    def selection_validated(self, result: CandidateSelection) -> None:
        """Persist only after pipeline composition and validation succeeded."""
        if result.metadata.get("cache_hit"):
            return
        entry = self._pending.pop(result.metadata["cache_key"], None)
        if entry is not None:
            _append_jsonl(self.cache_path, entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Let Azure choose one validated regimen from post-gate catalog products."
    )
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default="llm-recommendations.json")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--usage-log", default="runs/recsys/llm_usage.jsonl")
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--allow-signal-catalog-mismatch", action="store_true")
    parser.add_argument("--allow-unreviewed-policy", action="store_true")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    selector = AzureCatalogSelector(
        cache_path=args.cache or data_root / "cache" / "llm_catalog_selections.jsonl",
        usage_path=args.usage_log,
        timeout=args.timeout,
    )
    candidate_selector = selector
    try:
        analysis_schema = json.loads(Path(args.analysis).read_text()).get("schema_version")
    except (OSError, json.JSONDecodeError, AttributeError):
        analysis_schema = None  # run() reports the authoritative contract error
    if str(analysis_schema) == "3":
        def reject_legacy_analysis(*_args, **_kwargs):
            selector._require_configuration()
            raise SelectionUnavailable("schema4_analysis_required")

        candidate_selector = reject_legacy_analysis
    try:
        document = run(
            analysis_path=args.analysis,
            profile_path=args.profile,
            catalog_path=args.catalog,
            data_root=data_root,
            generated_at=args.generated_at,
            allow_signal_catalog_mismatch=args.allow_signal_catalog_mismatch,
            allow_unreviewed_policy=args.allow_unreviewed_policy,
            candidate_selector=candidate_selector,
        )
    except ContractViolation as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = emit(document, args.out)
    if document["status"] != "ok":
        print(f"unavailable: {document.get('reason', 'selection failed')} -> {output}")
        return 4
    if skipped_signal_warnings(document.get("warnings") or []):
        print(f"unavailable: signal store skipped -> {output}", file=sys.stderr)
        return 3
    selection = document["selection"]
    print(
        f"ok: {len(document['selected_products'])} products -> {output} "
        f"(cache_hit={selection['cache_hit']}, "
        f"candidates={selection['candidate_count']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
