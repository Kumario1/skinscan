"""Signal providers: pluggable, evidence-carrying scoring inputs.

Each provider returns a SignalScore(value 0..1, evidence, details) or None when
it has no data for a product. None is neutral (0.5) plus an uncertainty note —
missing data is never a hidden penalty or bonus.

Extensibility: store-backed providers are discovered through
data/signals/registry.json; a registry entry's `kind` maps to a class in
STORE_PROVIDERS. Unknown kinds are skipped with a warning so a newer store
never breaks an older engine. Adding a signal = new build tool + store file +
registry entry + one provider class here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import CatalogProduct
from .contracts import ContractViolation, Profile, sha256_file
from .knowledge import Knowledge

REGISTRY_SCHEMA_VERSION = "recsys-registry-1"
MIN_SKIN_TYPE_CELL = 20


@dataclass(frozen=True)
class SignalScore:
    name: str
    value: float  # 0..1
    evidence: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TargetConcern:
    concern: str
    severity: int
    confidence: float


@dataclass(frozen=True)
class ScoringContext:
    targets: tuple[TargetConcern, ...]
    profile: Profile
    knowledge: Knowledge
    category_prices: dict[str, tuple[float, ...]]  # sorted asc, per category


class ConcernFitSignal:
    """Catalog actives x concern->actives knowledge map, weighted by target
    severity. Built-in (no store)."""

    name = "concern_fit"
    version = "v0"

    def score(self, product: CatalogProduct, slot: str, ctx: ScoringContext):
        if not ctx.targets:
            return None
        actives = set(product.actives)
        matched: dict[str, list[str]] = {}
        total = sum(t.severity for t in ctx.targets) or 1
        hit_weight = 0
        for t in ctx.targets:
            overlap = sorted(actives & ctx.knowledge.concern_actives.get(t.concern, frozenset()))
            if overlap:
                matched[t.concern] = overlap
                hit_weight += t.severity
        if not matched:
            return SignalScore(self.name, 0.0, "no concern-targeting actives", {"matched": {}})
        by_active: dict[str, list[str]] = {}
        for concern, overlap in matched.items():
            phrase = ctx.knowledge.phrasing.get(concern, concern)
            for active in overlap:
                by_active.setdefault(active, []).append(phrase)
        evidence = "; ".join(
            f"{active.replace('_', ' ')} targets {', '.join(phrases)}"
            for active, phrases in sorted(by_active.items())
        )
        return SignalScore(self.name, round(hit_weight / total, 6), evidence, {"matched": matched})


class PriceValueSignal:
    """Cheaper than peers in the same category = higher value. Built-in
    (computed from the catalog itself, no store)."""

    name = "price_value"
    version = "v0"

    def score(self, product: CatalogProduct, slot: str, ctx: ScoringContext):
        prices = ctx.category_prices.get(product.category) or ()
        if product.price_usd is None or len(prices) < 2:
            return None
        more_expensive = sum(1 for p in prices if p > product.price_usd)
        value = round(more_expensive / (len(prices) - 1), 6)
        value = min(value, 1.0)
        return SignalScore(
            self.name, value,
            f"${product.price_usd:.2f} — cheaper than {value:.0%} of "
            f"{product.category} options in this catalog",
            {"price_usd": product.price_usd},
        )


class ReviewQualitySignal:
    """Bayesian-smoothed rating from the review_stats store, preferring the
    reviewer cell matching the user's skin type when it is large enough."""

    name = "review_quality"

    def __init__(self, store: dict, meta: dict):
        self.products = store.get("products") or {}
        self.version = meta.get("version", "v?")

    def score(self, product: CatalogProduct, slot: str, ctx: ScoringContext):
        entry = self.products.get(product.product_id)
        if not entry:
            return None
        skin = ctx.profile.skin_type
        cell = (entry.get("by_skin_type") or {}).get(skin) if skin != "unknown" else None
        if cell and cell.get("n", 0) >= MIN_SKIN_TYPE_CELL:
            value = (cell["smoothed"] - 1) / 4
            evidence = f"{cell['mean']:.1f}★ from {cell['n']} {skin}-skin reviewers"
            details = {"cell": skin, **cell}
        else:
            value = (entry["smoothed"] - 1) / 4
            evidence = f"{entry['mean']:.1f}★ across {entry['n']} reviews"
            details = {"cell": "all", "n": entry["n"], "mean": entry["mean"], "smoothed": entry["smoothed"]}
        return SignalScore(self.name, round(max(0.0, min(1.0, value)), 6), evidence, details)


class PopularitySignal:
    """Category loves-percentile from the popularity store (Sephora snapshot)."""

    name = "popularity"

    def __init__(self, store: dict, meta: dict):
        self.products = store.get("products") or {}
        self.snapshot = store.get("signal_age", "snapshot")
        self.version = meta.get("version", "v?")

    def score(self, product: CatalogProduct, slot: str, ctx: ScoringContext):
        entry = self.products.get(product.product_id)
        if not entry:
            return None
        pct = entry["category_percentile"]
        return SignalScore(
            self.name, round(pct, 6),
            f"more loved than {pct:.0%} of {product.category} products on Sephora "
            f"({self.snapshot})",
            dict(entry),
        )


class IngredientAnalysisSignal:
    """Model-derived INCI irritancy estimate; safety gates never consume it."""

    name = "ingredient_analysis"
    _VALUES = {"low": 1.0, "medium": 0.5, "high": 0.0}

    def __init__(self, store: dict, meta: dict):
        self.products = store.get("products") or {}
        self.version = meta.get("version", "v?")

    def score(self, product: CatalogProduct, slot: str, ctx: ScoringContext):
        entry = self.products.get(product.product_id)
        if not entry:
            return None
        tier = entry.get("irritancy_tier")
        if tier not in self._VALUES:
            return None
        observations = [f"model-estimated {tier} irritancy from the INCI list"]
        if entry.get("fragrance_or_essential_oils"):
            observations.append("fragrance or essential oils flagged")
        comedogenic = entry.get("comedogenic_ingredients") or []
        if comedogenic:
            observations.append("comedogenicity flags: " + ", ".join(comedogenic))
        return SignalScore(
            self.name, self._VALUES[tier], "; ".join(observations), dict(entry)
        )


class ConcernEfficacySignal:
    """Review-text outcomes conditioned on the report's target concerns.

    Bayesian smoothing happens in the offline builder. Inference additionally
    shrinks small cells toward neutral so a thin cell cannot dominate a
    routine merely because its observed help rate is extreme.
    """

    name = "concern_efficacy"

    def __init__(self, store: dict, meta: dict):
        self.products = store.get("products") or {}
        self.version = meta.get("version", "v?")
        self.confidence_n = float(store.get("confidence_n", 20))

    def score(self, product: CatalogProduct, slot: str, ctx: ScoringContext):
        product_cells = self.products.get(product.product_id) or {}
        matches = []
        weighted = []
        for target in ctx.targets:
            ladder = "exact"
            concern = target.concern
            entry = product_cells.get(concern)
            if entry is None and concern.startswith("acne_"):
                ladder = "acne_general"
                concern = "acne_general"
                entry = product_cells.get(concern)
            if entry is None:
                continue
            cell = (entry.get("by_skin_type") or {}).get(ctx.profile.skin_type)
            cell = cell or entry.get("all")
            if not cell or not cell.get("n"):
                continue
            n = int(cell["n"])
            smoothed = float(cell["smoothed"])
            reliability = n / (n + self.confidence_n)
            adjusted = 0.5 + (smoothed - 0.5) * reliability
            target_weight = target.severity * max(target.confidence, 0.01)
            weighted.append((adjusted, target_weight))
            matches.append({
                "target": target.concern,
                "cell_concern": concern,
                "ladder": ladder,
                "n": n,
                "help_rate": cell["help_rate"],
                "smoothed": smoothed,
                "reliability": round(reliability, 6),
            })
        if not weighted:
            return None
        value = sum(v * weight for v, weight in weighted) / sum(weight for _, weight in weighted)
        evidence = "; ".join(
            f"{m['help_rate']:.0%} of {m['n']} reviewers said it helped "
            f"{ctx.knowledge.phrasing.get(m['target'], m['target'])}"
            + (" (general-acne fallback)" if m["ladder"] != "exact" else "")
            for m in matches
        )
        return SignalScore(self.name, round(value, 6), evidence, {"matches": matches})


class MediaSignal:
    """Verified editorial/media evidence, isolated from safety decisions."""

    name = "media"

    def __init__(self, store: dict, meta: dict):
        self.products = store.get("products") or {}
        self.version = meta.get("version", "v?")

    def score(self, product: CatalogProduct, slot: str, ctx: ScoringContext):
        entry = self.products.get(product.product_id)
        if not entry:
            return None
        value = entry.get("value")
        evidence = entry.get("evidence")
        if not isinstance(value, (int, float)) or not 0 <= value <= 1 or not evidence:
            raise ContractViolation("media", f"invalid entry for {product.product_id}")
        details = {key: value for key, value in entry.items()
                   if key not in {"value", "evidence"}}
        return SignalScore(self.name, float(value), str(evidence), details)


STORE_PROVIDERS: dict[str, type] = {
    "concern_efficacy": ConcernEfficacySignal,
    "ingredient_analysis": IngredientAnalysisSignal,
    "media": MediaSignal,
    "review_stats": ReviewQualitySignal,
    "popularity": PopularitySignal,
}


def load_providers(data_root: str | Path) -> tuple[list, list[dict], list[str]]:
    """Instantiate built-in providers plus every active registry store whose
    sha256 matches the file on disk. Returns (providers, store_meta, warnings)."""
    data_root = Path(data_root)
    providers: list = [ConcernFitSignal(), PriceValueSignal()]
    meta: list[dict] = []
    warnings: list[str] = []
    registry_path = data_root / "signals" / "registry.json"
    if not registry_path.exists():
        warnings.append("no signal registry — built-in signals only")
        return providers, meta, warnings
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ContractViolation(
            "registry.schema_version",
            f"expected {REGISTRY_SCHEMA_VERSION!r}, got {registry.get('schema_version')!r}",
        )
    for entry in registry.get("stores") or []:
        if entry.get("status") != "active":
            continue
        kind = entry.get("kind")
        cls = STORE_PROVIDERS.get(kind)
        if cls is None:
            warnings.append(f"unknown signal kind {kind!r} — skipped")
            continue
        store_path = data_root / entry["path"]
        actual = sha256_file(store_path)
        if actual != entry.get("sha256"):
            raise ContractViolation(
                "registry.sha256",
                f"store {entry.get('name')!r} at {store_path} does not match its "
                f"registry entry (expected {entry.get('sha256')}, got {actual})",
            )
        store = json.loads(store_path.read_text(encoding="utf-8"))
        providers.append(cls(store, entry))
        meta.append({"name": entry.get("name"), "version": entry.get("version"),
                     "sha256": entry.get("sha256")})
    return providers, meta, warnings
