"""Executable versions of the two data contracts.

CONCERN_SCHEMA.md  -> ConcernReport (input to the recommender)
CATALOG_SCHEMA.md  -> Product        (the catalog)

Keeping these as dataclasses means the contract is enforced in code, not just
prose. If the CV side can produce a valid ConcernReport, Stage 3 works —
regardless of whether that report came from a real model or a hand-written test
fixture (D-007).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# --- closed vocabularies (must match the docs) -----------------------------
CONCERNS = {
    "acne_comedonal", "acne_inflammatory", "acne_cystic",
    "hyperpigmentation", "dryness",
}
REGIONS = {
    "forehead", "nose", "left_cheek", "right_cheek", "chin_jaw", "perioral",
}
CATEGORIES = ["cleanser", "treatment", "serum", "moisturizer", "spf"]  # ordered


# --- concern side (Stage 2 -> Stage 3) -------------------------------------
@dataclass
class Concern:
    concern: str
    region: str
    severity: int              # 0-4 ordinal
    confidence: float          # 0-1
    lesion_count: Optional[int] = None

    def __post_init__(self):
        assert self.concern in CONCERNS, f"unknown concern: {self.concern}"
        assert self.region in REGIONS, f"unknown region: {self.region}"
        assert 0 <= self.severity <= 4, "severity must be 0-4"
        assert 0.0 <= self.confidence <= 1.0, "confidence must be 0-1"


@dataclass
class ConcernReport:
    image_id: str
    concerns: list[Concern] = field(default_factory=list)
    clear_skin: bool = False
    low_light_flag: bool = False
    notes: str = ""

    @property
    def overall_severity(self) -> int:
        acne = [c.severity for c in self.concerns if c.concern.startswith("acne_")]
        return max(acne) if acne else 0

    @property
    def has_cystic(self) -> bool:
        return any(c.concern == "acne_cystic" for c in self.concerns)


# --- product side ----------------------------------------------------------
@dataclass
class Product:
    product_id: str
    name: str
    brand: str
    category: str
    actives: list[str] = field(default_factory=list)
    comedogenic_flags: list[str] = field(default_factory=list)
    price_usd: Optional[float] = None
    price_is_stale: bool = True
    # Ingredient-KB enrichment (spec 2026-07-10-ingredient-kb). All three
    # default to their tier-1/no-KB values so a catalog imported without the KB
    # serializes byte-identically to before (the importer omits keys at default).
    ingredient_match: dict[str, float] = field(default_factory=dict)  # concern -> [0,1]
    tier: int = 1                          # 1 = review-backed Sephora; 2 = beautyapi fallback
    no_outcome_data: bool = False          # True for tier-2 (no review outcomes exist)

    def __post_init__(self):
        assert self.category in CATEGORIES, f"unknown category: {self.category}"


# --- user profile (D-021) --------------------------------------------------
SKIN_TYPES = {"combination", "dry", "normal", "oily"}
TONE_BUCKETS = {"light", "medium", "deep"}
TONE_SOURCES = {"self_report", "photo", "unknown"}


@dataclass
class UserProfile:
    """Optional context that steers (never overrides) the rules (D-021)."""
    skin_type: str
    tone_bucket: Optional[str] = None
    tone_source: str = "unknown"           # self_report > photo > unknown
    pregnant_or_nursing: bool = False

    def __post_init__(self):
        assert self.skin_type in SKIN_TYPES, f"unknown skin_type: {self.skin_type}"
        assert self.tone_bucket is None or self.tone_bucket in TONE_BUCKETS, \
            f"unknown tone_bucket: {self.tone_bucket}"
        assert self.tone_source in TONE_SOURCES, f"unknown tone_source: {self.tone_source}"
