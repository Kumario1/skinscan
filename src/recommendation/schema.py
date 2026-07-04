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

    def __post_init__(self):
        assert self.category in CATEGORIES, f"unknown category: {self.category}"
