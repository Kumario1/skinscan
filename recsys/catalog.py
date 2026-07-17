"""Catalog core: product identity + details only. No scores, no stats —
those live in the signal stores (see ARCHITECTURE.md)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import SLOTS, ContractViolation
from .inci import parse_ingredients

CATALOG_SCHEMA_VERSION = "recsys-catalog-1"
# Matched as a literal prefix rather than parsed: the engine imports no urllib
# (see tests/test_no_network.py), and a prefix demands https into the bargain.
LABEL_PREFIX = "https://dailymed.nlm.nih.gov/"


def _cites_label(url: object) -> bool:
    return isinstance(url, str) and url.startswith(LABEL_PREFIX)


def _label_stated_actives(d: dict) -> tuple[str, ...] | None:
    """Actives a regulatory label states in its own structured ingredient data.

    A drug label publishes no INCI list, so the usual "actives must parse out of
    the INCI" derivation cannot apply. It does something stronger: it names each
    active, gives its exact strength, and cites the DailyMed label it came from.
    Rows that clear all of that derive their actives from drug_actives instead.
    Returns None for everything else, which keeps the INCI rule in force for
    every cosmetic. Evidence-byte hash binding belongs to verification.py, not
    this catalog door.
    """
    drug_actives = d.get("drug_actives") or []
    if not drug_actives or not _cites_label(d.get("label_source")):
        return None
    for active in drug_actives:
        if not isinstance(active, dict) or not active.get("name") or not active.get("strength"):
            return None
        if not _cites_label(active.get("source")):
            return None
    return tuple(sorted({str(active["name"]) for active in drug_actives}))


@dataclass(frozen=True)
class CatalogProduct:
    product_id: str
    name: str
    brand: str
    category: str  # one of SLOTS
    price_usd: float | None
    size: str | None
    format: str | None
    spf: int | None
    spf_source: str | None  # "name_parse" | "verified" | None
    inci: tuple[str, ...]
    inci_sha256: str
    actives: tuple[str, ...]
    broad_spectrum: bool | None = None
    cadence: str | None = None
    contraindications: tuple[str, ...] = ()
    contraindications_verified: bool = False
    intended_areas: tuple[str, ...] = ()
    routine_roles: tuple[str, ...] = ()
    exposure: str | None = None
    drug_actives: tuple[dict, ...] = ()
    otc_drug: bool | None = None
    label_source: str | None = None
    label_verified_at: str | None = None
    cadence_source: str | None = None
    amount: str | None = None
    amount_source: str | None = None
    evidence_roles: tuple[str, ...] = ()
    daily_support_verified: bool = False
    evidence_grade: str | None = None
    comedogenic_claim: str | None = None

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "price_usd": self.price_usd,
            "size": self.size,
            "format": self.format,
            "spf": self.spf,
            "spf_source": self.spf_source,
            "inci": list(self.inci),
            "inci_sha256": self.inci_sha256,
            "actives": list(self.actives),
            "broad_spectrum": self.broad_spectrum,
            "cadence": self.cadence,
            "contraindications": list(self.contraindications),
            "contraindications_verified": self.contraindications_verified,
            "intended_areas": list(self.intended_areas),
            "routine_roles": list(self.routine_roles),
            "exposure": self.exposure,
            "drug_actives": list(self.drug_actives),
            "otc_drug": self.otc_drug,
            "label_source": self.label_source,
            "label_verified_at": self.label_verified_at,
            "cadence_source": self.cadence_source,
            "amount": self.amount,
            "amount_source": self.amount_source,
            "evidence_roles": list(self.evidence_roles),
            "daily_support_verified": self.daily_support_verified,
            "evidence_grade": self.evidence_grade,
            "comedogenic_claim": self.comedogenic_claim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CatalogProduct":
        if d.get("category") not in {*SLOTS, "scar_care"}:
            raise ContractViolation(
                "catalog.category",
                f"product {d.get('product_id')!r}: unknown {d.get('category')!r}",
            )
        if not d.get("product_id"):
            raise ContractViolation("catalog.product_id", "missing")
        inci = tuple(d.get("inci") or [])
        expected_digest = hashlib.sha256(
            json.dumps(list(inci), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if d.get("inci_sha256") != expected_digest:
            raise ContractViolation(
                "catalog.inci_sha256", f"product {d['product_id']!r}: stale or invalid"
            )
        stated = _label_stated_actives(d)
        parsed_actives = (
            stated if stated is not None else tuple(parse_ingredients(",".join(inci))[0])
        )
        declared_actives = tuple(d.get("actives") or [])
        # Schema migration: iron-oxide CI parsing was added after existing
        # catalogs were built.  It is a deterministic fact from the unchanged
        # INCI bytes, so permit exactly that additive derivation without
        # weakening stale-active rejection for any other difference.
        additive_iron_migration = (
            set(parsed_actives) - set(declared_actives) == {"iron_oxides"}
            and set(declared_actives) == set(parsed_actives) - {"iron_oxides"}
        )
        if declared_actives != parsed_actives and not additive_iron_migration:
            raise ContractViolation(
                "catalog.actives", f"product {d['product_id']!r}: stale or invalid"
            )
        spf = d.get("spf")
        return cls(
            product_id=d["product_id"],
            name=d.get("name") or "",
            brand=d.get("brand") or "",
            category=d["category"],
            price_usd=d.get("price_usd"),
            size=d.get("size"),
            format=d.get("format"),
            spf=int(spf) if spf is not None else None,
            spf_source=d.get("spf_source"),
            inci=inci,
            inci_sha256=expected_digest,
            actives=parsed_actives,
            broad_spectrum=d.get("broad_spectrum"),
            cadence=d.get("cadence"),
            contraindications=tuple(d.get("contraindications") or []),
            contraindications_verified=False,
            intended_areas=tuple(d.get("intended_areas") or []),
            routine_roles=tuple(d.get("routine_roles") or []),
            exposure=d.get("exposure"),
            drug_actives=tuple(d.get("drug_actives") or []),
            otc_drug=d.get("otc_drug"),
            label_source=d.get("label_source"),
            label_verified_at=d.get("label_verified_at"),
            cadence_source=d.get("cadence_source"),
            amount=d.get("amount"),
            amount_source=d.get("amount_source"),
            evidence_roles=tuple(d.get("evidence_roles") or []),
            evidence_grade=d.get("evidence_grade"),
            comedogenic_claim=d.get("comedogenic_claim"),
        )


def load_catalog(path: str | Path) -> tuple[list[CatalogProduct], dict]:
    """Returns (products, header). Header carries schema_version, source and
    builder provenance as written by tools/build_catalog.py."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ContractViolation(
            "catalog.schema_version",
            f"expected {CATALOG_SCHEMA_VERSION!r}, got "
            f"{data.get('schema_version') if isinstance(data, dict) else type(data).__name__!r}",
        )
    products = [CatalogProduct.from_dict(row) for row in data.get("products") or []]
    seen: set[str] = set()
    for p in products:
        if p.product_id in seen:
            raise ContractViolation("catalog.product_id", f"duplicate {p.product_id!r}")
        seen.add(p.product_id)
    header = {k: v for k, v in data.items() if k != "products"}
    return products, header
