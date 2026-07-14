"""Catalog core: product identity + details only. No scores, no stats —
those live in the signal stores (see ARCHITECTURE.md)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import SLOTS, ContractViolation

CATALOG_SCHEMA_VERSION = "recsys-catalog-1"


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
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CatalogProduct":
        if d.get("category") not in SLOTS:
            raise ContractViolation(
                "catalog.category",
                f"product {d.get('product_id')!r}: unknown {d.get('category')!r}",
            )
        if not d.get("product_id"):
            raise ContractViolation("catalog.product_id", "missing")
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
            inci=tuple(d.get("inci") or []),
            inci_sha256=d.get("inci_sha256") or "",
            actives=tuple(d.get("actives") or []),
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
