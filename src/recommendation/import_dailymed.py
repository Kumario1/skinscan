"""Deterministic DailyMed SPL adapter for topical acne drugs.

OTC labels are admitted only when they fill an exact modeled therapy path.
Prescription labels are admitted on the acne-active allow-list alone: D-033 lets
the app surface prescription-strength options with a see-a-doctor note, and
these rows carry label facts only -- which therapy paths exist stays D-029
clinician-gated, so an Rx row is never eligible for a routine on its own.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from .schema import Product, VerifiedActive


MODELED_STRENGTHS = {
    (("azelaic_acid", "10%"),),
    (("benzoyl_peroxide", "2.5%"),),
    (("adapalene", "0.1%"), ("benzoyl_peroxide", "2.5%")),
}
ACTIVE_NAMES = {
    "AZELAIC ACID": "azelaic_acid",
    "BENZOYL PEROXIDE": "benzoyl_peroxide",
    "ADAPALENE": "adapalene",
    # Prescription acne actives. Salt forms map to the moiety the label doses;
    # an active missing from this table makes the whole SPL fail closed below.
    "TRETINOIN": "tretinoin",
    "TAZAROTENE": "tazarotene",
    "TRIFAROTENE": "trifarotene",
    "CLINDAMYCIN": "clindamycin",
    "CLINDAMYCIN PHOSPHATE": "clindamycin",
    "DAPSONE": "dapsone",
    "MINOCYCLINE": "minocycline",
    "MINOCYCLINE HYDROCHLORIDE": "minocycline",
    "ERYTHROMYCIN": "erythromycin",
    "CLASCOTERONE": "clascoterone",
    "SULFACETAMIDE SODIUM": "sulfacetamide_sodium",
    "SODIUM SULFACETAMIDE": "sulfacetamide_sodium",
    "SULFUR": "sulfur",
}
TOPICAL_FORMS = {"gel", "cream", "lotion", "foam", "solution"}
# SPL states an active's strength against its basis (ACTIB), its active moiety
# (ACTIM), or a reference substance (ACTIR). Reading ACTIB alone silently drops
# every label that doses by moiety -- most clindamycin products, for one.
ACTIVE_CLASS_CODES = {"ACTIB", "ACTIM", "ACTIR"}


def _local(element: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in element.iter() if node.tag.rsplit("}", 1)[-1] == name]


def _first_attr(root: ET.Element, element_name: str, attribute: str) -> str | None:
    for node in _local(root, element_name):
        value = node.get(attribute)
        if value:
            return value.strip()
    return None


def _strength(ingredient: ET.Element) -> str | None:
    numerator = next(iter(_local(ingredient, "numerator")), None)
    denominator = next(iter(_local(ingredient, "denominator")), None)
    if numerator is None or denominator is None:
        return None
    try:
        n = float(numerator.get("value", ""))
        d = float(denominator.get("value", ""))
    except ValueError:
        return None
    n_unit = (numerator.get("unit") or "").lower()
    d_unit = (denominator.get("unit") or "").lower()
    if n_unit == "mg" and d_unit in {"g", "ml"} and d:
        value = n / d / 10
    elif n_unit in {"ug", "mcg"} and d_unit in {"g", "ml"} and d:
        value = n / d / 10_000  # potent retinoids dose in micrograms (Aklief: 50 ug/g)
    elif n_unit == "g" and d_unit == "g" and d:
        value = n / d * 100
    else:
        return None
    return f"{value:g}%"


def parse_spl(
    xml_bytes: bytes,
    *,
    source_url: str,
    retrieved_at: str | None = None,
    current: bool = False,
    archived: bool = False,
) -> list[Product]:
    """Parse one current human topical SPL into exact-strength catalog rows."""
    source = urlparse(source_url)
    if (archived or not current or not retrieved_at or source.scheme != "https"
            or source.hostname != "dailymed.nlm.nih.gov"):
        return []
    root = ET.fromstring(xml_bytes)
    set_id = _first_attr(root, "setId", "root")
    version = _first_attr(root, "versionNumber", "value")
    effective = _first_attr(root, "effectiveTime", "value")
    title_node = next(
        (node for node in list(root) if node.tag.rsplit("}", 1)[-1] == "title"),
        None,
    )
    title = (" ".join("".join(title_node.itertext()).split())
             if title_node is not None else "") or "DailyMed topical drug"
    form_text = " ".join(
        filter(None, (node.get("displayName") for node in _local(root, "formCode")))
    ).lower()
    route_text = " ".join(
        filter(None, (node.get("displayName") for node in _local(root, "routeCode")))
    ).lower()
    marketing_text = " ".join(
        filter(None, (node.get("displayName") or node.get("code")
                      for node in _local(root, "marketingCategory")))
    ).lower()
    species_text = " ".join(
        filter(None, (node.get("displayName") or node.get("code")
                      for node in _local(root, "speciesCode")))
    ).lower()
    document_label_text = " ".join(
        filter(None, (node.get("displayName") or node.get("code")
                      for node in _local(root, "code")
                      if node.get("codeSystem") == "2.16.840.1.113883.6.1"))
    ).lower()
    document_text = " ".join("".join(root.itertext()).split()).lower()
    form = next((item for item in TOPICAL_FORMS if item in form_text), None)
    human_otc_document = "human otc drug label" in document_label_text
    # LOINC 34391-3 is the authoritative prescription marker (34390-5 is its OTC
    # counterpart); marketingCategory is absent on many real labels.
    human_rx_document = "human prescription drug label" in document_label_text
    human = "human" in species_text or human_otc_document or human_rx_document
    otc = "otc" in marketing_text or human_otc_document
    # DailyMed states the topical route as TOPICAL or CUTANEOUS (Tazorac, Azelex
    # use the latter); matching "topical" alone silently drops those labels.
    topical = any(word in route_text for word in ("topical", "cutaneous")) or (
        not route_text and "for external use only" in document_text
    )
    if not set_id or not form or not topical or not human:
        return []
    # A label is OTC or prescription, never both and never neither. Anything else
    # leaves legal status unknown, and unknown must not become a catalog fact.
    if otc == human_rx_document:
        return []

    seen: dict[tuple[str, str | None], VerifiedActive] = {}
    unmodeled = False
    active_ingredients = _local(root, "activeIngredient") + [
        node for node in _local(root, "ingredient")
        if node.get("classCode") in ACTIVE_CLASS_CODES
    ]
    for ingredient in active_ingredients:
        names = ["".join(node.itertext()).strip().upper()
                 for node in _local(ingredient, "name")]
        canonical = next((ACTIVE_NAMES[name] for name in names if name in ACTIVE_NAMES), None)
        if canonical:
            active = VerifiedActive(canonical, _strength(ingredient), source_url)
            seen[(active.name, active.strength)] = active  # SPLs may state one twice
        else:
            # An active we cannot name is either another product entirely or a
            # combination we would misreport by silently dropping an ingredient.
            unmodeled = True
    if unmodeled or not seen:
        return []
    actives = [seen[key] for key in sorted(seen, key=lambda key: (key[0], key[1] or ""))]
    if len({active.name for active in actives}) != len(actives):
        # One active at two strengths: this document describes more than one
        # product, and actives are read document-wide, so we cannot tell which
        # strength belongs to which NDC.
        return []
    if any(active.strength is None for active in actives):
        return []  # without a strength we cannot state what the drug contains
    exact = tuple(sorted((active.name, active.strength or "") for active in actives))
    if otc and exact not in {tuple(sorted(item)) for item in MODELED_STRENGTHS}:
        return []

    ndcs = sorted({
        node.get("code")
        for product in _local(root, "manufacturedProduct")
        for node in list(product)
        if node.tag.rsplit("}", 1)[-1] == "code"
        and node.get("codeSystem") == "2.16.840.1.113883.6.69"
        and node.get("code")
    })
    if not ndcs:
        return []
    source_hash = hashlib.sha256(xml_bytes).hexdigest()
    products = []
    for ndc in ndcs:
        active_key = "+".join(f"{item.name}-{item.strength}" for item in actives)
        products.append(Product(
            product_id=f"dailymed:{set_id}:{ndc}:{active_key}",
            name=title, brand="DailyMed SPL", category="treatment",
            actives=sorted(active.name for active in actives),
            intended_areas=["face"], routine_roles=[],
            format=form, exposure="leave_on", drug_actives=actives,
            otc_drug=otc, label_source=source_url, label_verified_at=retrieved_at,
            evidence_roles=[], evidence_grade="pending_review",
            cadence="per_label", cadence_source=source_url,
            source_set_id=set_id, ndc_product_code=ndc, label_version=version,
            label_effective_date=effective, source_hash=source_hash,
            catalog_schema_version="2",
        ))
    return products


def fetch_current_spl(set_id: str, *, opener=urlopen) -> bytes:
    """Fetch through the versioned DailyMed v2 SPL endpoint; injectable in tests."""
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
    with opener(url) as response:
        return response.read()


def import_current_set_ids(
    set_ids: list[str], out_path: Path, *, opener=urlopen,
    clock=lambda: datetime.now(timezone.utc),
) -> dict[str, object]:
    """Fetch current official SPLs into a quarantined, review-ready catalog."""
    products: list[Product] = []
    for set_id in sorted(set(set_ids)):
        url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
        retrieved_at = clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        products.extend(parse_spl(
            fetch_current_spl(set_id, opener=opener), source_url=url,
            retrieved_at=retrieved_at, current=True,
        ))
    unique = {product.product_id: product for product in products}
    rows = [unique[key].to_dict() for key in sorted(unique)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"requested_set_ids": len(set(set_ids)), "kept_for_review": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-id", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(import_current_set_ids(args.set_id, args.out))


if __name__ == "__main__":
    main()
