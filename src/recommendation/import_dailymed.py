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
    # Prescription acne actives. Every key must name the substance the numerator
    # beside it measures, because that name is what the row goes on to claim the
    # product contains. Salt names are absent on purpose: a label dosing
    # clindamycin phosphate states 12 mg/g of the salt, and restating 12 mg/g
    # under "clindamycin" claims a fifth more drug than the product holds. The
    # salt and its moiety differ by a mass no label element carries, so a salt
    # falls through to the fail-closed path below rather than being converted.
    "TRETINOIN": "tretinoin",
    "TAZAROTENE": "tazarotene",
    "TRIFAROTENE": "trifarotene",
    "CLINDAMYCIN": "clindamycin",
    "DAPSONE": "dapsone",
    "MINOCYCLINE": "minocycline",
    "ERYTHROMYCIN": "erythromycin",
    "CLASCOTERONE": "clascoterone",
    # Sulfacetamide sodium is the salt the label doses and the substance this
    # name states, so its numerator needs no conversion to be true.
    "SULFACETAMIDE SODIUM": "sulfacetamide_sodium",
    "SODIUM SULFACETAMIDE": "sulfacetamide_sodium",
    "SULFUR": "sulfur",
}
TOPICAL_FORMS = {"gel", "cream", "lotion", "foam", "solution"}
# SPL states an active's strength against its basis (ACTIB), its active moiety
# (ACTIM), or a reference substance (ACTIR). All three mark an active, so all
# three are collected here: an ACTIR ingredient left out of this set would not
# fail its label closed, it would disappear from it, and a combination product
# would import one active short. Which of them can be read is _dosed_name's
# question, not this one's.
ACTIVE_CLASS_CODES = {"ACTIB", "ACTIM", "ACTIR"}


def _local(element: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in element.iter() if node.tag.rsplit("}", 1)[-1] == name]


def _direct(element: ET.Element, name: str) -> list[ET.Element]:
    """Immediate children only: a product's own facts, not its packaging's."""
    return [node for node in element if node.tag.rsplit("}", 1)[-1] == name]


def _text(element: ET.Element) -> str:
    return "".join(element.itertext()).strip().upper()


def _dosed_name(ingredient: ET.Element) -> str | None:
    """The substance this ingredient's numerator measures, or None if unreadable.

    An SPL quantity is meaningless without its basis: 12 mg/g against ACTIB
    measures clindamycin phosphate, the same 12 mg/g against ACTIM would measure
    clindamycin, and the two are different amounts of different substances. So
    the basis, not document order, has to pick which name the number belongs to.
    Reading whichever name appears first inside the ingredient reports the salt's
    mass under the moiety's name, which is how a 1% drug came to be listed at
    1.2%. ACTIR measures against a reference substance the ingredient never
    identifies, leaving no name to hand the number to at all.
    """
    tag = ingredient.tag.rsplit("}", 1)[-1]
    if tag == "activeIngredient":
        # An <activeIngredient> doses the substance it names, like ACTIB. Labels
        # spell the holder either way, so read both -- but only its own <name>,
        # never a moiety nested under it.
        names = {_text(name)
                 for holder_name in ("activeIngredientSubstance", "ingredientSubstance")
                 for holder in _direct(ingredient, holder_name)
                 for name in _direct(holder, "name")}
    elif ingredient.get("classCode") == "ACTIB":
        names = {_text(name) for holder in _direct(ingredient, "ingredientSubstance")
                 for name in _direct(holder, "name")}
    elif ingredient.get("classCode") == "ACTIM":
        # The moiety sits one or two <activeMoiety> deep depending on the label.
        names = {_text(name) for holder in _local(ingredient, "activeMoiety")
                 for name in _local(holder, "name")}
    else:
        return None
    if len(names) != 1:
        return None  # unnamed, or several names and no way to say which is dosed
    return names.pop()


def _product_actives(node: ET.Element, source_url: str) -> list[VerifiedActive] | None:
    """This product's actives, or None if the label leaves them ambiguous."""
    seen: dict[tuple[str, str | None], VerifiedActive] = {}
    ingredients = _direct(node, "activeIngredient") + [
        child for child in _direct(node, "ingredient")
        if child.get("classCode") in ACTIVE_CLASS_CODES
    ]
    for ingredient in ingredients:
        name = _dosed_name(ingredient)
        canonical = ACTIVE_NAMES.get(name) if name else None
        if not canonical:
            # An active we cannot name is either another product entirely, a
            # combination we would misreport by dropping an ingredient, or a
            # strength stated against something this parser cannot read.
            return None
        active = VerifiedActive(canonical, _strength(ingredient), source_url)
        seen[(active.name, active.strength)] = active  # a label may state one twice
    if not seen:
        return None
    actives = [seen[key] for key in sorted(seen, key=lambda key: (key[0], key[1] or ""))]
    if len({active.name for active in actives}) != len(actives):
        return None  # one active at two strengths inside a single product
    if any(active.strength is None for active in actives):
        return None  # without a strength we cannot state what the drug contains
    return actives


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
    if not set_id or not topical or not human:
        return []
    # A label is OTC or prescription, never both and never neither. Anything else
    # leaves legal status unknown, and unknown must not become a catalog fact.
    if otc == human_rx_document:
        return []

    # One document may describe several products -- Retin-A Micro carries four
    # strengths, Retin-A a cream and a gel. Read each product's own facts from
    # its own node; the packaging levels above it carry no NDC and are skipped.
    source_hash = hashlib.sha256(xml_bytes).hexdigest()
    modeled = {tuple(sorted(item)) for item in MODELED_STRENGTHS}
    products = []
    for node in _local(root, "manufacturedProduct"):
        ndc = next((child.get("code") for child in _direct(node, "code")
                    if child.get("codeSystem") == "2.16.840.1.113883.6.69"
                    and child.get("code")), None)
        if not ndc:
            continue
        form_text = next(
            (child.get("displayName") or "" for child in _direct(node, "formCode")), ""
        ).lower()
        form = next((item for item in sorted(TOPICAL_FORMS) if item in form_text), None)
        actives = _product_actives(node, source_url)
        if not form or actives is None:
            continue
        exact = tuple(sorted((active.name, active.strength or "") for active in actives))
        if otc and exact not in modeled:
            continue  # OTC rows are admitted only against an exact modeled path
        active_key = "+".join(f"{item.name}-{item.strength}" for item in actives)
        # A prescription document's <title> is the HIGHLIGHTS OF PRESCRIBING
        # INFORMATION preamble, never a product name; the label names each
        # product on its own node, so prefer that. Only an OTC document titles
        # itself with its product, so only there can the title stand in -- an Rx
        # row falling back to the title is named "These highlights do not
        # include all the information needed to use RETIN-A." The active key
        # names nothing the label did not state, which the preamble cannot say.
        name = next((" ".join("".join(child.itertext()).split())
                     for child in _direct(node, "name")), "")
        name = name or (title if otc else "") or active_key
        products.append(Product(
            product_id=f"dailymed:{set_id}:{ndc}:{active_key}",
            name=name, brand="DailyMed SPL", category="treatment",
            actives=sorted(active.name for active in actives),
            # An SPL states its target as "cover the entire affected area" and
            # never names the face, so leave the area unstated (D-034) rather
            # than stamp every drug row with a claim no label carries.
            intended_areas=[], routine_roles=[],
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
