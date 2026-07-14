from datetime import datetime, timezone
from pathlib import Path

from src.recommendation.import_dailymed import import_current_set_ids, parse_spl


FIXTURE = Path(__file__).parent / "fixtures" / "dailymed_adapalene_bpo.xml"


def test_current_exact_topical_spl_becomes_quarantined_v2_candidate():
    products = parse_spl(
        FIXTURE.read_bytes(), source_url="https://dailymed.nlm.nih.gov/spl.xml",
        retrieved_at="2026-07-13T00:00:00Z", current=True,
    )
    assert len(products) == 1
    product = products[0]
    assert product.catalog_schema_version == "2"
    assert product.source_set_id == "fixture-set-id"
    assert product.ndc_product_code == "00000-0001"
    assert product.label_version == "3"
    assert [(item.name, item.strength) for item in product.drug_actives] == [
        ("adapalene", "0.1%"), ("benzoyl_peroxide", "2.5%")
    ]
    assert product.routine_roles == []
    assert product.evidence_grade == "pending_review"


def test_current_dailymed_document_level_human_otc_metadata_is_supported():
    xml = b'''<?xml version="1.0" encoding="UTF-8"?>
    <document xmlns="urn:hl7-org:v3">
      <code code="34390-5" codeSystem="2.16.840.1.113883.6.1"
            displayName="HUMAN OTC DRUG LABEL" />
      <setId root="current-set-id" />
      <versionNumber value="7" />
      <effectiveTime value="20260204" />
      <title>Acne Clearing Treatment</title>
      <text>For external use only</text>
      <manufacturedProduct>
        <code codeSystem="2.16.840.1.113883.6.69" code="14222-1620" />
        <formCode displayName="Lotion" />
        <ingredient classCode="ACTIB">
          <quantity><numerator value="25" unit="mg"/>
            <denominator value="1" unit="mL"/></quantity>
          <ingredientSubstance><name>BENZOYL PEROXIDE</name></ingredientSubstance>
        </ingredient>
      </manufacturedProduct>
    </document>'''
    products = parse_spl(
        xml, source_url="https://dailymed.nlm.nih.gov/current.xml",
        retrieved_at="2026-07-14T00:00:00Z", current=True,
    )
    assert len(products) == 1
    assert products[0].format == "lotion"
    assert products[0].otc_drug is True
    assert products[0].name == "Acne Clearing Treatment"


def test_archived_or_non_exact_label_is_excluded():
    assert parse_spl(
        FIXTURE.read_bytes(), source_url="https://dailymed.nlm.nih.gov/spl.xml",
        retrieved_at="2026-07-13T00:00:00Z", current=True, archived=True
    ) == []
    wrong = FIXTURE.read_bytes().replace(b'value="25"', b'value="50"')
    assert parse_spl(
        wrong, source_url="https://dailymed.nlm.nih.gov/spl.xml",
        retrieved_at="2026-07-13T00:00:00Z", current=True,
    ) == []


def test_unverified_or_local_spl_cannot_become_eligible():
    assert parse_spl(
        FIXTURE.read_bytes(), source_url="https://dailymed.nlm.nih.gov/spl.xml"
    ) == []
    assert parse_spl(
        FIXTURE.read_bytes(), source_url="file:///fixture.xml",
        retrieved_at="2026-07-13T00:00:00Z", current=True,
    ) == []


def test_official_fetch_workflow_writes_review_ready_catalog(tmp_path):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return FIXTURE.read_bytes()

    seen = []

    def opener(url):
        seen.append(url)
        return Response()

    out = tmp_path / "catalog_drug.json"
    report = import_current_set_ids(
        ["fixture-set-id"], out, opener=opener,
        clock=lambda: datetime(2026, 7, 13, tzinfo=timezone.utc),
    )
    assert report == {"requested_set_ids": 1, "kept_for_review": 1}
    assert seen == [
        "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/fixture-set-id.xml"
    ]
    assert '"routine_roles": []' in out.read_text()
