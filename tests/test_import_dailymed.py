from datetime import datetime, timezone
from pathlib import Path

from src.recommendation.import_dailymed import import_current_set_ids, parse_spl


FIXTURE = Path(__file__).parent / "fixtures" / "dailymed_adapalene_bpo.xml"

RX_SPL = """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3">
  <code code="{doc_code}" codeSystem="2.16.840.1.113883.6.1" displayName="{doc_name}" />
  <setId root="rx-set-id" />
  <versionNumber value="2" />
  <effectiveTime value="20260301" />
  <title>Tretinoin Cream USP 0.05%</title>
  <manufacturedProduct>
    <code codeSystem="2.16.840.1.113883.6.69" code="00000-0002" />
    <formCode displayName="Cream" />
    <routeCode displayName="TOPICAL" />
    <subject><speciesCode displayName="Human" /></subject>
    <ingredient classCode="ACTIB">
      <quantity><numerator value="0.5" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>TRETINOIN</name></ingredientSubstance>
    </ingredient>
    {extra}
  </manufacturedProduct>
</document>"""


def _rx_spl(*, extra="", doc_code="34391-3", doc_name="HUMAN PRESCRIPTION DRUG LABEL"):
    return RX_SPL.format(doc_code=doc_code, doc_name=doc_name, extra=extra).encode()


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


def test_prescription_label_is_imported_and_recorded_as_not_otc():
    # D-033: prescription-strength options may be surfaced with a referral, so the
    # Rx label imports -- and otc_drug must record what the label actually says.
    products = parse_spl(
        _rx_spl(), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert len(products) == 1
    product = products[0]
    assert product.otc_drug is False
    assert [(item.name, item.strength) for item in product.drug_actives] == [
        ("tretinoin", "0.05%")
    ]
    assert product.format == "cream"
    assert product.evidence_grade == "pending_review"


def test_unmodeled_active_fails_closed_instead_of_misreporting_a_combination():
    # Tretinoin + an active we cannot name: importing it would silently drop an
    # ingredient and describe a combination product as tretinoin-only.
    extra = """<ingredient classCode="ACTIB">
      <quantity><numerator value="40" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>HYDROQUINONE</name></ingredientSubstance>
    </ingredient>"""
    assert parse_spl(
        _rx_spl(extra=extra), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    ) == []


def test_active_dosed_against_its_moiety_is_read_not_dropped():
    # Most clindamycin labels state strength against the active moiety (ACTIM)
    # rather than the basis (ACTIB); reading ACTIB alone dropped them silently.
    moiety = """<ingredient classCode="ACTIM">
      <quantity><numerator value="10" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>CLINDAMYCIN PHOSPHATE</name>
        <activeMoiety><name>CLINDAMYCIN</name></activeMoiety></ingredientSubstance>
    </ingredient>"""
    products = parse_spl(
        _rx_spl(extra=moiety), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert len(products) == 1
    assert ("clindamycin", "1%") in [
        (item.name, item.strength) for item in products[0].drug_actives
    ]


def test_document_holding_two_strengths_of_one_active_is_excluded():
    # Actives are read document-wide, so a label covering 0.05% and 0.1% gives no
    # way to say which strength belongs to which NDC. Reject rather than guess.
    second = """<ingredient classCode="ACTIB">
      <quantity><numerator value="1" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>TRETINOIN</name></ingredientSubstance>
    </ingredient>"""
    assert parse_spl(
        _rx_spl(extra=second), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    ) == []


def test_active_without_a_parseable_strength_is_excluded():
    unstated = RX_SPL.format(
        doc_code="34391-3", doc_name="HUMAN PRESCRIPTION DRUG LABEL", extra=""
    ).replace('<numerator value="0.5" unit="mg" />', '<numerator value="0.5" unit="iu" />')
    assert parse_spl(
        unstated.encode(), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    ) == []


def test_label_that_is_neither_otc_nor_prescription_is_excluded():
    # Unknown legal status must never become a catalog fact.
    assert parse_spl(
        _rx_spl(doc_code="99999-9", doc_name="SOME OTHER DOCUMENT"),
        source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    ) == []


def test_prescription_label_skips_the_otc_modeled_path_gate():
    # tretinoin 0.05% fills no modeled path; it still catalogs as an Rx fact row,
    # because which therapy paths exist stays D-029 clinician-gated elsewhere.
    products = parse_spl(
        _rx_spl(), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert products and products[0].routine_roles == []


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
