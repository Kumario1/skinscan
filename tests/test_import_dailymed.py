from datetime import datetime, timezone
import json
from pathlib import Path

from src.recommendation.import_dailymed import import_current_set_ids, parse_spl


FIXTURE = Path(__file__).parent / "fixtures" / "dailymed_adapalene_bpo.xml"
POOL = Path(__file__).parents[1] / "data" / "verification" / "dailymed-pool.json"

# Every strength the committed pool states, per canonical active, reviewed
# against what the substance we name is actually marketed at. An active
# legitimately ships at several strengths -- tretinoin at seven here -- so
# "more than one strength" is not itself the bug. The bug is a strength that is
# the same real drug as another entry, wearing a number from a different basis:
# clindamycin sat at both 1% and 1.2% while all eight products were the same 1%
# drug, because two labels dose the phosphate salt and the parser handed the
# salt's 12 mg/g to the moiety's name.
#
# A diff here is not a formatting nit to be regenerated away. It means either a
# genuinely new product -- confirm the label states that number against the
# substance this table names, then add it -- or a basis confusion that has
# already reached the catalog, in which case fix the parser and re-derive.
# The pool is append-only and never re-parses its SPLs, so nothing else looks.
POOL_STRENGTHS = {
    "adapalene": {"0.1%", "0.3%"},
    "azelaic_acid": {"20%"},
    "benzoyl_peroxide": {"2.5%", "3%", "3.75%", "5%"},
    "clascoterone": {"1%"},
    "clindamycin": {"1%"},
    "dapsone": {"5%", "7.5%"},
    "erythromycin": {"2%"},
    "minocycline": {"4%"},
    "sulfacetamide_sodium": {"10%"},
    "sulfur": {"5%"},
    "tazarotene": {"0.045%", "0.05%", "0.1%"},
    "tretinoin": {"0.01%", "0.025%", "0.04%", "0.05%", "0.06%", "0.08%", "0.1%"},
    "trifarotene": {"0.005%"},
}


def test_committed_pool_states_each_active_only_at_reviewed_strengths():
    """The catalog must not claim one substance at two strengths for one drug.

    This reads the committed rows alone, with no label bytes: a wrong strength
    frozen into the pool outlives the code that wrote it, so the parser's own
    tests cannot be what guards it. Nothing re-parses these SPLs.
    """
    strengths: dict[str, set[str]] = {}
    for row in json.loads(POOL.read_text()):
        for active in row.get("drug_actives") or []:
            strengths.setdefault(active["name"], set()).add(active["strength"])
    assert strengths == POOL_STRENGTHS

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
    # D-034: an SPL states "cover the entire affected area" and never names the
    # face, so the importer must not stamp one on. A fabricated "face" here
    # would be indistinguishable from a verified one downstream, and would let
    # a body-only drug pass the intended-area veto.
    assert product.intended_areas == []
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


def test_moiety_dosed_active_is_read_through_the_doubled_activemoiety_wrapper():
    # A real SPL nests <activeMoiety><activeMoiety><name>, not the single wrapper
    # the fixture above uses. Reading only one level deep would find no moiety to
    # name an ACTIM row after, and would fail every clindamycin label closed.
    doubled = """<ingredient classCode="ACTIM">
      <quantity><numerator value="10" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>CLINDAMYCIN PHOSPHATE</name>
        <activeMoiety><activeMoiety><code code="3U02EL437C" />
          <name>CLINDAMYCIN</name></activeMoiety></activeMoiety></ingredientSubstance>
    </ingredient>"""
    products = parse_spl(
        _rx_spl(extra=doubled), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert ("clindamycin", "1%") in [
        (item.name, item.strength) for item in products[0].drug_actives
    ]


def test_salt_dosed_active_is_rejected_not_restated_under_its_moietys_name():
    # ZIANA and Cabtreo dose the salt (ACTIB): 12 mg/g measures clindamycin
    # PHOSPHATE, which is clindamycin 1% -- the label spells the equivalence out,
    # "12 mg (1.2%) of clindamycin phosphate equivalent to 10 mg (1%)
    # clindamycin". Handing that 12 to the moiety's name shipped "clindamycin
    # 1.2%" to real users, a fifth more drug than the tube holds, and the two
    # differ by a mass no element of the label carries. Converting would be a
    # guess, so the label fails closed instead.
    salt = """<ingredient classCode="ACTIB">
      <quantity><numerator value="12" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>CLINDAMYCIN PHOSPHATE</name>
        <activeMoiety><activeMoiety><name>CLINDAMYCIN</name>
          </activeMoiety></activeMoiety></ingredientSubstance>
    </ingredient>"""
    assert parse_spl(
        _rx_spl(extra=salt), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    ) == []


def test_reference_substance_dosed_active_fails_its_label_closed():
    # ACTIR states a strength against a reference substance the ingredient never
    # identifies, so the numerator measures nothing this parser can name. It has
    # to reject the label rather than pass the number to whichever name is
    # nearest -- and rather than ignore the ingredient, which would import this
    # two-active combination as tretinoin alone and never say so.
    reference = """<ingredient classCode="ACTIR">
      <quantity><numerator value="12" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>CLINDAMYCIN PHOSPHATE</name>
        <activeMoiety><activeMoiety><name>CLINDAMYCIN</name>
          </activeMoiety></activeMoiety></ingredientSubstance>
    </ingredient>"""
    assert parse_spl(
        _rx_spl(extra=reference), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    ) == []


def test_one_product_stating_an_active_at_two_strengths_is_excluded():
    # A single product cannot be both 0.05% and 0.1% tretinoin. Reject rather
    # than pick one. (Two *products* at two strengths is normal -- see below.)
    second = """<ingredient classCode="ACTIB">
      <quantity><numerator value="1" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>TRETINOIN</name></ingredientSubstance>
    </ingredient>"""
    assert parse_spl(
        _rx_spl(extra=second), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    ) == []


def test_document_with_several_products_imports_each_at_its_own_strength():
    # Retin-A Micro states four strengths in one document and Retin-A a cream
    # plus a gel, each on its own product node with its own NDC and form.
    second = """</manufacturedProduct>
  <manufacturedProduct>
    <code codeSystem="2.16.840.1.113883.6.69" code="00000-0003" />
    <name>Retin-A</name>
    <formCode displayName="Gel" />
    <ingredient classCode="ACTIB">
      <quantity><numerator value="1" unit="mg" /><denominator value="1" unit="g" /></quantity>
      <ingredientSubstance><name>TRETINOIN</name></ingredientSubstance>
    </ingredient>"""
    products = parse_spl(
        _rx_spl(extra=second), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert [(p.format, p.drug_actives[0].strength) for p in products] == [
        ("cream", "0.05%"), ("gel", "0.1%")
    ]
    assert len({p.product_id for p in products}) == 2


def test_active_without_a_parseable_strength_is_excluded():
    unstated = RX_SPL.format(
        doc_code="34391-3", doc_name="HUMAN PRESCRIPTION DRUG LABEL", extra=""
    ).replace('<numerator value="0.5" unit="mg" />', '<numerator value="0.5" unit="iu" />')
    assert parse_spl(
        unstated.encode(), source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    ) == []


def test_product_is_named_from_the_label_not_the_highlights_preamble():
    # A prescription document's <title> is the HIGHLIGHTS OF PRESCRIBING
    # INFORMATION boilerplate, so titling the row from it names every Rx product
    # "These highlights do not include...".
    spl = _rx_spl().replace(
        b'<formCode displayName="Cream" />',
        b'<name>Retin-A</name><formCode displayName="Cream" />',
    ).replace(
        b"<title>Tretinoin Cream USP 0.05%</title>",
        b"<title>These highlights do not include all the information needed"
        b" to use RETIN-A safely and effectively.</title>",
    )
    products = parse_spl(
        spl, source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert products[0].name == "Retin-A"


def test_prescription_product_without_its_own_name_is_not_named_from_the_preamble():
    # The fix above prefers the product's <name> but fell back to <title>, and an
    # Rx <title> IS the preamble -- so the fallback landed exactly where the fix
    # was meant to stop. A product that does not name itself takes the active key
    # it was built from, which states only what the label states.
    spl = _rx_spl().replace(
        b"<title>Tretinoin Cream USP 0.05%</title>",
        b"<title>These highlights do not include all the information needed"
        b" to use RETIN-A safely and effectively.</title>",
    )
    products = parse_spl(
        spl, source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert products[0].name == "tretinoin-0.05%"
    assert "highlights" not in products[0].name.lower()


def test_cutaneous_route_counts_as_topical():
    # Tazorac and Azelex state the route as CUTANEOUS; matching "topical" alone
    # silently dropped them.
    cutaneous = _rx_spl().replace(b'displayName="TOPICAL"', b'displayName="CUTANEOUS"')
    products = parse_spl(
        cutaneous, source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert len(products) == 1


def test_microgram_dosed_active_is_parsed():
    # Aklief doses trifarotene at 50 ug/g; only mg and g units were handled.
    micrograms = _rx_spl().replace(
        b'<numerator value="0.5" unit="mg" />', b'<numerator value="50" unit="ug" />'
    )
    products = parse_spl(
        micrograms, source_url="https://dailymed.nlm.nih.gov/rx.xml",
        retrieved_at="2026-07-15T00:00:00Z", current=True,
    )
    assert products[0].drug_actives[0].strength == "0.005%"


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
