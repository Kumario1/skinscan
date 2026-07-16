import hashlib
import json
from datetime import datetime, timezone

import pytest

from recsys.catalog import CatalogProduct
from recsys.contracts import ContractViolation
from recsys.verification import apply_verification, load_verification_overlay


def _multi_row_overlay(tmp_path, rows):
    """Write an overlay verbatim from `rows`, snapshotting each row's evidence."""
    root = tmp_path / "verification"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    products = []
    for product_id, body, approved_at, facts in rows:
        digest = hashlib.sha256(body).hexdigest()
        (evidence / digest).write_bytes(body)
        products.append({"product_id": product_id, "assertions": [{
            "status": "approved",
            "reviewer_id": "reviewer-1",
            "reviewer_type": "agent",
            "approved_at": approved_at,
            "retrieved_at": approved_at,
            "source_url": f"https://example.test/{approved_at}",
            "source_sha256": digest,
            "facts": facts,
        }]})
    (root / "approved.json").write_text(json.dumps({
        "schema_version": "recsys-verification-1", "products": products,
    }))
    return root


@pytest.mark.parametrize("order", ["newest_first", "oldest_first"])
def test_newest_approval_wins_when_one_product_spans_several_rows(tmp_path, order):
    """approved_at decides precedence -- that is what the loader sorts on. A
    product re-reviewed in a later batch can land in a second row for the same
    product_id, and document row order must not be able to resurrect the
    superseded facts. Both approvals are recent, so neither is dropped as stale.
    """
    newest = ("p1", b"july label", "2026-07-10T00:00:00Z", {"otc_drug": True})
    oldest = ("p1", b"june label", "2026-06-01T00:00:00Z", {"otc_drug": False})
    rows = [newest, oldest] if order == "newest_first" else [oldest, newest]
    root = _multi_row_overlay(tmp_path, rows)

    overlay, warnings, _ = load_verification_overlay(
        root, now=datetime(2026, 7, 15, tzinfo=timezone.utc))

    assert warnings == []
    assert overlay["p1"]["otc_drug"] is True
    # provenance keeps both sources, oldest approval first
    assert [s["url"] for s in overlay["p1"]["_sources"]] == [
        "https://example.test/2026-06-01T00:00:00Z",
        "https://example.test/2026-07-10T00:00:00Z",
    ]


def _product():
    return CatalogProduct(
        product_id="p1", name="SPF 15", brand="Test", category="spf",
        price_usd=10, size=None, format=None, spf=15, spf_source="name_parse",
        inci=(), inci_sha256="", actives=(),
    )


def _write_approved_overlay(tmp_path, facts, assertion_overrides=None):
    root = tmp_path / "verification"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    body = b"regulatory label bytes"
    digest = hashlib.sha256(body).hexdigest()
    (evidence / digest).write_bytes(body)
    (root / "approved.json").write_text(json.dumps({
        "schema_version": "recsys-verification-1",
        "products": [{
            "product_id": "p1",
            "assertions": [{
                "status": "approved",
                "reviewer_id": "reviewer-1",
                "reviewer_type": "agent",
                "approved_at": "2026-07-14T00:00:00Z",
                "retrieved_at": "2026-07-14T00:00:00Z",
                "source_url": "https://example.test/label",
                "source_sha256": digest,
                "facts": facts,
                **(assertion_overrides or {}),
            }],
        }],
    }))
    return root


def test_verified_facts_override_name_parse_and_stale_facts_do_not(tmp_path):
    root = tmp_path / "verification"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    body = b"regulatory label bytes"
    digest = hashlib.sha256(body).hexdigest()
    (evidence / digest).write_bytes(body)
    (root / "approved.json").write_text(json.dumps({
        "schema_version": "recsys-verification-1",
        "products": [{
            "product_id": "p1",
            "assertions": [{
                "status": "approved",
                "reviewer_id": "reviewer-1",
                "reviewer_type": "agent",
                "approved_at": "2026-07-14T00:00:00Z",
                "retrieved_at": "2026-07-14T00:00:00Z",
                "source_url": "https://example.test/label",
                "source_sha256": digest,
                "facts": {
                    "spf": 50,
                    "broad_spectrum": True,
                    "cadence": "per_label",
                    "evidence_grade": "regulatory_label",
                    "intended_areas": ["face"],
                    "routine_roles": ["sunscreen"],
                    "format": "lotion",
                    "exposure": "leave_on",
                    "otc_drug": True,
                    "label_source": "https://dailymed.nlm.nih.gov/dailymed/spl/p1",
                    "label_verified_at": "2026-07-14",
                    "cadence_source": "https://example.test/label",
                    "amount": "thin_layer",
                    "amount_source": "https://example.test/label",
                    "evidence_roles": ["daily_support"],
                    "drug_actives": [{
                        "name": "benzoyl_peroxide",
                        "strength": "2.5%",
                        "source": "https://dailymed.nlm.nih.gov/dailymed/spl/p1",
                    }],
                },
            }],
        }],
    }))

    overlay, warnings, _meta = load_verification_overlay(
        root, now=datetime(2026, 7, 15, tzinfo=timezone.utc)
    )
    verified = apply_verification([_product()], overlay)[0]
    assert warnings == []
    assert verified.spf == 50
    assert verified.spf_source == "verified"
    assert verified.broad_spectrum is True
    assert verified.cadence == "per_label"
    assert verified.intended_areas == ("face",)
    assert verified.routine_roles == ("sunscreen",)
    assert verified.format == "lotion"
    assert verified.exposure == "leave_on"
    assert verified.drug_actives == ({
        "name": "benzoyl_peroxide",
        "strength": "2.5%",
        "source": "https://dailymed.nlm.nih.gov/dailymed/spl/p1",
    },)
    assert verified.otc_drug is True
    assert verified.label_source == "https://dailymed.nlm.nih.gov/dailymed/spl/p1"
    assert verified.label_verified_at == "2026-07-14"
    assert verified.cadence_source == "https://example.test/label"
    assert verified.amount == "thin_layer"
    assert verified.amount_source == "https://example.test/label"
    assert verified.evidence_roles == ("daily_support",)
    assert "benzoyl_peroxide" in verified.actives

    stale, warnings, _meta = load_verification_overlay(
        root, now=datetime(2027, 2, 1, tzinfo=timezone.utc)
    )
    assert stale == {}
    assert warnings == ["verification_stale:p1:https://example.test/label"]


def test_approved_assertion_requires_d032_provenance(tmp_path):
    root = tmp_path / "verification"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    body = b"regulatory label bytes"
    digest = hashlib.sha256(body).hexdigest()
    (evidence / digest).write_bytes(body)
    (root / "approved.json").write_text(json.dumps({
        "schema_version": "recsys-verification-1",
        "products": [{
            "product_id": "p1",
            "assertions": [{
                "status": "approved",
                "reviewer_type": "agent",
                "approved_at": "2026-07-14T00:00:00Z",
                "retrieved_at": "2026-07-14T00:00:00Z",
                "source_url": "https://example.test/label",
                "source_sha256": digest,
                "facts": {"spf": 50},
            }],
        }],
    }))

    with pytest.raises(ContractViolation, match="reviewer_id"):
        load_verification_overlay(root, now=datetime(2026, 7, 15, tzinfo=timezone.utc))


def test_approved_assertion_rejects_unknown_fact_keys(tmp_path):
    root = _write_approved_overlay(tmp_path, {"untrusted_field": True})

    with pytest.raises(ContractViolation, match="untrusted_field"):
        load_verification_overlay(root, now=datetime(2026, 7, 15, tzinfo=timezone.utc))


@pytest.mark.parametrize("field", ["reviewer_id", "reviewer_type", "approved_at", "source_url"])
def test_approved_assertion_requires_each_d032_provenance_field(tmp_path, field):
    root = _write_approved_overlay(tmp_path, {"spf": 50}, {field: None})

    with pytest.raises(ContractViolation, match=field):
        load_verification_overlay(root, now=datetime(2026, 7, 15, tzinfo=timezone.utc))


@pytest.mark.parametrize(("overrides", "field"), [
    ({"reviewer_type": "robot"}, "reviewer_type"),
    ({"approved_at": "2026-07-14T00:00:00"}, "approved_at"),
    ({"source_url": 123}, "source_url"),
])
def test_approved_assertion_rejects_invalid_d032_provenance(tmp_path, overrides, field):
    root = _write_approved_overlay(tmp_path, {"spf": 50}, overrides)

    with pytest.raises(ContractViolation, match=field):
        load_verification_overlay(root, now=datetime(2026, 7, 15, tzinfo=timezone.utc))


@pytest.mark.parametrize(("facts", "field"), [
    ({"spf": True}, "spf"),
    ({"spf": 30.5}, "spf"),
    ({"spf": -1}, "spf"),
    ({"broad_spectrum": "true"}, "broad_spectrum"),
    ({"intended_areas": ["moon"]}, "intended_areas"),
    ({"routine_roles": ["unknown_role"]}, "routine_roles"),
    ({"routine_roles": ["cleanser", 1]}, "routine_roles"),
    ({"exposure": "weekly"}, "exposure"),
    ({"comedogenic_claim": "maybe"}, "comedogenic_claim"),
    ({"drug_actives": [{"name": "benzoyl_peroxide", "strength": 2}]}, "drug_actives"),
])
def test_approved_assertion_rejects_invalid_fact_shapes(tmp_path, facts, field):
    root = _write_approved_overlay(tmp_path, facts)

    with pytest.raises(ContractViolation, match=field):
        load_verification_overlay(root, now=datetime(2026, 7, 15, tzinfo=timezone.utc))


def test_a_tampered_evidence_snapshot_stops_its_facts_from_applying(tmp_path):
    """The hash binding is checked at application time, every run -- not once at
    import. A reviewer deleted the four checking lines and 169 tests stayed
    green, which is how a refactor ships facts sourced from bytes nobody
    approved. These two tests are the tension that was missing."""
    root = _multi_row_overlay(tmp_path, [
        ("p1", b"approved label bytes", "2026-07-14T00:00:00Z", {"spf": 50}),
    ])
    digest = hashlib.sha256(b"approved label bytes").hexdigest()
    (root / "evidence" / digest).write_bytes(b"edited after approval")

    with pytest.raises(ContractViolation, match="hash mismatch"):
        load_verification_overlay(root, now=datetime(2026, 7, 15, tzinfo=timezone.utc))


def test_a_deleted_evidence_snapshot_stops_its_facts_from_applying(tmp_path):
    root = _multi_row_overlay(tmp_path, [
        ("p1", b"approved label bytes", "2026-07-14T00:00:00Z", {"spf": 50}),
    ])
    digest = hashlib.sha256(b"approved label bytes").hexdigest()
    (root / "evidence" / digest).unlink()

    with pytest.raises(ContractViolation, match="missing snapshot"):
        load_verification_overlay(root, now=datetime(2026, 7, 15, tzinfo=timezone.utc))
