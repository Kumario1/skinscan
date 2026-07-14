import hashlib
import json
from datetime import datetime, timezone

from recsys.catalog import CatalogProduct
from recsys.verification import apply_verification, load_verification_overlay


def _product():
    return CatalogProduct(
        product_id="p1", name="SPF 15", brand="Test", category="spf",
        price_usd=10, size=None, format=None, spf=15, spf_source="name_parse",
        inci=(), inci_sha256="", actives=(),
    )


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
                "retrieved_at": "2026-07-14T00:00:00Z",
                "source_url": "https://example.test/label",
                "source_sha256": digest,
                "facts": {
                    "spf": 50,
                    "broad_spectrum": True,
                    "cadence": "per_label",
                    "evidence_grade": "regulatory_label",
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

    stale, warnings, _meta = load_verification_overlay(
        root, now=datetime(2027, 2, 1, tzinfo=timezone.utc)
    )
    assert stale == {}
    assert warnings == ["verification_stale:p1:https://example.test/label"]
