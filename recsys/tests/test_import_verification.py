"""The legacy -> recsys verification bridge.

Contract (recsys/README.md): "Import only already-approved assertions; this
command never approves facts." Every approved assertion must carry evidence
whose bytes hash to its recorded source_sha256, and the emitted overlay must
be loadable by recsys.verification.load_verification_overlay.
"""
import datetime as dt
import hashlib
import json

import pytest

from recsys.tools.import_verification import build
from recsys.verification import SCHEMA_VERSION, load_verification_overlay

NOW = dt.datetime(2026, 7, 15, tzinfo=dt.timezone.utc)


def _evidence(root, body: bytes) -> str:
    """Write an evidence snapshot named by its own digest; return the digest."""
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(body).hexdigest()
    (root / digest).write_bytes(body)
    return digest


def _assertion(digest, *, status="approved", approved_at="2026-07-14T00:00:00Z", facts=None):
    return {
        "status": status,
        "reviewer_id": "reviewer-1",
        "reviewer_type": "agent",
        "approved_at": approved_at,
        "retrieved_at": "2026-07-14T00:00:00Z",
        "source_url": "https://example.test/label",
        "source_sha256": digest,
        "facts": facts or {"routine_roles": ["treatment"]},
    }


def _source(tmp_path, products, name="source.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": "2", "products": products}), encoding="utf-8")
    return path


def _approved(out_root):
    return json.loads((out_root / "approved.json").read_text(encoding="utf-8"))


def test_imports_approved_assertion_with_evidence(tmp_path):
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [_assertion(digest)]}])
    out_root = tmp_path / "out"

    stats = build(source, src_evidence, out_root)

    assert stats == {"products": 1, "dropped_facts": [], "evidence_snapshots": 1}
    value = _approved(out_root)
    assert value["schema_version"] == SCHEMA_VERSION
    assert [row["product_id"] for row in value["products"]] == ["p1"]
    # the evidence snapshot travels with the overlay, byte-for-byte
    assert (out_root / "evidence" / digest).read_bytes() == b"label bytes"


def test_drops_assertions_that_are_not_approved(tmp_path):
    """"never approves facts": pending/rejected assertions must not be imported."""
    src_evidence = tmp_path / "evidence"
    approved_digest = _evidence(src_evidence, b"approved label")
    pending_digest = _evidence(src_evidence, b"pending label")
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [
        _assertion(approved_digest),
        _assertion(pending_digest, status="pending"),
        _assertion(pending_digest, status="rejected"),
    ]}])
    out_root = tmp_path / "out"

    stats = build(source, src_evidence, out_root)

    assert stats == {"products": 1, "dropped_facts": [], "evidence_snapshots": 1}
    [row] = _approved(out_root)["products"]
    assert [a["status"] for a in row["assertions"]] == ["approved"]
    assert [a["source_sha256"] for a in row["assertions"]] == [approved_digest]
    # evidence for a non-approved assertion has no business in the overlay
    assert not (out_root / "evidence" / pending_digest).exists()


def test_omits_products_with_no_approved_assertions(tmp_path):
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"pending label")
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [
        _assertion(digest, status="pending"),
    ]}])
    out_root = tmp_path / "out"

    stats = build(source, src_evidence, out_root)

    assert stats == {"products": 0, "dropped_facts": [], "evidence_snapshots": 0}
    assert _approved(out_root)["products"] == []


def test_rejects_assertion_whose_evidence_is_missing(tmp_path):
    src_evidence = tmp_path / "evidence"
    src_evidence.mkdir(parents=True)
    absent = hashlib.sha256(b"never stored").hexdigest()
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [_assertion(absent)]}])

    with pytest.raises(SystemExit, match="missing evidence"):
        build(source, src_evidence, tmp_path / "out")


def test_rejects_tampered_evidence(tmp_path):
    """The snapshot's bytes must hash to its recorded digest."""
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"original label")
    (src_evidence / digest).write_bytes(b"tampered label")  # same name, different bytes
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [_assertion(digest)]}])

    with pytest.raises(SystemExit, match="hash mismatch"):
        build(source, src_evidence, tmp_path / "out")


def test_rejects_assertion_with_no_digest(tmp_path):
    src_evidence = tmp_path / "evidence"
    src_evidence.mkdir(parents=True)
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [_assertion(None)]}])

    with pytest.raises(SystemExit, match="source_sha256"):
        build(source, src_evidence, tmp_path / "out")


def test_rejects_row_with_no_product_id(tmp_path):
    """Malformed input gets the same clean refusal as every other bad row, not a
    raw KeyError. Mirrors the consumer, which raises on a missing product_id."""
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    source = _source(tmp_path, [{"assertions": [_assertion(digest)]}])

    with pytest.raises(SystemExit, match="product_id"):
        build(source, src_evidence, tmp_path / "out")


def test_absolute_digest_cannot_escape_the_evidence_directory(tmp_path):
    """A digest is a filename, not a path: pathlib lets an absolute string
    swallow the join, so the hash check is what must stop it."""
    src_evidence = tmp_path / "evidence"
    src_evidence.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [_assertion(str(outside))]}])

    with pytest.raises(SystemExit):
        build(source, src_evidence, tmp_path / "out")
    assert not (tmp_path / "out" / "evidence" / "outside.txt").exists()


@pytest.mark.parametrize("products", [[], None], ids=["empty", "null"])
def test_empty_source_yields_an_empty_overlay(tmp_path, products):
    src_evidence = tmp_path / "evidence"
    src_evidence.mkdir(parents=True)
    source = _source(tmp_path, products)
    out_root = tmp_path / "out"

    stats = build(source, src_evidence, out_root)

    assert stats == {"products": 0, "dropped_facts": [], "evidence_snapshots": 0}
    value = _approved(out_root)
    assert value == {"schema_version": SCHEMA_VERSION, "products": []}


def test_products_are_sorted_by_product_id(tmp_path):
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    source = _source(tmp_path, [
        {"product_id": pid, "assertions": [_assertion(digest)]}
        for pid in ("p3", "p1", "p2")
    ])
    out_root = tmp_path / "out"

    build(source, src_evidence, out_root)

    assert [row["product_id"] for row in _approved(out_root)["products"]] == ["p1", "p2", "p3"]


def test_evidence_shared_by_two_products_is_copied_once(tmp_path):
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"shared label")
    source = _source(tmp_path, [
        {"product_id": "p1", "assertions": [_assertion(digest)]},
        {"product_id": "p2", "assertions": [_assertion(digest)]},
    ])
    out_root = tmp_path / "out"

    stats = build(source, src_evidence, out_root)

    assert stats == {"products": 2, "dropped_facts": [], "evidence_snapshots": 1}
    assert [p.name for p in (out_root / "evidence").iterdir()] == [digest]


def test_import_is_idempotent(tmp_path):
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [_assertion(digest)]}])
    out_root = tmp_path / "out"

    first = build(source, src_evidence, out_root)
    first_bytes = (out_root / "approved.json").read_bytes()
    second = build(source, src_evidence, out_root)

    assert first == second
    assert (out_root / "approved.json").read_bytes() == first_bytes


def _committed_with_intended_areas(tmp_path, src_evidence, digest, out_root):
    """An overlay that asserts intended_areas, plus a source that no longer does
    -- the P188306 shape: a re-verification supersedes an assertion and silently
    fails to re-asserts one of its facts."""
    build(_source(tmp_path, [{"product_id": "p1", "assertions": [
        _assertion(digest, facts={"routine_roles": ["treatment"], "intended_areas": ["face"]}),
    ]}]), src_evidence, out_root)
    return _source(tmp_path, [{"product_id": "p1", "assertions": [
        _assertion(digest, facts={"routine_roles": ["treatment"]}),
    ]}], name="narrowed.json")


def test_refuses_to_drop_a_fact_the_committed_overlay_asserts(tmp_path):
    """One lost intended_areas quarantines the product and zeroes out every
    routine downstream. That must never be something a red test suite tells us."""
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    out_root = tmp_path / "out"
    narrowed = _committed_with_intended_areas(tmp_path, src_evidence, digest, out_root)
    before = (out_root / "approved.json").read_bytes()

    with pytest.raises(SystemExit, match="intended_areas"):
        build(narrowed, src_evidence, out_root)

    # refused *before* writing: no half-applied overlay
    assert (out_root / "approved.json").read_bytes() == before


def test_refuses_to_drop_a_product_entirely(tmp_path):
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    out_root = tmp_path / "out"
    build(_source(tmp_path, [{"product_id": "p1", "assertions": [_assertion(digest)]}]),
          src_evidence, out_root)
    emptied = _source(tmp_path, [], name="emptied.json")

    with pytest.raises(SystemExit, match="product dropped entirely"):
        build(emptied, src_evidence, out_root)


def test_allow_fact_loss_records_the_loss_deliberately(tmp_path):
    """The loss is legitimate exactly when no source states the fact -- but it
    has to be said out loud, not discovered."""
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    out_root = tmp_path / "out"
    narrowed = _committed_with_intended_areas(tmp_path, src_evidence, digest, out_root)

    stats = build(narrowed, src_evidence, out_root, allow_fact_loss=True)

    [row] = _approved(out_root)["products"]
    assert "intended_areas" not in row["assertions"][0]["facts"]
    # The flag's promise is that the loss is recorded, not merely permitted: it
    # must come back in the stats a caller sees, naming the product and fact.
    assert any("intended_areas" in loss for loss in stats["dropped_facts"])


def test_a_first_import_has_no_overlay_to_regress_against(tmp_path):
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [_assertion(digest)]}])

    assert build(source, src_evidence, tmp_path / "fresh")["products"] == 1


def test_imported_overlay_loads_through_the_real_consumer(tmp_path):
    """The whole point of the bridge: what it writes, recsys must be able to read."""
    src_evidence = tmp_path / "evidence"
    digest = _evidence(src_evidence, b"label bytes")
    source = _source(tmp_path, [{"product_id": "p1", "assertions": [
        _assertion(digest, facts={"routine_roles": ["treatment"], "otc_drug": True}),
    ]}])
    out_root = tmp_path / "out"

    build(source, src_evidence, out_root)
    overlay, warnings, provenance = load_verification_overlay(out_root, now=NOW)

    assert warnings == []
    assert overlay["p1"]["routine_roles"] == ["treatment"]
    assert overlay["p1"]["otc_drug"] is True
    assert provenance["products"] == 1
