import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from src.recommendation.approve_verification import approve_batch, main
from src.recommendation.import_catalog import load_verification_overlay


FIXTURE = Path(__file__).parent / "fixtures" / "catalog_verification_sample.json"


def proposed_overlay(tmp_path: Path) -> Path:
    value = json.loads(FIXTURE.read_text())
    for product in value["products"]:
        for assertion in product["assertions"]:
            assertion["status"] = "proposed"
            assertion.pop("reviewer_id", None)
            assertion.pop("reviewer_type", None)
            assertion.pop("approved_at", None)
    source = tmp_path / "proposed.json"
    source.write_text(json.dumps(value))
    return source


def test_approval_requires_explicit_reviewer_attestation(tmp_path):
    with pytest.raises(ValueError, match="acknowledge-reviewed"):
        approve_batch(
            proposed_overlay(tmp_path), tmp_path / "approved.json",
            reviewer_id="reviewer", approved_at="2026-07-14T00:00:00Z",
            acknowledged_review=False,
        )


def test_approval_signs_every_assertion_and_produces_valid_overlay(tmp_path):
    destination = tmp_path / "approved.json"
    report = approve_batch(
        proposed_overlay(tmp_path), destination,
        reviewer_id="reviewer-1", approved_at="2026-07-14T00:00:00Z",
        acknowledged_review=True, reviewer_type="agent",
    )
    assert report == {
        "approved_assertions": 2,
        "approved_products": 2,
        "reviewer_id": "reviewer-1",
        "reviewer_type": "agent",
        "approved_at": "2026-07-14T00:00:00Z",
    }
    value = json.loads(destination.read_text())
    assert all(
        assertion["status"] == "approved"
        and assertion["reviewer_id"] == "reviewer-1"
        and assertion["reviewer_type"] == "agent"
        and assertion["approved_at"] == "2026-07-14T00:00:00Z"
        for product in value["products"]
        for assertion in product["assertions"]
    )
    assert set(load_verification_overlay(destination)) == {"P480274", "P504987"}


def test_approval_rejects_unknown_reviewer_type(tmp_path):
    with pytest.raises(ValueError, match="reviewer_type"):
        approve_batch(
            proposed_overlay(tmp_path), tmp_path / "approved.json",
            reviewer_id="reviewer", reviewer_type="robot",
            approved_at="2026-07-14T00:00:00Z", acknowledged_review=True,
        )


def test_approval_refuses_mixed_or_already_signed_input(tmp_path):
    with pytest.raises(ValueError, match="only proposed assertions"):
        approve_batch(
            FIXTURE, tmp_path / "approved.json",
            reviewer_id="reviewer", approved_at="2026-07-14T00:00:00Z",
            acknowledged_review=True,
        )


def test_approval_refuses_overwrite_or_naive_timestamp(tmp_path):
    source = proposed_overlay(tmp_path)
    with pytest.raises(ValueError, match="must not overwrite"):
        approve_batch(
            source, source, reviewer_id="reviewer",
            approved_at="2026-07-14T00:00:00Z", acknowledged_review=True,
        )
    with pytest.raises(ValueError, match="include a timezone"):
        approve_batch(
            source, tmp_path / "approved.json", reviewer_id="reviewer",
            approved_at="2026-07-14T00:00:00", acknowledged_review=True,
        )


# --- malformed approval inputs are refused before anything is written --------

def _approve(tmp_path, source, **overrides):
    kwargs = {
        "reviewer_id": "reviewer-1", "approved_at": "2026-07-14T00:00:00Z",
        "acknowledged_review": True, "reviewer_type": "agent",
    }
    kwargs.update(overrides)
    return approve_batch(source, tmp_path / "approved.json", **kwargs)


def _write(tmp_path, payload, name="proposed.json"):
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


@pytest.mark.parametrize("reviewer_id", ["", "   ", "\t\n"])
def test_approval_refuses_a_blank_reviewer_id(tmp_path, reviewer_id):
    """Attestation is worthless without an identifiable reviewer."""
    with pytest.raises(ValueError, match="reviewer_id must be non-empty"):
        _approve(tmp_path, proposed_overlay(tmp_path), reviewer_id=reviewer_id)


@pytest.mark.parametrize("approved_at", ["not-a-date", "2026-13-45T00:00:00Z", ""])
def test_approval_refuses_an_unparseable_timestamp(tmp_path, approved_at):
    with pytest.raises(ValueError, match="ISO-8601"):
        _approve(tmp_path, proposed_overlay(tmp_path), approved_at=approved_at)


@pytest.mark.parametrize("payload, match", [
    ("{not json", "invalid verification JSON"),
    ({"schema_version": "1", "products": []}, "schema_version must be '2'"),
    ([], "schema_version must be '2'"),
    ({"schema_version": "2"}, "must contain products"),
    ({"schema_version": "2", "products": []}, "must contain products"),
    ({"schema_version": "2", "products": ["p1"]}, r"products\[0\] must be an object"),
    ({"schema_version": "2", "products": [{"product_id": "p1"}]},
     r"products\[0\]\.assertions must be non-empty"),
    ({"schema_version": "2", "products": [{"product_id": "p1", "assertions": []}]},
     r"products\[0\]\.assertions must be non-empty"),
    ({"schema_version": "2", "products": [{"product_id": "p1", "assertions": ["x"]}]},
     r"assertions\[0\] must be an object"),
], ids=lambda v: None)
def test_approval_refuses_malformed_overlays(tmp_path, payload, match):
    with pytest.raises(ValueError, match=match):
        _approve(tmp_path, _write(tmp_path, payload))


def test_approval_leaves_no_artifact_when_validation_fails(tmp_path):
    """A batch that cannot produce a valid overlay must leave the destination
    absent and no temp file behind -- approval is all-or-nothing."""
    value = json.loads(proposed_overlay(tmp_path).read_text())
    # strip the facts a product needs, so it yields no patch and the count check trips
    value["products"][0]["assertions"][0]["facts"] = {"unknown_field": True}
    source = _write(tmp_path, value, name="broken.json")
    destination = tmp_path / "approved.json"

    with pytest.raises(ValueError):
        approve_batch(source, destination, reviewer_id="reviewer-1",
                      approved_at="2026-07-14T00:00:00Z", acknowledged_review=True,
                      reviewer_type="agent")

    assert not destination.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_approval_does_not_clobber_an_existing_output_when_it_fails(tmp_path):
    destination = tmp_path / "approved.json"
    destination.write_text('{"previous": "approval"}')
    value = json.loads(proposed_overlay(tmp_path).read_text())
    value["products"][0]["assertions"][0]["facts"] = {"unknown_field": True}

    with pytest.raises(ValueError):
        approve_batch(_write(tmp_path, value, name="broken.json"), destination,
                      reviewer_id="reviewer-1", approved_at="2026-07-14T00:00:00Z",
                      acknowledged_review=True, reviewer_type="agent")

    assert json.loads(destination.read_text()) == {"previous": "approval"}


def test_approval_does_not_mutate_the_proposed_source(tmp_path):
    source = proposed_overlay(tmp_path)
    before = source.read_text()
    _approve(tmp_path, source)
    assert source.read_text() == before


def test_reviewer_id_is_recorded_stripped(tmp_path):
    destination = tmp_path / "approved.json"
    report = approve_batch(proposed_overlay(tmp_path), destination,
                           reviewer_id="  reviewer-1  ", approved_at="2026-07-14T00:00:00Z",
                           acknowledged_review=True, reviewer_type="agent")
    assert report["reviewer_id"] == "reviewer-1"
    value = json.loads(destination.read_text())
    assert value["products"][0]["assertions"][0]["reviewer_id"] == "reviewer-1"


# --- CLI ----------------------------------------------------------------------

def test_cli_refuses_without_the_attestation_flag(tmp_path, monkeypatch):
    """--acknowledge-reviewed is the whole point: absent it, nothing is signed."""
    source = proposed_overlay(tmp_path)
    destination = tmp_path / "approved.json"
    monkeypatch.setattr(sys, "argv", [
        "approve_verification", str(source), "--out", str(destination),
        "--reviewer-id", "reviewer-1", "--reviewer-type", "human",
    ])
    with pytest.raises(ValueError, match="acknowledge-reviewed"):
        main()
    assert not destination.exists()


def test_cli_stamps_the_current_time_when_no_timestamp_is_given(tmp_path, monkeypatch, capsys):
    source = proposed_overlay(tmp_path)
    destination = tmp_path / "approved.json"
    monkeypatch.setattr(sys, "argv", [
        "approve_verification", str(source), "--out", str(destination),
        "--reviewer-id", "reviewer-1", "--reviewer-type", "human",
        "--acknowledge-reviewed",
    ])
    assert main() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["reviewer_type"] == "human"
    # a real, timezone-aware stamp the overlay loader will accept
    stamped = datetime.fromisoformat(report["approved_at"].replace("Z", "+00:00"))
    assert stamped.tzinfo is not None
    assert load_verification_overlay(destination)


def test_cli_passes_an_explicit_timestamp_through(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "approve_verification", str(proposed_overlay(tmp_path)),
        "--out", str(tmp_path / "approved.json"),
        "--reviewer-id", "reviewer-1", "--reviewer-type", "agent",
        "--approved-at", "2026-07-14T00:00:00Z", "--acknowledge-reviewed",
    ])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["approved_at"] == "2026-07-14T00:00:00Z"
