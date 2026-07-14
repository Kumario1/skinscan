import json
from pathlib import Path

import pytest

from src.recommendation.approve_verification import approve_batch
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
