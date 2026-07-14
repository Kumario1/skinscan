"""The verification-loop orchestrator advances the batch-001 process end to end."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.recommendation.verification_loop import main, now_iso

CSV = """product_id,name,brand,category,ingredients,price
C1,Gentle Wash,BrandA,cleanser,"Water, Glycerin",9
M1,Daily Lotion,BrandA,moisturizer,"Water, Ceramides",12
S1,Sun Shield,BrandB,spf,"Water, Zinc Oxide",15
T1,Azelaic Gel,BrandC,treatment,"Azelaic Acid",20
T2,BP Lotion,BrandC,treatment,"Benzoyl Peroxide",18
T3,Combo Gel,BrandC,treatment,"Adapalene, Benzoyl Peroxide",25
"""

SUPPORT_FACTS = {
    "C1": {"intended_areas": ["face"], "routine_roles": ["cleanser"],
           "format": "cleanser", "exposure": "rinse_off",
           "cadence": "am_pm", "cadence_source": "https://branda.example/wash"},
    "M1": {"intended_areas": ["face"], "routine_roles": ["moisturizer"],
           "format": "lotion", "exposure": "leave_on", "cadence": "am_pm",
           "cadence_source": "https://branda.example/lotion",
           "comedogenic_claim": "claimed_noncomedogenic"},
    "S1": {"intended_areas": ["face"], "routine_roles": ["sunscreen"],
           "format": "lotion", "exposure": "leave_on", "cadence": "am",
           "cadence_source": "https://brandb.example/spf", "broad_spectrum": True,
           "spf": 40, "label_source": "https://brandb.example/spf",
           "label_verified_at": "2026-07-14", "comedogenic_claim": "claimed_noncomedogenic"},
}
TREATMENT_ACTIVES = {
    "T1": [{"name": "azelaic_acid", "strength": "10%"}],
    "T2": [{"name": "benzoyl_peroxide", "strength": "2.5%"}],
    "T3": [{"name": "adapalene", "strength": "0.1%"},
           {"name": "benzoyl_peroxide", "strength": "2.5%"}],
}
for pid, actives in TREATMENT_ACTIVES.items():
    url = f"https://dailymed.example/{pid}"
    SUPPORT_FACTS[pid] = {
        "intended_areas": ["face"], "routine_roles": ["treatment"],
        "format": "gel", "exposure": "leave_on", "cadence": "per_label",
        "cadence_source": url, "otc_drug": True, "evidence_grade": "regulatory_label",
        "drug_actives": [dict(a, source=url) for a in actives],
        "label_source": url, "label_verified_at": "2026-07-14",
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "raw.csv").write_text(CSV, encoding="utf-8")
    (tmp_path / "data" / "verification").mkdir(parents=True)
    return tmp_path


def loop(root: Path, *argv: str) -> int:
    return main(["--root", str(root), *argv])


def rebuild(root: Path, *extra: str) -> int:
    return loop(root, "rebuild", "--sephora-csv", "raw.csv", "--sephora-format",
                "simple", "--beautyapi-jsonl", "missing.jsonl",
                "--support-minimum", "1", *extra)


def manifest(root: Path) -> dict:
    return json.loads((root / "data/verification/loop_manifest.json").read_text())


def snapshot(root: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    evidence = root / "data/verification/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / digest).write_bytes(content)
    return digest


def assertion(root: Path, pid: str, facts: dict, retrieved_at: str | None = None) -> dict:
    url = str(facts.get("cadence_source", f"https://example.com/{pid}"))
    return {"status": "proposed", "source_url": url,
            "retrieved_at": retrieved_at or now_iso(),
            "source_sha256": snapshot(root, f"page for {pid}".encode()),
            "facts": facts}


def write_proposed(root: Path, batch: str, pids: list[str]) -> Path:
    path = root / "data/verification/batches" / batch / "proposed.json"
    path.write_text(json.dumps({"schema_version": "2", "products": [
        {"product_id": pid, "assertions": [assertion(root, pid, SUPPORT_FACTS[pid])]}
        for pid in pids]}), encoding="utf-8")
    return path


def approve(root: Path, batch: str) -> int:
    return loop(root, "approve", "--batch", batch, "--reviewer-id", "test-agent",
                "--reviewer-type", "agent", "--acknowledge-reviewed")


def test_full_cycle_to_stopping_criteria(root: Path, capsys):
    assert rebuild(root) == 0
    report = json.loads((root / "data/verification/catalog_completeness.json").read_text())
    assert not report["complete"] and report["shortfalls"] == {
        "cleanser": 1, "moisturizer": 1, "sunscreen": 1}

    assert loop(root, "select", "--batch-size", "6") == 0
    m = manifest(root)
    batch = sorted(m["batches"])[0]
    members = {p for p, e in m["products"].items() if e.get("batch") == batch}
    assert members == {"C1", "M1", "S1", "T1", "T2", "T3"}
    assert all(e["state"] == "researching" for e in m["products"].values())
    brief = (root / "data/verification/batches" / batch / "RESEARCH_BRIEF.md").read_text()
    assert "routine_role_not_verified" in brief and "azelaic_acid 10%" in brief

    write_proposed(root, batch, sorted(members))
    assert loop(root, "ingest", "--batch", batch) == 0
    assert all(e["state"] == "proposed" for e in manifest(root)["products"].values())

    assert approve(root, batch) == 2  # no REVIEW.md yet: refuse to sign
    (root / "data/verification/batches" / batch / "REVIEW.md").write_text(
        "checked every source and fact", encoding="utf-8")
    assert approve(root, batch) == 0
    assert all(e["state"] == "approved" for e in manifest(root)["products"].values())

    assert rebuild(root) == 0
    m = manifest(root)
    assert all(e["state"] == "eligible" for e in m["products"].values())
    assert m["last_rebuild"]["complete"] and m["last_rebuild"]["unmatched"] == []

    assert loop(root, "status") == 1  # audit still outstanding
    assert loop(root, "audit", "--sample", "3", "--seed", "7") == 0
    assert loop(root, "audit", "--record", "pass", "--notes", "spot-checked") == 0
    assert loop(root, "status") == 0  # every stopping criterion met


def test_ingest_fails_closed(root: Path, capsys):
    assert rebuild(root) == 0
    assert loop(root, "select", "--batch-size", "2") == 0
    batch = sorted(manifest(root)["batches"])[0]
    members = sorted(p for p, e in manifest(root)["products"].items()
                     if e.get("batch") == batch)
    good, bad = members[0], members[1]
    rows = [
        {"product_id": good, "assertions": [assertion(root, good, SUPPORT_FACTS[good])]},
        {"product_id": bad, "assertions": [
            {"status": "proposed", "source_url": "http://insecure.example",
             "retrieved_at": "not-a-date", "source_sha256": "ff" * 32,
             "facts": {"format": "gel"}},
            {"status": "proposed", "source_url": "https://ok.example",
             "retrieved_at": now_iso(),
             "source_sha256": snapshot(root, b"other page"),
             "facts": {"format": "lotion"}},  # overlaps first assertion
        ]},
    ]
    path = root / "data/verification/batches" / batch / "proposed.json"
    path.write_text(json.dumps({"schema_version": "2", "products": rows}))
    assert loop(root, "ingest", "--batch", batch) == 1
    states = manifest(root)["products"]
    assert states[good]["state"] == "proposed"
    assert states[bad]["state"] == "researching"
    joined = ";".join(states[bad]["reasons"])
    assert "source_url_not_https" in joined and "snapshot_missing" in joined
    assert "facts_overlap_between_assertions" in joined


def test_approve_enforces_batch_membership(root: Path):
    assert rebuild(root) == 0
    assert loop(root, "select", "--batch-size", "2") == 0
    batch = sorted(manifest(root)["batches"])[0]
    members = sorted(p for p, e in manifest(root)["products"].items()
                     if e.get("batch") == batch)
    write_proposed(root, batch, members)
    assert loop(root, "ingest", "--batch", batch) == 0
    (root / "data/verification/batches" / batch / "REVIEW.md").write_text("reviewed")
    assert loop(root, "reject", "--batch", batch, "--product", members[1],
                "--reason", "variant mismatch") == 0
    assert approve(root, batch) == 2  # rejected product still in proposed.json
    write_proposed(root, batch, members[:1])
    assert approve(root, batch) == 0
    assert manifest(root)["products"][members[1]]["state"] == "rejected"

    # rejected products stay out of later selections
    assert rebuild(root) == 0
    assert loop(root, "select", "--batch-size", "6") == 0
    entry = manifest(root)["products"][members[1]]
    assert entry["state"] == "rejected" and entry["batch"] == batch


def test_unmatched_approved_id_is_flagged(root: Path, capsys):
    overlay = {"schema_version": "2", "products": [{"product_id": "GHOST", "assertions": [{
        "status": "approved", "source_url": "https://x.example",
        "retrieved_at": now_iso(), "source_sha256": snapshot(root, b"ghost"),
        "reviewer_id": "r", "reviewer_type": "agent", "approved_at": now_iso(),
        "facts": {"routine_roles": ["cleanser"]}}]}]}
    (root / "data/verification/catalog-verification-batch-000-approved.json").write_text(
        json.dumps(overlay))
    assert rebuild(root) == 0
    m = manifest(root)
    assert m["last_rebuild"]["unmatched"] == ["GHOST"]
    assert m["products"]["GHOST"]["state"] == "quarantined"
    assert loop(root, "status") == 1


def test_refresh_marks_stale_and_reselects(root: Path):
    assert rebuild(root) == 0
    assert loop(root, "select", "--batch-size", "1") == 0
    batch = sorted(manifest(root)["batches"])[0]
    pid = next(p for p, e in manifest(root)["products"].items() if e.get("batch") == batch)
    path = root / "data/verification/batches" / batch / "proposed.json"
    path.write_text(json.dumps({"schema_version": "2", "products": [
        {"product_id": pid, "assertions": [
            assertion(root, pid, SUPPORT_FACTS[pid], retrieved_at="2020-01-01T00:00:00Z")]}]}))
    assert loop(root, "ingest", "--batch", batch) == 0
    (root / "data/verification/batches" / batch / "REVIEW.md").write_text("reviewed")
    assert approve(root, batch) == 0
    assert rebuild(root) == 0
    assert loop(root, "refresh") == 0
    assert manifest(root)["products"][pid]["state"] == "refresh_due"
    approved = json.loads(
        (root / "data/verification/batches" / batch / "approved.json").read_text())
    assert all(a["status"] == "stale"
               for row in approved["products"] for a in row["assertions"])

    assert loop(root, "select", "--batch-size", "1") == 0
    entry = manifest(root)["products"][pid]
    assert entry["state"] == "researching" and entry["batch"] != batch
    brief = (root / "data/verification/batches" / entry["batch"]
             / "RESEARCH_BRIEF.md").read_text()
    assert "Re-verification" in brief and "routine_roles" in brief

    # the re-researched batch re-asserts the same facts without colliding
    write_proposed(root, entry["batch"], [pid])
    assert loop(root, "ingest", "--batch", entry["batch"]) == 0
    (root / "data/verification/batches" / entry["batch"] / "REVIEW.md").write_text("ok")
    assert approve(root, entry["batch"]) == 0
    assert rebuild(root) == 0
    assert manifest(root)["products"][pid]["state"] == "eligible"
    assert loop(root, "refresh", "--dry-run") == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
