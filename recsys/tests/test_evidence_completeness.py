"""Every fact the overlay asserts must ship with the bytes that back it.

`load_verification_overlay` re-checks each snapshot's sha256 at application time,
so a snapshot that exists only in one developer's working tree passes the whole
suite on that machine and raises `contract_violation:verification.evidence` on
every other one. Existence on disk is therefore not the property worth testing --
being committed is. These tests fail on the author's machine, which is the only
place the omission is still cheap to fix.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPROVED = ROOT / "recsys" / "data" / "verification" / "approved.json"
EVIDENCE = ROOT / "recsys" / "data" / "verification" / "evidence"


def _asserted_digests() -> set[str]:
    approved = json.loads(APPROVED.read_text())
    return {
        assertion["source_sha256"]
        for product in approved["products"]
        for assertion in product["assertions"]
    }


def _tracked(paths: list[Path]) -> set[Path]:
    if not paths:
        return set()
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", *(str(p) for p in paths)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {ROOT / name for name in completed.stdout.split("\0") if name}


def test_every_asserted_source_has_its_evidence_snapshot() -> None:
    missing = sorted(d for d in _asserted_digests() if not (EVIDENCE / d).is_file())
    assert missing == [], f"approved.json asserts facts with no evidence bytes: {missing}"


def test_every_evidence_snapshot_is_committed() -> None:
    referenced = [EVIDENCE / d for d in sorted(_asserted_digests())]
    untracked = sorted(p.name for p in referenced if p not in _tracked(referenced))
    assert untracked == [], (
        "evidence is referenced by approved.json but not tracked by git; the engine "
        f"will raise contract_violation:verification.evidence on a fresh clone: {untracked}"
    )
