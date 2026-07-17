"""Import approved legacy assertions into the standalone recsys overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from ..contracts import ContractViolation
from ..verification import SCHEMA_VERSION, _validate_approved_assertion
from .common import write_json


def _fact_keys(products: list[dict]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for row in products:
        keys = merged.setdefault(row["product_id"], set())
        for assertion in row.get("assertions") or []:
            keys |= set(assertion.get("facts") or {})
    return merged


def _fact_losses(out_root: Path, products: list[dict]) -> list[str]:
    """Facts the committed overlay asserts that this build would drop.

    A re-verification that supersedes an assertion without re-asserting all of
    its facts silently narrows the overlay, and one lost intended_areas is
    enough to quarantine a product and zero out every routine downstream. The
    loss is legitimate exactly when the new source does not state the fact --
    but it is never something to discover from a red test suite.
    """
    approved_path = out_root / "approved.json"
    if not approved_path.exists():
        return []
    committed = json.loads(approved_path.read_text(encoding="utf-8"))
    before = _fact_keys(committed.get("products") or [])
    after = _fact_keys(products)
    losses = []
    for product_id, keys in sorted(before.items()):
        lost = keys - after.get(product_id, set())
        if lost:
            dropped = "" if product_id in after else "; product dropped entirely"
            losses.append(f"{product_id}: {', '.join(sorted(lost))}{dropped}")
    return losses


def build(
    source: Path,
    source_evidence: Path,
    out_root: Path,
    *,
    allow_fact_loss: bool = False,
) -> dict:
    value = json.loads(source.read_text(encoding="utf-8"))
    products = []
    digests = set()
    for row in value.get("products") or []:
        assertions = [a for a in row.get("assertions") or [] if a.get("status") == "approved"]
        if not assertions:
            continue
        if not row.get("product_id"):
            raise SystemExit(f"missing product_id for row with {len(assertions)} approved assertions")
        for assertion in assertions:
            # Validate D-032 provenance here, at the boundary. The loader
            # validates too, but by then the malformed assertion is a committed
            # artifact and every `recsys recommend` dies three steps from the
            # cause; the importer is where the fix is still one command away.
            try:
                _validate_approved_assertion(assertion, row["product_id"])
            except ContractViolation as violation:
                raise SystemExit(
                    f"refusing to import {row['product_id']}: {violation}"
                ) from violation
            digest = assertion.get("source_sha256")
            snapshot = source_evidence / str(digest)
            if not digest or not snapshot.exists():
                raise SystemExit(f"missing evidence for {row.get('product_id')}: {digest}")
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                raise SystemExit(f"evidence hash mismatch: {digest}")
            digests.add(digest)
        products.append({"product_id": row["product_id"], "assertions": assertions})
    losses = _fact_losses(out_root, products)
    if losses and not allow_fact_loss:
        raise SystemExit(
            "refusing to drop facts the committed overlay asserts:\n  "
            + "\n  ".join(losses)
            + "\nRe-assert each fact against a source that states it, or pass"
            " --allow-fact-loss to record the loss deliberately."
        )
    if losses:
        # The refusal message promises the flag "records the loss deliberately";
        # a loss that leaves no trace but a git diff is not recorded.
        print("dropping facts the committed overlay asserts:", file=sys.stderr)
        for loss in losses:
            print(f"  {loss}", file=sys.stderr)
    evidence_out = out_root / "evidence"
    evidence_out.mkdir(parents=True, exist_ok=True)
    for digest in sorted(digests):
        shutil.copyfile(source_evidence / digest, evidence_out / digest)
    write_json(out_root / "approved.json", {
        "schema_version": SCHEMA_VERSION,
        "products": sorted(products, key=lambda row: row["product_id"]),
    })
    return {
        "products": len(products),
        "evidence_snapshots": len(digests),
        "dropped_facts": losses,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--allow-fact-loss", action="store_true",
        help="proceed even though the rebuild drops facts the committed overlay asserts",
    )
    args = parser.parse_args(argv)
    print(build(
        args.source, args.source_evidence, args.out_root,
        allow_fact_loss=args.allow_fact_loss,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
