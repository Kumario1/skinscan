"""Sign a reviewed catalog verification batch without hand-editing JSON."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .import_catalog import load_verification_overlay


def approve_batch(
    source: Path,
    destination: Path,
    *,
    reviewer_id: str,
    reviewer_type: str = "human",
    approved_at: str,
    acknowledged_review: bool,
) -> dict[str, int | str]:
    """Approve every proposed assertion after an identified reviewer attestation."""
    if not acknowledged_review:
        raise ValueError("approval requires --acknowledge-reviewed")
    if not reviewer_id.strip():
        raise ValueError("reviewer_id must be non-empty")
    if reviewer_type not in {"human", "agent"}:
        raise ValueError("reviewer_type must be 'human' or 'agent'")
    if source.resolve() == destination.resolve():
        raise ValueError("approved output must not overwrite the proposed overlay")
    try:
        parsed_approval_time = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("approved_at must be an ISO-8601 timestamp") from exc
    if parsed_approval_time.tzinfo is None:
        raise ValueError("approved_at must include a timezone")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid verification JSON: {exc}") from exc
    if not isinstance(value, dict) or str(value.get("schema_version")) != "2":
        raise ValueError("verification overlay schema_version must be '2'")
    products = value.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("verification overlay must contain products")

    approved = 0
    for product_index, product in enumerate(products):
        if not isinstance(product, dict):
            raise ValueError(f"products[{product_index}] must be an object")
        assertions = product.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            raise ValueError(f"products[{product_index}].assertions must be non-empty")
        for assertion_index, assertion in enumerate(assertions):
            if not isinstance(assertion, dict):
                raise ValueError(
                    f"products[{product_index}].assertions[{assertion_index}] must be an object"
                )
            if assertion.get("status") != "proposed":
                raise ValueError(
                    "approval input must contain only proposed assertions; "
                    f"found {assertion.get('status')!r}"
                )
            assertion["status"] = "approved"
            assertion["reviewer_id"] = reviewer_id.strip()
            assertion["reviewer_type"] = reviewer_type
            assertion["approved_at"] = approved_at
            approved += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        patches = load_verification_overlay(temporary)
        if len(patches) != len(products):
            raise ValueError("not every product produced an approved verification patch")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "approved_assertions": approved,
        "approved_products": len(products),
        "reviewer_id": reviewer_id.strip(),
        "reviewer_type": reviewer_type,
        "approved_at": approved_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument(
        "--reviewer-type", choices=("human", "agent"), required=True,
        help="kind of reviewer responsible for the evidence decision",
    )
    parser.add_argument(
        "--approved-at", default=None,
        help="ISO-8601 approval time (default: current UTC time)",
    )
    parser.add_argument(
        "--acknowledge-reviewed", action="store_true",
        help="attest that the named reviewer checked every source and asserted fact",
    )
    args = parser.parse_args()
    approved_at = args.approved_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    print(json.dumps(approve_batch(
        args.overlay,
        args.out,
        reviewer_id=args.reviewer_id,
        reviewer_type=args.reviewer_type,
        approved_at=approved_at,
        acknowledged_review=args.acknowledge_reviewed,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
