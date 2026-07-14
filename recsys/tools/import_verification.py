"""Import approved legacy assertions into the standalone recsys overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ..verification import SCHEMA_VERSION
from .common import write_json


def build(source: Path, source_evidence: Path, out_root: Path) -> dict:
    value = json.loads(source.read_text(encoding="utf-8"))
    products = []
    digests = set()
    for row in value.get("products") or []:
        assertions = [a for a in row.get("assertions") or [] if a.get("status") == "approved"]
        if not assertions:
            continue
        for assertion in assertions:
            digest = assertion.get("source_sha256")
            snapshot = source_evidence / str(digest)
            if not digest or not snapshot.exists():
                raise SystemExit(f"missing evidence for {row.get('product_id')}: {digest}")
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                raise SystemExit(f"evidence hash mismatch: {digest}")
            digests.add(digest)
        products.append({"product_id": row["product_id"], "assertions": assertions})
    evidence_out = out_root / "evidence"
    evidence_out.mkdir(parents=True, exist_ok=True)
    for digest in sorted(digests):
        shutil.copyfile(source_evidence / digest, evidence_out / digest)
    write_json(out_root / "approved.json", {
        "schema_version": SCHEMA_VERSION,
        "products": sorted(products, key=lambda row: row["product_id"]),
    })
    return {"products": len(products), "evidence_snapshots": len(digests)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(build(args.source, args.source_evidence, args.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
