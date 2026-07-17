"""CLI: python -m recsys recommend --analysis <analysis.json> [...]"""
from __future__ import annotations

import argparse
import sys

from .contracts import ContractViolation
from .pipeline import DEFAULT_DATA_ROOT, emit, run, skipped_signal_warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recommend", help="analysis.json + profile -> recommendations.json")
    rec.add_argument(
        "--analysis", required=True,
        help="analysis.json schema 4 (schema 3 accepted for one migration release)",
    )
    rec.add_argument("--profile", default=None, help="profile JSON (falls back to analysis input_profile)")
    rec.add_argument("--catalog", default=None, help="catalog JSON (default: bundled seed catalog)")
    rec.add_argument("--data-root", default=None, help=f"recsys data dir (default: {DEFAULT_DATA_ROOT})")
    rec.add_argument("--out", default="recommendations.json", help="output path")
    rec.add_argument("--generated-at", default=None, help="pin the timestamp (deterministic runs)")
    rec.add_argument("--eligibility-mode", default="hybrid", choices=("strict", "hybrid"),
                     help="hybrid: D-035 safety gates plus verification-aware ranking "
                          "(default); strict: deprecated compatibility alias")
    rec.add_argument("--allow-signal-catalog-mismatch", action="store_true",
                     help="accept a run whose signal stores are bound to a different "
                          "catalog: mismatched stores are skipped with a warning, every "
                          "store-backed signal scores a neutral 0.5, and the document is "
                          "still written -- but the exit code is 3 because the ranking "
                          "is blind (same as any skipped-store run)")
    rec.add_argument("--allow-unreviewed-policy", action="store_true",
                     help="legacy schema-3 dev only: accept an unreviewed single-primary "
                          "therapy policy; schema 4 uses the audited synthetic-MVP lesion "
                          "policy and all product safety gates remain active")
    args = parser.parse_args(argv)

    try:
        document = run(
            analysis_path=args.analysis,
            profile_path=args.profile,
            catalog_path=args.catalog,
            data_root=args.data_root,
            generated_at=args.generated_at,
            eligibility_mode=args.eligibility_mode,
            allow_signal_catalog_mismatch=args.allow_signal_catalog_mismatch,
            allow_unreviewed_policy=args.allow_unreviewed_policy,
        )
    except ContractViolation as exc:
        # The contract is the product: a violated one is a clean non-zero exit
        # with the reason on stderr, not a traceback the caller has to parse.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = emit(document, args.out)
    routines = document.get("routines") or []
    # ARCHITECTURE.md tells operators to check that data_versions.signals is
    # populated and warnings is empty. It said so because this CLI withheld both
    # and always returned 0, which made every automated caller -- including the
    # integrated pipeline, which shells out to this very command -- read a blind
    # run as a success. Report them here so the check is the tool's job.
    warnings = document.get("warnings") or []
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    print(f"{document['status']}: {len(routines)} routines -> {out}")
    for routine in routines:
        total = routine.get("total_price_usd")
        price = f"${total:.2f}" if total is not None else "n/a"
        print(f"  - {routine['archetype']}: {routine['slot_count']} steps, {price}")

    skipped = skipped_signal_warnings(warnings)
    if skipped:
        print(
            f"error: {len(skipped)} signal store(s) were skipped, so the ranking "
            f"scored blind on a neutral 0.5 for every store-backed signal. "
            f"data_versions.signals: {[s['name'] for s in document['data_versions']['signals']]}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
