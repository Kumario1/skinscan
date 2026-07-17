"""Catalog verification loop orchestrator: one resumable command.

Owns the per-product state machine the batch-001 process ran by hand:

    (candidate pool) -> researching -> proposed -> approved -> eligible
                             |             |                      |
                             v             v                      v
                          rejected     researching           quarantined
                                      (failed ingest)       refresh_due (stale)

The manifest (data/verification/loop_manifest.json) records every product the
loop has touched. Products never touched are the implicit "queued" pool: every
catalog row whose target role is still quarantined, plus DailyMed discoveries.

The loop never fabricates evidence and never approves anything itself:
research and review stay with an identified human or agent (D-032). Each
`run` advances every mechanical step and then prints exactly which manual
step blocks the next transition.

Typical cycle:

    python -m src.recommendation.verification_loop run
    # -> research the printed brief, write batches/NNN/proposed.json,
    #    save evidence bytes to data/verification/evidence/<sha256>
    python -m src.recommendation.verification_loop run          # ingests
    # -> review, write batches/NNN/REVIEW.md
    python -m src.recommendation.verification_loop approve \
        --batch NNN --reviewer-id you --reviewer-type agent
    python -m src.recommendation.verification_loop run          # rebuild+measure
    # repeat until `status` exits 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .import_catalog import (
    Product,
    apply_verification_overlay,
    build_completeness_report,
    build_quarantine_report,
    import_beautyapi,
    import_csv,
    load_catalog,
    load_verification_overlay,
    product_dict,
)
from .approve_verification import approve_batch

SUPPORT_ROLES = ("cleanser", "moisturizer", "sunscreen")
# ponytail: third copy of the modeled-path table (import_dailymed.MODELED_STRENGTHS,
# build_completeness_report.modeled); consolidate in schema.py if a fourth appears.
PATH_SPECS = {
    # azelaic_acid_10 dropped 2026-07-16: no drug product exists at 10% (Rx is
    # 15/20%), so the path could never be filled and blocked coverage forever
    "benzoyl_peroxide_2_5": (("benzoyl_peroxide", "2.5%"),),
    "adapalene_0_1_benzoyl_peroxide_2_5": (
        ("adapalene", "0.1%"), ("benzoyl_peroxide", "2.5%"),
    ),
}
FRESHNESS_DAYS = {"regulatory_label": 180, "default": 90}
ACTIVE_STATES = {"researching", "proposed", "approved"}

REASON_HINTS = {
    "intended_area_not_face": (
        "facts.intended_areas names a non-face area; if the product really is "
        'not for the face, reject it -- never assert "face" without a source'
    ),
    "routine_role_not_verified": "facts.routine_roles must include the target role",
    "format_unknown": "facts.format (e.g. gel, lotion, cream, cleanser)",
    "exposure_unknown": 'facts.exposure ("leave_on" or "rinse_off")',
    "non_daily_format": "mask/scrub/peel cannot fill a daily role - likely reject",
    "instruction_cadence_unknown": "facts.cadence plus facts.cadence_source",
    "instruction_cadence_source_missing": "facts.cadence_source (URL stating the cadence)",
    "drug_active_not_verified": "facts.drug_actives [{name, strength, source}]",
    "drug_active_strength_missing": "every facts.drug_actives entry needs strength",
    "drug_active_source_missing": "every facts.drug_actives entry needs source",
    "label_source_missing": "facts.label_source (authoritative label URL)",
    "label_verification_timestamp_missing": "facts.label_verified_at",
    "broad_spectrum_not_verified": "facts.broad_spectrum true per Drug Facts label",
    "spf_below_30_or_unknown": "facts.spf (integer >= 30) per Drug Facts label",
    "noncomedogenic_claim_not_verified":
        'facts.comedogenic_claim "claimed_noncomedogenic" only if the source claims it',
}


# --- paths ------------------------------------------------------------------

class Paths:
    def __init__(self, root: Path):
        self.root = root
        self.verification = root / "data" / "verification"
        self.manifest = self.verification / "loop_manifest.json"
        self.batches = self.verification / "batches"
        self.evidence = self.verification / "evidence"
        self.audits = self.verification / "audits"
        self.dailymed_pool = self.verification / "dailymed-pool.json"
        self.completeness = self.verification / "catalog_completeness.json"
        self.processed = root / "data" / "processed"
        self.catalog = self.processed / "catalog.json"
        self.catalog_tier2 = self.processed / "catalog_tier2.json"
        self.catalog_drug = self.processed / "catalog_drug.json"
        self.quarantine = self.processed / "catalog_quarantine.json"
        self.quarantine_tier2 = self.processed / "catalog_tier2_quarantine.json"
        self.quarantine_drug = self.processed / "catalog_drug_quarantine.json"
        self.review_stats = self.processed / "review_stats.json"

    def approved_overlays(self) -> list[Path]:
        legacy = sorted(self.verification.glob("catalog-verification-batch-*-approved.json"))
        batch = sorted(self.batches.glob("*/approved.json"))
        return legacy + batch

    def batch_dir(self, batch: str) -> Path:
        return self.batches / batch


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# --- manifest ---------------------------------------------------------------

def load_manifest(paths: Paths) -> dict:
    if paths.manifest.exists():
        return read_json(paths.manifest)
    return {"schema_version": "1", "batch_seq": 1, "products": {}, "batches": {},
            "audits": [], "last_rebuild": None}


def save_manifest(paths: Paths, manifest: dict) -> None:
    write_json(paths.manifest, manifest)


def set_state(manifest: dict, product_id: str, state: str, **fields) -> None:
    entry = manifest["products"].setdefault(product_id, {})
    entry.update(fields)
    entry["state"] = state
    entry["updated_at"] = now_iso()


# --- shared loaders ---------------------------------------------------------

def load_all_products(paths: Paths) -> list[Product]:
    products: list[Product] = []
    for path in (paths.catalog, paths.catalog_tier2, paths.catalog_drug):
        if path.exists():
            products.extend(load_catalog(path))
    return products


def merged_overlay_value(paths: Paths) -> dict:
    """Concatenate every approved overlay into one schema-v2 overlay document."""
    merged: dict[str, list] = {}
    for overlay_path in paths.approved_overlays():
        value = read_json(overlay_path)
        for row in value.get("products", []):
            merged.setdefault(row["product_id"], []).extend(row["assertions"])
    return {"schema_version": "2", "products": [
        {"product_id": pid, "assertions": assertions}
        for pid, assertions in sorted(merged.items())
    ]}


def iter_approved_assertions(paths: Paths):
    """Yield (product_id, assertion) for every approved assertion on disk."""
    for row in merged_overlay_value(paths)["products"]:
        for assertion in row["assertions"]:
            if assertion.get("status") == "approved":
                yield row["product_id"], assertion


def prior_fact_keys(paths: Paths, product_id: str) -> list[str]:
    """Fact keys ever asserted for a product, any status - what a re-research must cover."""
    keys: set[str] = set()
    for row in merged_overlay_value(paths)["products"]:
        if row["product_id"] == product_id:
            for assertion in row["assertions"]:
                keys.update(assertion.get("facts") or {})
    return sorted(keys)


def mark_assertions_stale(paths: Paths, product_id: str) -> int:
    """Flip a product's approved assertions to status "stale" in every overlay file.

    Stale evidence stops granting eligibility (load_verification_overlay skips
    it) and stops colliding with the re-researched batch's fresh assertions."""
    flipped = 0
    for overlay_path in paths.approved_overlays():
        value = read_json(overlay_path)
        changed = False
        for row in value.get("products", []):
            if row["product_id"] != product_id:
                continue
            for assertion in row["assertions"]:
                if assertion.get("status") == "approved":
                    assertion["status"] = "stale"
                    changed = True
                    flipped += 1
        if changed:
            write_json(overlay_path, value)
    return flipped


def evidence_issues(paths: Paths, assertion: dict, *, now: datetime | None = None) -> list[str]:
    """Freshness + snapshot problems for one approved assertion. Empty = healthy."""
    issues = []
    digest = str(assertion.get("source_sha256", "")).lower()
    snapshot = paths.evidence / digest
    if not snapshot.exists():
        issues.append("snapshot_missing")
    elif hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
        issues.append("snapshot_hash_mismatch")
    grade = (assertion.get("facts") or {}).get("evidence_grade")
    window = FRESHNESS_DAYS.get(str(grade), FRESHNESS_DAYS["default"])
    age = ((now or datetime.now(timezone.utc)) - parse_iso(assertion["retrieved_at"])).days
    if age > window:
        issues.append(f"evidence_stale_{age}d_window_{window}d")
    return issues


# --- status -----------------------------------------------------------------

def stopping_criteria(paths: Paths, manifest: dict) -> list[tuple[str, bool, str]]:
    criteria: list[tuple[str, bool, str]] = []

    if paths.completeness.exists():
        report = read_json(paths.completeness)
        detail = (f"eligible {report['eligible_by_role']}, "
                  f"missing paths {report['missing_treatment_paths']}")
        criteria.append(("coverage_complete", bool(report["complete"]), detail))
    else:
        criteria.append(("coverage_complete", False, "no completeness report - run rebuild"))

    rebuild = manifest.get("last_rebuild") or {}
    unmatched = rebuild.get("unmatched", None)
    criteria.append(("no_unmatched_approved_ids",
                     unmatched == [],
                     "never rebuilt" if unmatched is None else f"unmatched: {unmatched}"))

    stuck = {pid: e["state"] for pid, e in manifest["products"].items()
             if e["state"] in {"researching", "proposed", "approved",
                               "quarantined", "refresh_due"}}
    criteria.append(("no_products_in_flight", not stuck, f"{len(stuck)} pending: "
                     + ", ".join(f"{p}={s}" for p, s in sorted(stuck.items())[:6])
                     if stuck else "all terminal"))

    problems = {}
    latest_approval = None
    for product_id, assertion in iter_approved_assertions(paths):
        approved_at = parse_iso(assertion["approved_at"])
        latest_approval = max(latest_approval or approved_at, approved_at)
        issues = evidence_issues(paths, assertion)
        if issues:
            problems.setdefault(product_id, []).extend(issues)
    fresh = {p: i for p, i in problems.items() if any("stale" in x for x in i)}
    snaps = {p: i for p, i in problems.items() if any("snapshot" in x for x in i)}
    criteria.append(("evidence_snapshots_present", not snaps,
                     f"{len(snaps)} products missing snapshots" if snaps else "all snapshotted"))
    criteria.append(("evidence_fresh", not fresh,
                     f"{len(fresh)} products stale" if fresh else "all within window"))

    audits = [a for a in manifest["audits"] if a.get("result") == "pass"]
    last_audit = max((parse_iso(a["at"]) for a in audits), default=None)
    audits_current = (latest_approval is None
                      or (last_audit is not None and last_audit >= latest_approval))
    criteria.append(("audit_after_latest_approval", audits_current,
                     f"last pass {last_audit}, last approval {latest_approval}"))
    return criteria


def cmd_status(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    counts: dict[str, int] = {}
    for entry in manifest["products"].values():
        counts[entry["state"]] = counts.get(entry["state"], 0) + 1
    print(f"manifest: {len(manifest['products'])} products "
          + json.dumps(counts, sort_keys=True))
    criteria = stopping_criteria(paths, manifest)
    for name, passed, detail in criteria:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    done = all(passed for _, passed, _ in criteria)
    print("loop complete" if done else "loop incomplete - run "
          "`python -m src.recommendation.verification_loop run` to advance")
    return 0 if done else 1


# --- select -----------------------------------------------------------------

BRIEF_HEADER = """# Research brief - verification batch {batch}

Rules (fail closed - see data/verification/README.md and D-032):
- Sources must be the manufacturer's own product page or a regulatory label
  (DailyMed SPL). HTTPS only. Never retailer listings, search snippets, or
  name-based inference.
- Match the exact brand, product, size, strength, and variant of the catalog
  row. Discontinued or mismatched variant => reject the product
  (`verification_loop reject --batch {batch} --product <ID> --reason ...`).
- For every source: save the exact retrieved bytes to
  `data/verification/evidence/<sha256-of-bytes>` and record `source_url`,
  `retrieved_at` (UTC ISO-8601), `source_sha256`.
- Assert only facts the source explicitly states, in
  `data/verification/batches/{batch}/proposed.json` (schema below). Facts may
  not repeat across a product's assertions.
- When done run: `python -m src.recommendation.verification_loop ingest --batch {batch}`

```json
{{"schema_version": "2", "products": [
  {{"product_id": "<ID>", "assertions": [
    {{"status": "proposed", "source_url": "https://...",
      "retrieved_at": "2026-01-01T00:00:00Z", "source_sha256": "<64 hex>",
      "facts": {{"routine_roles": ["cleanser"], "...": "..."}}}}]}}]}}
```
"""


def candidate_rank(paths: Paths):
    loves = {}
    if paths.review_stats.exists():
        loves = read_json(paths.review_stats).get("loves", {})
    return lambda product: (-loves.get(product.product_id, 0), product.product_id)


def cmd_select(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    if not paths.completeness.exists() or not paths.quarantine.exists():
        print("error: no completeness/quarantine report yet - run rebuild first")
        return 2
    completeness = read_json(paths.completeness)
    quarantine = {}
    for path in (paths.quarantine, paths.quarantine_tier2, paths.quarantine_drug):
        if path.exists():
            quarantine.update(read_json(path)["products"])
    products = {p.product_id: p for p in load_all_products(paths)}
    rank = candidate_rank(paths)
    # rejected products need a manual requeue (edit the manifest) to re-enter
    taken = {pid for pid, e in manifest["products"].items()
             if e["state"] in ACTIVE_STATES or e["state"] == "rejected"}

    picks: list[tuple[str, str, list[str]]] = []  # (product_id, target, reasons)

    def quarantined_for(product_id: str, role: str) -> list[str]:
        return quarantine.get(product_id, {}).get("quarantined_roles", {}).get(role, [])

    def pick(product_id: str, target: str) -> None:
        role = target if target in SUPPORT_ROLES + ("serum",) else "treatment"
        picks.append((product_id, target, quarantined_for(product_id, role)))
        taken.add(product_id)

    # 1. re-research stale evidence first
    for product_id, entry in sorted(manifest["products"].items()):
        if entry["state"] == "refresh_due" and len(picks) < args.batch_size:
            pick(product_id, entry.get("target", "unknown"))

    # 2. one candidate per missing treatment path
    for path_key in completeness.get("missing_treatment_paths", []):
        if len(picks) >= args.batch_size:
            break
        wanted = {name for name, _ in PATH_SPECS[path_key]}
        pool = [p for p in products.values()
                if p.product_id not in taken and p.category in ("treatment", "serum")
                and (wanted <= set(p.actives)
                     or {(a.name, a.strength) for a in p.drug_actives} == set(PATH_SPECS[path_key]))]
        if not pool:
            print(f"no catalog candidate for treatment path {path_key}: ingest a base "
                  "row first, e.g. `verification_loop discover --set-id <DailyMed SET ID>`")
            continue
        for candidate in sorted(pool, key=rank)[:1]:
            pick(candidate.product_id, path_key)

    # 3. support roles, largest shortfall first, round-robin
    shortfalls = sorted(completeness.get("shortfalls", {}).items(),
                        key=lambda item: -item[1])
    pools = {
        role: iter(sorted(
            (p for p in products.values()
             if p.product_id not in taken and quarantined_for(p.product_id, role)
             and (role in p.routine_roles
                  or p.category == {"sunscreen": "spf"}.get(role, role))),
            key=rank))
        for role, _ in shortfalls
    }
    while len(picks) < args.batch_size and pools:
        for role, _ in list(shortfalls):
            if len(picks) >= args.batch_size or role not in pools:
                continue
            candidate = next((c for c in pools[role] if c.product_id not in taken), None)
            if candidate is None:
                del pools[role]
                continue
            pick(candidate.product_id, role)
        if not any(True for _ in pools):
            break

    if not picks:
        print("nothing to select: no shortfalls, or candidate pool exhausted")
        return 0

    batch = f"{manifest['batch_seq'] + 1:03d}"
    manifest["batch_seq"] += 1
    batch_dir = paths.batch_dir(batch)
    batch_dir.mkdir(parents=True, exist_ok=True)

    lines = [BRIEF_HEADER.format(batch=batch)]
    for product_id, target, reasons in picks:
        product = products.get(product_id)
        name = f"{product.name} ({product.brand})" if product else "(dailymed pool)"
        prior_state = manifest["products"].get(product_id, {}).get("state")
        set_state(manifest, product_id, "researching", batch=batch, target=target,
                  reasons=reasons, name=getattr(product, "name", product_id))
        lines.append(f"\n## {product_id} - {name} -> {target}\n")
        if prior_state == "refresh_due":
            keys = ", ".join(prior_fact_keys(paths, product_id))
            lines.append("- Re-verification: prior evidence went stale or lacks a "
                         "snapshot. Re-fetch every source and re-assert these "
                         f"facts fresh: {keys}\n")
        if target in PATH_SPECS:
            spec = ", ".join(f"{n} {s}" for n, s in PATH_SPECS[target])
            # DailyMed only, not "or the manufacturer's own page": recsys's drug
            # door (recsys/catalog.py LABEL_PREFIX) refuses any per-active source
            # that is not a DailyMed label, so a batch researched against a
            # manufacturer page imports here and then fails there. The brief
            # states the stricter contract both engines can honour.
            lines.append(f"- Treatment path target: exactly [{spec}] verified via a "
                         "current DailyMed SPL label; label_source and every "
                         "facts.drug_actives[].source must cite the DailyMed label "
                         "(https://dailymed.nlm.nih.gov/...) -- a manufacturer page "
                         "does not qualify for drug rows "
                         "(D-033: OTC status recorded but not required).\n")
        for reason in reasons:
            lines.append(f"- {reason}: {REASON_HINTS.get(reason, 'resolve from source')}\n")
    brief = batch_dir / "RESEARCH_BRIEF.md"
    brief.write_text("".join(lines), encoding="utf-8")
    manifest["batches"][batch] = {"created_at": now_iso(), "brief": str(brief.relative_to(paths.root))}
    save_manifest(paths, manifest)
    print(f"batch {batch}: {len(picks)} products selected -> {brief}")
    return 0


# --- discover (new base rows: DailyMed) --------------------------------------

def cmd_discover(paths: Paths, args) -> int:
    """Ingest new base product rows from DailyMed so overlays have an ID to enrich."""
    out = paths.verification / f".dailymed-fetch-{os.getpid()}.json"
    command = [sys.executable, "-m", "src.recommendation.import_dailymed", "--out", str(out)]
    for set_id in args.set_id:
        command += ["--set-id", set_id]
    result = subprocess.run(command, cwd=paths.root)
    if result.returncode != 0 or not out.exists():
        print("error: import_dailymed failed")
        return 2
    fetched = read_json(out)
    out.unlink()
    pool = read_json(paths.dailymed_pool) if paths.dailymed_pool.exists() else []
    known = {row["product_id"] for row in pool}
    added = [row for row in fetched if row["product_id"] not in known]
    if added:
        write_json(paths.dailymed_pool, pool + added)
    print(f"dailymed: fetched {len(fetched)}, added {len(added)} new base rows "
          f"(pool {len(known) + len(added)}); they join the catalog on next rebuild")
    if not fetched:
        print("warning: zero candidates survived the SPL gates - check the SET IDs")
    return 0


# --- ingest -----------------------------------------------------------------

def validate_proposed_assertion(paths: Paths, assertion: dict) -> list[str]:
    errors = []
    if assertion.get("status") != "proposed":
        errors.append("status_not_proposed")
    url = assertion.get("source_url", "")
    if not isinstance(url, str) or not url.startswith("https://"):
        errors.append("source_url_not_https")
    try:
        parse_iso(assertion["retrieved_at"])
    except Exception:
        errors.append("retrieved_at_not_iso8601")
    digest = str(assertion.get("source_sha256", "")).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        errors.append("source_sha256_not_hex_digest")
    else:
        snapshot = paths.evidence / digest
        if not snapshot.exists():
            errors.append(f"snapshot_missing:{digest[:12]}")
        elif hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            errors.append(f"snapshot_hash_mismatch:{digest[:12]}")
    facts = assertion.get("facts")
    if not isinstance(facts, dict) or not facts:
        errors.append("facts_empty")
    return errors


def cmd_ingest(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    batch_dir = paths.batch_dir(args.batch)
    proposed_path = batch_dir / "proposed.json"
    if not proposed_path.exists():
        print(f"error: {proposed_path} not found - research is not done")
        return 2
    value = read_json(proposed_path)
    members = {pid for pid, e in manifest["products"].items() if e.get("batch") == args.batch}
    prior_facts: dict[str, set] = {}
    for product_id, assertion in iter_approved_assertions(paths):
        prior_facts.setdefault(product_id, set()).update(assertion["facts"])

    failures: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in value.get("products", []):
        product_id = row.get("product_id", "?")
        seen.add(product_id)
        errors = []
        if product_id not in members:
            errors.append("not_a_member_of_this_batch")
        fact_keys: list[str] = []
        for assertion in row.get("assertions", []):
            errors += validate_proposed_assertion(paths, assertion)
            fact_keys += list(assertion.get("facts", {}) or {})
        if len(fact_keys) != len(set(fact_keys)):
            errors.append("facts_overlap_between_assertions")
        overlap = set(fact_keys) & prior_facts.get(product_id, set())
        if overlap:
            errors.append(f"facts_already_approved_in_earlier_batch:{sorted(overlap)}")
        if not row.get("assertions"):
            errors.append("no_assertions")
        if errors:
            failures[product_id] = errors
            if product_id in members:
                set_state(manifest, product_id, "researching", reasons=errors)
        else:
            set_state(manifest, product_id, "proposed", reasons=[])

    if str(value.get("schema_version")) != "2":
        print("error: proposed.json schema_version must be '2'")
        return 2
    missing = sorted(pid for pid in members
                     if pid not in seen and manifest["products"][pid]["state"] == "researching")
    save_manifest(paths, manifest)
    for product_id, errors in sorted(failures.items()):
        print(f"FAIL {product_id}: {'; '.join(errors)}")
    if missing:
        print(f"still researching (not in proposed.json): {', '.join(missing)}")
    proposed = [pid for pid in seen if pid not in failures]
    print(f"ingest batch {args.batch}: {len(proposed)} proposed, "
          f"{len(failures)} failed, {len(missing)} outstanding")
    if proposed and not failures and not missing:
        print(f"next: review the evidence, write {batch_dir / 'REVIEW.md'}, then run approve")
    return 0 if not failures else 1


def cmd_reject(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    entry = manifest["products"].get(args.product)
    if not entry or entry.get("batch") != args.batch:
        print(f"error: {args.product} is not part of batch {args.batch}")
        return 2
    set_state(manifest, args.product, "rejected", reasons=[args.reason])
    save_manifest(paths, manifest)
    print(f"rejected {args.product}: {args.reason} "
          f"(remove it from proposed.json before approving)")
    return 0


# --- approve ----------------------------------------------------------------

def cmd_approve(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    batch_dir = paths.batch_dir(args.batch)
    review = batch_dir / "REVIEW.md"
    if not review.exists() or not review.read_text(encoding="utf-8").strip():
        print(f"error: {review} missing or empty - the reviewer must record the "
              "review before signing (see BATCH_001_REVIEW.md for the shape)")
        return 2
    members = {pid: e for pid, e in manifest["products"].items()
               if e.get("batch") == args.batch and e["state"] != "rejected"}
    not_proposed = sorted(pid for pid, e in members.items() if e["state"] != "proposed")
    if not members or not_proposed:
        print(f"error: batch not fully ingested - non-proposed members: {not_proposed}")
        return 2
    proposed_path = batch_dir / "proposed.json"
    file_ids = {row["product_id"] for row in read_json(proposed_path)["products"]}
    if file_ids != set(members):
        print(f"error: proposed.json products {sorted(file_ids)} != "
              f"batch members {sorted(members)} (remove rejected products from the file)")
        return 2
    result = approve_batch(
        proposed_path, batch_dir / "approved.json",
        reviewer_id=args.reviewer_id, reviewer_type=args.reviewer_type,
        approved_at=now_iso(), acknowledged_review=args.acknowledge_reviewed,
    )
    for product_id in members:
        set_state(manifest, product_id, "approved")
    manifest["batches"].setdefault(args.batch, {}).update(
        approved_file=str((batch_dir / "approved.json").relative_to(paths.root)),
        review_file=str(review.relative_to(paths.root)),
        approved_at=result["approved_at"], reviewer_id=result["reviewer_id"],
    )
    save_manifest(paths, manifest)
    print(json.dumps(result, sort_keys=True))
    print("next: run rebuild to import and measure")
    return 0


# --- rebuild ----------------------------------------------------------------

def cmd_rebuild(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    combined_path = paths.verification / "approved-combined.json"
    overlay_value = merged_overlay_value(paths)
    write_json(combined_path, overlay_value)
    overlay = load_verification_overlay(combined_path)  # validates the merge

    logs = {}
    unmatched_sets = []
    if Path(paths.root / args.sephora_csv).exists():
        logs["tier1"] = import_csv(
            paths.root / args.sephora_csv, paths.catalog, fmt=args.sephora_format,
            verification=combined_path, quarantine_out=paths.quarantine)
        unmatched_sets.append(set(read_json(paths.quarantine)["unmatched_verification_ids"]))
    if Path(paths.root / args.beautyapi_jsonl).exists():
        logs["tier2"] = import_beautyapi(
            paths.root / args.beautyapi_jsonl, paths.catalog_tier2,
            verification=combined_path, quarantine_out=paths.quarantine_tier2)
        unmatched_sets.append(set(read_json(paths.quarantine_tier2)["unmatched_verification_ids"]))
    if paths.dailymed_pool.exists():
        rows = [Product.from_dict(row) for row in read_json(paths.dailymed_pool)]
        rows, unmatched = apply_verification_overlay(rows, overlay)
        write_json(paths.catalog_drug, [product_dict(p) for p in rows])
        write_json(paths.quarantine_drug, build_quarantine_report(rows, unmatched))
        logs["drug"] = {"kept": len(rows)}
        unmatched_sets.append(set(unmatched))

    if not logs:
        print("error: no catalog sources found (raw data missing)")
        return 2

    products = load_all_products(paths)
    truly_unmatched = sorted(set.intersection(*unmatched_sets)) if unmatched_sets else []
    completeness = build_completeness_report(products, support_minimum=args.support_minimum)
    write_json(paths.completeness, completeness)
    write_json(paths.processed / "catalog_completeness.json", completeness)

    union_quarantine = build_quarantine_report(products, truly_unmatched)["products"]

    def outcome(product_id: str, target: str) -> list[str]:
        role = target if target in SUPPORT_ROLES + ("serum",) else "treatment"
        reasons = union_quarantine.get(product_id, {}).get(
            "quarantined_roles", {}).get(role, [])
        if product_id in truly_unmatched:
            reasons = ["approved_id_unmatched_in_all_catalogs"] + reasons
        return reasons

    for product_id, entry in manifest["products"].items():
        if entry["state"] not in {"approved", "eligible", "quarantined"}:
            continue
        reasons = outcome(product_id, entry.get("target", ""))
        set_state(manifest, product_id, "quarantined" if reasons else "eligible",
                  reasons=reasons)
    # adopt legacy overlay products (batch 001 predates the manifest)
    for product_id, patch in overlay.items():
        if product_id not in manifest["products"]:
            roles = patch.get("routine_roles") or ["treatment"]
            target = roles[0] if roles[0] in SUPPORT_ROLES + ("serum",) else "treatment"
            reasons = outcome(product_id, target)
            set_state(manifest, product_id, "quarantined" if reasons else "eligible",
                      batch="legacy", target=target, reasons=reasons)
    manifest["last_rebuild"] = {
        "at": now_iso(), "unmatched": truly_unmatched,
        "complete": completeness["complete"],
        "eligible_by_role": completeness["eligible_by_role"],
        "missing_treatment_paths": completeness["missing_treatment_paths"],
        "logs": {k: {x: v[x] for x in v if x in ("rows", "kept")} for k, v in logs.items()},
    }
    save_manifest(paths, manifest)
    print(f"rebuild: sources={sorted(logs)}, unmatched={truly_unmatched}, "
          f"complete={completeness['complete']}")
    return 0


# --- refresh / audit ---------------------------------------------------------

def cmd_refresh(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    flagged: dict[str, list[str]] = {}
    for product_id, assertion in iter_approved_assertions(paths):
        issues = evidence_issues(paths, assertion)
        if issues:
            flagged.setdefault(product_id, []).extend(issues)
    changed = 0
    for product_id, issues in sorted(flagged.items()):
        entry = manifest["products"].get(product_id, {})
        if entry.get("state") in {"eligible", "quarantined", "refresh_due"}:
            if not args.dry_run:
                mark_assertions_stale(paths, product_id)
                set_state(manifest, product_id, "refresh_due", reasons=sorted(set(issues)))
            changed += 1
            print(f"refresh_due {product_id}: {sorted(set(issues))}")
    if not args.dry_run:
        save_manifest(paths, manifest)
    print(f"refresh: {changed} products marked stale (of {len(flagged)} with issues)"
          + (" [dry-run]" if args.dry_run else "")
          + ("; coverage reflects it after the next rebuild" if changed and not args.dry_run else ""))
    return 0


def cmd_audit(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    if args.record:
        pending = [a for a in manifest["audits"] if a.get("result") is None]
        if not pending:
            print("error: no pending audit to record")
            return 2
        pending[-1].update(result=args.record, notes=args.notes or "", at=now_iso())
        save_manifest(paths, manifest)
        print(f"audit recorded: {args.record}")
        return 0
    assertions = list(iter_approved_assertions(paths))
    if not assertions:
        print("nothing to audit: no approved assertions")
        return 0
    sample = random.Random(args.seed).sample(assertions, min(args.sample, len(assertions)))
    number = sum(1 for _ in manifest["audits"]) + 1
    brief = paths.audits / f"{number:03d}" / "AUDIT_BRIEF.md"
    brief.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Random audit {number:03d}\n\nRe-fetch each source and confirm every "
             "fact is still explicitly supported. Record with:\n"
             "`python -m src.recommendation.verification_loop audit --record pass|fail --notes ...`\n"]
    for product_id, assertion in sample:
        lines.append(f"\n## {product_id}\n- source: {assertion['source_url']}\n"
                     f"- retrieved_at: {assertion['retrieved_at']}\n"
                     f"- sha256: {assertion['source_sha256']}\n"
                     f"- facts: `{json.dumps(assertion['facts'], sort_keys=True)}`\n")
    brief.write_text("".join(lines), encoding="utf-8")
    manifest["audits"].append({"at": now_iso(), "brief": str(brief.relative_to(paths.root)),
                               "products": sorted({p for p, _ in sample}), "result": None})
    save_manifest(paths, manifest)
    print(f"audit brief ({len(sample)} assertions) -> {brief}")
    return 0


# --- run (the resumable driver) ----------------------------------------------

def cmd_run(paths: Paths, args) -> int:
    manifest = load_manifest(paths)
    if not paths.completeness.exists() or not (manifest.get("last_rebuild")):
        print("== rebuild (no measured catalog yet)")
        if cmd_rebuild(paths, args) != 0:
            return 2
        manifest = load_manifest(paths)

    print("== refresh")
    cmd_refresh(paths, argparse.Namespace(dry_run=False))
    manifest = load_manifest(paths)

    open_batches = sorted({e["batch"] for e in manifest["products"].values()
                           if e["state"] in {"researching", "proposed"} and e.get("batch")})
    blocked = []
    for batch in open_batches:
        states = {pid: e["state"] for pid, e in manifest["products"].items()
                  if e.get("batch") == batch and e["state"] in {"researching", "proposed"}}
        proposed_file = paths.batch_dir(batch) / "proposed.json"
        if "researching" in states.values() and proposed_file.exists():
            print(f"== ingest batch {batch}")
            cmd_ingest(paths, argparse.Namespace(batch=batch))
            manifest = load_manifest(paths)
            states = {pid: e["state"] for pid, e in manifest["products"].items()
                      if e.get("batch") == batch and e["state"] in {"researching", "proposed"}}
        if "researching" in states.values():
            blocked.append(f"batch {batch}: research per "
                           f"{paths.batch_dir(batch) / 'RESEARCH_BRIEF.md'}")
        elif states:
            blocked.append(
                f"batch {batch}: review evidence, write {paths.batch_dir(batch) / 'REVIEW.md'}, "
                f"then: python -m src.recommendation.verification_loop approve --batch {batch} "
                "--reviewer-id <you> --reviewer-type agent --acknowledge-reviewed")

    if any(e["state"] == "approved" for e in manifest["products"].values()):
        print("== rebuild (new approvals)")
        cmd_rebuild(paths, args)
        manifest = load_manifest(paths)

    criteria_met = all(passed for _, passed, _ in stopping_criteria(paths, manifest))
    # ponytail: one batch in flight at a time; parallel batches when throughput matters
    if not blocked and not criteria_met:
        needs_research = any(e["state"] == "refresh_due" for e in manifest["products"].values())
        report = read_json(paths.completeness)
        if needs_research or not report["complete"]:
            print("== select")
            seq_before = manifest["batch_seq"]
            if cmd_select(paths, argparse.Namespace(batch_size=args.batch_size)) == 0:
                manifest = load_manifest(paths)
                if manifest["batch_seq"] > seq_before:
                    batch = f"{manifest['batch_seq']:03d}"
                    blocked.append(f"batch {batch}: research per "
                                   f"{paths.batch_dir(batch) / 'RESEARCH_BRIEF.md'}")

    print("== status")
    code = cmd_status(paths, args)
    for line in blocked:
        print(f"NEXT: {line}")
    if not blocked and code != 0:
        print("NEXT: see FAIL lines above (audit/refresh/discover as indicated)")
    return code


# --- entry -------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    p = sub.add_parser("select")
    p.add_argument("--batch-size", type=int, default=8)
    p = sub.add_parser("discover")
    p.add_argument("--set-id", action="append", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--batch", required=True)
    p = sub.add_parser("reject")
    p.add_argument("--batch", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("approve")
    p.add_argument("--batch", required=True)
    p.add_argument("--reviewer-id", required=True)
    p.add_argument("--reviewer-type", choices=("human", "agent"), required=True)
    p.add_argument("--acknowledge-reviewed", action="store_true")
    p = sub.add_parser("refresh")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("audit")
    p.add_argument("--sample", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--record", choices=("pass", "fail"), default=None)
    p.add_argument("--notes", default=None)
    for name in ("rebuild", "run"):
        p = sub.add_parser(name)
        p.add_argument("--sephora-csv", default="data/raw/sephora/product_info.csv")
        p.add_argument("--sephora-format", default="sephora", choices=("sephora", "simple"))
        p.add_argument("--beautyapi-jsonl", default="data/raw/beautyapi/beauty_data.jsonl")
        p.add_argument("--support-minimum", type=int, default=25)
        if name == "run":
            p.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args(argv)
    paths = Paths(args.root.resolve())
    handler = {
        "status": cmd_status, "select": cmd_select, "discover": cmd_discover,
        "ingest": cmd_ingest, "reject": cmd_reject, "approve": cmd_approve,
        "rebuild": cmd_rebuild, "refresh": cmd_refresh, "audit": cmd_audit,
        "run": cmd_run,
    }[args.command]
    return handler(paths, args)


if __name__ == "__main__":
    raise SystemExit(main())
