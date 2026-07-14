---
name: catalog-loop
summary: Drive the catalog verification loop - research, propose, review, approve, rebuild - until release inventory criteria pass.
---

# Catalog verification loop

One cycle (repeat until `status` exits 0):

1. `python -m src.recommendation.verification_loop run` - it rebuilds, stales
   old evidence, ingests/rebuilds whatever it can, then prints one `NEXT:` line.
2. **Research** (when NEXT points at a `RESEARCH_BRIEF.md`): for each product,
   fetch only the manufacturer's own product page or a DailyMed SPL over HTTPS.
   Save the exact retrieved bytes and record their hash:
   `curl -s "$URL" -o page && d=$(shasum -a 256 page | cut -d' ' -f1) && mv page data/verification/evidence/$d`
   Confirm the source matches the exact brand, product, size, strength, and
   variant. Assert only facts the page explicitly states. Write
   `data/verification/batches/<N>/proposed.json` per the brief's schema.
   Unavailable page, discontinued SKU, or variant mismatch =>
   `verification_loop reject --batch <N> --product <ID> --reason "..."` and
   drop it from proposed.json. Never approve from a product name or snippet.
3. `verification_loop ingest --batch <N>` - fix any FAIL lines it prints.
4. **Review**: re-open every snapshot, re-check each fact against it, write
   `batches/<N>/REVIEW.md` (follow `BATCH_001_REVIEW.md`'s shape: per-product
   evidence result, rejects with reasons, checklist).
5. `verification_loop approve --batch <N> --reviewer-id <your identity> --reviewer-type agent --acknowledge-reviewed`
   (D-032: agent approval covers factual catalog evidence only, never the
   D-029 clinician gates.)
6. `verification_loop run` again - it rebuilds catalogs, updates
   eligible/quarantined states, and selects the next batch.

Also:
- Missing treatment path with no catalog candidate: find the OTC product's
  DailyMed SET ID, then `verification_loop discover --set-id <SETID>` (creates
  the base row; overlays can only enrich existing product IDs).
- When status FAILs `audit_after_latest_approval`: `verification_loop audit`,
  re-verify the sampled assertions against fresh fetches, then
  `verification_loop audit --record pass|fail --notes "..."`.
- Quarantined after rebuild means the approved facts still don't clear the
  role's reason codes - read `reasons` in `data/verification/loop_manifest.json`.
