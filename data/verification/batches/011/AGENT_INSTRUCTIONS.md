# Batch 011 research agent instructions

You verify skincare catalog rows against manufacturer sources. Fail closed.

Files:
- Brief (rules + per-product reason codes): data/verification/batches/011/RESEARCH_BRIEF.md (read the header and YOUR products' sections)
- Catalog rows (exact name/brand/size/variant to match): data/verification/batches/011/catalog_extract.json

Per product:
1. Find the manufacturer's OWN product page (never retailers, never sephora.com).
2. Fetch via Firecrawl first:
   curl -s -X POST https://api.firecrawl.dev/v2/scrape -H "Authorization: Bearer fc-f6647fb2bc644e97956595aae9237e5a" -H "Content-Type: application/json" -d '{"url":"<URL>","formats":["rawHtml","markdown"]}'
   Save the data.rawHtml bytes EXACTLY to a temp file, then:
   d=$(shasum -a 256 tmp | cut -d' ' -f1) && mv tmp data/verification/evidence/$d
   Fallback: curl -s --http1.1 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15" "<URL>" -o tmp
3. CHECK the fetched page's <title>/content actually matches the exact brand, product name, and size variant from catalog_extract.json. Old slugs can 301 to WRONG products. Wrong product / discontinued / size variant no longer sold / page unavailable => reject (record it, step 5).
4. Assert ONLY facts the page explicitly states, per the reason codes in your product's brief section:
   - routine_roles must include the target role (e.g. page usage says cleanser/moisturizer/serum/sunscreen use)
   - cadence — MUST be one of exactly: "am", "pm", "am_pm", "daily", "once_daily", "twice_daily", "per_label" (the recsys import contract rejects anything else) + cadence_source = the URL that states it. Page MUST state usage frequency.
   - comedogenic_claim: "claimed_noncomedogenic" ONLY if the page states non-comedogenic / won't clog pores. If a moisturizer-target page makes no such claim, the product cannot clear quarantine => reject with reason "no noncomedogenic claim on manufacturer page".
   - Sunscreen targets additionally need a SECOND assertion from a current DailyMed SPL (manufacturer pages rarely state Broad Spectrum): search https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name=<name> , fetch the SPL XML (https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/<SETID>.xml), save it as evidence the same sha256 way. Check <marketingAct><statusCode>: "completed" on ALL lots = discontinued => reject. Facts from the SPL: broad_spectrum true, spf (integer), label_source = the SPL URL, label_verified_at (UTC now).
   Every reason code listed for your product must be clearable by your facts; if any cannot be, reject with the specific gap.
5. Output ONE file: data/verification/batches/011/fragment_<agentname>.json
   {"products":[{"product_id":"...","assertions":[{"status":"proposed","source_url":"https://...","retrieved_at":"<UTC ISO8601 from `date -u +%Y-%m-%dT%H:%M:%SZ`>","source_sha256":"<64 hex of saved evidence file>","facts":{...}}]}],
    "rejects":[{"product_id":"...","reason":"..."}]}
   Facts may not repeat across a product's assertions (sunscreens: page assertion carries routine_roles/cadence/comedogenic_claim; SPL assertion carries broad_spectrum/spf/label_source/label_verified_at).
6. Do NOT run any verification_loop command. Do NOT edit any other file. Your final message: one line per product, verified or rejected+reason.
