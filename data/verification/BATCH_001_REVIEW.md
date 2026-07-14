# Catalog verification batch 001

Status: **approved by agent `codex-agent` at 2026-07-14T17:07:00.729296Z**

This batch was researched and prepared by Codex on 2026-07-14. The production
overlay intentionally retains `status: "proposed"` as the immutable research
input. The separately signed approved overlay grants catalog eligibility.

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| Cleanser | Paula's Choice RESIST Perfectly Balanced Foaming Cleanser | `P469520` | Official page identifies a daily face foaming cleanser, rinse-off directions, and AM/PM use. |
| Moisturizer | Paula's Choice CLEAR Oil-Free Moisturizer | `P469517` | Official page identifies a sheer lotion, face use, AM/PM use, and states that it will not clog pores or cause breakouts. |
| Sunscreen | Supergoop! Mineral Mattescreen SPF 40 | `P476733` | Official product page identifies non-comedogenic face sunscreen, SPF 40, broad-spectrum Drug Facts directions, and leave-on use. |
| Treatment | Clinique Acne Solutions All-Over Clearing Treatment | `P188306` | Clinique verifies face treatment identity; current DailyMed SPL verifies human OTC lotion, benzoyl peroxide 2.5%, NDC `49527-117`, directions, and label version 3. |

Every assertion records the authoritative URL, retrieval timestamp, SHA-256
content hash, and only the facts supported by that source. The Clinique product
uses separate non-overlapping manufacturer and DailyMed assertions.

## Rejected during research

Paula's Choice CLEAR Regular Strength Daily Skin Clearing Treatment 2.5%
(`P469515`) is the 2.25 oz / 67 mL full-size SKU. Paula's Choice lists that
full size as discontinued in June 2026 while explicitly noting that the travel
size remains available. The live product page, recent reviews, and remaining
travel-size availability therefore do not verify this catalog SKU as currently
available. A different brand's 2.5% DailyMed label was not substituted for that
SKU; the separately cataloged Clinique product was verified against its own
manufacturer page and regulatory label.

Paula's Choice CLEAR Pore Normalizing Cleanser (`P469513`) was replaced in this
batch because it carries salicylic acid. The recommender correctly prevents a
treatment active from entering through a support-product role.

## Non-production preview result

An ephemeral approved copy was imported into the full 1,634-product primary
catalog solely to test the pending assertions. It was not written to the
production catalog.

- All four proposed products had zero role-quarantine reasons.
- No verification product IDs were unmatched.
- Completeness increased to one eligible cleanser, moisturizer, sunscreen, and
  benzoyl-peroxide 2.5% treatment.
- A recommendation preview selected exactly these four products with no missing
  roles and no regimen validation errors.
- The release catalog remains incomplete: each support role still needs 24 more
  products, and the azelaic-acid and adapalene/benzoyl-peroxide paths remain
  unfilled.

The preview also exposed and fixed two importer/eligibility defects: current
DailyMed SPL document-level human-OTC metadata and `ACTIB` ingredient structure
are now supported, and benign repeated support ingredients such as glycerin no
longer count as duplicate therapy.

## Approval checklist

For every assertion in `catalog-verification-batch-001.json`, confirm:

1. The source page is the same brand, product, strength, and variant as the
   catalog product ID.
2. Each fact is explicitly supported by the source; no search snippet or product
   name inference is being used as proof.
3. The DailyMed SET ID, NDC product code, strength, directions, effective date,
   and current-label status match the Clinique treatment.
4. The source URL, retrieval timestamp, and SHA-256 are present.
5. The reviewer attaches their identity, reviewer type, and approval time.

Batch 001 passed these checks and was signed as a separate approved overlay.
The full primary catalog was re-imported with that overlay. All four products
have zero role-quarantine reasons, and completeness now recognizes one eligible
cleanser, moisturizer, sunscreen, and benzoyl-peroxide 2.5% treatment. The
catalog is still below the release inventory target: each support role needs 24
more products, and the azelaic-acid and adapalene/benzoyl-peroxide paths remain
unfilled.
