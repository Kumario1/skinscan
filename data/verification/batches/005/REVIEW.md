# Catalog verification batch 005

Status: **approved by agent `claude-fable-5` at 2026-07-16**

Researched 2026-07-16 via Firecrawl (rendered manufacturer pages saved as
rawHtml evidence bytes; Firecrawl cleared the Clinique bot wall that blocked
curl in earlier batches). Every snapshot re-opened; hashes recomputed from the
stored bytes and matched (also enforced by ingest). All sources are the
manufacturer's own HTTPS product pages.

## Proposed products

| Role | Catalog product | Product ID | Evidence result |
|---|---|---|---|
| moisturizer | The True Cream Aqua Bomb (belif), 50 mL | `P394639` | Successor manufacturer domain states daily AM/PM face use. |
| cleanser | Take The Day Off Cleansing Balm (CLINIQUE), 125 mL | `P126301` | Page states face use, rinse-off, non-comedogenic; **no usage frequency stated → no cadence asserted** (stays quarantined on cadence). |
| cleanser | Salicylic Acid Acne + Pore Cleanser (The INKEY List), 150 mL | `P443833` | Page states AM/PM use, rinse-off, facial cleanser also usable on back/chest. |
| sunscreen | Glow Stick Sunscreen SPF 50 (Supergoop!), 0.70 oz | `P429953` | Drug-facts directions + non-comedogenic stated; **no product-level "Broad Spectrum SPF" claim or numeric SPF in Drug Facts in the fetched page → broad_spectrum/spf/label facts NOT asserted** (stays quarantined on those; follow up via DailyMed SPL). |
| sunscreen | (Re)setting Mineral Powder SPF 35 (Supergoop!), 0.15 oz | `P467976` | Same limitation as P429953; face+neck application stated; zinc oxide 24.7% shown but SPF facts deferred to a DailyMed follow-up. |

## Evidence re-check

### P394639 — `9361d83914c604958187529ae517424cff3520465fc81561a43355867d2b45d6` (843,977 bytes)
Final URL `https://lgbeauty.com/products/the-true-cream-aqua-bomb`. **Domain
note:** belifusa.com is decommissioned and 301-redirects wholesale to
lgbeauty.com — LG H&H is belif's parent and this is the brand's official DTC
site, not a retailer. Page: "After cleansing and toner, apply dime-sized or
desired amount to face in a gentle patting motion. Can be used daily, morning
and night." Size selector offers 25/50/100 mL — the catalog's 50 mL variant is
offered. Facts asserted: moisturizer role, cream format (product name "The
True Cream"), leave_on, am_pm cadence, face, daily_support evidence role.
No non-comedogenic claim on the page → not asserted.

### P126301 — `0e06eba0fc574cbd65822f743db185e67794cdaf552be170c2a8fc1d49690394` (964,437 bytes)
H1 "Take The Day Off™ Cleansing Balm". "With dry skin and dry hands, scoop a
small amount and massage it over your face... Then rinse thoroughly."
"Non-comedogenic" stated. Sizes include 3.8 oz / 125 mL (catalog variant).
The page states no application frequency, so cadence/cadence_source are
deliberately absent.

### P443833 — `4522b19ed3cb3d78d941c079ea1ddb154db1e6b6efcb6108dafceb412207219a` (2,332,217 bytes)
Title "2% Salicylic Acid Cleanser for Acne ... 150ml" (catalog variant).
"Use AM and PM. Gently massage a raspberry-sized amount onto damp skin for 60
seconds... then rinse thoroughly." "Not just a facial cleanser for oily skin -
can also be used on acne-prone areas like back and chest" → intended_areas
face + body.

### P429953 — `b4af36232ee8876652a70c15be165ec6ce69a7d7b302e604425ba3e2ce8cf1ad` (3,477,701 bytes)
H1 "Glow Stick SPF 50", 0.70 oz (catalog variant). Drug-facts directions
("Apply generously and evenly 15 minutes before sun exposure • Reapply at
least every 2 hours...") → cadence per_label. "Non-comedogenic" stated.
The fetched page contains only the FDA boilerplate sentence about broad
spectrum ("regularly use a sunscreen with broad spectrum SPF of 15 or
higher"), which is generic advice, not a product claim — so broad_spectrum,
spf, label_source, label_verified_at were NOT asserted (fail closed).

### P467976 — `92befeb7eb991b2b2815da7803360b18e291330b8f1d9556bb1f820f1acef232` (3,400,080 bytes)
H1 "(Re)setting Mineral Powder SPF 35". "Brush (Re)setting Powder generously
and evenly across your face and neck" → face + neck. "Non-comedogenic" stated;
active zinc oxide 24.7%. Single-size product (0.15 oz catalog row); the page
prints no net weight in the rendered content — identity matched on the unique
product name. Same broad-spectrum/spf fail-closed treatment as P429953.

## Rejected during research

- **P429242 — Clear Sunscreen Stick SPF 50+ (Shiseido). Substantive reject.**
  Page states: "This product is being discontinued and will no longer be
  available once it sells out" (recommends Ultimate Sun Protector Clear Stick
  SPF 60+). The loop drops discontinued SKUs.
- **P427415 — 100% Organic Cold-Pressed Rose Hip Seed Oil (The Ordinary).
  Substantive reject.** Page presents a face oil ("Apply a few drops to the
  face once daily, ideally at bedtime"), states no moisturizer role and no
  non-comedogenic claim; oil cannot clear the moisturizer quarantine. Requeue
  if claim standards change.

## Checklist

- [x] Every source is the manufacturer's own page over HTTPS (lgbeauty.com
      accepted as belif's official successor domain, documented above)
- [x] Brand, product, strength, and size variant match the catalog row (or rejected)
- [x] Evidence bytes snapshotted under their sha256; hashes re-verified
- [x] Every asserted fact restates something the page explicitly says;
      unstated facts (cadence for P126301, SPF facts for both Supergoop rows,
      non-comedogenic for P394639) deliberately omitted
- [x] Rejected products removed from proposed.json
