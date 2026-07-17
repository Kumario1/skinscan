# CONTEXT — SkinScan glossary

The shared vocabulary every issue, doc, and module uses. Definitions only —
mechanisms and rationale live in `docs/DECISIONS.md` and the schema docs, not
here.

## Terms

**concern** — A deprecated grouped cosmetic display summary, never a medical
condition (D-002). Closed vocabulary: `acne_comedonal`, `acne_inflammatory`,
`acne_cystic`, `acne_scarring`, `hyperpigmentation`, `dryness`
(`docs/CONCERN_SCHEMA.md`). Schema 4 keeps it for one compatibility release,
but care and product logic may not read it (D-038).

**lesion finding** — One schema-4 entry for an exact detector label: count,
regions, mean/max detector confidence, and evidence source. The closed ten-label
vocabulary is `closed_comedo`, `open_comedo`, `papule`, `pustule`, `nodule`,
`atrophic_scar`, `hypertrophic_scar`, `melasma`, `nevus`, and `other`.
Confidence describes detector evidence quality, never diagnosis probability.

**care pathway** — The product-independent policy result for one exact lesion
finding. It carries retail active specifications and roles, separately
channelled clinician options, unknown intake, reason/source IDs, and one of
`not_detected`, `retail_eligible`, `clinician_only`, `deferred`,
`monitoring_only`, or `unsupported` (D-038).

**region** — A named area of the face a concern is localized to. Closed
vocabulary: `forehead`, `nose`, `left_cheek`, `right_cheek`, `chin_jaw`,
`perioral` (D-008 / D-020).

**active** — A canonical skincare ingredient the rules reason over (e.g.
`salicylic_acid`, `niacinamide`, `retinol`). A carried active is a safety fact,
not proof that products are therapeutically interchangeable; role, exposure,
strength, and source must also match (D-006/D-029). The closed list lives in
`docs/CATALOG_SCHEMA.md`.

**ConcernReport** — The legacy grouped contract between the CV side and the
decision side: one aggregated entry per concern with a `regions` list,
severity, lesion count, raw score aggregate, and evidence, plus clear-skin and
low-light meta flags (D-008, `docs/CONCERN_SCHEMA.md`). Schema 4 builds exact
lesion findings directly from retained detections; ConcernReport remains a
display/compatibility projection.

**bridge** — The step that turns raw per-lesion model output into a ConcernReport;
the join from the CV side to the rules side.

**tile pipeline** — The v2 identification funnel (D-026): the photo is chunked
into native-resolution 1024px tiles with guaranteed overlap, every tile goes
through the SA-RPN detector, and seam duplicates are removed before the bridge.
Its rejected alternative, the **zoom pipeline** (YOLO areas upscaled to 1024px),
stays defined only as the dead end it is (README §8).

**profile** — The exact, normalized safety intake used for an inference
(D-021/D-029): unknown-capable skin/tone/pregnancy, age, allergies,
sensitivity conditions, current actives/medications, treatment history,
duration, pain/deep-lesion report, prior scarring, new/changing spot and
separate symptom answers, acne-control state, scar duration/wound/confirmation
state, pigment onset context, and optional budget. Unknown is data and is never
collapsed to a favorable default.

**tone bucket** — A coarse skin-tone band — `light`, `medium`, `deep`, or
`unknown` — self-reported or estimated from the photo via ITA, used to
personalize ranking and to disaggregate evaluation. Triage, not a claim (D-021 /
D-016); `unknown` is always reported, never dropped.

**slot (AM/PM)** — Which routine an active or product belongs in: `AM` (morning)
or `PM` (evening). Conflicting actives are split across slots (e.g. retinoids PM,
SPF AM-only) rather than one being dropped (`docs/RULES.md`).

**ranker** — An optional ordering mechanism that sees only products that have
already passed hard role eligibility (D-005/D-022/D-029). Concern outcome
evidence and tolerability precede evidence completeness/budget; pooled
`StatsRanker` review/popularity data is only a final deterministic tie-break.
It never admits, repairs, schedules, or overrides safety.

**care decision** — The complete exact-label result whose independent axes are
`triage_level`/referral reasons and `therapy_disposition`. A review referral
does not inherently suppress eligible treatment. Raw detector confidence is
not a calibrated probability (D-029).

**therapy plan** — Product-independent intent. Schema 4 preserves plural exact
lesion types and clinician options while retail active specifications remain in
each care pathway; its legacy single `primary` is always null. Schema 3 keeps
the reviewed single-primary compatibility shape.

**routine role** — A closed functional position (`cleanser`, `treatment`,
`moisturizer`, `sunscreen`/recsys `spf`, and dedicated `scar_care`) that an
eligible product may occupy. Category or an ingredient-list match does not
prove lesion coverage; exact active, exposure, role, and safety facts must also
match.

**eligibility** — A hard, role-specific veto over area, role, format, exposure,
therapy/strength/label source, sunscreen claims, profile, and every carried
active. Unknown required data is ineligible, not a score penalty.

**selected regimen** — At most one selected product per requested role plus
sourced AM/PM instructions. Independently eligible lower-ranked choices live
under `alternatives` and never appear in selected steps.

**provenance envelope** — Semantic input identities, code commit/dirty state,
schema version, generation time, and a deterministic **replay key**. Volatile
render/attempt fields do not change that key. Stale or mixed envelopes are not
release-comparable.

**popularity** — How much a product is bought/wanted, proxied by Sephora
`loves_count` (no purchase counts exist in the data). Distinct from
**well-reviewed** (rating quality): a product can be popular and mediocre, or
excellent and obscure. Feeds the ranker as a small deliberate bias (D-028),
never selection or safety.

**review-stats** — Per-product × skin-type summary statistics from the Sephora
reviews (count, average rating, % recommended) that feed both the report's
per-product "why" line and a ranking baseline (D-015 / D-022).

**concern-stats** — Legacy per-product × grouped-concern efficacy aggregates mined from
review text via one-time LLM labeling (helped/worsened counts, smoothed help
rate; D-023). They may provide pooled ranking evidence, but cannot create an
exact-label target or active match.

## Artifact map

Where the built artifacts live; paths are configurable in `configs/default.yaml`.

| Artifact | Location |
|----------|----------|
| Stage 1 detector (YOLO) | `models/detection/acne04_yolov8m_best.pt` |
| Stage 2 type classifier (Keras) | `models/classification/acne_model.keras` |
| FaceLandmarker bundle (MediaPipe Tasks) | `models/face_landmarker.task` |
| Normalized product catalog | `data/processed/catalog.json` |
| Catalog role quarantine | `data/processed/catalog_quarantine.json` |
| Clinician-reviewed therapy policy | external; none bundled for production |
| Ranker model (sklearn) | `models/ranker/ranker.joblib` |
| Review-stats table | `data/processed/review_stats.json` |
| Raw source data (Sephora, ACNE04) | `data/raw/` |
