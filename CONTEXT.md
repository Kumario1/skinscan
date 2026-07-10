# CONTEXT — SkinScan glossary

The shared vocabulary every issue, doc, and module uses. Definitions only —
mechanisms and rationale live in `docs/DECISIONS.md` and the schema docs, not
here.

## Terms

**concern** — A cosmetic skin issue the system reports on, never a medical
condition (D-002). Closed vocabulary: `acne_comedonal`, `acne_inflammatory`,
`acne_cystic`, `hyperpigmentation`, `dryness` (`docs/CONCERN_SCHEMA.md`). The CV
pipeline currently produces only the acne concerns.

**region** — A named area of the face a concern is localized to. Closed
vocabulary: `forehead`, `nose`, `left_cheek`, `right_cheek`, `chin_jaw`,
`perioral` (D-008 / D-020).

**active** — A canonical skincare ingredient the rules reason over (e.g.
`salicylic_acid`, `niacinamide`, `retinol`). Products are interchangeable
carriers of actives (D-006); the closed ~30-item list lives in
`docs/CATALOG_SCHEMA.md`.

**ConcernReport** — The fixed JSON contract between the CV side (Stage 2) and the
rules side (Stage 3): one entry per (concern, region) with severity, lesion
count, and confidence, plus clear-skin and low-light meta flags (D-008,
`docs/CONCERN_SCHEMA.md`). Neither side reaches around it.

**bridge** — The step that turns raw per-lesion model output into a ConcernReport;
the join from the CV side to the rules side.

**profile** — The inference-time facts about the user that personalize a report:
skin_type (required; combination/dry/normal/oily), tone bucket (optional), and a
pregnancy/nursing flag (D-021). Intake asks only what something downstream uses.

**tone bucket** — A coarse skin-tone band — `light`, `medium`, `deep`, or
`unknown` — self-reported or estimated from the photo via ITA, used to
personalize ranking and to disaggregate evaluation. Triage, not a claim (D-021 /
D-016); `unknown` is always reported, never dropped.

**slot (AM/PM)** — Which routine an active or product belongs in: `AM` (morning)
or `PM` (evening). Conflicting actives are split across slots (e.g. retinoids PM,
SPF AM-only) rather than one being dropped (`docs/RULES.md`).

**ranker** — The learned re-ranker that reorders rule-approved candidate products
within a category by predicted fit for a profile (D-005 / D-022). It only
reorders — it never selects, gates, or overrides safety.

**review-stats** — Per-product × skin-type summary statistics from the Sephora
reviews (count, average rating, % recommended) that feed both the report's
per-product "why" line and a ranking baseline (D-015 / D-022).

## Artifact map

Where the built artifacts live; paths are configurable in `configs/default.yaml`.

| Artifact | Location |
|----------|----------|
| Stage 1 detector (YOLO) | `models/detection/acne04_yolov8m_best.pt` |
| Stage 2 type classifier (Keras) | `models/classification/acne_model.keras` |
| FaceLandmarker bundle (MediaPipe Tasks) | `models/face_landmarker.task` |
| Normalized product catalog | `data/processed/catalog.json` |
| Ranker model (sklearn) | `models/ranker/ranker.joblib` |
| Review-stats table | `data/processed/review_stats.json` |
| Raw source data (Sephora, ACNE04) | `data/raw/` |
