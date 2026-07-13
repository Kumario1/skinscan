E2E run verdict: the pipeline logic checks out end-to-end — every number recomputes exactly — but the run exposes two data problems (missing landmarker artifact, catalog gaps) that make the final routine weaker than the code intends.

The run is runs/e2e/sarpn-v2-smoke/: one 3448×4600 clinical-rig photo through the SA-RPN native-tile pipeline (20 tiles of 1024², local endpoint), producing analysis.json, routine.json, and three overlay images. Here's the reasoning chain, layer by layer.

Detection layer — internally consistent, verified by recomputation

- 103 detections, all above the 0.3 score floor (minimum observed 0.307), every box inside its source tile.
- Dedupe worked: I computed IoU across all 5,253 detection pairs — zero pairs above even 0.3 (threshold is 0.5), so no double-counted lesions survived the overlapping tiles.
- Concern aggregation is exact: lesion counts (52 comedonal / 36 inflammatory / 7 scarring / 2 hyperpigmentation), mean confidences, max confidences, and region lists all recompute to precisely the JSON values.
- Visually, the detection overlay and lesion sheet are credible: pustules show white heads, open comedones dark centers, nevi are dark macules. A couple of crops in hair/eyebrow areas are marginal, but the label mix is sane.
- One nit: analysis.json reports overlap: 128, but the actual tile stride (808/894) gives an effective overlap of 216/130 px — the tiler spreads tiles evenly to cover the image and the metadata records the requested minimum, not the effective value. Not a bug, just worth knowing when reading the JSON.

Severity — recomputed by hand against configs/default.yaml, all correct

- Comedonal 52 lesions → bisect over [1,8,20,40] → 4 ✓. Inflammatory 36 → [1,6,15,30] → 4 ✓.
- Scarring: 7 lesions would give only 2, but 4 affected regions ≥ broad_region_count: 3 floors it to 3 ✓.
- Hyperpigmentation: 2 lesions give 1, floored to 2 by the 2-region rule; max confidence 0.64 ≥ 0.5 so the low-confidence cap didn't trigger ✓.
- 6 nevi ≥ min_count: 3 → professional_review: true ✓.

Region mapping — the one degraded component of this run

models/face_landmarker.task is missing, so regions came from the grid fallback (face_detected: false). The overlay shows the grid bands roughly fit because the rig photo fills the frame, but there are visible misassignments (eyebrow-area detections labeled right_cheek, lower-forehead papules split across bands). This matters selectively: the two severity-4 acne scores are count-driven and robust to region noise, but the scarring severity of 3 exists only because of the region-spread floor over grid-guessed regions — it's the least trustworthy number in the report. To make the smoke test exercise the production path, the landmarker artifact needs to be dropped in place.

Recommendation layer — behaves exactly as coded, and that's the interesting part

Overall severity 4 triggers the escalation short-circuit at src/recommendation/engine.py:127: flag see a dermatologist, skip all treatment actives, target centella/ceramides/hyaluronic_acid, gentle-only catalog filter, everything in both slots, SPF AM-only. routine.json matches all of it (SPF present in AM, absent in PM; no "possible — verify" flags since all four concern confidences ≥ 0.5). recommendation_status: complete.

But thinking through the actual products surfaced three real findings:

1. Centella is a phantom target. The tier-1 catalog (1,634 products) contains zero products 3, but tier-2 only fills a slot×category when tier-1 has none — and hyaluronic acid fills
everything. So the soothe path's signature calming active can never appear in any recommendamport_catalog.py:66-67 maps only the strings "centella" / "centella asiatica", while SephoraINCI typically reads "Centella Asiatica (Hydrocotyl) Extract", "cica", or "madecassoside". Worth checking whether the importer's matching misses those variants.
2. Top-5 selection is degenerate on this path. With ranker=None and no ingredient-match scores, sort order collapses to catalog order — alphabetical by brand. That's why the "gentle routine" for a derm-referral
face leads with a $390 Augustinus Bader serum and $290 cream while $20–40 equivalents sit lo"the first five alphabetical HA products per category."
3. Gentle-filter leaks via catalog data. Clinique "Clarifying Lotion 2" (a high-alcohol exfoliating toner) passes _gentle_only because its catalog actives are just glycerin/HA and the name-hint regex doesn't
cover "clarifying". Similarly, wash-off masks and a cleansing gel sit in the "treatment" catleanser" — catalog categorization noise, not engine logic.

Bottom line

Mechanics — tiling, dedupe, aggregation, severity math, escalation gating, slot/SPF rules, a verified correct. The weaknesses are environmental and data-side: the missing FaceLandmarkerartifact (regions are guesses), the centella import gap (the soothe routine is effectively a), and catalog category/ordering quality. None of them block the cutover; the centellaimporter gap is the one I'd chase first since it silently guts the severity-4 path's intende