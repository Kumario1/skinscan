# SkinScan

A learning project for computer vision: scan a face, detect skin concerns (acne
location + type + severity, hyperpigmentation), and recommend real skincare
products based on active ingredients.

**This is a learn-CV-by-building project, not a production app and not medical
software.** See `docs/DECISIONS.md` D-001 and D-002 for what that means in
practice.

## The pipeline

```
                  ┌──────────────────────────────────────────────┐
   face photo ──> │  STAGE 1: DETECTION (ML)                     │
                  │  ACNE04-trained detector                     │
                  │  output: boxes {location, severity, count}   │
                  └──────────────────────────────────────────────┘
                                    │
                                    v  crop each detected region
                  ┌──────────────────────────────────────────────┐
                  │  STAGE 2: CLASSIFICATION (ML)                 │
                  │  small classifier on center-cropped lesion   │
                  │  output: acne type (comedonal/inflammatory/  │
                  │          cystic), + concern tags             │
                  └──────────────────────────────────────────────┘
                                    │
                                    v  concern + location + severity
                  ┌──────────────────────────────────────────────┐
                  │  STAGE 3: RECOMMENDATION (rules, NOT ML)     │
                  │  concern -> ingredient rules (hand-written)  │
                  │  ingredient -> product lookup (catalog)      │
                  │  output: ranked routine                      │
                  └──────────────────────────────────────────────┘
```

**Core principle (carried over from the DormRoom project):** trust lives in the
logic layer, never in the ML output. The recommender reasons over a small,
auditable, hand-written rules table — not a learned mapping. The CV model is
allowed to be uncertain; the rules table is where correctness is guaranteed.

## Repo layout

```
data/
  raw/              # downloaded datasets, untouched (gitignored)
  processed/        # converted to training format (gitignored)
  self_collected/   # our own phone photos — TEST SET ONLY, never train
src/
  detection/        # Stage 1: acne detection + severity
  classification/   # Stage 2: acne-type / concern classifier
  recommendation/   # Stage 3: rules engine + catalog lookup
  evaluation/       # metrics, especially skin-tone-disaggregated eval
notebooks/          # cell-by-cell experiments (curriculum style)
configs/            # dataset paths, hyperparams, thresholds
docs/               # DECISIONS.md, data contracts, ADRs
models/             # saved weights (gitignored)
```

## Build order

Each stage is independently testable. Build and validate one before the next.

1. **Data prep** — ACNE04 -> YOLO format. Look at the boxes. Distrust them.
2. **Stage 1 detection** — single-class acne first, then severity. New metric
   family: mAP, IoU.
3. **Stage 3 recommender** — build this EARLY with hand-faked Stage 1/2 output.
   It's rules + a CSV; it needs no model. Lets us lock the data contract.
4. **Stage 2 classification** — acne type on cropped lesions.
5. **Evaluation** — Fitzpatrick-disaggregated error rates. The real lesson.
6. **Domain gap** — self-collected phone photos as a held-out test set.

See `docs/DECISIONS.md` for every locked choice and why.
