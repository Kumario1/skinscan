# Research brief - verification batch 007 (manually queued: serum role)

Purpose: recsys now fills the `serum` slot with concern-matched cosmetics
(candidates.py), but no serum row carries the overlay facts the gates require
(routine_roles, format, exposure, cadence + cadence_source, intended_areas).
The loop's `select` only covers support roles and treatment paths, so this
batch was queued by hand (manifest requeue, per data/verification/README.md).

Rules are unchanged (fail closed, D-032): manufacturer's own product page over
HTTPS, exact brand/product/strength/variant match, evidence bytes snapshotted
by sha256, assert only facts the page explicitly states. Cosmetic facts only —
no drug_actives, no treatment-path claims (P427411 was rejected in batch 003
as an azelaic_acid_10 drug-path candidate and stays rejected for that purpose;
this batch asserts only its cosmetic serum-role facts).

## P427417 - Niacinamide 10% + Zinc 1% Oil Control Serum (The Ordinary) -> serum
- routine_role_not_verified: facts.routine_roles must include serum
- instruction_cadence_unknown: facts.cadence plus facts.cadence_source
- instruction_cadence_source_missing: facts.cadence_source (URL stating the cadence)

## P427411 - Azelaic Acid 10% Suspension Brightening Cream (The Ordinary) -> serum
- same facts as above; format per the page (suspension)

## P427412 - Alpha Arbutin 2% + HA Hyperpigmentation Serum (The Ordinary) -> serum
- same facts as above

## P479732 - Salicylic Acid 2% Anhydrous Solution Pore Clearing Serum (The Ordinary) -> serum
- same facts as above; contraindication (sensitive) if the page states it

## P427419 - Hyaluronic Acid 2% + B5 Hydrating Serum (The Ordinary) -> serum
- same facts as above; catalog row is the 2 oz / 60 mL variant — confirm the
  manufacturer page still offers that size
