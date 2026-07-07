# Plan 009: Restore the Stage 1 detector training notebook from git history

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. Your reviewer maintains `plans/README.md` — do
> not update it.
>
> **Drift check (run first)**: `ls notebooks/ 2>/dev/null` — if a
> `notebooks/01_acne04_detector.md` already exists at HEAD, STOP (someone
> restored it already).

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: tech-debt
- **Planned at**: commit `1ebd544`, 2026-07-06

## Why this matters

Stage 2 training is reproducible (`src/classification/train_type_classifier.py`),
but Stage 1's YOLOv8m detector weights are a binary with zero training
provenance in the repo: no script, no notebook, no hyperparameters. The
training notebook `notebooks/01_acne04_detector.md` existed at the initial
commit and was deleted in commit `362899e` ("docs: document SkinScan process
and notebook results"). Anyone (including the maintainer in six months) who
needs to retrain, tweak, or audit the detector has nothing to start from.
Git still has the file; restoring it is one command.

## Current state

- `git ls-tree -r --name-only 362899e~1 | grep notebooks` shows (verified at
  planning time):
  - `notebooks/01_acne04_detector.md` ← the file to restore
  - `notebooks/02_type_classifier.ipynb` / `.md` — do NOT restore (Stage 2 is
    covered by the checked-in trainer; restoring superseded notebooks adds
    confusion)
- Current HEAD has no `notebooks/` directory.
- `README.md` §1 describes the detector but names no training source.
- Worktrees share the repo object store — `git show 362899e~1:...` works from
  your worktree.

## Environment facts

- Fresh git worktree; git history fully available.
- No interpreter needed beyond git + grep.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Confirm source exists | `git cat-file -e 362899e~1:notebooks/01_acne04_detector.md && echo found` | prints `found` |
| Restore | see Step 1 | file created |

## Scope

**In scope**:
- `notebooks/01_acne04_detector.md` (restore from history)
- `README.md` (one provenance line in §1)
- `data/README.md` (one pointer line)

**Out of scope**:
- `notebooks/02_type_classifier.*` and `notebooks/acne_model (1).ipynb` —
  superseded by `train_type_classifier.py`; leave them in history.
- Editing the restored notebook's content — restore verbatim, even if parts
  are stale; it is a historical record. (Exception: none.)
- Re-running any training.

## Git workflow

- Stay on the worktree's branch. Commit style:
  `docs: restore stage 1 detector training notebook from history`
- Do NOT push.

## Steps

### Step 1: Restore the file verbatim

```bash
mkdir -p notebooks
git show 362899e~1:notebooks/01_acne04_detector.md > notebooks/01_acne04_detector.md
```

**Verify**: `wc -l notebooks/01_acne04_detector.md` → > 50 lines;
`head -3 notebooks/01_acne04_detector.md` → starts with a markdown heading or
front matter (eyeball: it should read as a detector-training walkthrough; if
it is empty or binary, STOP).

### Step 2: Link it from the READMEs

- `README.md` §1 (the "Stage 1 - lesion locator" section): after the operating
  point code block, add one line:
  `Training provenance: [notebooks/01_acne04_detector.md](notebooks/01_acne04_detector.md) (Colab walkthrough that produced these weights).`
- `data/README.md`: in the "Detector-only location check expects" area, add a
  pointer line: `Stage 1 training walkthrough: ../notebooks/01_acne04_detector.md`.

**Verify**: `grep -c "01_acne04_detector" README.md data/README.md` → 1 each

## Test plan

None (docs restore). Verification is the greps + line count.

## Done criteria

- [ ] `notebooks/01_acne04_detector.md` exists, > 50 lines, verbatim from
      `362899e~1` (`git show 362899e~1:notebooks/01_acne04_detector.md | diff - notebooks/01_acne04_detector.md` → empty)
- [ ] `grep -c "01_acne04_detector" README.md` → 1
- [ ] `grep -c "01_acne04_detector" data/README.md` → 1
- [ ] `git status --porcelain` clean outside the in-scope list

## STOP conditions

- `git cat-file -e` fails (object missing — history rewritten since planning).
- The restored file is empty, binary, or clearly not a detector-training
  document.
- `notebooks/01_acne04_detector.md` already exists at HEAD.

## Maintenance notes

- If the detector is retrained (e.g. after plan 010's negative-class work or
  any future size change), the notebook should be updated or superseded by a
  committed training script — this restore is the floor, not the ceiling.
