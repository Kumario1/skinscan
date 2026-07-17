"""Regression tests for the deterministic ingredient parser every safety gate
keys off. Two properties here were previously held closed by coincidence rather
than enforcement, which is why each case below is pinned individually:

  * the allergen gate matched whole tokens against the declared word alone, so a
    declared "parabens" allergy — a category no label ever spells the singular
    way — was a gate that could not fire;
  * the retinoid vocabulary lived in two places that had already drifted apart,
    and the pregnancy gate stayed closed only because the drifted-away names
    happened not to be listed as candidates anywhere else.
"""
import json
from pathlib import Path

import pytest

from recsys.inci import (
    ALLERGY_FAMILIES,
    RETINOID_MARKERS,
    allergy_matches,
    contains_retinoid,
    parse_ingredients,
)

SAFETY_RULES = json.loads(
    (Path(__file__).parents[1] / "data" / "knowledge" / "safety_rules.json").read_text(
        encoding="utf-8"
    )
)
KNOWLEDGE_RETINOIDS = SAFETY_RULES["retinoids"]


# (declared allergy, label INCI). Every row was a verified NO-VETO before the
# families table and plural normalization landed.
ALLERGEN_MISSES = [
    ("parabens", ("Water", "Methylparaben", "Propylparaben")),
    ("paraben", ("Water", "Methylparaben", "Propylparaben")),
    ("sulfates", ("Water", "Sodium Lauryl Sulfate")),
    ("fragrances", ("Water", "Fragrance")),
    ("aloe vera", ("Water", "Aloe Barbadensis Leaf Juice")),
    ("vitamin e", ("Water", "Tocopherol")),
    ("nuts", ("Water", "Prunus Amygdalus Dulcis (Sweet Almond) Oil")),
    ("almonds", ("Water", "Prunus Amygdalus Dulcis (Sweet Almond) Oil")),
    ("salicylates", ("Water", "Salicylic Acid")),
    ("benzoates", ("Water", "Sodium Benzoate")),
]

# Rows that already vetoed. Pinned so the fix stays additive.
ALLERGEN_ALREADY_WORKING = [
    ("sulfate", ("Water", "Sodium Lauryl Sulfate")),
    ("fragrance", ("Water", "Fragrance")),
    ("aloe", ("Water", "Aloe Barbadensis Leaf Juice")),
    ("almond", ("Water", "Prunus Amygdalus Dulcis (Sweet Almond) Oil")),
    ("niacinamide", ("Water", "Niacinamide")),
    ("fragrance", ("Water", "Parfum (Fragrance)", "Limonene")),
]


@pytest.mark.parametrize(
    "allergy,inci", ALLERGEN_MISSES, ids=[f"{a}-{i[1]}" for a, i in ALLERGEN_MISSES]
)
def test_declared_allergen_family_vetoes_the_member_the_label_actually_names(allergy, inci):
    """A person declares a category; a label names a member. Nobody is allergic
    to the word "paraben" — they are allergic to the methyl- and propyl- esters
    that every label lists instead, so matching only the declared word made the
    most commonly declared allergy in skincare a permanently dead gate."""
    assert allergy_matches(allergy, inci, ()), f"{allergy!r} did not veto {inci}"


@pytest.mark.parametrize(
    "allergy,inci",
    ALLERGEN_ALREADY_WORKING,
    ids=[f"{a}-{i[1]}" for a, i in ALLERGEN_ALREADY_WORKING],
)
def test_allergen_spellings_that_already_vetoed_still_veto(allergy, inci):
    """The families table and the de-pluralizer only ever add candidate
    spellings. Any row that regresses here means the fix stopped being additive
    and started replacing the token-run match it was supposed to widen."""
    assert allergy_matches(allergy, inci, ())


def test_precise_allergy_declaration_is_never_weaker_than_the_vague_one():
    """The inversion that gave this defect away: "aloe" vetoed and "aloe vera"
    did not, so the user who answered the intake form carefully was protected
    less than the one who typed a single word."""
    inci = ("Water", "Aloe Barbadensis Leaf Juice")
    assert allergy_matches("aloe", inci, ())
    assert allergy_matches("aloe vera", inci, ())


# Whole-token matching on family members is load-bearing: these are the exact
# over-vetoes a substring scan for "nut" would cause.
NON_MEMBERS = [
    ("nuts", ("Water", "Coconut Alkanes")),
    ("nut", ("Water", "Cocos Nucifera (Coconut) Oil")),
    ("tree nuts", ("Water", "Myristica Fragrans (Nutmeg) Oil")),
    ("almonds", ("Water", "Glycerin", "Squalane")),
    ("parabens", ("Water", "Phenoxyethanol")),
    ("vitamin e", ("Water", "Ascorbic Acid")),
]


@pytest.mark.parametrize(
    "allergy,inci", NON_MEMBERS, ids=[f"{a}-{i[1]}" for a, i in NON_MEMBERS]
)
def test_allergen_families_do_not_over_veto_lookalike_ingredients(allergy, inci):
    """Coconut is a drupe and nutmeg a seed; neither is a tree nut, and both
    contain "nut" as a substring. Blanket substring matching would veto them and
    is why whole-token equality was chosen in the first place — the families
    table has to buy recall without spending that precision."""
    assert not allergy_matches(allergy, inci, ()), f"{allergy!r} over-vetoed {inci}"


def test_allergy_vetoes_a_drug_row_that_carries_actives_but_no_inci():
    """Drug rows ship inci: [], so the raw-label scan is blind to them and the
    synonym table is the only route an allergy has to reach them at all."""
    assert allergy_matches("salicylates", (), ("salicylic_acid",))
    assert allergy_matches("bha", (), ("salicylic_acid",))


def test_allergen_family_keys_resolve_to_members_that_tokenize():
    """A member phrase is matched as a token run, so one containing punctuation
    or a stray empty string would be silently unmatchable — a dead row in a
    safety table reads exactly like a live one."""
    for family, members in ALLERGY_FAMILIES.items():
        assert members, f"{family} has no members"
        for member in members:
            assert member == member.lower().strip(), f"{family}: {member!r} not normalized"
            assert member.replace(" ", "").isalnum(), f"{family}: {member!r} will not tokenize"


@pytest.mark.parametrize("retinoid", KNOWLEDGE_RETINOIDS)
def test_every_retinoid_in_the_knowledge_table_is_also_a_raw_inci_marker(retinoid):
    """The pregnancy gate is `actives & knowledge.retinoids or
    contains_retinoid(inci)`. A name known to only one arm is a hole in the
    other, so the two vocabularies are derived from one list rather than
    maintained in parallel and hoped to agree."""
    assert retinoid.replace("_", " ") in RETINOID_MARKERS


@pytest.mark.parametrize("retinoid", KNOWLEDGE_RETINOIDS)
def test_every_retinoid_is_caught_by_the_raw_inci_scan_when_a_label_names_it(retinoid):
    """Cosmetic labels spell drugs out in full. If the knowledge table lists a
    retinoid the raw scan cannot see, a product naming it in INCI reaches a
    pregnant user whenever it also fails to parse to a canonical active."""
    assert contains_retinoid((f"Water, {retinoid.replace('_', ' ').title()} 0.1%",))


@pytest.mark.parametrize(
    "drug,product",
    [
        ("tazarotene", "Tazorac / Arazlo — pregnancy category X"),
        ("trifarotene", "AKLIEF"),
        ("isotretinoin", "oral/topical isotretinoin"),
    ],
)
def test_prescription_retinoids_in_the_drug_catalog_are_known_to_the_pregnancy_gate(drug, product):
    """These are real rows in catalog_drug.json (3x tazarotene, 1x trifarotene).
    They were absent from safety_rules.retinoids and — because drug rows carry
    inci: [] — the raw-INCI backstop could never cover for that. They stayed
    unreachable only because concern_actives.json happened not to list them, so
    the safety property was resting on an unrelated table. Adding "tazarotene"
    there is a plausible one-line edit; it must not be the thing standing
    between a pregnant user and a category X teratogen."""
    assert drug in KNOWLEDGE_RETINOIDS, f"{product} would not be vetoed for pregnancy"


def test_retinoid_markers_are_derived_from_the_knowledge_file_not_hardcoded():
    """The dangerous direction is knowledge-name-without-marker. Extra markers
    are fine and expected — they are label spellings like "retinyl" that never
    parse to a canonical active — but a retinoid the knowledge table names and
    the scan cannot see is exactly how tazarotene went missing."""
    stems = set(SAFETY_RULES["retinoid_inci_stems"])
    names = {r.replace("_", " ") for r in KNOWLEDGE_RETINOIDS}
    assert names - set(RETINOID_MARKERS) == set()
    assert set(RETINOID_MARKERS) == names | stems


@pytest.mark.parametrize(
    "field,path",
    [
        ("treatment_actives", ("treatment_actives",)),
        ("pm_pinned_actives", ("session_preferences", "pm_pinned_actives")),
        ("gentle.excluded_actives", ("gentle", "excluded_actives")),
    ],
)
def test_every_retinoid_stays_listed_everywhere_a_retinoid_must_be_listed(field, path):
    """These three lists each independently re-name the retinoids, and each was
    a superset of safety_rules.retinoids by coincidence. Adding a retinoid to
    the pregnancy table alone would silently drop it out of PM-pinning (a
    photosensitizer scheduled for AM) and out of the gentle exclusion."""
    listed = SAFETY_RULES
    for key in path:
        listed = listed[key]
    missing = sorted(set(KNOWLEDGE_RETINOIDS) - set(listed))
    assert not missing, f"retinoids missing from {field}: {missing}"


def test_retinoid_ester_that_parses_to_no_active_is_still_caught_by_the_raw_scan():
    """P393718 in the real catalog. The "(and)" separator defeats the comma
    split and the ester is in no synonym table, so it yields ZERO canonical
    actives — the actives set cannot be trusted for pregnancy and the raw-label
    substring scan is the only thing vetoing it. This is the backstop working as
    designed; it must survive any change to the marker source."""
    inci = ("Water", "Dimethyl Isosorbide (and) Hydroxypinacolone Retinoate")
    actives, _ = parse_ingredients(", ".join(inci))
    assert actives == []
    assert contains_retinoid(inci)


def test_raw_scan_does_not_fire_on_ingredients_that_merely_spell_like_a_retinoid():
    """Phloretin is an apple polyphenol, not a vitamin A derivative. It carries
    the substring "retin" and appears in 4 real catalog products; a scan for the
    bare stem would veto all of them for pregnancy on a spelling coincidence."""
    assert not contains_retinoid(("Water", "Phloretin", "Ferulic Acid"))
