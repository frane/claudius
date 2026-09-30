import random

from claudius.normalize import is_normalized, normalize
from claudius.sft.errors import (
    Corruption,
    corrupt,
    corrupt_case,
    corrupt_order,
    corrupt_verb,
)

S = normalize("Puella rosam amat.")
LONG = normalize("Magister discipulos in schola docet.")


def test_case_error_changes_one_word_to_a_wrong_ending():
    c = corrupt_case(S, random.Random(0))
    assert c is not None and c.kind == "case"
    assert c.corrupted != c.original
    diff = [(a, b) for a, b in zip(c.original.split(), c.corrupted.split()) if a != b]
    assert len(diff) == 1, "exactly one word should be wrong"
    assert c.right in c.original.split() and c.wrong in c.corrupted.split()


def test_verb_error_swaps_person_or_tense():
    c = corrupt_verb(S, random.Random(0))
    assert c is not None and c.kind == "verb"
    # AMAT -> AMANT: singular subject with a plural verb.
    assert c.right.startswith("AMA") and c.wrong != c.right


def test_errors_are_real_latin_letters_only():
    """A corrupted sentence must stay in the model's orthography."""
    rng = random.Random(1)
    for _ in range(50):
        c = corrupt(LONG, rng)
        if c:
            assert is_normalized(c.corrupted), c.corrupted


def test_function_words_are_never_corrupted():
    """No learner writes ESTT for EST; those errors would teach nothing."""
    s = normalize("Et in est non cum ad ut sed.")
    for seed in range(20):
        assert corrupt_case(s, random.Random(seed)) is None
        assert corrupt_verb(s, random.Random(seed)) is None


def test_order_error_moves_the_final_verb_forward():
    c = corrupt_order(LONG, random.Random(0))
    assert c is not None and c.kind == "order"
    assert c.original.split()[-1].rstrip(".") == "DOCET"
    assert c.corrupted.split()[1] == "DOCET", c.corrupted
    # Same words, different order: a register slip, not a morphological one.
    strip = lambda t: sorted(w.strip(".") for w in t.split())
    assert strip(c.original) == strip(c.corrupted)


def test_corruption_is_deterministic_given_a_seed():
    a = corrupt(LONG, random.Random(42))
    b = corrupt(LONG, random.Random(42))
    assert a == b


def test_every_corruption_knows_its_answer():
    """The correctness guard checks the buddy's fix against this record."""
    rng = random.Random(7)
    for _ in range(30):
        c = corrupt(LONG, rng)
        if c:
            assert isinstance(c, Corruption)
            assert c.right and c.wrong
            assert 0 <= c.position < len(c.original.split())


def test_uncorruptible_sentence_returns_none():
    assert corrupt("ET IN", random.Random(0)) is None


def test_sentence_final_verb_with_punctuation_is_corruptible():
    """The regression: AMAT. carries a full stop and was being skipped.

    Latin puts the verb last, so it is the word most likely to carry
    punctuation. Skipping it meant verb errors almost never got generated.
    """
    c = corrupt_verb(normalize("Puella rosam amat."), random.Random(0))
    assert c is not None
    assert c.corrupted.endswith("."), "the full stop must survive"
    assert c.right == "AMAT" and "." not in c.wrong, "record bare words"


def test_verb_errors_actually_occur_on_real_prose():
    """Across real sentences, verb errors must be generated at a real rate."""
    sents = [normalize(s) for s in (
        "Caesar Galliam vincit.", "Puellae rosas amant.", "Magister librum legit.",
        "Milites urbem oppugnant.", "Poeta carmen scribit.", "Servus aquam portat.",
    )]
    hits = sum(1 for i, s in enumerate(sents) if corrupt_verb(s, random.Random(i)))
    assert hits >= 5, f"only {hits}/6 sentences yielded a verb error"
