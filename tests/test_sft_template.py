import random

import pytest

from claudius.normalize import is_normalized, normalize
from claudius.sft.errors import corrupt
from claudius.sft.template import (
    LEVELS, STUDENT, SYSTEM, TEACHER, Turn, _WHY,
    correction_example, correction_reply, prompt_for_reply, render,
)

SENTS = [normalize(s) for s in (
    "Puella rosam pulchram amat.", "Caesar Galliam in partes tres dividit.",
    "Magister discipulos bonos docet.", "Milites urbem magnam oppugnant.",
)]


def test_every_level_has_a_normalised_system_prompt():
    for level in LEVELS:
        assert is_normalized(SYSTEM[level]), level


def test_render_is_normalised_end_to_end():
    text = render([Turn(STUDENT, "Salve, magister!"), Turn(TEACHER, "Salve!")], "tiro")
    body = text.replace("<|endoftext|>", "")
    assert is_normalized(body)
    assert text.endswith("<|endoftext|>")


def test_prompt_ends_ready_for_the_teacher():
    p = prompt_for_reply([Turn(STUDENT, "Salve!")], "medius")
    assert p.endswith("MAGISTER:")
    assert "<|endoftext|>" not in p


def test_unknown_level_and_role_are_rejected():
    with pytest.raises(ValueError):
        render([Turn(STUDENT, "x")], "expertissimus")
    with pytest.raises(ValueError):
        render([Turn("ROBOT", "x")], "tiro")


def test_tiro_gets_no_grammatical_explanation():
    """A beginner told WHY in Latin grammatical terms learns nothing from it."""
    rng = random.Random(0)
    for s in SENTS:
        c = corrupt(s, rng)
        if c and c.kind != "order":
            reply = correction_reply(c, "tiro", random.Random(1))
            assert _WHY[c.kind] not in reply


def test_medius_and_peritus_get_the_explanation():
    rng = random.Random(0)
    c = corrupt(SENTS[0], rng)
    for level in ("medius", "peritus"):
        assert _WHY[c.kind] in correction_reply(c, level, random.Random(1))


def test_correction_always_names_the_right_form():
    """Whatever else it says, the reply must contain the correct word."""
    rng = random.Random(3)
    for _ in range(40):
        s = rng.choice(SENTS)
        c = corrupt(s, rng, kinds=("case", "verb"))
        if c:
            reply = correction_reply(c, rng.choice(LEVELS), rng)
            assert f'"{c.right}"' in reply, (c, reply)


def test_explanations_make_no_category_specific_claim():
    """The regression. Specific explanations were sometimes false.

    "Noun and adjective must agree in case" was given for ROSAM -> ROSA, a
    case-role error. "Agree in number" would be given for AMAUIT -> AMABAT, a
    tense error. A tutor that mis-corrects is worse than none.
    """
    for kind in ("case", "verb"):
        why = _WHY[kind]
        for false_claim in ("ADIECTIUUM", "NUMERO", "SUBIECTO", "CONSENTIRE"):
            assert false_claim not in why, f"{kind}: {why!r} claims {false_claim}"


def test_full_example_round_trips():
    rng = random.Random(9)
    c = corrupt(SENTS[1], rng)
    ex = correction_example(c, "medius", rng)
    assert f"DISCIPULUS: {c.corrupted}" in ex
    assert "MAGISTER:" in ex
