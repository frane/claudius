"""The correctness guard. Skipped where LatinCy is not installed."""
import pytest

pytest.importorskip("la_core_web_sm")

from claudius.sft.checker import AGREEMENT_FEATURES, check, is_clean  # noqa: E402


def test_uppercase_is_lowercased_before_parsing():
    """On raw uppercase, LatinCy reads every word as a Roman numeral (NUM).

    The checker must lowercase, or it produces garbage on the only orthography
    the model ever emits.
    """
    assert is_clean("PUELLA ROSAM PULCHRAM AMAT.")


def test_catches_adjective_case_disagreement():
    errs = check("PUELLA ROSAM PULCHRA AMAT.")
    assert any(e.rule == "adjective-noun" and e.feature == "Case"
               and e.word == "PULCHRA" and e.head == "ROSAM" for e in errs), errs


def test_catches_subject_verb_number_disagreement():
    errs = check("MAGISTER DISCIPULOS BONOS DOCENT.")
    assert any(e.rule == "subject-verb" and e.word == "MAGISTER" for e in errs), errs


def test_correct_sentences_are_clean():
    for s in ("PUELLA ROSAM PULCHRAM AMAT.", "MAGISTER DISCIPULOS BONOS DOCET.",
              "CAESAR GALLIAM DIUIDIT.", "UIR BONUS UENIT."):
        assert is_clean(s), (s, check(s))


def test_gender_is_never_checked():
    """Gender caused 16 of 41 false positives on correct Vulgate text.

    Latin does not mark gender on most third-declension endings, so the parser
    guesses: DIE SEPTIMO had DIES tagged feminine.
    """
    assert "Gender" not in AGREEMENT_FEATURES
    assert not any(e.feature == "Gender" for e in check("SEPTIMO DIE CESSABIS."))


def test_self_contradictory_subject_is_not_trusted():
    """nsubj tagged accusative is the parser contradicting itself.

    QUAE and MOSES were both labelled subject while tagged Case=Acc. A student
    must never be told they are wrong on the strength of a confused parse.
    """
    assert not any(e.rule == "subject-verb" and e.word == "QUAE"
                   for e in check("FECIT ERGO NOE OMNIA QUAE MANDAUERAT EI DOMINUS."))
    assert not any(e.rule == "subject-verb" and e.word == "MOSES"
                   for e in check("DESCENDIT MOSES AD POPULUM ET OMNIA NARRAUIT EIS."))
