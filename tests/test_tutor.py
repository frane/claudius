import random

import pytest

pytest.importorskip("tokenizers")

from claudius import tutor  # noqa: E402
from claudius.tutor import Finding, candidates, correction_text  # noqa: E402


def test_candidates_can_undo_an_agreement_error():
    fixes = {(w, r) for _, w, r in candidates("PUELLA ROSAM AMANT.")}
    assert ("AMANT", "AMAT") in fixes
    assert ("ROSAM", "ROSA") in fixes


def test_candidates_keep_punctuation_off_the_word():
    for sent, wrong, right in candidates("MILITES URBEM DEFENDIT."):
        assert sent.endswith(".")
        assert "." not in wrong and "." not in right


def test_firm_correction_only_when_confirmed():
    rng = random.Random(0)
    firm = correction_text(Finding("AMANT", "AMAT", 3.0, True), "medius", rng)
    hint = correction_text(Finding("AMANT", "AMAT", 9.0, False), "medius", rng)
    assert 'DICENDUM EST "AMAT", NON "AMANT"' in firm
    assert hint.startswith("NUM") and "AMAT" in hint and "NON" not in hint


def _fake(monkeypatch, scores, flagged):
    import claudius.sft.checker as checker
    import claudius.sft.scorer as scorer

    class D:
        def __init__(self, w, h):
            self.word, self.head = w, h
            self.rule, self.feature = "subject-verb", "Number"
            self.word_value, self.head_value = "Sing", "Plur"

    monkeypatch.setattr(scorer, "sentence_logprob", lambda m, t, s: scores.get(s, 0.0))
    monkeypatch.setattr(checker, "check", lambda s: [D(*p) for p in flagged])
    return D


def test_checker_plus_model_gives_firm_fix(monkeypatch):
    _fake(monkeypatch, {"PUELLA ROSAM AMAT.": 4.0}, [("PUELLA", "AMANT")])
    f = tutor.detect_swaps(None, None, "PUELLA ROSAM AMANT.")
    assert f.confirmed and (f.wrong, f.right) == ("AMANT", "AMAT")
    assert f.rule == "PUELLA NUMERI SINGULARIS EST, ERGO ET UERBUM: AMAT."


def test_model_alone_below_hint_margin_says_nothing(monkeypatch):
    _fake(monkeypatch, {"PUELLA ROSAM AMAT.": 5.0}, [])
    assert tutor.detect_swaps(None, None, "PUELLA ROSAM AMANT.") is None


def test_checker_alone_says_nothing(monkeypatch):
    _fake(monkeypatch, {}, [("PUELLA", "AMANT")])   # model prefers no fix
    assert tutor.detect_swaps(None, None, "PUELLA ROSAM AMANT.") is None


def test_roman_display_writes_u_as_v():
    from claudius.generate import roman
    from claudius.normalize import normalize

    assert roman("SALUE! UERBUM UULT UITAM.") == "SALVE! VERBVM VVLT VITAM."
    # Round trip: display form normalises back to the internal U form.
    assert normalize(roman("SALUE, MAGISTER")) == "SALUE, MAGISTER"


def test_rule_text_names_the_rule_for_each_kind():
    from claudius.sft.checker import Disagreement
    from claudius.tutor import rule_text

    sv = Disagreement("subject-verb", "Number", "MILITES", "DEFENDIT", "Plur", "Sing")
    assert rule_text(sv, "DEFENDIT", "DEFENDUNT") == \
        "MILITES numeri pluralis est, ergo et verbum: DEFENDUNT."
    assert rule_text(sv, "MILITES", "MILES") == \
        "Verbum DEFENDIT numeri singularis est, ergo et subiectum: MILES."
    an = Disagreement("adjective-noun", "Case", "PULCHRA", "ROSAM", "Nom", "Acc")
    assert "casu consentire debet" in rule_text(an, "PULCHRA", "PULCHRAM")


def test_trim_keeps_one_turn_up_to_the_first_question():
    from claudius.tutor import trim

    assert trim("BENE. QUID AGIS? ET TU? \nDISCIPULUS: X") == "BENE. QUID AGIS?"
    assert trim("BENE DIXISTI.\nDISCIPULUS: X") == "BENE DIXISTI."


def _fake_learned(monkeypatch, fixed, margin, flagged, scores=None):
    import claudius.sft.correct as correct

    _fake(monkeypatch, scores or {}, flagged)
    monkeypatch.setattr(correct, "propose", lambda m, t, s, samples=0: correct.Correction(
        s, fixed, margin, correct.diff_words(s, fixed)))


def test_firm_when_both_detectors_agree_and_checker_objects(monkeypatch):
    _fake_learned(monkeypatch, "MILITES URBEM DEFENDUNT.", 1.0, [("MILITES", "DEFENDIT")],
                  {"MILITES URBEM DEFENDUNT.": 3.0})
    f = tutor.detect(None, None, "MILITES URBEM DEFENDIT.")
    assert f.confirmed and (f.wrong, f.right) == ("DEFENDIT", "DEFENDUNT")
    assert f.rule.startswith("MILITES NUMERI")


def test_learned_alone_is_only_a_question(monkeypatch):
    # No swap candidate produces UENERUNT, so the detectors cannot agree.
    _fake_learned(monkeypatch, "AMICI MEI HERI UENERUNT.", 5.0, [("AMICI", "UENIT")])
    f = tutor.detect(None, None, "AMICI MEI HERI UENIT.")
    assert not f.confirmed and f.right == "UENERUNT"


def test_nothing_when_neither_detector_is_confident(monkeypatch):
    _fake_learned(monkeypatch, "AMICI MEI HERI UENERUNT.", 1.0, [])
    assert tutor.detect(None, None, "AMICI MEI HERI UENIT.") is None


def test_swaps_alone_is_only_a_question(monkeypatch):
    _fake_learned(monkeypatch, "PUELLA ROSAM AMANT.", 0.0, [("PUELLA", "AMANT")],
                  {"PUELLA ROSAM AMAT.": 4.0})
    monkeypatch.setattr(__import__("claudius.sft.correct", fromlist=["x"]), "propose",
                        lambda m, t, s, samples=0: None)
    f = tutor.detect(None, None, "PUELLA ROSAM AMANT.")
    assert not f.confirmed and (f.wrong, f.right) == ("AMANT", "AMAT")


def test_echo_question_is_detected_but_asking_back_is_fine():
    from claudius.tutor import echoes

    assert echoes("EGO CANEM HABEO.", "TU CANEM HABES?")
    assert echoes("PATER MEUS AGRICOLA EST.", "PATER TUUS AGRICOLA EST?")
    assert not echoes("EGO CANEM HABEO.", "OPTIME! QUOD NOMEN EST CANI TUO?")
    assert not echoes("QUID AGIS?", "BENE AGO. ET TU, QUID AGIS?")
