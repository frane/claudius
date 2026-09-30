import pytest

from claudius.segment import split_word

# A small stand-in lexicon with the frequencies that matter.
F = {"DEDECORE": 614, "ATQUE": 157573, "CAPTIUORUM": 661, "QUISPIAM": 811,
     "NON": 900000, "EGREDITUR": 900, "AD": 800000, "DIUERSOS": 700,
     "MITIGATUS": 300, "MITIGATA": 200, "MITIGAT": 180, "OS": 5000,
     "RESPUBLICA": 418, "RES": 90000, "PUBLICA": 8000, "QUE": 3000}


@pytest.mark.parametrize("fused,parts", [
    ("DEDECOREATQUE", ["DEDECORE", "ATQUE"]),
    ("CAPTIUORUMQUISPIAM", ["CAPTIUORUM", "QUISPIAM"]),
    ("NONEGREDITUR", ["NON", "EGREDITUR"]),
    ("ADDIUERSOS", ["AD", "DIUERSOS"]),
])
def test_splits_real_fusions(fused, parts):
    assert split_word(fused, F) == parts


def test_does_not_split_an_inflection_of_a_common_word():
    """MITIGATOS is a participle, not MITIGAT + OS."""
    assert split_word("MITIGATOS", F) == ["MITIGATOS"]


def test_does_not_split_a_common_compound():
    assert split_word("RESPUBLICA", F) == ["RESPUBLICA"]


def test_never_splits_off_an_enclitic():
    """DEDECOREQUE is correct Latin: dedecore + -que."""
    assert split_word("DEDECOREQUE", F) == ["DEDECOREQUE"]


def test_two_letter_halves_must_be_function_words():
    f = {**F, "BA": 400, "LATRONIBUS": 300}
    assert split_word("BALATRONIBUS", f) == ["BALATRONIBUS"]
