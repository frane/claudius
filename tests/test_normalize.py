import pytest

from claudius.normalize import ALPHABET, LETTERS, fold_token, is_normalized, normalize


def test_classical_alphabet_only():
    raw = "Iūlius Cæsar, «Dē Bellō Gallicō» — ſcripsit anno MDCXLII (1642)!"
    out = normalize(raw)
    assert is_normalized(out), out
    assert "J" not in out and "V" not in out and "W" not in out
    assert not any(c.islower() for c in out)


def test_v_and_j_fold():
    assert normalize("Vir iustus, Julius vivit") == "UIR IUSTUS, IULIUS UIUIT"


def test_ligatures_and_long_s():
    assert normalize("Cæſar Œconomia") == "CAESAR OECONOMIA"


def test_diacritics_stripped():
    assert normalize("praeclārē éxīstimō") == "PRAECLARE EXISTIMO"


def test_hyphenation_joined():
    assert normalize("propo-\nsitum est") == "PROPOSITUM EST"
    # A hyphen that is not at a line break survives.
    assert "-" in normalize("prae-clarus")


def test_greek_dropped():
    out = normalize("dicit λόγος esse uerbum")
    assert out == "DICIT ESSE UERBUM"


def test_digits_and_punctuation_survive():
    out = normalize("Anno 1642; caput 3: 'quid?'")
    assert "1642" in out and ";" in out and ":" in out and "'" in out


def test_editorial_symbols_removed():
    assert "§" not in normalize("§ 12 caput")
    assert "†" not in normalize("† obiit anno")


def test_idempotent():
    raw = "Iūlius Cæſar — vī et armīs, λόγος, propo-\nsitum §3"
    once = normalize(raw)
    assert normalize(once) == once


def test_empty():
    assert normalize("") == ""
    assert normalize("   \n\n ") == ""


def test_fold_token_matches_normalize():
    assert fold_token("Jūlius") == "IULIUS"
    assert fold_token("vērō") == "UERO"
    assert fold_token("123") == ""


def test_alphabet_is_23_letters():
    assert len(LETTERS) == 23
    assert set("JVW").isdisjoint(LETTERS)
    assert set(ALPHABET) >= set(LETTERS)


@pytest.mark.parametrize("text", ["ELLIPSIS...", "DASH---DASH", "QUID???"])
def test_repeated_punctuation_bounded(text):
    out = normalize(text)
    assert "...." not in out


def test_missing_space_after_punctuation_is_restored():
    """OCR drops it ~4,000x per million chars; the 400M learned 'RABIE,IN'."""
    assert normalize("conueniat,cuius") == "CONUENIAT, CUIUS"
    assert normalize("probandum;atque") == "PROBANDUM; ATQUE"
    assert normalize("M.Tullius") == "M. TULLIUS"
    assert normalize("anno 1642.5") == "ANNO 1642.5", "decimals must be untouched"
    assert normalize("ita. sed") == "ITA. SED", "an existing space is not doubled"


def test_dropped_foreign_word_leaves_no_double_separator():
    from claudius.normalize import normalize

    out = normalize("Wolfgangum, Ἀθῆναι, «dīxit»")
    assert ",," not in out and out.startswith("UUOLFGANGUM,")
    assert normalize(out) == out
