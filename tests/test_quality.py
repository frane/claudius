from claudius.quality import church_rate, fused_rate, number_rate, strip_edition_numbers


def test_strips_line_and_section_numbers_but_not_roman_numerals():
    text = "NEC PATER EX ILLO RERUM DECEPTUS AB AEUO 355\n12 SED PATER [...] UT SECUM GERERET.\nLIBER IV [3]"
    out = strip_edition_numbers(text)
    assert "355" not in out and "12" not in out and "[...]" not in out and "3" not in out
    assert "LIBER IV" in out
    assert "SED PATER UT SECUM GERERET." in out


def test_sentence_with_a_content_number_is_dropped_not_holed():
    text = "AUGUSTUS REGNAUIT. INTER ANNOS 27 ET 14 A.C. MULTA FECIT. ROMA FLORUIT."
    out = strip_edition_numbers(text)
    assert "INTER ANNOS" not in out
    assert out == "AUGUSTUS REGNAUIT. MULTA FECIT. ROMA FLORUIT."


def test_no_digit_survives():
    for text in ("A3B CAESAR", "ANNO 1644 NATUS EST.", "CAPUT 12\nGALLIA EST OMNIS DIUISA."):
        assert not any(c.isdigit() for c in strip_edition_numbers(text))


def test_number_rate():
    assert number_rate("ARMA UIRUMQUE CANO 1 TROIAE QUI PRIMUS 2") == 1000 * 2 / 6


def test_church_rate_separates_registers():
    sermon = "DOMINUS DEUS NOSTER ET SANCTA ECCLESIA CHRISTI PECCATA DIMITTIT"
    caesar = "GALLIA EST OMNIS DIUISA IN PARTES TRES QUARUM UNAM INCOLUNT BELGAE"
    assert church_rate(sermon) > 500 and church_rate(caesar) == 0


def test_fused_rate_counts_glued_long_words():
    freq = {"UIDEO": 5000, "PUELLAM": 3000, "IMPERATOR": 5000}
    assert fused_rate("UIDEOPUELLAM IMPERATOR", freq) == 0.5
    assert fused_rate("IMPERATOR", freq) == 0.0


def test_strips_numbers_glued_to_words():
    out = strip_edition_numbers("EST SCIENTIA NATURALIS.N. 4SED QUIA EA QUAE")
    assert out == "EST SCIENTIA NATURALIS. SED QUIA EA QUAE"
