import random

from claudius.normalize import normalize
from claudius.sft.build import build, extract_sentences, summary


def test_praenomen_abbreviations_do_not_split_sentences():
    """P. is Publius, M. is Marcus -- not sentence ends.

    Splitting there turned 'Accusator erat eius P. Scipio' into the fragment
    'ACCUSATOR ERAT EIUS P.' and the orphan 'SCIPIO, UIR CLARUS.'
    """
    t = normalize("Accusator erat eius P. Scipio, vir clarus. "
                  "M. Tullius Cicero orationem habuit. Puella rosam amat.")
    got = list(extract_sentences([t], min_words=2))
    assert got == [
        "ACCUSATOR ERAT EIUS P. SCIPIO, UIR CLARUS.",
        "M. TULLIUS CICERO ORATIONEM HABUIT.",
        "PUELLA ROSAM AMAT.",
    ]


def test_apparatus_and_fragments_are_rejected():
    t = normalize("Vide cap. 12 supra. [lacuna] hic. Sine puncto finali")
    assert list(extract_sentences([t], min_words=2)) == []


def test_length_bounds():
    t = normalize("Ita. Puella rosam pulchram in horto amat. " + "Et " * 20 + "amat.")
    got = list(extract_sentences([t], min_words=4, max_words=12))
    assert got == ["PUELLA ROSAM PULCHRAM IN HORTO AMAT."]


def test_dataset_is_not_all_corrections():
    """A model trained only on corrections 'corrects' perfectly good Latin."""
    sents = [normalize(s) for s in (
        "Puella rosam pulchram amat.", "Magister discipulos bonos docet.",
        "Caesar Galliam in partes tres dividit.", "Milites urbem magnam oppugnant.",
    )]
    ex = build(sents, n=400, seed=3, correct_share=0.4)
    share = summary(ex)["by_kind"]["correct"] / len(ex)
    assert 0.3 < share < 0.5, share


def test_correct_examples_contain_no_correction():
    sents = [normalize("Puella rosam pulchram amat.")]
    for ex in build(sents, n=100, seed=1, correct_share=1.0):
        assert ex["kind"] == "correct"
        assert 'NON "' not in ex["text"], "a correct sentence was corrected"


def test_order_correction_does_not_stammer():
    """'BENE! BENE INTELLEGO' -- praise must appear once."""
    sents = [normalize("Magister discipulos bonos in schola docet.")]
    for ex in build(sents, n=200, seed=2, correct_share=0.0):
        if ex["kind"] == "order":
            reply = ex["text"].split("MAGISTER:")[-1]
            assert "BENE! BENE" not in reply, reply


def test_vulgate_source_never_reads_the_english_column(monkeypatch):
    """Every Vulgate row carries an English translation in `etext`.

    The project's premise is that the model has only ever seen Latin. The
    English column must not reach the output -- not even as a fallback.
    """
    import datasets

    from claudius.sft import build as b

    rows = [
        {"ltext": "in principio creavit Deus caelum et terram",
         "etext": "In the beginning God created heaven and earth"},
        {"ltext": "", "etext": "An English-only row that must be skipped entirely"},
        {"ltext": "dixitque Deus fiat lux et facta est lux",
         "etext": "And God said be light made"},
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: iter(rows))

    got = list(b.vulgate_sentences(min_words=2))
    assert got == ["IN PRINCIPIO CREAUIT DEUS CAELUM ET TERRAM.",
                   "DIXITQUE DEUS FIAT LUX ET FACTA EST LUX."]
    joined = " ".join(got)
    for english in ("THE", "BEGINNING", "CREATED", "HEAVEN", "SAID", "ENGLISH"):
        assert english not in joined.split(), f"English leaked: {english}"
