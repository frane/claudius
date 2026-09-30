import math

from claudius.evaluate import INCIPITS, InterpolatedBigram, MemorisationIndex
from claudius.normalize import ALPHABET, is_normalized

LATIN = (
    "GALLIA EST OMNIS DIUISA IN PARTES TRES, QUARUM UNAM INCOLUNT BELGAE, "
    "ALIAM AQUITANI, TERTIAM QUI IPSORUM LINGUA CELTAE, NOSTRA GALLI APPELLANTUR. "
    "HI OMNES LINGUA, INSTITUTIS, LEGIBUS INTER SE DIFFERUNT. "
) * 40

OTHER = (
    "ARMA UIRUMQUE CANO, TROIAE QUI PRIMUS AB ORIS ITALIAM FATO PROFUGUS "
    "LAUINIAQUE UENIT LITORA, MULTUM ILLE ET TERRIS IACTATUS ET ALTO. "
) * 40


def test_incipits_are_normalised():
    assert all(is_normalized(i) for i in INCIPITS)
    assert INCIPITS[0].startswith("GALLIA EST OMNIS DIUISA")


def test_floor_beats_uniform():
    floor = InterpolatedBigram().fit([LATIN])
    bpb = floor.bits_per_byte([LATIN])
    assert 0 < bpb < math.log2(len(set(ALPHABET)))


def test_floor_is_worse_on_unseen_text():
    floor = InterpolatedBigram().fit([LATIN])
    assert floor.bits_per_byte([OTHER]) > floor.bits_per_byte([LATIN])


def test_floor_requires_fit():
    try:
        InterpolatedBigram().bits_per_byte([LATIN])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_memorisation_detects_verbatim_copy():
    index = MemorisationIndex(n=8).add(LATIN)
    res = index.check(generated=[LATIN[:600]], control=[OTHER])
    assert res.rate > 0.9
    assert res.control_rate == 0.0


def test_memorisation_clean_generation():
    index = MemorisationIndex(n=13).add(LATIN)
    res = index.check(generated=[OTHER], control=[OTHER])
    assert res.rate == 0.0


def test_memorisation_report_shape():
    index = MemorisationIndex(n=8).add(LATIN)
    d = index.check([LATIN[:600]], [OTHER]).as_dict()
    assert d["ngram"] == 8 and d["generated_verbatim"] > 0


def test_repeat_rate_detects_loops():
    from claudius.generate import repeat_rate

    assert repeat_rate("QUID SIBI UULT QUID SIBI UULT QUID SIBI UULT") > 0.5
    assert repeat_rate("GALLIA EST OMNIS DIUISA IN PARTES TRES") == 0.0


def test_clean_output_restores_space_after_punctuation():
    from claudius.generate import clean_output

    assert clean_output("RABIE,IN TAM") == "RABIE, IN TAM"
    assert clean_output("ANNO 1642.5") == "ANNO 1642.5"


def test_incipits_use_anti_loop_settings_by_default():
    import inspect

    from claudius import evaluate

    assert "ANTI_LOOP" in inspect.getsource(evaluate.sample_incipits)
