import pytest

from claudius.normalize import normalize
from claudius.tokenizer import (
    Fertility,
    measure_fertility,
    pick_elbow,
    sweep,
    train_bpe,
    truncate_bpe,
)

CORPUS = [
    normalize(
        "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, "
        "aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur. "
        "Hi omnes lingua, institutis, legibus inter se differunt."
    ),
    normalize(
        "Arma virumque cano, Troiae qui primus ab oris Italiam fato profugus "
        "Laviniaque venit litora, multum ille et terris iactatus et alto."
    ),
    normalize(
        "In principio creavit Deus caelum et terram. Terra autem erat inanis et vacua, "
        "et tenebrae erant super faciem abyssi."
    ),
] * 40


@pytest.fixture(scope="module")
def base():
    return train_bpe(CORPUS, vocab_size=800, min_frequency=1)


def test_roundtrip(base):
    text = CORPUS[0]
    assert base.decode(base.encode(text).ids) == text


def test_truncation_is_a_prefix_of_the_larger_vocabulary(base):
    small = truncate_bpe(base, 400)
    assert small.get_vocab_size() == 400
    big_vocab = base.get_vocab()
    for token, idx in small.get_vocab().items():
        assert big_vocab[token] == idx


def test_truncation_still_roundtrips(base):
    small = truncate_bpe(base, 400)
    text = CORPUS[1]
    assert small.decode(small.encode(text).ids) == text


def test_truncation_raises_when_too_large(base):
    with pytest.raises(ValueError):
        truncate_bpe(base, 10_000)


def test_smaller_vocabulary_is_never_more_efficient(base):
    big = measure_fertility(base, CORPUS[:3], vocab_size=800)
    small = measure_fertility(truncate_bpe(base, 400), CORPUS[:3], vocab_size=400)
    assert small.tokens_per_word >= big.tokens_per_word


def test_sweep_returns_one_row_per_size():
    results, chosen, base = sweep(CORPUS, CORPUS[:3], sizes=(300, 400, 450), min_frequency=1)
    assert [r.vocab_size for r in results][:3] == [300, 400, 450]
    assert chosen in [r.vocab_size for r in results]
    # The trained tokenizer comes back so the caller need not retrain it.
    assert base.get_vocab_size() == 450


def test_sweep_clamps_to_the_merges_the_corpus_actually_yields():
    # The requested ceiling is unreachable on this sample; the sweep must report
    # the vocabulary that was actually learned instead of failing.
    results, chosen, base = sweep(CORPUS, CORPUS[:3], sizes=(300, 50_000), min_frequency=1)
    assert max(r.vocab_size for r in results) < 50_000
    assert chosen in [r.vocab_size for r in results]
    assert base.get_vocab_size() == max(r.vocab_size for r in results)


def test_pick_elbow_stops_when_gains_flatten():
    results = [
        Fertility(vocab_size=4096, tokens=200, words=100, bytes=600),   # 2.00
        Fertility(vocab_size=8192, tokens=150, words=100, bytes=600),   # 1.50
        Fertility(vocab_size=16384, tokens=148, words=100, bytes=600),  # 1.48
        Fertility(vocab_size=32768, tokens=147, words=100, bytes=600),
    ]
    assert pick_elbow(results, min_gain=0.03) == 8192


def test_pick_elbow_takes_the_largest_when_gains_persist():
    results = [
        Fertility(vocab_size=4096, tokens=400, words=100, bytes=600),
        Fertility(vocab_size=8192, tokens=200, words=100, bytes=600),
        Fertility(vocab_size=16384, tokens=100, words=100, bytes=600),
    ]
    assert pick_elbow(results, min_gain=0.03) == 16384


def test_load_reports_a_missing_directory_clearly(tmp_path):
    from claudius.tokenizer import load

    with pytest.raises(SystemExit) as e:
        load(tmp_path / "not-there")
    assert "no tokenizer directory" in str(e.value)


def test_load_reports_an_incomplete_directory(tmp_path):
    from claudius.tokenizer import load

    (tmp_path / "half").mkdir()
    with pytest.raises(SystemExit) as e:
        load(tmp_path / "half")
    assert "tokenizer.json" in str(e.value)


def test_find_elbow_returns_none_when_the_curve_never_flattens():
    """The Latin case: every doubling still pays, so there is no elbow."""
    from claudius.tokenizer import find_elbow, format_sweep, pick_elbow

    # Measured on clean Latin: 9.9%, 8.3%, 7.0%, 5.6% gains per doubling.
    latin = [
        Fertility(vocab_size=4096, tokens=1914, words=1000, bytes=7030),
        Fertility(vocab_size=8192, tokens=1725, words=1000, bytes=7030),
        Fertility(vocab_size=16384, tokens=1582, words=1000, bytes=7030),
        Fertility(vocab_size=32768, tokens=1472, words=1000, bytes=7030),
        Fertility(vocab_size=65536, tokens=1389, words=1000, bytes=7030),
    ]
    assert find_elbow(latin) is None
    # pick_elbow still returns a number, but the report must not call it a choice.
    assert pick_elbow(latin) == 65536
    report = format_sweep(latin, pick_elbow(latin))
    assert "NO ELBOW" in report
    assert "not a choice" in report


def test_card_records_whether_an_elbow_was_found():
    from claudius.tokenizer import tokenizer_card

    flat = [
        Fertility(vocab_size=4096, tokens=2000, words=1000, bytes=6000),
        Fertility(vocab_size=8192, tokens=1500, words=1000, bytes=6000),
        Fertility(vocab_size=16384, tokens=1490, words=1000, bytes=6000),
    ]
    card = tokenizer_card(flat, 8192)
    assert card["elbow_found"] is True

    steep = [
        Fertility(vocab_size=4096, tokens=2000, words=1000, bytes=6000),
        Fertility(vocab_size=8192, tokens=1500, words=1000, bytes=6000),
        Fertility(vocab_size=16384, tokens=1000, words=1000, bytes=6000),
    ]
    card = tokenizer_card(steep, 16384)
    assert card["elbow_found"] is False
    assert "rather than a decision" in card["elbow_note"]


def test_embedding_cost_report():
    from claudius.model import SIZES, param_count
    from claudius.tokenizer import embedding_cost

    p400 = param_count(SIZES["400m"], 32_000)
    assert "10%" in embedding_cost(32_000, 1024, p400)
    # At 64k the table is nearly a fifth of the on-device model.
    p400_64 = param_count(SIZES["400m"], 65_536)
    assert "18%" in embedding_cost(65_536, 1024, p400_64)
