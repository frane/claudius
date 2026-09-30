from claudius.filters import (
    FilterConfig,
    clean_lines,
    drop_foreign_blocks,
    greek_ratio,
    is_latin,
    latin_score,
    line_is_text,
    prepare_document,
    rival_score,
)

LATIN = (
    "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, "
    "aliam Aquitani, tertiam qui ipsorum lingua Celtae, nostra Galli appellantur. "
    "Hi omnes lingua, institutis, legibus inter se differunt. Gallos ab Aquitanis "
    "Garumna flumen, a Belgis Matrona et Sequana dividit. Horum omnium fortissimi "
    "sunt Belgae, propterea quod a cultu atque humanitate provinciae longissime absunt. "
) * 6

FRENCH = (
    "La Gaule est divisee en trois parties, dont l'une est habitee par les Belges, "
    "l'autre par les Aquitains, et la troisieme par ceux qui dans leur langue se "
    "nomment Celtes et dans la notre Gaulois. Tous ces peuples different entre eux "
    "par le langage, les institutions et les lois. "
) * 6


def test_latin_scores_above_threshold():
    assert latin_score(LATIN) > 0.10


GERMAN = (
    "Gallien ist in drei Teile geteilt, von denen der eine von den Belgern "
    "bewohnt wird, der andere von den Aquitaniern, der dritte von denen, die in "
    "ihrer Sprache Kelten und in der unseren Gallier heissen. "
) * 6

ENGLISH = (
    "All Gaul is divided into three parts, one of which the Belgae inhabit, the "
    "Aquitani another, and the third those who in their own language are called "
    "Celts, but in ours Gauls. "
) * 6


def test_french_shares_stopwords_with_latin():
    # The bare stopword ratio is not enough: French clears the 0.06 threshold.
    assert latin_score(FRENCH) > FilterConfig().min_latin


def test_rival_check_rejects_french():
    assert rival_score(FRENCH) > latin_score(FRENCH)
    assert not is_latin(FRENCH)


def test_rival_check_rejects_german_and_english():
    assert not is_latin(GERMAN)
    assert not is_latin(ENGLISH)


def test_latin_accepted_by_the_full_rule():
    assert is_latin(LATIN)
    assert rival_score(LATIN) < 0.02


def test_french_preface_is_cut_and_the_latin_kept():
    mixed = FRENCH + "\n\n" + LATIN * 4
    out = prepare_document(mixed)
    assert out is not None
    assert out.startswith("GALLIA EST OMNIS DIUISA")
    assert "GAULE" not in out and "PARTIES" not in out


def test_german_footnotes_are_cut():
    mixed = LATIN * 4 + "\n\n" + GERMAN
    out = prepare_document(mixed)
    assert out is not None
    assert "GALLIEN" not in out


def test_short_text_scores_zero():
    assert latin_score("et in est") == 0.0


def test_greek_ratio():
    assert greek_ratio("λόγος λόγος λόγος") > 0.5
    assert greek_ratio(LATIN) == 0.0


def test_line_filtering():
    assert line_is_text("Gallia est omnis divisa")
    assert not line_is_text("12")
    assert not line_is_text("|| .. --- 34 ||")
    assert not line_is_text("")


def test_clean_lines_keeps_paragraphs():
    text = "Gallia est omnis divisa\n|| 34 ||\n\nHi omnes lingua differunt"
    out = clean_lines(text)
    assert "34" not in out
    assert out.count("\n\n") == 1


def test_prepare_document_accepts_latin():
    out = prepare_document(LATIN)
    assert out is not None
    assert out.startswith("GALLIA EST OMNIS DIUISA")


def test_prepare_document_rejects_french():
    assert prepare_document(FRENCH) is None


def test_prepare_document_rejects_german():
    assert prepare_document(GERMAN) is None


def test_prepare_document_rejects_short():
    assert prepare_document("Gallia est omnis divisa in partes tres.") is None


def test_prepare_document_rejects_greek_text():
    greek = "λόγος περι της φυσεως " * 400 + LATIN[:500]
    assert prepare_document(greek) is None


def test_drop_foreign_blocks_keeps_short_lines():
    text = "CAPUT I\n\n" + LATIN
    assert drop_foreign_blocks(text).startswith("CAPUT I")


def test_drop_foreign_blocks_on_line_separated_text():
    text = "\n".join([LATIN[:200], FRENCH[:200], LATIN[200:400]])
    out = drop_foreign_blocks(text)
    assert "Gaule" not in out
    assert "Gallia" in out


def test_sample_text_skips_front_matter():
    from claudius.filters import sample_text

    front = "TITLE PAGE LIBRARY PLATE GOOGLE SCAN ARTIFACT " * 400
    body = LATIN * 40
    cfg = FilterConfig()
    sampled = sample_text(front + body, cfg)
    assert len(sampled) <= cfg.sample_chars + cfg.sample_slices
    assert "Gallia" in sampled


def test_head_only_front_matter_no_longer_sinks_the_score():
    # A scanned book: garbled front matter, then a long Latin body. Scoring the
    # head rejects it; scoring throughout the body keeps it.
    front = "Goo< itized by X Bibliotheca ori ColL Rom ^^ociet Jesu 4 " * 350
    doc = front + LATIN * 30
    assert latin_score(doc[:20_000]) < FilterConfig().min_latin
    assert latin_score(doc) > FilterConfig().min_latin
    assert is_latin(doc)


def test_short_documents_are_scored_whole():
    from claudius.filters import sample_text

    assert sample_text(LATIN[:500], FilterConfig()) == LATIN[:500]
