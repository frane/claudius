import gzip
import json

import pytest

from claudius.mixing import CorpusSpec, iter_mixed, mix_report, mixed_texts


def _corpus(root, name, n, text="GALLIA EST OMNIS DIUISA IN PARTES TRES"):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "train-00000.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"text": f"{text} {name} {i}", "author": f"a{i}",
                                 "identifier": f"{name}-{i}", "source": name}) + "\n")
    return CorpusSpec(path=str(d), name=name)


def test_reads_several_corpora(tmp_path):
    a = _corpus(tmp_path, "alpha", 10)
    b = _corpus(tmp_path, "beta", 10)
    labels = [lab for lab, _ in iter_mixed([a, b], limit=20)]
    assert set(labels) == {"alpha", "beta"}


def test_weights_control_the_share(tmp_path):
    big = CorpusSpec(**{**_corpus(tmp_path, "big", 200).__dict__, "weight": 1.0})
    small = CorpusSpec(**{**_corpus(tmp_path, "small", 20).__dict__, "weight": 4.0})
    labels = [lab for lab, _ in iter_mixed([big, small], limit=200)]
    share = labels.count("small") / len(labels)
    assert 0.65 < share < 0.85, share


def test_small_corpus_repeats_to_hold_its_share(tmp_path):
    big = _corpus(tmp_path, "big", 300)
    small = CorpusSpec(**{**_corpus(tmp_path, "small", 5).__dict__, "weight": 2.0})
    labels = [lab for lab, _ in iter_mixed([big, small], limit=150)]
    # Only 5 documents exist, so holding a 2:1 share requires repetition.
    assert labels.count("small") > 5


def test_stream_ends_when_finite_corpora_are_exhausted(tmp_path):
    a = CorpusSpec(**{**_corpus(tmp_path, "a", 7).__dict__, "repeat": False})
    b = CorpusSpec(**{**_corpus(tmp_path, "b", 3).__dict__, "repeat": False})
    assert len(list(iter_mixed([a, b]))) == 10


def test_mix_report_shares_sum_to_one(tmp_path):
    a = _corpus(tmp_path, "a", 50)
    b = CorpusSpec(**{**_corpus(tmp_path, "b", 50).__dict__, "weight": 3.0})
    rep = mix_report([a, b], sample=100)
    total = sum(v["share_of_chars"] for v in rep["by_corpus"].values())
    assert 0.99 < total < 1.01
    assert rep["by_corpus"]["b"]["share_of_chars"] > rep["by_corpus"]["a"]["share_of_chars"]


def test_mixed_texts_yields_strings(tmp_path):
    a = _corpus(tmp_path, "a", 4)
    assert all(isinstance(t, str) for t in mixed_texts([a], limit=4))


def test_empty_spec_list():
    assert list(iter_mixed([])) == []


@pytest.mark.parametrize("raw,path,weight", [
    ("data/corpus:4", "data/corpus", 4.0),
    ("data/corpus", "data/corpus", 1.0),
    ("/abs/path:2.5", "/abs/path", 2.5),
])
def test_parse_spec(raw, path, weight):
    s = CorpusSpec.parse(raw)
    assert s.path == path and s.weight == weight


def test_nested_corpus_is_found(tmp_path):
    """A corpus downloaded from the Hub may sit in a subdirectory."""
    from claudius.prepare import read_shards

    _corpus(tmp_path / "nested" / "deep", "c", 3)
    assert len(list(read_shards(tmp_path / "nested", "train"))) == 3


def test_tokenizer_stage_samples_the_mix_not_one_corpus(tmp_path):
    """A 30:1 corpus must not swamp the tokenizer's training sample."""
    from claudius.pipeline import _as_specs
    from claudius.mixing import mixed_texts

    _corpus(tmp_path, "huge", 300, text="OCR NOISE GARBLED TEXT")
    _corpus(tmp_path, "clean", 20, text="GALLIA EST OMNIS DIUISA")
    specs = _as_specs([f"{tmp_path}/huge:1", f"{tmp_path}/clean:6"])
    assert [s.weight for s in specs] == [1.0, 6.0]

    sample = list(mixed_texts(specs, "train", limit=120))
    clean = sum(1 for t in sample if "GALLIA" in t)
    assert clean / len(sample) > 0.6, clean / len(sample)


def test_as_specs_accepts_a_bare_path(tmp_path):
    from claudius.pipeline import _as_specs

    specs = _as_specs(str(tmp_path))
    assert len(specs) == 1 and specs[0].weight == 1.0


def test_document_weighting_is_not_character_weighting(tmp_path):
    """The trap: big documents swamp a document-weighted mix.

    Latin-PD documents are whole books (~800KB); clean documents average ~50KB.
    A 6:1 document weighting realised as 1:2.7 by characters on the real corpus.
    """
    big = CorpusSpec(**{**_corpus(tmp_path, "big", 60, text="X" * 4000).__dict__,
                        "weight": 1.0})
    small = CorpusSpec(**{**_corpus(tmp_path, "small", 60, text="Y" * 100).__dict__,
                          "weight": 6.0})
    rep = mix_report([big, small], sample=120)
    # Six times as many documents...
    assert rep["by_corpus"]["small"]["documents"] > rep["by_corpus"]["big"]["documents"] * 3
    # ...but the big corpus still dominates the text.
    assert rep["by_corpus"]["big"]["share_of_chars"] > 0.5


def test_by_chars_weighting_matches_the_requested_share(tmp_path):
    """`by_chars` divides out mean document size so the share is of text."""
    big = CorpusSpec(**{**_corpus(tmp_path, "big2", 60, text="X" * 4000).__dict__,
                        "weight": 1.0, "by_chars": True})
    small = CorpusSpec(**{**_corpus(tmp_path, "small2", 60, text="Y" * 100).__dict__,
                          "weight": 3.0, "by_chars": True})
    rep = mix_report([big, small], sample=150)
    share = rep["by_corpus"]["small2"]["share_of_chars"]
    assert share > 0.55, f"asked for 3:1 by chars, realised {share:.2f}"


def test_parse_accepts_the_chars_suffix():
    s = CorpusSpec.parse("data/clean:6:chars")
    assert s.path == "data/clean" and s.weight == 6.0 and s.by_chars is True
    s2 = CorpusSpec.parse("data/clean:6")
    assert s2.by_chars is False
