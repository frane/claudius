import random

from claudius.dedup import Deduper, MinHasher, _bands, jaccard, shingles

random.seed(7)
WORDS = "GALLIA EST OMNIS DIUISA IN PARTES TRES QUARUM UNAM INCOLUNT BELGAE ALIAM AQUITANI".split()


def doc(n=400, seed=0):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n))


def test_shingles_stable():
    a = shingles("GALLIA EST OMNIS DIUISA IN PARTES TRES")
    b = shingles("GALLIA EST OMNIS DIUISA IN PARTES TRES")
    assert a.size == 3 and (a == b).all()


def test_shingles_too_short():
    assert shingles("GALLIA EST").size == 0


def test_identical_documents_have_identical_signatures():
    h = MinHasher(num_perm=64)
    text = doc(seed=1)
    assert jaccard(h.signature(text), h.signature(text)) == 1.0


def test_different_documents_differ():
    h = MinHasher(num_perm=64)
    assert jaccard(h.signature(doc(seed=1)), h.signature(doc(seed=2))) < 0.5


def test_bands_factorise_num_perm():
    b, r = _bands(128, 0.8)
    assert b * r == 128


def test_exact_duplicate_dropped():
    d = Deduper(num_perm=64)
    text = doc(seed=3)
    assert d.add(text) is not None
    assert d.add(text) is None
    assert d.stats.dup_docs == 1


def test_near_duplicate_dropped():
    d = Deduper(num_perm=128, threshold=0.7)
    text = doc(1200, seed=4)
    words = text.split()
    # Simulate an OCR variant: perturb 3% of the tokens.
    rng = random.Random(11)
    for i in range(0, len(words), 33):
        words[i] = rng.choice(WORDS)
    assert d.add(text) is not None
    assert d.add(" ".join(words)) is None


def test_distinct_documents_kept():
    d = Deduper(num_perm=64)
    assert d.add(doc(seed=5)) is not None
    assert d.add(doc(seed=6)) is not None
    assert d.stats.kept == 2


def test_boilerplate_paragraph_stripped():
    d = Deduper(num_perm=64)
    plate = "EX BIBLIOTHECA REGIA MONACENSI " * 20
    first = plate + "\n\n" + doc(seed=7)
    second = plate + "\n\n" + doc(seed=8)
    assert d.add(first) is not None
    kept = d.add(second)
    assert kept is not None
    assert "BIBLIOTHECA" not in kept
    assert d.stats.dup_paragraphs == 1


def test_short_repeated_paragraphs_kept():
    d = Deduper(num_perm=64)
    assert d.add("CAPUT I\n\n" + doc(seed=9)) is not None
    kept = d.add("CAPUT I\n\n" + doc(seed=10))
    assert kept.startswith("CAPUT I")


def test_unhashable_document():
    d = Deduper(num_perm=64)
    assert d.add("ET IN") is None
    assert d.stats.empty == 1


# -- parallel path: fingerprint now, decide later ---------------------------
def test_fingerprint_matches_streaming_deduper():
    from claudius.dedup import GlobalDedup, MinHasher, apply_paragraph_drops, fingerprint

    plate = "EX BIBLIOTHECA REGIA MONACENSI " * 20
    docs = [doc(seed=20), doc(seed=20), plate + "\n\n" + doc(seed=21),
            plate + "\n\n" + doc(seed=22), "ET IN"]

    streaming = Deduper(num_perm=64)
    stream_out = [streaming.add(d) for d in docs]

    hasher = MinHasher(num_perm=64)
    g = GlobalDedup(num_perm=64)
    parallel_out = []
    for d in docs:
        keep, drop = g.decide(fingerprint(d, hasher))
        parallel_out.append(apply_paragraph_drops(d, drop) if keep else None)

    assert [o is None for o in stream_out] == [o is None for o in parallel_out]
    assert stream_out == parallel_out
    assert g.stats.kept == streaming.stats.kept
    assert g.stats.dup_docs == streaming.stats.dup_docs
    assert g.stats.dup_paragraphs == streaming.stats.dup_paragraphs


def test_fingerprint_is_position_aligned():
    from claudius.dedup import paragraph_hashes

    long_para = "A" * 300
    text = "\n\n".join(["short", long_para, "also short", long_para])
    h = paragraph_hashes(text)
    assert len(h) == 4
    assert h[0] == 0 and h[2] == 0, "short paragraphs are never deduplicated"
    assert h[1] == h[3] != 0


def test_apply_paragraph_drops():
    from claudius.dedup import apply_paragraph_drops

    text = "one\n\ntwo\n\nthree"
    assert apply_paragraph_drops(text, {1}) == "one\n\nthree"
    assert apply_paragraph_drops(text, set()) == text


def test_global_dedup_is_order_deterministic():
    from claudius.dedup import GlobalDedup, MinHasher, fingerprint

    hasher = MinHasher(num_perm=64)
    docs = [doc(seed=s) for s in (30, 30, 31)]
    fps = [fingerprint(d, hasher) for d in docs]
    a = [GlobalDedup(num_perm=64).decide(f)[0] for f in fps]
    g = GlobalDedup(num_perm=64)
    b = [g.decide(f)[0] for f in fps]
    assert b == [True, False, True]
    assert a[0] is True


def test_banding_catches_pairs_at_the_stated_threshold():
    """The bug: banding targeted a 50% coin flip at the threshold.

    128 permutations at 0.8 chose 8x16, which catches 20.4% of true
    near-duplicates. The deduper claimed threshold 0.8 and behaved like 0.9.
    """
    from claudius.dedup import _bands

    bands, rows = _bands(128, 0.8)
    recall = 1.0 - (1.0 - 0.8**rows) ** bands
    assert recall >= 0.9, f"{bands}x{rows} catches only {recall:.1%} at J=0.8"
    assert (bands, rows) == (16, 8)


def test_banding_is_the_cheapest_that_meets_recall():
    """More rows = fewer candidate pairs to verify. Take the largest that works."""
    from claudius.dedup import _bands

    for threshold in (0.7, 0.8, 0.9):
        bands, rows = _bands(128, threshold)
        assert bands * rows == 128
        assert 1.0 - (1.0 - threshold**rows) ** bands >= 0.9
        # One more row would fall short.
        bigger = [r for r in (rows * 2,) if 128 % r == 0]
        for r in bigger:
            assert 1.0 - (1.0 - threshold**r) ** (128 // r) < 0.9
