from claudius.corpora import (
    HOLDOUT_FRACTION,
    Document,
    Source,
    holdout_bucket,
    interleave,
    is_holdout,
)


def test_holdout_is_stable_and_case_insensitive():
    assert holdout_bucket("Cicero") == holdout_bucket("  cicero ")


def test_holdout_fraction_is_approximately_right():
    authors = [f"author-{i}" for i in range(20_000)]
    held = sum(is_holdout(a) for a in authors)
    assert 0.5 * HOLDOUT_FRACTION < held / len(authors) < 1.5 * HOLDOUT_FRACTION


def test_authors_never_straddle_the_split():
    author = next(a for a in (f"a{i}" for i in range(1000)) if is_holdout(a))
    assert all(is_holdout(author) for _ in range(5))


def test_blank_author_is_never_held_out():
    assert not is_holdout("")
    assert not is_holdout("   ")


def _fake(name, n, weight, repeat):
    src = Source(name=name, kind="fake", path="", weight=weight, repeat=repeat)
    return src, [Document(text=f"{name}-{i}", source=name) for i in range(n)]


def test_interleave_respects_weights(monkeypatch):
    import claudius.corpora as c

    big, big_docs = _fake("big", 100, 1.0, False)
    small, small_docs = _fake("small", 5, 4.0, True)
    pools = {"big": big_docs, "small": small_docs}
    monkeypatch.setattr(c, "iter_source", lambda s: iter(list(pools[s.name])))

    out = list(c.interleave([big, small]))
    counts = {"big": 0, "small": 0}
    for d in out:
        counts[d.source] += 1
    assert counts["big"] == 100
    # weight 4 vs 1 -> the repeating source should appear roughly 4x as often.
    assert counts["small"] > 2 * counts["big"]


def test_interleave_terminates_with_only_repeating_sources(monkeypatch):
    import claudius.corpora as c

    src, docs = _fake("only", 3, 2.0, True)
    monkeypatch.setattr(c, "iter_source", lambda s: iter(list(docs)))
    assert len(list(c.interleave([src]))) == 3


def test_interleave_empty():
    assert list(interleave([])) == []


def test_spread_files_picks_evenly_across_shards(monkeypatch):
    import huggingface_hub

    from claudius.corpora import spread_files

    names = [f"latin_pd_{i}.parquet" for i in range(118)] + ["README.md"]

    class FakeApi:
        def list_repo_files(self, repo, repo_type=None):
            return names

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    picked = spread_files("PleIAs/Latin-PD", 24)
    assert len(picked) == 24
    assert len(set(picked)) == 24
    assert all(p.endswith(".parquet") for p in picked)
    # Spread, not the first 24 shards of the sorted list.
    assert picked != tuple(sorted(n for n in names if n.endswith(".parquet"))[:24])


def test_spread_files_returns_everything_when_asked_for_more(monkeypatch):
    import huggingface_hub

    from claudius.corpora import spread_files

    class FakeApi:
        def list_repo_files(self, repo, repo_type=None):
            return ["a.parquet", "b.parquet"]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert spread_files("x", 24) == ("a.parquet", "b.parquet")


def test_interleave_does_not_collapse_sources_sharing_a_name(monkeypatch):
    """Spreading a sample across shards makes many sources with one name."""
    import claudius.corpora as c

    shards = {
        f"s{i}": [Document(text=f"shard{i}-{k}", source="latin-pd") for k in range(5)]
        for i in range(4)
    }
    srcs = [
        Source(name="latin-pd", kind="fake", path="", data_files=(f"s{i}",))
        for i in range(4)
    ]
    monkeypatch.setattr(c, "iter_source", lambda s: iter(list(shards[s.data_files[0]])))

    out = [d.text for d in c.interleave(srcs)]
    assert len(out) == 20
    assert {t.split("-")[0] for t in out} == {"shard0", "shard1", "shard2", "shard3"}
    # And they are drawn round-robin, not one shard drained at a time.
    assert {t.split("-")[0] for t in out[:4]} == {"shard0", "shard1", "shard2", "shard3"}


def test_pick_resolves_alternative_column_names():
    from claudius.corpora import _pick

    assert _pick({"identifier": "abc"}, ("identifier", "file_id")) == "abc"
    assert _pick({"file_id": "xyz"}, ("identifier", "file_id")) == "xyz"
    assert _pick({"identifier": "", "file_id": "xyz"}, ("identifier", "file_id")) == "xyz"
    assert _pick({}, ("identifier", "file_id")) == ""
    assert _pick({"a": 1642}, "a") == "1642"          # int64 dates coerce
    assert _pick({"a": "x"}, None) == ""


def test_latin_pd_accepts_both_identifier_schemas():
    from claudius.corpora import LATIN_PD, _pick

    assert _pick({"file_id": "bub_1"}, LATIN_PD.id_column) == "bub_1"
    assert _pick({"identifier": "bub_2"}, LATIN_PD.id_column) == "bub_2"


def test_hf_shards_uses_data_files_when_given():
    from claudius.corpora import Source, hf_shards

    src = Source(name="x", kind="hf", path="repo", data_files=("a.parquet",))
    assert hf_shards(src) == ["a.parquet"]


def test_iter_hf_streams_each_shard_separately(monkeypatch):
    """Shards must not be unified into one schema -- that raises CastError."""
    import claudius.corpora as c

    shards = {
        "s0.parquet": [{"identifier": "a", "text": "ALPHA", "creator": "Cicero"}],
        "s1.parquet": [{"file_id": "b", "text": "BETA", "creator": "Vergilius",
                        "__index_level_0__": 3}],
    }
    seen = []

    def fake_load_dataset(path, split=None, streaming=None, data_files=None):
        assert data_files is not None and len(data_files) == 1, "one shard at a time"
        seen.append(data_files[0])
        return iter(shards[data_files[0]])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(c, "hf_shards", lambda src: list(shards))

    docs = list(c._iter_hf(c.LATIN_PD))
    assert seen == ["s0.parquet", "s1.parquet"]
    assert [d.identifier for d in docs] == ["a", "b"]
    assert [d.author for d in docs] == ["Cicero", "Vergilius"]


def test_source_can_override_min_chars():
    """A threshold tuned for books must not govern an encyclopaedia."""
    import dataclasses

    from claudius.corpora import SOURCES
    from claudius.filters import FilterConfig

    assert SOURCES["vicipaedia"].min_chars == 500
    assert SOURCES["latin-pd"].min_chars is None, "books keep the corpus default"

    cfg = FilterConfig()
    override = dataclasses.replace(cfg, min_chars=SOURCES["vicipaedia"].min_chars)
    assert override.min_chars == 500
    assert cfg.min_chars == 2000, "the corpus-wide default is untouched"
