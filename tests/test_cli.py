import pytest

from claudius.cli import build_parser, main


def test_every_subcommand_parses():
    p = build_parser()
    for cmd in ("sources", "normalize", "filter-report", "prepare", "tokenizer",
                "pack", "shapes", "plan", "train", "eval"):
        args = p.parse_args([cmd])
        assert callable(args.func)


def test_defaults_match_the_brief():
    p = build_parser()
    assert p.parse_args(["prepare"]).min_latin == 0.06
    assert p.parse_args(["pack"]).seq_len == 2048
    assert p.parse_args(["train"]).lr == 3e-4
    assert p.parse_args(["eval"]).ngram == 13
    assert p.parse_args(["tokenizer"]).sweep == [4096, 8192, 16384, 32768, 65536]


def test_unknown_source_is_rejected():
    with pytest.raises(SystemExit):
        main(["prepare", "--source", "no-such-corpus", "--out", "/tmp/nope"])


def test_local_source_shorthand(tmp_path):
    from claudius.cli import _sources

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("GALLIA")
    srcs = _sources([f"mine={tmp_path}"])
    assert srcs[0].name == "mine" and srcs[0].kind == "textdir"


def test_plan_without_data(capsys):
    main(["plan", "--size", "1b", "--tokens", "10000000000", "--vocab-size", "32000"])
    out = capsys.readouterr().out
    assert '"steps": 19073' in out
    assert '"params": 1034512384' in out


def test_shapes_output(capsys):
    main(["shapes"])
    out = capsys.readouterr().out
    assert "1,034,512,384" in out


def test_normalize_stdin(monkeypatch, capsys, tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("Iūlius Cæſar vīcit", encoding="utf-8")
    main(["normalize", str(path)])
    assert capsys.readouterr().out.strip() == "IULIUS CAESAR UICIT"


def test_prepare_accepts_shards():
    assert build_parser().parse_args(["prepare"]).shards == 0
    assert build_parser().parse_args(["prepare", "--shards", "24"]).shards == 24


def test_spread_expands_hub_sources_one_per_shard(monkeypatch):
    import claudius.cli as cli
    from claudius.corpora import LATIN_PD, Source

    monkeypatch.setattr(cli.corpora, "spread_files",
                        lambda repo, n: tuple(f"s{i}.parquet" for i in range(n)))
    local = Source(name="local", kind="textdir", path="/tmp/x")
    out = cli._spread([LATIN_PD, local], 4)
    assert len(out) == 5
    assert [s.data_files for s in out[:4]] == [("s0.parquet",), ("s1.parquet",),
                                               ("s2.parquet",), ("s3.parquet",)]
    assert out[4] is local, "local sources are not sharded"


def test_spread_is_a_noop_when_disabled():
    import claudius.cli as cli
    from claudius.corpora import LATIN_PD

    assert cli._spread([LATIN_PD], 0) == [LATIN_PD]


def test_no_job_script_shadows_a_stdlib_module():
    """A job script sits first on sys.path inside the job container.

    A file named `tokenize.py` shadows the standard library's `tokenize`, which
    `linecache` and `inspect` import internally -- the job then dies with
    `AttributeError: module 'tokenize' has no attribute 'open'` from inside an
    unrelated dependency, with nothing in the traceback pointing at the script.
    """
    import pathlib
    import sys

    jobs = pathlib.Path(__file__).resolve().parent.parent / "jobs"
    stdlib = set(sys.stdlib_module_names)
    clashes = [p.name for p in jobs.glob("*.py") if p.stem in stdlib]
    assert clashes == [], f"job scripts shadow stdlib modules: {clashes}"


def test_memorisation_scores_continuations_not_prompts():
    """Every incipit is a verbatim classical quotation.

    Scoring prompt+continuation made the annealed 400M look like it memorised
    the Aeneid (6-gram overlap 7.1% vs 0.1% control); continuations alone
    scored 0.7%. The prompt was matching, not the model.
    """
    import inspect
    import pathlib

    from claudius import cli

    src = inspect.getsource(cli.cmd_eval)
    job = (pathlib.Path(__file__).resolve().parent.parent / "jobs" / "train.py").read_text()
    for code in (src, job):
        assert 's["prompt"] + s["continuation"]' not in code
        assert '[s["continuation"] for s in samples]' in code
