"""The whole pipeline on a synthetic corpus: prepare, tokenize, pack, train, eval.

Tiny -- a 5M model for two steps -- but it exercises every stage boundary, which
is where pipelines actually break.
"""

import json
import random

import pytest

from claudius.corpora import Source
from claudius.prepare import read_shards, read_texts, run as prepare_run

torch = pytest.importorskip("torch")

SENTENCES = [
    "Gallia est omnis divisa in partes tres quarum unam incolunt Belgae",
    "Arma virumque cano Troiae qui primus ab oris Italiam fato profugus venit",
    "In principio creavit Deus caelum et terram et terra erat inanis et vacua",
    "Quousque tandem abutere Catilina patientia nostra quamdiu etiam furor iste eludet",
    "Omnia vincit amor et nos cedamus amori atque ita res se habet",
    "Hi omnes lingua institutis legibus inter se differunt et de his rebus dicemus",
    "Senatus populusque Romanus decrevit ut consules rem publicam defenderent",
    "Non est ad astra mollis e terris via sed per aspera itur ad gloriam",
    "Beatus ille qui procul negotiis paterna rura bobus exercet suis",
    "Igitur cum de omnibus rebus quae ad rem pertinent dictum sit reliqua videamus",
]


def _make_corpus(root, n_authors=120):
    """120 authors so the 2% holdout has something to hold out, plus one
    deliberate reprint of an existing book under a different author."""
    rng = random.Random(4)
    first = None
    for a in range(n_authors):
        d = root / f"auctor{a:03d}"
        d.mkdir(parents=True, exist_ok=True)
        lines = [rng.choice(SENTENCES) for _ in range(120)]
        text = "\n".join(lines) + "\n"
        (d / "liber.txt").write_text(text, encoding="utf-8")
        if first is None:
            first = text
    (root / "auctor000" / "reprint.txt").write_text(first, encoding="utf-8")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    _make_corpus(root)
    out = tmp_path_factory.mktemp("work") / "corpus"
    src = Source(name="synthetic", kind="textdir", path=str(root))
    stats = prepare_run([src], out, log_every=0)
    assert stats.kept_train > 0
    return out, stats


def test_prepare_writes_shards_and_stats(corpus):
    out, stats = corpus
    assert list(out.glob("train-*.jsonl.gz"))
    written = json.loads((out / "stats.json").read_text())
    assert written["kept_train"] == stats.kept_train
    assert written["author_overlap"] == 0


def test_prepare_output_is_normalised(corpus):
    from claudius.normalize import is_normalized

    out, _ = corpus
    texts = list(read_texts(out, "train", limit=5))
    assert texts and all(is_normalized(t) for t in texts)


def test_dedup_dropped_the_reprint(corpus):
    """The deliberate reprint must be gone; nothing else is guaranteed exactly.

    This used to assert exactly one drop. It passed only because the old LSH
    banding had 20% recall and never surfaced the other candidates. With correct
    banding, this pathological corpus (120 docs drawn from 10 sentences) yields
    pairs whose 128-permutation ESTIMATE is 0.80-0.85 while the exact Jaccard
    is 0.69-0.77: verification uses the estimate. On real Latin-PD, 404,550
    pairs contained none above 0.5, so it does not bite there -- but a test
    that depends on a bug's side effect is worse than one that is honest.
    """
    out, stats = corpus
    assert stats.rejected_dedup >= 1, "the exact reprint was not removed"
    assert stats.rejected_dedup < 10, "dedup is dropping far more than near-duplicates"
    texts = [r["text"] for split in ("train", "val") for r in read_shards(out, split)]
    assert len(texts) == len(set(texts)), "an identical text survived in the output"


def test_author_holdout_produced_a_validation_split(corpus):
    out, stats = corpus
    assert stats.kept_val > 0
    assert not (stats.authors_train & stats.authors_val)
    assert list(read_texts(out, "val", limit=1))


@pytest.fixture(scope="module")
def trained(corpus, tmp_path_factory):
    from claudius.pipeline import stage_pack, stage_tokenizer
    from claudius.train import TrainConfig, run as train_run

    out, _ = corpus
    work = tmp_path_factory.mktemp("trained")
    tok_dir, packed, model_dir = work / "tokenizer", work / "packed", work / "model"

    card = stage_tokenizer(out, tok_dir, vocab_size=512, train_docs=100)
    summary = stage_pack(out, tok_dir, packed, seq_len=64)

    cfg = TrainConfig(
        size="test",
        data_dir=str(packed),
        tokenizer_dir=str(tok_dir),
        out_dir=str(model_dir),
        seq_len=64,
        per_device_batch=2,
        grad_accum=1,
        max_steps=2,
        save_steps=1000,
        eval_steps=1000,
        bf16=False,
    )
    final = train_run(cfg)
    return card, summary, tok_dir, packed, final


def test_tokenizer_card(trained):
    card, *_ = trained
    assert card["vocab_size"] == 512
    assert card["fertility_tokens_per_word"] > 0


def test_packing_shapes(trained):
    _, summary, _, packed, _ = trained
    assert summary["seq_len"] == 64
    assert summary["splits"]["train"]["sequences"] > 0

    from claudius.pipeline import load_packed

    ds = load_packed(packed, "train")
    assert len(ds[0]["input_ids"]) == 64


def test_training_produced_a_model(trained):
    from transformers import AutoModelForCausalLM

    *_, final = trained
    model = AutoModelForCausalLM.from_pretrained(final)
    assert model.config.vocab_size == 512
    plan = json.loads((__import__("pathlib").Path(final) / "training_plan.json").read_text())
    assert plan["steps"] == 2


def test_eval_reports_bits_per_byte(corpus, trained):
    from transformers import AutoModelForCausalLM

    from claudius.evaluate import InterpolatedBigram, bits_per_byte
    from claudius.tokenizer import load as load_tokenizer

    out, _ = corpus
    *_, final = trained
    val = list(read_texts(out, "val", limit=3)) or list(read_texts(out, "train", limit=3))
    tok = load_tokenizer(final)
    model = AutoModelForCausalLM.from_pretrained(final)

    res = bits_per_byte(model, tok, val, seq_len=64)
    assert res.tokens > 0 and res.bits_per_byte > 0

    floor = InterpolatedBigram().fit(list(read_texts(out, "train", limit=20)))
    assert floor.bits_per_byte(val) > 0


def test_generation_is_in_the_alphabet(trained):
    from transformers import AutoModelForCausalLM

    from claudius.evaluate import sample_incipits
    from claudius.normalize import is_normalized
    from claudius.tokenizer import load as load_tokenizer

    *_, final = trained
    tok = load_tokenizer(final)
    model = AutoModelForCausalLM.from_pretrained(final)
    samples = sample_incipits(model, tok, max_new_tokens=8)
    assert len(samples) == 4
    for s in samples:
        assert is_normalized(s["prompt"])


def test_parallel_build_matches_the_streaming_build(tmp_path_factory):
    """map+reduce must produce exactly what the single-process build produces."""
    import json

    from claudius.corpora import Source
    from claudius.parallel import run_map, run_reduce
    from claudius.prepare import read_shards, run as prepare_run

    root = tmp_path_factory.mktemp("raw_par")
    _make_corpus(root, n_authors=40)
    src = Source(name="synthetic", kind="textdir", path=str(root))

    serial_dir = tmp_path_factory.mktemp("serial") / "corpus"
    serial_stats = prepare_run([src], serial_dir, log_every=0)

    par_dir = tmp_path_factory.mktemp("parallel") / "corpus"
    run_map([src], par_dir, workers=1)
    par_stats = run_reduce(par_dir)

    assert par_stats.kept_train == serial_stats.kept_train
    assert par_stats.kept_val == serial_stats.kept_val
    assert par_stats.rejected_dedup == serial_stats.rejected_dedup

    for split in ("train", "val"):
        a = [r["text"] for r in read_shards(serial_dir, split)]
        b = [r["text"] for r in read_shards(par_dir, split)]
        assert a == b, f"{split} split differs between serial and parallel builds"

    assert json.loads((par_dir / "stats.json").read_text())["author_overlap"] == 0


def test_map_stage_is_resumable(tmp_path_factory):
    from claudius.corpora import Source
    from claudius.parallel import part_dir, run_map

    root = tmp_path_factory.mktemp("raw_resume")
    _make_corpus(root, n_authors=8)
    src = Source(name="synthetic", kind="textdir", path=str(root))
    out = tmp_path_factory.mktemp("resume") / "corpus"

    first = run_map([src], out, workers=1)
    marker = part_dir(out, src, "") / "DONE"
    assert marker.exists()

    # A second run must not redo the work.
    again = run_map([src], out, workers=1)
    assert again[0]["written"] == first[0]["written"]
    assert again[0]["seconds"] == first[0]["seconds"], "finished shard was reprocessed"


def test_reduce_can_free_parts_as_it_goes(tmp_path_factory):
    from claudius.corpora import Source
    from claudius.parallel import PARTS, run_map, run_reduce
    from claudius.prepare import read_shards

    root = tmp_path_factory.mktemp("raw_drop")
    _make_corpus(root, n_authors=12)
    src = Source(name="synthetic", kind="textdir", path=str(root))
    out = tmp_path_factory.mktemp("drop") / "corpus"

    run_map([src], out, workers=1)
    stats = run_reduce(out, drop_parts=True)

    assert stats.kept_train > 0
    assert list(read_shards(out, "train")), "final shards must still be written"
    leftovers = list((out / PARTS).rglob("docs.jsonl.gz"))
    assert leftovers == [], "part documents should have been freed"
    # The fingerprints and the DONE marker stay, so the map stage is not redone.
    assert list((out / PARTS).rglob("sigs.npy"))
    assert list((out / PARTS).rglob("DONE"))


def test_every_source_produces_at_least_one_task():
    """A source with no parquet shards is one task, never zero.

    `hf_shards` returns [] for a config-based repo or a jsonl repo, meaning
    "let datasets resolve it". Treating that as an empty task list drops the
    source from the build without any error.
    """
    from claudius.corpora import Source
    from claudius.filters import FilterConfig
    from claudius.parallel import build_tasks

    sources = [
        Source(name="config-repo", kind="hf", path="wikimedia/wikipedia", config="20231101.la"),
        Source(name="jsonl-repo", kind="hf", path="someone/jsonl-only"),
        Source(name="csv-repo", kind="hf-csv", path="someone/csv", data_files=("a.csv",)),
        Source(name="local", kind="textdir", path="/tmp/whatever"),
    ]
    # No network: an unshardable source must still yield exactly one task.
    import claudius.corpora as c

    # build_tasks imports hf_shards from the module at call time, so patching
    # the module attribute is what takes effect.
    original = c.hf_shards
    c.hf_shards = lambda src: []
    try:
        tasks = build_tasks(sources, "/tmp/out", FilterConfig())
    finally:
        c.hf_shards = original

    names = sorted(t[0]["name"] for t in tasks)
    assert names == ["config-repo", "csv-repo", "jsonl-repo", "local"], names


def test_annealing_continues_from_existing_weights(trained, tmp_path_factory):
    """The annealing phase: continue a finished model, do not start over."""
    import torch
    from transformers import AutoModelForCausalLM

    from claudius.train import TrainConfig, run as train_run

    _, _, tok_dir, packed, final = trained
    before = AutoModelForCausalLM.from_pretrained(final)
    w_before = before.model.layers[0].mlp.down_proj.weight.detach().clone()

    out = tmp_path_factory.mktemp("anneal") / "model"
    cfg = TrainConfig(
        size="test", data_dir=str(packed), tokenizer_dir=str(tok_dir),
        out_dir=str(out), seq_len=64, per_device_batch=2, grad_accum=1,
        max_steps=2, save_steps=1000, eval_steps=1000, bf16=False,
        lr=1e-5, init_from=str(final),
    )
    annealed_path = train_run(cfg)
    after = AutoModelForCausalLM.from_pretrained(annealed_path)
    w_after = after.model.layers[0].mlp.down_proj.weight.detach()

    # Continued, not reinitialised: close to the base weights but not identical.
    assert not torch.equal(w_before, w_after), "annealing did not train"
    delta = (w_after - w_before).abs().mean().item()
    scale = w_before.abs().mean().item()
    assert delta < scale, "weights moved more than a from-scratch init would leave"


def test_annealing_rejects_a_tokenizer_mismatch(trained, tmp_path_factory):
    """A vocabulary mismatch silently produces gibberish; fail loudly instead."""
    import pytest
    from transformers import AutoModelForCausalLM

    from claudius.train import TrainConfig, run as train_run

    _, _, tok_dir, packed, final = trained

    # Save a copy of the model with a deliberately different vocabulary size.
    wrong = tmp_path_factory.mktemp("wrongvocab") / "model"
    model = AutoModelForCausalLM.from_pretrained(final)
    model.resize_token_embeddings(model.config.vocab_size + 8)
    model.save_pretrained(str(wrong))

    cfg = TrainConfig(size="test", data_dir=str(packed), tokenizer_dir=str(tok_dir),
                      out_dir=str(tmp_path_factory.mktemp("anneal_bad") / "m"),
                      seq_len=64, per_device_batch=2, grad_accum=1, max_steps=1,
                      save_steps=1000, eval_steps=1000, bf16=False,
                      init_from=str(wrong))
    with pytest.raises(SystemExit) as e:
        train_run(cfg)
    assert "tokenizer mismatch" in str(e.value)
