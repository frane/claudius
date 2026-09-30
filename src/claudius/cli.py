"""Command line entry point: `claudius <stage> ...`.

Stages run in order:

    claudius prepare    # corpora -> filtered, normalised, deduped shards
    claudius tokenizer  # vocabulary sweep -> chosen BPE
    claudius pack       # shards -> packed uint16 sequences
    claudius plan       # what the run will cost, before paying for it
    claudius train      # pretrain from scratch
    claudius eval       # bits-per-byte vs the floor, memorisation, samples
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from . import corpora
from .filters import FilterConfig


def _sources(names: list[str]) -> list[corpora.Source]:
    out = []
    for name in names:
        if name in corpora.SOURCES:
            src = corpora.SOURCES[name]
        elif "=" in name:  # name=path, for a local directory of .txt files
            label, _, path = name.partition("=")
            src = corpora.Source(name=label, kind="textdir", path=path, weight=6.0, repeat=True)
        else:
            raise SystemExit(f"unknown source: {name} (see `claudius sources`)")
        if not corpora.available(src):
            print(f"[warn] skipping {src.name}: {src.path} not present", file=sys.stderr)
            continue
        out.append(src)
    if not out:
        raise SystemExit("no usable sources")
    return out


def _spread(sources: list, shards: int) -> list:
    """Expand each Hub source into one source per shard, spread across the repo.

    Without this a sample is drawn from the head of the stream, which on
    Latin-PD is a single shard -- one library, one period, one press.
    """
    if not shards:
        return sources
    out = []
    for src in sources:
        if src.kind != "hf":
            out.append(src)
            continue
        files = corpora.spread_files(src.path, shards)
        print(f"[sample] {src.name}: drawing evenly from {len(files)} of the "
              f"repo's shards", file=sys.stderr)
        out.extend(replace(src, data_files=(f,)) for f in files)
    return out


def _filter_cfg(a) -> FilterConfig:
    return FilterConfig(min_latin=a.min_latin, min_chars=a.min_chars)


# -- commands --------------------------------------------------------------
def cmd_sources(a):
    for name, src in corpora.SOURCES.items():
        mark = "ok " if corpora.available(src) else "-- "
        print(f"{mark}{name:<24} weight={src.weight:<4} {src.register}")
        print(f"   {src.path}  [{src.license}]")
        if src.note:
            print(f"   {src.note}")


def cmd_normalize(a):
    from .normalize import normalize

    text = Path(a.path).read_text(encoding="utf-8") if a.path else sys.stdin.read()
    sys.stdout.write(normalize(text) + "\n")


def cmd_filter_report(a):
    from .filters import filter_report

    sources = _spread(_sources(a.source), a.shards)
    docs = []
    for doc in corpora.interleave(sources):
        docs.append((f"{doc.source}:{doc.identifier or doc.author}", doc.text))
        if len(docs) >= a.sample:
            break
    rep = filter_report(docs, _filter_cfg(a), limit=a.show)
    cut = rep["paragraph_cut_of_accepted"]
    print(f"sampled {rep['n']}: kept {rep['kept']}, rejected {rep['rejected']} "
          f"-> accept rate {rep['accept_rate']:.1%} "
          f"(latin >= {rep['threshold']}, rival <= {rep['max_rival']})")
    print(f"paragraph cut on accepted docs: mean {cut['mean']:.2%}, median "
          f"{cut['median']:.2%}, p90 {cut['p90']:.2%}, "
          f"{cut['docs_cut_over_5pct']} docs cut by more than 5%")
    for label, rows in (("JUST REJECTED", rep["just_rejected"]), ("JUST ACCEPTED", rep["just_accepted"])):
        print(f"\n=== {label} ===")
        for latin, rival, greek, cut_ratio, ident, head in rows:
            print(f"\n[latin {latin:.4f} | rival {rival:.4f} | greek {greek:.3f} "
                  f"| cut {cut_ratio:.1%}] {ident}\n{head.strip()[:300]}")


def cmd_prepare(a):
    sources = _spread(_sources(a.source), a.shards)
    if a.workers == 1 and not a.map_only and not a.reduce_only:
        from .prepare import run

        stats = run(sources, out_dir=a.out, cfg=_filter_cfg(a),
                    max_docs=a.max_docs, max_chars=a.max_chars)
    else:
        from .parallel import run_map, run_reduce

        if not a.reduce_only:
            run_map(sources, a.out, _filter_cfg(a), workers=a.workers,
                    drop_cache=a.drop_cache)
        if a.map_only:
            print("map stage complete; run with --reduce-only to dedup and write shards")
            return
        stats = run_reduce(a.out, drop_parts=a.drop_parts)
    print(json.dumps(stats.as_dict(), indent=2))
    if a.push:
        from .prepare import push

        print("pushed to", push(a.out, a.push))


def cmd_tokenizer(a):
    from .pipeline import stage_tokenizer

    card = stage_tokenizer(
        a.corpus,
        a.out,
        vocab_size=a.vocab_size,
        sweep_sizes=a.sweep,
        train_docs=a.train_docs,
    )
    print(json.dumps(card, indent=2))


def cmd_pack(a):
    from .pipeline import stage_pack

    print(json.dumps(stage_pack(a.corpus, a.tokenizer, a.out, seq_len=a.seq_len,
                                max_tokens=a.max_tokens), indent=2))


def cmd_shapes(a):
    from .model import describe

    print(describe(a.vocab_size))


def cmd_plan(a):
    from .train import DEFAULT_MFU, TrainConfig, plan

    if a.mfu is None:
        a.mfu = DEFAULT_MFU

    if a.tokens:
        n_sequences = a.tokens // a.seq_len
    else:
        from .pipeline import load_packed

        n_sequences = len(load_packed(a.data, "train"))
    cfg = TrainConfig(size=a.size, data_dir=a.data, epochs=a.epochs, seq_len=a.seq_len)
    print(json.dumps(plan(cfg, n_sequences, a.vocab_size, mfu=a.mfu, gpu=a.gpu), indent=2))


def cmd_train(a):
    from .train import TrainConfig, run

    cfg = TrainConfig(
        size=a.size,
        data_dir=a.data,
        tokenizer_dir=a.tokenizer,
        out_dir=a.out,
        epochs=a.epochs,
        lr=a.lr,
        per_device_batch=a.batch,
        grad_accum=a.accum,
        max_steps=a.max_steps,
        save_steps=a.save_steps,
        hub_repo=a.hub_repo,
        trackio=a.trackio,
        run_name=a.run_name,
        resume=a.resume,
        init_from=a.init_from,
    )
    print("final model:", run(cfg))


def cmd_chat(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .generate import roman
    from .sft.template import STUDENT, TEACHER, Turn
    from .tutor import reply

    path = a.model
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = os.path.join(snapshot_download(a.model, allow_patterns=["final/*"]), "final")
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    print(f"Claudius tutor ({a.level}). Scribe Latine; Ctrl-D to quit.\n")
    turns: list = []
    while True:
        try:
            text = input("DISCIPULUS: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        turns.append(Turn(STUDENT, text))
        answer, finding = reply(model, tok, turns[-a.history:], a.level)
        turns.append(Turn(TEACHER, answer))
        print(f"MAGISTER: {roman(answer)}")
        if a.verbose and finding:
            kind = "checker+model" if finding.confirmed else "model only"
            print(f"  [{kind}: {roman(finding.wrong)} -> {roman(finding.right)}, "
                  f"{finding.margin:.1f} nats]")
        print()


def cmd_serve(a):
    import uvicorn

    os.environ["CLAUDIUS_MODEL"] = a.model
    print(f"Claudius on http://{a.host}:{a.port}  (model: {a.model})", flush=True)
    uvicorn.run("claudius.web.server:app", host=a.host, port=a.port, log_level="warning")


def cmd_eval(a):
    from .evaluate import InterpolatedBigram, MemorisationIndex, bits_per_byte, sample_incipits
    from .prepare import read_texts
    from .tokenizer import load as load_tokenizer

    val = list(read_texts(a.corpus, "val", limit=a.val_docs))
    train_sample = list(read_texts(a.corpus, "train", limit=a.train_docs))
    if not val:
        raise SystemExit(
            f"no validation documents in {a.corpus}: the whole-author holdout is empty. "
            "Check `authors_val` in the corpus stats.json -- on a small pilot the held-out "
            "authors can all be removed by deduplication. Prepare more documents rather "
            "than evaluating on training authors."
        )

    report: dict = {"val_documents": len(val)}

    floor = InterpolatedBigram().fit(train_sample)
    report["bigram_floor_bits_per_byte"] = round(floor.bits_per_byte(val), 4)

    if a.model:
        from transformers import AutoModelForCausalLM

        tok = load_tokenizer(a.tokenizer or a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model)
        if a.device:
            model = model.to(a.device)
        res = bits_per_byte(model, tok, val, seq_len=a.seq_len)
        report["model_bits_per_byte"] = round(res.bits_per_byte, 4)
        report["model_perplexity_per_token"] = round(res.perplexity, 3)
        report["tokens_scored"] = res.tokens
        report["improvement_over_floor"] = round(
            report["bigram_floor_bits_per_byte"] - res.bits_per_byte, 4
        )

        samples = sample_incipits(model, tok, max_new_tokens=a.max_new_tokens)
        report["samples"] = samples

        index = MemorisationIndex(n=a.ngram).add_all(train_sample)
        # Continuations only: every incipit is itself a verbatim quotation,
        # so windows spanning the prompt 'match' on any model trained on it.
        mem = index.check([s["continuation"] for s in samples], val[: a.control_docs])
        report["memorisation"] = mem.as_dict()
        report["memorisation"]["index_ngrams"] = len(index)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# -- parser ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="claudius", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sources", help="list the corpus registry")
    s.set_defaults(func=cmd_sources)

    s = sub.add_parser("normalize", help="normalise a file or stdin")
    s.add_argument("path", nargs="?")
    s.set_defaults(func=cmd_normalize)

    def add_filter_args(sp):
        sp.add_argument("--source", nargs="+", default=["latin-pd"],
                        help="registry names, or name=path for a local .txt directory")
        sp.add_argument("--min-latin", type=float, default=0.06)
        sp.add_argument("--min-chars", type=int, default=2_000)

    s = sub.add_parser("filter-report", help="eyeball the language threshold on a sample")
    add_filter_args(s)
    s.add_argument("--sample", type=int, default=500)
    s.add_argument("--show", type=int, default=20, help="documents to print per side")
    s.add_argument("--shards", type=int, default=24,
                   help="spread the sample across this many Hub shards (0 = head of stream)")
    s.set_defaults(func=cmd_filter_report)

    s = sub.add_parser("prepare", help="filter, normalise, dedup, split by author")
    add_filter_args(s)
    s.add_argument("--out", default="out/corpus")
    s.add_argument("--shards", type=int, default=0,
                   help="draw evenly from this many Hub shards (0 = every shard, in order)")
    s.add_argument("--max-docs", type=int, default=0)
    s.add_argument("--max-chars", type=int, default=0)
    s.add_argument("--workers", type=int, default=0,
                   help="0 = all cores but two; 1 = single-process streaming build")
    s.add_argument("--drop-cache", action="store_true",
                   help="delete each shard's downloaded parquet once it is processed")
    s.add_argument("--drop-parts", action="store_true",
                   help="free each part as the reduce consumes it (halves peak disk)")
    s.add_argument("--map-only", action="store_true", help="filter and fingerprint, then stop")
    s.add_argument("--reduce-only", action="store_true", help="global dedup over finished parts")
    s.add_argument("--push", default=None, help="dataset repo id to upload to")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("tokenizer", help="vocabulary sweep and BPE training")
    s.add_argument("--corpus", nargs="+", default=["out/corpus"],
                   help="one or more prepared corpora, each optionally path:weight")
    s.add_argument("--out", default="out/tokenizer")
    s.add_argument("--vocab-size", type=int, default=None, help="skip the sweep and force a size")
    s.add_argument("--sweep", type=int, nargs="+", default=list((4096, 8192, 16384, 32768, 65536)))
    s.add_argument("--train-docs", type=int, default=20_000)
    s.set_defaults(func=cmd_tokenizer)

    s = sub.add_parser("pack", help="encode and pack into fixed-length sequences")
    s.add_argument("--corpus", nargs="+", default=["out/corpus"],
                   help="one or more prepared corpora, each optionally path:weight")
    s.add_argument("--tokenizer", default="out/tokenizer")
    s.add_argument("--out", default="out/packed")
    s.add_argument("--seq-len", type=int, default=2048)
    s.add_argument("--max-tokens", type=int, default=0)
    s.set_defaults(func=cmd_pack)

    s = sub.add_parser("shapes", help="parameter counts for each size")
    s.add_argument("--vocab-size", type=int, default=32_000)
    s.set_defaults(func=cmd_shapes)

    s = sub.add_parser("plan", help="token budget, FLOPs and GPU hours for a run")
    s.add_argument("--size", choices=list(("150m", "400m", "1b", "test")), default="400m")
    s.add_argument("--data", default="out/packed")
    s.add_argument("--vocab-size", type=int, default=32_000)
    s.add_argument("--seq-len", type=int, default=2048)
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--tokens", type=int, default=0,
                   help="budget a hypothetical run instead of reading packed data")
    s.add_argument("--gpu", default="a100", choices=["a10g", "l4", "a100", "h200"])
    s.add_argument("--mfu", type=float, default=None,
                   help="model FLOPs utilisation; default is the measured 0.17, not a textbook 0.4")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("train", help="pretrain from scratch")
    s.add_argument("--size", choices=list(("150m", "400m", "1b", "test")), default="400m")
    s.add_argument("--data", default="out/packed")
    s.add_argument("--tokenizer", default="out/tokenizer")
    s.add_argument("--out", default="out/model")
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--accum", type=int, default=0, help="0 derives it from 0.5M tokens/step")
    s.add_argument("--max-steps", type=int, default=-1)
    s.add_argument("--save-steps", type=int, default=500)
    s.add_argument("--hub-repo", default=None)
    s.add_argument("--trackio", action="store_true")
    s.add_argument("--run-name", default="claudius")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--init-from", default=None,
                   help="continue from these weights instead of initialising from scratch "
                        "(the annealing phase)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="bits-per-byte, floor, memorisation, samples")
    s.add_argument("--corpus", default="out/corpus")
    s.add_argument("--model", default=None, help="omit to report the floor only")
    s.add_argument("--tokenizer", default=None)
    s.add_argument("--val-docs", type=int, default=200)
    s.add_argument("--train-docs", type=int, default=2_000)
    s.add_argument("--control-docs", type=int, default=20)
    s.add_argument("--ngram", type=int, default=13)
    s.add_argument("--seq-len", type=int, default=2048)
    s.add_argument("--max-new-tokens", type=int, default=80)
    s.add_argument("--device", default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("chat", help="talk to the tutor in the terminal")
    s.add_argument("--model", default="fbandov/claudius-tutor-v7",
                   help="local model directory or private hub repo")
    s.add_argument("--level", default="tiro", choices=["tiro", "medius", "peritus"])
    s.add_argument("--history", type=int, default=6, help="turns kept as context")
    s.add_argument("--device", default=None)
    s.add_argument("--verbose", action="store_true", help="show what the detector found")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("serve", help="run the tutor web app locally")
    s.add_argument("--model", default="fbandov/claudius-tutor-v7",
                   help="local model directory or private hub repo")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
