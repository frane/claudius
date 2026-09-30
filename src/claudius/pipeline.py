"""Stage 2: prepared shards -> tokenizer -> packed token sequences on disk.

Sequences are written as parquet with a uint16 column, which is what a vocabulary
of at most 65,536 needs: 10B tokens is 20 GB rather than the 80 GB an int64
column would cost, and the training loader casts on the fly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from . import tokenizer as tk
from .mixing import CorpusSpec, mix_report, mixed_texts
from .model import SEQ_LEN
from .packing import pack
from .prepare import read_texts


def _sample(texts: Iterable[str], n: int) -> list[str]:
    out = []
    for t in texts:
        out.append(t)
        if len(out) >= n:
            break
    return out


def _as_specs(corpus) -> list[CorpusSpec]:
    """Accept a single directory, a `path:weight` string, or a list of either."""
    if isinstance(corpus, (str, Path)):
        return [CorpusSpec.parse(str(corpus))]
    out = []
    for item in corpus:
        out.append(item if isinstance(item, CorpusSpec) else CorpusSpec.parse(str(item)))
    return out


def stage_tokenizer(
    corpus_dir: str | Path,
    out_dir: str | Path,
    vocab_size: int | None = None,
    sweep_sizes: Sequence[int] = tk.DEFAULT_SWEEP,
    train_docs: int = 20_000,
    holdout_docs: int = 500,
    min_frequency: int = 2,
) -> dict:
    """Train the tokenizer, running the vocabulary sweep unless a size is forced.

    The sweep measures fertility on the *validation* split -- held-out authors,
    unseen by BPE training -- so the chosen size is not fitted to the text it is
    scored on.
    """
    specs = _as_specs(corpus_dir)
    out_dir = Path(out_dir)
    # Sample the MIX, not one corpus after another: a tokenizer trained on the
    # first N documents of a 30:1 corpus is a tokenizer for OCR noise.
    train_texts = list(mixed_texts(specs, "train", limit=train_docs))
    holdout = list(mixed_texts(specs, "val", limit=holdout_docs))
    if not holdout:
        # Small pilots may have no held-out author yet; fall back to a tail slice
        # of training text and say so in the card.
        holdout = train_texts[-max(1, len(train_texts) // 20):]
        holdout_note = "fallback: tail of train (no held-out authors in this run)"
    else:
        holdout_note = "whole-author holdout"
    if not train_texts:
        raise SystemExit(f"no prepared documents in {[s.path for s in specs]}")

    if vocab_size is None:
        results, chosen, base = tk.sweep(train_texts, holdout, sweep_sizes, min_frequency)
        print(tk.format_sweep(results, chosen), flush=True)
        # Truncate the tokenizer the sweep already trained; do not train again.
        tok = base if chosen == base.get_vocab_size() else tk.truncate_bpe(base, chosen)
    else:
        chosen = vocab_size
        tok = tk.train_bpe(train_texts, vocab_size=chosen, min_frequency=min_frequency)
        results = [tk.measure_fertility(tok, holdout, vocab_size=chosen)]

    tk.save(tok, out_dir)
    card = tk.tokenizer_card(results, chosen, corpus_note=str(corpus_dir))
    card["holdout"] = holdout_note
    card["train_documents"] = len(train_texts)
    card["mix"] = {s.label: s.weight for s in specs}
    if len(specs) > 1:
        card["realised_mix"] = mix_report(specs, "train", sample=min(2000, train_docs))
    (out_dir / "tokenizer_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"[tokenizer] vocab={chosen} -> {out_dir}", flush=True)
    return card


def _encode_stream(tok, texts: Iterable[str], batch: int = 128) -> Iterator[list[int]]:
    buf = []
    for text in texts:
        buf.append(text)
        if len(buf) >= batch:
            yield from tok(buf)["input_ids"]
            buf = []
    if buf:
        yield from tok(buf)["input_ids"]


def stage_pack(
    corpus_dir: str | Path,
    tokenizer_dir: str | Path,
    out_dir: str | Path,
    seq_len: int = SEQ_LEN,
    max_tokens: int = 0,
    shard_seqs: int = 50_000,
    splits: Sequence[str] = ("train", "val"),
    max_val_tokens: int = 20_000_000,
) -> dict:
    """Pack a corpus into fixed-length sequences.

    `max_val_tokens` exists because an uncapped validation split is expensive in
    a way that is invisible until it bills. The 8B-token mix produced 427M
    validation tokens -- 208,575 sequences -- and a training run evaluating all
    of it spent 45 minutes per check. 20M tokens estimates held-out loss just as
    well and costs seconds.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    specs = _as_specs(corpus_dir)
    out_dir = Path(out_dir)
    tok = tk.load(tokenizer_dir)
    if tok.vocab_size > 65_536:
        raise ValueError("uint16 storage requires vocab_size <= 65536")
    eos = tok.eos_token_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Training reads the tokenizer from <packed>/tokenizer; ship it with the data
    # (a run failed at startup once because it was left out).
    if Path(tokenizer_dir).resolve() != (out_dir / "tokenizer").resolve():
        import shutil
        shutil.copytree(tokenizer_dir, out_dir / "tokenizer", dirs_exist_ok=True)
    summary = {"seq_len": seq_len, "vocab_size": tok.vocab_size,
               "mix": {s.label: s.weight for s in specs}, "splits": {}}

    for split in splits:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([("input_ids", pa.list_(pa.uint16(), seq_len))])
        n_seq = shard = 0
        buf: list[list[int]] = []
        t0 = time.time()
        budget = max_tokens if split == "train" else max_val_tokens

        def flush():
            nonlocal buf, shard
            if not buf:
                return
            table = pa.table({"input_ids": pa.array(buf, type=pa.list_(pa.uint16(), seq_len))}, schema=schema)
            pq.write_table(table, split_dir / f"{split}-{shard:05d}.parquet", compression="zstd")
            shard += 1
            buf = []

        stream = mixed_texts(specs, split) if len(specs) > 1 else read_texts(specs[0].path, split)
        for seq in pack(_encode_stream(tok, stream), eos, seq_len):
            buf.append(seq)
            n_seq += 1
            if len(buf) >= shard_seqs:
                flush()
                rate = n_seq * seq_len / max(time.time() - t0, 1e-9)
                print(
                    f"[pack:{split}] {n_seq * seq_len / 1e9:.3f}B tokens "
                    f"({rate / 1e6:.1f}M tok/s)",
                    flush=True,
                )
            if budget and n_seq * seq_len >= budget:
                break
        flush()
        summary["splits"][split] = {"sequences": n_seq, "tokens": n_seq * seq_len}
        print(
            f"[pack:{split}] done: {n_seq} sequences, {n_seq * seq_len:,} tokens "
            f"({n_seq * seq_len / 1e9:.3f}B)",
            flush=True,
        )

    (out_dir / "pack_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_packed(path: str | Path, split: str = "train"):
    from datasets import load_dataset

    files = sorted(str(p) for p in (Path(path) / split).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no packed shards for split {split} in {path}")
    return load_dataset("parquet", data_files=files, split="train")
