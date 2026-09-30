"""Stage 1: corpus -> filtered, normalised, deduplicated, author-split shards.

Output layout (local or pushed to the Hub):

    out/corpus/train-00000.jsonl.gz ...
    out/corpus/val-00000.jsonl.gz ...
    out/corpus/stats.json

Each line is `{"text", "source", "author", "identifier", "date"}`. Keeping the
metadata means the author holdout stays auditable and the memorisation check has
something to index.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .corpora import Document, Source, interleave, is_holdout
from .dedup import Deduper
from .filters import FilterConfig, prepare_document


@dataclass
class PrepareStats:
    seen: int = 0
    rejected_filter: int = 0
    rejected_dedup: int = 0
    kept_train: int = 0
    kept_val: int = 0
    chars_train: int = 0
    chars_val: int = 0
    by_source: dict = field(default_factory=dict)
    authors_train: set = field(default_factory=set)
    authors_val: set = field(default_factory=set)
    dedup: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "documents_seen": self.seen,
            "rejected_by_filter": self.rejected_filter,
            "rejected_as_duplicate": self.rejected_dedup,
            "kept_train": self.kept_train,
            "kept_val": self.kept_val,
            "chars_train": self.chars_train,
            "chars_val": self.chars_val,
            "approx_words_train": self.chars_train // 6,
            "authors_train": len(self.authors_train),
            "authors_val": len(self.authors_val),
            "author_overlap": len(self.authors_train & self.authors_val),
            "by_source": self.by_source,
            "dedup": self.dedup,
        }


#: gzip.open defaults to level 9. Measured on real filtered Latin, not on
#: synthetic text (which compresses to nothing and hides the difference):
#:
#:     level 9   20.8 MB/s   43.6% of original
#:     level 6   21.7 MB/s   43.6%   -- no faster than 9, so not worth choosing
#:     level 4   54.0 MB/s   45.6%   -- 2.6x faster for 2% more bytes
#:     level 1   94.2 MB/s   50.2%
#:
#: Writing an 88 GB corpus is 70 minutes at level 9 and 27 at level 4. The
#: output is read back by a tokenizer, not shipped to users, so 2% of storage
#: is the right thing to spend for 43 minutes of every build.
GZIP_LEVEL = 4


class ShardWriter:
    """Gzipped JSONL shards, rolled at a fixed uncompressed size."""

    def __init__(self, out_dir: Path, split: str, shard_chars: int = 256 * 1024 * 1024,
                 compresslevel: int = GZIP_LEVEL):
        self.dir = out_dir
        self.split = split
        self.shard_chars = shard_chars
        self.compresslevel = compresslevel
        self.dir.mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._written = 0
        self._fh = None
        self.paths: list[Path] = []

    def _roll(self):
        if self._fh:
            self._fh.close()
        path = self.dir / f"{self.split}-{self._n:05d}.jsonl.gz"
        self._fh = gzip.open(path, "wt", encoding="utf-8",
                             compresslevel=self.compresslevel)
        self.paths.append(path)
        self._n += 1
        self._written = 0

    def write(self, row: dict):
        if self._fh is None or self._written >= self.shard_chars:
            self._roll()
        line = json.dumps(row, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._written += len(line)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def iter_prepared(
    sources: list[Source],
    cfg: FilterConfig = FilterConfig(),
    deduper: Deduper | None = None,
    stats: PrepareStats | None = None,
    max_docs: int = 0,
    max_chars: int = 0,
) -> Iterator[tuple[Document, str, str]]:
    """Yield `(document, cleaned_text, split)` for documents that survive."""
    deduper = deduper if deduper is not None else Deduper()
    stats = stats if stats is not None else PrepareStats()
    kept_chars = 0

    by_name = {s.name: s for s in sources}
    for doc in interleave(sources):
        stats.seen += 1
        src = by_name.get(doc.source)
        doc_cfg = cfg
        if src is not None and src.min_chars is not None:
            doc_cfg = dataclasses.replace(cfg, min_chars=src.min_chars)
        cleaned = prepare_document(doc.text, doc_cfg)
        if cleaned is None:
            stats.rejected_filter += 1
            continue
        cleaned = deduper.add(cleaned)
        if cleaned is None:
            stats.rejected_dedup += 1
            continue

        split = "val" if is_holdout(doc.author) else "train"
        bucket = stats.by_source.setdefault(doc.source, {"docs": 0, "chars": 0})
        bucket["docs"] += 1
        bucket["chars"] += len(cleaned)
        if split == "train":
            stats.kept_train += 1
            stats.chars_train += len(cleaned)
            if doc.author:
                stats.authors_train.add(doc.author.strip().lower())
        else:
            stats.kept_val += 1
            stats.chars_val += len(cleaned)
            if doc.author:
                stats.authors_val.add(doc.author.strip().lower())
        kept_chars += len(cleaned)

        yield doc, cleaned, split

        if max_docs and (stats.kept_train + stats.kept_val) >= max_docs:
            return
        if max_chars and kept_chars >= max_chars:
            return


def run(
    sources: list[Source],
    out_dir: str | Path,
    cfg: FilterConfig = FilterConfig(),
    max_docs: int = 0,
    max_chars: int = 0,
    shard_chars: int = 256 * 1024 * 1024,
    log_every: int = 1000,
) -> PrepareStats:
    out_dir = Path(out_dir)
    stats = PrepareStats()
    writers = {s: ShardWriter(out_dir, s, shard_chars) for s in ("train", "val")}
    t0 = time.time()
    try:
        for doc, text, split in iter_prepared(sources, cfg, stats=stats, max_docs=max_docs, max_chars=max_chars):
            writers[split].write(
                {
                    "text": text,
                    "source": doc.source,
                    "author": doc.author,
                    "identifier": doc.identifier,
                    "date": doc.date,
                }
            )
            kept = stats.kept_train + stats.kept_val
            if log_every and kept % log_every == 0:
                rate = stats.chars_train / max(time.time() - t0, 1e-9) / 1e6
                print(
                    f"[prepare] seen={stats.seen} kept={kept} "
                    f"({stats.chars_train / 1e9:.3f}GB train, {rate:.1f}MB/s)",
                    flush=True,
                )
    finally:
        for w in writers.values():
            w.close()

    (out_dir / "stats.json").write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    return stats


def read_shards(path: str | Path, split: str = "train", limit: int = 0) -> Iterator[dict]:
    """Read back prepared shards."""
    path = Path(path)
    n = 0
    # Recursive: a corpus downloaded from the Hub may sit in a subdirectory,
    # and several corpora can share one root without their shard names
    # colliding.
    shards = sorted(path.glob(f"{split}-*.jsonl.gz")) or sorted(
        path.rglob(f"{split}-*.jsonl.gz"))
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
                n += 1
                if limit and n >= limit:
                    return


def read_texts(path: str | Path, split: str = "train", limit: int = 0) -> Iterator[str]:
    for row in read_shards(path, split, limit):
        yield row["text"]


def push(out_dir: str | Path, repo_id: str, private: bool = True,
         include_parts: bool = False) -> str:
    """Upload prepared shards to the Hub as a dataset repo.

    The intermediate `parts/` directory holds the same documents as the final
    shards, so uploading both doubles the transfer -- ~85 GB instead of ~42 GB
    for the full corpus. Parts are only worth pushing when a later job needs to
    resume the reduce stage from them.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
        ignore_patterns=None if include_parts else ["parts/*", "parts/**"],
    )
    return repo_id
