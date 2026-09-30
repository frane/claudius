"""Parallel, resumable corpus build.

Single-process `prepare` runs at roughly 45 documents a minute, which is about
37 hours for all of Latin-PD -- past every job timeout, and unrecoverable if it
dies at hour 30. This splits the work the only way it actually splits:

* **map** (parallel, one worker per shard) -- filter, normalise and fingerprint.
  This is all of the CPU cost, and it needs no coordination: the author holdout
  is a hash, so a worker knows a document's split without asking anyone.
* **reduce** (one process, cheap) -- decide duplicates globally and write the
  final shards. Near-duplicate detection cannot be done per-shard, because the
  reprint of a book sits in a different shard than the original. But it only
  needs 512 bytes of MinHash signature per document, so the whole corpus of
  fingerprints is a few dozen megabytes.

Both halves are resumable: a finished shard leaves a DONE marker and is skipped.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .corpora import Source, is_holdout, iter_source
from .dedup import (
    GlobalDedup,
    MinHasher,
    apply_paragraph_drops,
    fingerprint,
)
from .filters import FilterConfig, prepare_document
from .prepare import GZIP_LEVEL, PrepareStats, ShardWriter

PARTS = "parts"


def drop_shard_cache(src: Source, shard: str) -> int:
    """Delete a shard's downloaded parquet once it has been processed.

    All 118 Latin-PD shards is ~28 GB of parquet, and a job that also has to
    hold ~22 GB of parts and ~22 GB of final output can run out of disk. Each
    shard is read exactly once, so its blob is dead the moment its part is
    written. Returns bytes freed.
    """
    if src.kind != "hf" or not shard:
        return 0
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(src.path, shard, repo_type="dataset")
        if not isinstance(hit, str):
            return 0
        path = Path(hit)
        blob = path.resolve()
        freed = blob.stat().st_size if blob.exists() else 0
        if blob.exists():
            blob.unlink()
        if path.is_symlink() or path.exists():
            path.unlink()
        return freed
    except Exception as exc:  # never fail the shard over a cleanup problem
        print(f"[map] could not drop cache for {shard}: {exc!r}", flush=True)
        return 0


def part_dir(out_dir: Path, source: Source, shard: str) -> Path:
    stem = Path(shard).stem if shard else source.name
    return Path(out_dir) / PARTS / f"{source.name}--{stem}"


# -- map -------------------------------------------------------------------
def map_shard(task: tuple) -> dict:
    """Filter and fingerprint one shard. Returns its stats.

    Runs in a worker process, so everything it touches must be picklable and it
    must not share state with any other worker.
    """
    (source_kwargs, shard, out_dir, cfg_kwargs, num_perm, shingle_k, min_para,
     drop_cache) = task
    src = Source(**source_kwargs)
    cfg = FilterConfig(**cfg_kwargs)
    if src.min_chars is not None:
        cfg = replace(cfg, min_chars=src.min_chars)
    out = part_dir(Path(out_dir), src, shard)

    if (out / "DONE").exists():
        return json.loads((out / "stats.json").read_text())

    out.mkdir(parents=True, exist_ok=True)
    hasher = MinHasher(num_perm=num_perm)
    sigs: list[np.ndarray] = []
    para_flat: list[int] = []
    para_offsets: list[int] = [0]
    seen = rejected = unhashable = 0
    t0 = time.time()

    # A shard that is one 3.4 GB stream can run for an hour with nothing to
    # show. Per-shard completion is too coarse to tell "working" from "hung".
    log_every = 2000
    with gzip.open(out / "docs.jsonl.gz", "wt", encoding="utf-8",
                   compresslevel=GZIP_LEVEL) as fh:
        for doc in iter_source(src):
            seen += 1
            if log_every and seen % log_every == 0:
                print(f"[map:{out.name}] seen={seen:,} kept={len(sigs):,} "
                      f"({seen / max(time.time() - t0, 1e-9) * 60:.0f} docs/min)",
                      flush=True)
            text = prepare_document(doc.text, cfg)
            if text is None:
                rejected += 1
                continue
            fp = fingerprint(text, hasher, shingle_k, min_para)
            if fp.signature is None:
                unhashable += 1
                continue
            fh.write(json.dumps({
                "text": text,
                "source": src.name,
                "author": doc.author,
                "identifier": doc.identifier,
                "date": doc.date,
                "split": "val" if is_holdout(doc.author) else "train",
            }, ensure_ascii=False) + "\n")
            sigs.append(fp.signature)
            para_flat.extend(fp.paragraphs)
            para_offsets.append(len(para_flat))

    np.save(out / "sigs.npy",
            np.array(sigs, dtype=np.uint32) if sigs else np.zeros((0, num_perm), np.uint32))
    np.savez(out / "paras.npz",
             flat=np.array(para_flat, dtype=np.uint32),
             offsets=np.array(para_offsets, dtype=np.int64))
    freed = drop_shard_cache(src, shard) if drop_cache else 0
    stats = {
        "part": out.name, "shard": shard, "seen": seen, "written": len(sigs),
        "rejected_by_filter": rejected, "unhashable": unhashable,
        "seconds": round(time.time() - t0, 1), "cache_freed_bytes": freed,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    (out / "DONE").write_text("")
    return stats


def build_tasks(sources: list[Source], out_dir: str | Path, cfg: FilterConfig,
                num_perm: int = 128, shingle_k: int = 5, min_para: int = 200,
                drop_cache: bool = False) -> list[tuple]:
    """One task per shard, so work distributes at shard granularity."""
    from .corpora import hf_shards

    tasks = []
    for src in sources:
        # An empty shard list means "no parquet to split on" -- a config-based
        # repo, a jsonl repo, a local directory. That is one task for the whole
        # source, NOT zero tasks. Getting this wrong drops the source silently.
        shards = (hf_shards(src) or [""]) if src.kind == "hf" else [""]
        for shard in shards:
            per_shard = replace(src, data_files=(shard,)) if shard else src
            tasks.append((asdict(per_shard), shard, str(out_dir),
                          asdict(cfg), num_perm, shingle_k, min_para, drop_cache))
    return tasks


def run_map(sources: list[Source], out_dir: str | Path, cfg: FilterConfig = FilterConfig(),
            workers: int = 0, num_perm: int = 128, drop_cache: bool = False) -> list[dict]:
    import multiprocessing as mp

    tasks = build_tasks(sources, out_dir, cfg, num_perm=num_perm, drop_cache=drop_cache)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    done = sum(1 for t in tasks if (part_dir(Path(out_dir), Source(**t[0]), t[1]) / "DONE").exists())
    print(f"[map] {len(tasks)} shards, {done} already done, {workers} workers", flush=True)

    results = []
    t0 = time.time()
    if workers == 1:
        for task in tasks:
            results.append(map_shard(task))
            _log_map(results, len(tasks), t0)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for stats in pool.imap_unordered(map_shard, tasks):
                results.append(stats)
                _log_map(results, len(tasks), t0)
    return results


def _log_map(results: list[dict], total: int, t0: float):
    written = sum(r["written"] for r in results)
    seen = sum(r["seen"] for r in results)
    freed = sum(r.get("cache_freed_bytes", 0) for r in results)
    rate = written / max(time.time() - t0, 1e-9) * 60
    print(f"[map] {len(results)}/{total} shards, kept {written:,}/{seen:,} docs "
          f"({rate:.0f} docs/min, {freed / 1e9:.1f}GB cache freed)", flush=True)


# -- reduce ----------------------------------------------------------------
def run_reduce(out_dir: str | Path, threshold: float = 0.8, num_perm: int = 128,
               shard_chars: int = 256 * 1024 * 1024, log_every: int = 5000,
               drop_parts: bool = False) -> PrepareStats:
    """Global dedup over the fingerprints, then write the final shards.

    `drop_parts` deletes each part's documents once it has been folded into the
    output. The parts and the final shards hold the same text, so keeping both
    doubles peak disk -- about 85 GB for the full corpus instead of 45 GB. The
    cost is that a reduce which dies halfway cannot be restarted without
    re-running the map stage, so it is off by default and on in the job, where
    disk is the binding constraint and the output is pushed to the Hub anyway.
    """
    out_dir = Path(out_dir)
    parts = sorted((out_dir / PARTS).glob("*"))
    parts = [p for p in parts if (p / "DONE").exists()]
    if not parts:
        raise SystemExit(f"no completed parts in {out_dir / PARTS} -- run the map stage first")

    g = GlobalDedup(threshold=threshold, num_perm=num_perm)
    stats = PrepareStats()
    writers = {s: ShardWriter(out_dir, s, shard_chars) for s in ("train", "val")}
    n = 0
    t0 = time.time()
    try:
        for part in parts:
            sigs = np.load(part / "sigs.npy")
            paras = np.load(part / "paras.npz")
            flat, offsets = paras["flat"], paras["offsets"]
            part_stats = json.loads((part / "stats.json").read_text())
            stats.seen += part_stats["seen"]
            stats.rejected_filter += part_stats["rejected_by_filter"]

            with gzip.open(part / "docs.jsonl.gz", "rt", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    row = json.loads(line)
                    fp = _row_fingerprint(sigs, flat, offsets, i)
                    keep, drop = g.decide(fp)
                    n += 1
                    if not keep:
                        stats.rejected_dedup += 1
                        continue
                    text = apply_paragraph_drops(row["text"], drop)
                    if not text:
                        stats.rejected_dedup += 1
                        continue
                    split = row["split"]
                    bucket = stats.by_source.setdefault(row["source"], {"docs": 0, "chars": 0})
                    bucket["docs"] += 1
                    bucket["chars"] += len(text)
                    if split == "train":
                        stats.kept_train += 1
                        stats.chars_train += len(text)
                        if row["author"]:
                            stats.authors_train.add(row["author"].strip().lower())
                    else:
                        stats.kept_val += 1
                        stats.chars_val += len(text)
                        if row["author"]:
                            stats.authors_val.add(row["author"].strip().lower())
                    row["text"] = text
                    row.pop("split", None)
                    writers[split].write(row)
                    if log_every and (stats.kept_train + stats.kept_val) % log_every == 0:
                        print(f"[reduce] {n:,} scanned, {stats.kept_train + stats.kept_val:,} kept "
                              f"({stats.chars_train / 1e9:.1f}GB train, "
                              f"{time.time() - t0:.0f}s)", flush=True)
            if drop_parts:
                (part / "docs.jsonl.gz").unlink(missing_ok=True)
    finally:
        for w in writers.values():
            w.close()

    stats.dedup = g.stats.as_dict()
    (out_dir / "stats.json").write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    return stats


def _row_fingerprint(sigs, flat, offsets, i):
    from .dedup import Fingerprint

    return Fingerprint(
        signature=sigs[i],
        paragraphs=tuple(int(x) for x in flat[offsets[i]:offsets[i + 1]]),
    )
