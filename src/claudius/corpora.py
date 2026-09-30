"""Corpus registry, streaming and the author holdout.

Latin-PD is the bulk but it is 17th-century scholastic prose read by a scanner.
The clean, born-digital corpora are two orders of magnitude smaller, so they are
upweighted (repeated) to shift the register towards classical and textbook Latin
without waiting for data that does not exist.

Licences differ per source and some are per-collection. `Source.license` records
what we believe; check before redistributing a derived dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: Fraction of *authors* held out for validation. Whole-author holdout: no
#: author in validation appears in training, so validation loss cannot be won
#: by memorising an author's idiolect.
HOLDOUT_FRACTION = 0.02
_HOLDOUT_BUCKETS = 1000


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    identifier: str = ""
    author: str = ""
    date: str = ""


@dataclass(frozen=True)
class Source:
    name: str
    kind: str  # "hf" | "jsonl" | "textdir"
    path: str
    #: For `kind="hf"`: the dataset config name, when the repo has several.
    config: str | None = None
    #: Column names. Tuples are candidates tried in order -- Latin-PD does not
    #: use one schema across its shards.
    text_column: str | tuple[str, ...] = "text"
    author_column: str | tuple[str, ...] | None = None
    id_column: str | tuple[str, ...] | None = None
    date_column: str | tuple[str, ...] | None = None
    #: Relative token share target. Latin-PD is 1.0; a corpus with weight 3.0
    #: is repeated until it has contributed ~3x its share of the stream.
    weight: float = 1.0
    repeat: bool = False
    #: Override the corpus-wide minimum document length. A threshold tuned for
    #: whole scanned books throws away most of an encyclopaedia: at 2,000
    #: characters, 94% of Vicipaedia is discarded for being short rather than
    #: for being bad.
    min_chars: int | None = None
    #: For `kind="hf"`: stream only these files instead of the whole repo. Used
    #: to spread a sample across shards -- the head of the stream is one shard,
    #: not a sample of the corpus.
    data_files: tuple[str, ...] | None = None
    license: str = "unknown"
    register: str = ""
    note: str = ""


#: The one source that needs no local setup. Everything else is opt-in because
#: it needs a download, a licence check, or both.
LATIN_PD = Source(
    name="latin-pd",
    kind="hf",
    path="PleIAs/Latin-PD",
    text_column="text",
    author_column="creator",
    id_column=("identifier", "file_id"),
    date_column="publication_date",
    weight=1.0,
    license="public domain (pre-1884 scans)",
    register="neo-latin, scholastic, heavy OCR noise",
    note="16.5B words claimed; expect 5-10B tokens after filtering and dedup.",
)

#: Clean, born-digital corpora. Point `--source` at a local directory of .txt
#: or .jsonl files after downloading them; none of these are a single HF repo.
EXTRA_SOURCES = [
    Source(
        name="latin-literature",
        kind="hf-csv",
        path="Fece228/latin-literature-dataset-170M",
        data_files=("latin_raw.csv",),
        text_column="text",
        author_column="author",
        id_column="title",
        date_column="year",
        weight=4.0,
        repeat=True,
        license="UNSTATED on the repo -- fine to train on, check before redistributing",
        register="classical, medieval and neo-latin prose; born-digital, no OCR",
        note=("3.4GB of raw text with author and year columns. XML-derived (the "
              "row ids are latin_xmls/*.xml), so this is very likely Corpus "
              "Corporum material already mirrored -- which is why no scrape of "
              "mlat.uzh.ch is needed. The repo also ships a lemma file we ignore."),
    ),
    Source(
        name="corpus-corporum",
        kind="textdir",
        path="data/corpus-corporum",
        weight=3.0,
        repeat=True,
        license="per-collection; check mlat.uzh.ch before redistributing",
        register="classical, medieval, neo-latin; TEI, born-digital",
        note="Only needed if latin-literature turns out not to cover it.",
    ),
    Source(
        name="classical-core",
        kind="hf",
        path="fbandov/claudius-clean-local",
        text_column="text",
        author_column="author",
        id_column="identifier",
        weight=6.0,
        repeat=True,
        license="CC BY-SA / public domain per text (Latin Library + Perseus)",
        register="classical prose and verse -- the register we actually want",
        note=("The Latin Library (2,141 texts) and Perseus (362 TEI works, 84 "
              "authors) exported to jsonl so jobs can read them. Small -- 19M "
              "words -- but it is the classical core the buddy phase needs."),
    ),
    Source(
        name="latin-library",
        kind="textdir",
        path="data/latin_text_latin_library",
        weight=6.0,
        repeat=True,
        license="CC BY-SA / public domain, per text",
        register="classical prose and verse -- the register we actually want",
        note="github cltk/latin_text_latin_library -- 2,141 texts, author per directory.",
    ),
    Source(
        name="perseus",
        kind="tei",
        path="data/perseus-latinlit",
        weight=6.0,
        repeat=True,
        license="CC BY-SA 4.0 (per text; check individual works)",
        register="classical canon, critically edited",
        note="github PerseusDL/canonical-latinLit -- TEI XML, converted on read.",
    ),
    Source(
        name="varro",
        kind="textdir",
        path="data/varro",
        weight=6.0,
        repeat=True,
        license="licence-pinned per text (github y-a-v-a/varro)",
        register="classical",
        note="~5M words, already curated.",
    ),
    Source(
        name="vicipaedia",
        kind="hf",
        path="wikimedia/wikipedia",
        config="20231101.la",
        text_column="text",
        id_column="id",
        # Vicipaedia has no authors, so the holdout keys on the article title:
        # a deterministic 2% of articles are held out, and no article can
        # straddle the split. Same guarantee, different unit.
        author_column="title",
        weight=4.0,
        repeat=True,
        # An encyclopaedia is not a library of books. At the corpus-wide 2,000
        # characters only 5.9% of articles survive -- discarded for being short,
        # not for being bad.
        min_chars=500,
        license="CC BY-SA 4.0",
        register="modern expository Latin, native-written -- the buddy register",
        note="Latin Wikipedia. Clean, born-digital, no OCR.",
    ),
    Source(
        name="vulgate",
        kind="textdir",
        path="data/vulgate",
        weight=4.0,
        repeat=True,
        license="public domain",
        register="simple narrative Latin; the most widely-read Latin text there is",
    ),
    Source(
        name="readers-colloquia",
        kind="textdir",
        path="data/readers",
        weight=6.0,
        repeat=True,
        license="public domain (pre-1930 school readers)",
        register="graded textbook Latin -- upweight hard, this is the target register",
        note="19th-century readers and colloquia scholastica.",
    ),
]

SOURCES = {s.name: s for s in [LATIN_PD, *EXTRA_SOURCES]}


# -- author holdout --------------------------------------------------------
def holdout_bucket(author: str) -> int:
    """Stable bucket in [0, 1000) for an author string."""
    key = " ".join(author.lower().split()) or "<unknown>"
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big") % _HOLDOUT_BUCKETS


def is_holdout(author: str, fraction: float = HOLDOUT_FRACTION) -> bool:
    """Whole-author validation split.

    Documents with no author are never held out: an empty creator field is not
    an author, and pooling them would put unrelated texts in validation.
    """
    if not author or not author.strip():
        return False
    return holdout_bucket(author) < fraction * _HOLDOUT_BUCKETS


# -- readers ---------------------------------------------------------------
def spread_files(repo: str, n: int, pattern: str = ".parquet") -> tuple[str, ...]:
    """`n` shard filenames spread evenly across a Hub dataset repo.

    Latin-PD is 118 parquet shards and the rows are not shuffled across them, so
    the first 5,000 rows of the stream are 5,000 rows of one shard -- one press,
    one library, one period. Anything measured on that does not describe the
    corpus.
    """
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.endswith(pattern))
    if not files or n >= len(files):
        return tuple(files)
    step = len(files) / n
    return tuple(files[int(i * step)] for i in range(n))


def _pick(row: dict, columns) -> str:
    """First non-empty value among the candidate column names."""
    if not columns:
        return ""
    if isinstance(columns, str):
        columns = (columns,)
    for col in columns:
        value = row.get(col)
        if value not in (None, ""):
            return str(value)
    return ""


def hf_shards(src: Source) -> list[str]:
    """The parquet files to stream, one at a time.

    Latin-PD does not have one schema across its 118 shards: some use `file_id`
    where others use `identifier`, some carry `language` columns or a stray
    `__index_level_0__`, and `publication_date` is int64 in one shard and string
    in another. Handing the whole repo to `load_dataset` makes it unify those
    schemas, and it raises CastError partway through -- about 1,000 rows in, on
    the first boundary. That failure would land hours into a 24-hour CPU job, so
    each shard is streamed as its own dataset and the columns are resolved per
    row instead.
    """
    if src.data_files:
        return list(src.data_files)
    if src.config:
        # A multi-config repo holds every config's parquet side by side; listing
        # them all would mix languages. Let `datasets` resolve the config.
        return []
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(src.path, repo_type="dataset")
    except Exception as exc:
        # A listing failure must not kill a multi-source build. Fall back to
        # one task for the whole source and let `load_dataset` report the real
        # problem against the source that actually has it.
        print(f"[corpora] could not list {src.path}: {exc!r}", flush=True)
        return []
    return sorted(f for f in files if f.endswith(".parquet"))


def _iter_hf(src: Source, streaming: bool = True) -> Iterator[Document]:
    from datasets import load_dataset

    shards = hf_shards(src)
    # A repo that is not sharded parquet (or one we cannot list) still works.
    batches = [{"data_files": [f]} for f in shards] if shards else [{}]
    for kwargs in batches:
        if src.config:
            kwargs["name"] = src.config
        ds = load_dataset(src.path, split="train", streaming=streaming, **kwargs)
        for row in ds:
            text = _pick(row, src.text_column)
            if not text:
                continue
            yield Document(
                text=text,
                source=src.name,
                identifier=_pick(row, src.id_column),
                author=_pick(row, src.author_column),
                date=_pick(row, src.date_column),
            )


def _iter_jsonl(src: Source) -> Iterator[Document]:
    for path in sorted(Path(src.path).rglob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get(src.text_column) or ""
                if text:
                    yield Document(
                        text=text,
                        source=src.name,
                        identifier=str(row.get("identifier") or path.name),
                        author=str(row.get("author") or row.get("creator") or ""),
                        date=str(row.get("date") or ""),
                    )


def _iter_textdir(src: Source) -> Iterator[Document]:
    """One document per .txt file.

    The author is taken from the first path component below the root, which is
    how Perseus, the Latin Library and Varro all lay their texts out
    (`root/cicero/de_officiis.txt`). Without that the author holdout is
    meaningless for these sources.
    """
    root = Path(src.path)
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        rel = path.relative_to(root)
        author = rel.parts[0] if len(rel.parts) > 1 else path.stem
        yield Document(text=text, source=src.name, identifier=str(rel), author=author)


#: TEI elements whose contents are editorial, not text: apparatus, notes,
#: bibliography, and the header. Their text would otherwise land in the corpus
#: in whatever modern language the editor wrote it.
_TEI_DROP = ("note", "bibl", "biblStruct", "teiHeader", "app", "rdg", "witness",
             "figure", "graphic", "milestone", "gap", "orig")
_TEI_NS = "{http://www.tei-c.org/ns/1.0}"


_ENTITY = __import__("re").compile(r"&([a-zA-Z][a-zA-Z0-9]{1,31});")
#: XML's own five entities must survive; everything else is HTML and would make
#: the parser throw "undefined entity".
_XML_BUILTIN = {"amp", "lt", "gt", "quot", "apos"}


def _resolve_entities(xml: str) -> str:
    """Substitute HTML entities that TEI files use but XML does not define.

    Perseus files are full of `&dagger;`, `&mdash;`, `&aelig;` and friends.
    ElementTree rejects them outright, and 436 of 777 Latin files -- 56% of the
    classical canon, including most of Cicero's letters -- fail to parse
    without this.
    """
    from html.entities import html5

    def sub(m):
        name = m.group(1)
        if name in _XML_BUILTIN:
            return m.group(0)
        ch = html5.get(name + ";") or html5.get(name)
        return ch if ch else " "

    return _ENTITY.sub(sub, xml)


def _tei_text(path) -> tuple[str, str]:
    """Return `(author, body_text)` for one TEI file.

    Perseus ships English translations beside the Latin, so callers must select
    the Latin files -- `iter_source` does. Inside a file, editorial matter is
    dropped by element name rather than by language test, which is both exact
    and cheap.
    """
    import xml.etree.ElementTree as ET

    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    root = ET.fromstring(_resolve_entities(raw))

    # Perseus mixes TEI P5 (namespaced) with older P4 files that have no
    # namespace at all. Matching on the local name handles both.
    def local(elem) -> str:
        return elem.tag.rsplit("}", 1)[-1] if isinstance(elem.tag, str) else ""

    author = ""
    for elem in root.iter():
        if local(elem) == "author" and elem.text and elem.text.strip():
            author = " ".join(elem.text.split())
            break

    body = next((e for e in root.iter() if local(e) == "text"), None)
    if body is None:
        return author, ""
    drop = set(_TEI_DROP)
    for parent in body.iter():
        for child in list(parent):
            if local(child) in drop:
                parent.remove(child)
    parts = []
    for elem in body.iter():
        for chunk in (elem.text, elem.tail):
            if chunk and chunk.strip():
                parts.append(" ".join(chunk.split()))
    return author, "\n".join(parts)


def _iter_tei(src: Source) -> Iterator[Document]:
    root = Path(src.path)
    for path in sorted(root.rglob("*.xml")):
        # Latin only. Perseus names translations `...perseus-eng1.xml`.
        if "-lat" not in path.name:
            continue
        try:
            author, text = _tei_text(path)
        except Exception as exc:
            print(f"[tei] skipping {path.name}: {exc!r}")
            continue
        if not text.strip():
            continue
        rel = path.relative_to(root)
        fallback = rel.parts[-2] if len(rel.parts) > 1 else path.stem
        yield Document(text=text, source=src.name, identifier=str(rel),
                       author=author or fallback)


def _iter_hf_csv(src: Source) -> Iterator[Document]:
    """Stream a CSV file from the Hub without downloading it whole.

    Two reasons not to use the `datasets` CSV loader here. It materialises the
    file -- 3.4 GB on job disk for a file we only ever read once, start to
    finish. And these rows are whole books: Python's csv module refuses fields
    over 128 KB until the limit is raised, and a loader that hits that fails on
    the largest and most valuable documents rather than the smallest.
    """
    import csv
    import sys

    import requests
    from huggingface_hub import get_token, hf_hub_url

    csv.field_size_limit(sys.maxsize)
    headers = {}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for name in (src.data_files or ()):
        url = hf_hub_url(src.path, name, repo_type="dataset")
        # A 3.4 GB stream over a multi-hour job will eventually meet a blip, and
        # without a retry that kills the whole job with no resume. This retries
        # the CONNECTION only. A failure mid-stream still raises: resuming would
        # need Range requests from a tracked byte offset, and a CSV row split
        # across the boundary makes that genuinely fiddly. Known gap.
        last_exc = None
        for attempt in range(4):
            try:
                resp = requests.get(url, headers=headers, stream=True, timeout=120)
                resp.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                print(f"[csv] {name} attempt {attempt + 1} failed: {exc!r}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"could not stream {name}") from last_exc
        with resp:
            resp.encoding = resp.encoding or "utf-8"
            lines = resp.iter_lines(decode_unicode=True)
            for row in csv.DictReader(lines):
                text = _pick(row, src.text_column)
                if not text:
                    continue
                yield Document(
                    text=text,
                    source=src.name,
                    identifier=_pick(row, src.id_column),
                    author=_pick(row, src.author_column),
                    date=_pick(row, src.date_column),
                )


def iter_source(src: Source) -> Iterator[Document]:
    if src.kind == "hf":
        yield from _iter_hf(src)
    elif src.kind == "jsonl":
        yield from _iter_jsonl(src)
    elif src.kind == "textdir":
        yield from _iter_textdir(src)
    elif src.kind == "tei":
        yield from _iter_tei(src)
    elif src.kind == "hf-csv":
        yield from _iter_hf_csv(src)
    else:
        raise ValueError(f"unknown source kind: {src.kind}")


def available(src: Source) -> bool:
    return src.kind in ("hf", "hf-csv") or os.path.isdir(src.path)


def interleave(sources: list[Source]) -> Iterator[Document]:
    """Weighted round-robin over sources.

    Each source gets a deficit counter proportional to its weight; the source
    with the largest deficit yields next. Sources with `repeat=True` restart
    when exhausted, which is how the small clean corpora get upweighted. The
    stream ends when every non-repeating source is exhausted.
    """
    if not sources:
        return
    # Keyed by position, not by name: a sample spread across shards is many
    # sources that share one name, and keying by name silently collapses them
    # into whichever came last.
    streams = {i: iter_source(s) for i, s in enumerate(sources)}
    credit = {i: 0.0 for i in range(len(sources))}
    live = set(streams)

    while live:
        # Give everyone credit proportional to weight, serve the hungriest.
        for i in live:
            credit[i] += sources[i].weight
        i = max(live, key=lambda k: credit[k])
        credit[i] -= sum(sources[k].weight for k in live)
        try:
            yield next(streams[i])
        except StopIteration:
            if sources[i].repeat and any(not sources[k].repeat for k in live if k != i):
                streams[i] = iter_source(sources[i])
                try:
                    yield next(streams[i])
                except StopIteration:  # source is genuinely empty
                    live.discard(i)
            else:
                live.discard(i)
