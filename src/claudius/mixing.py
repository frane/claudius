"""Weighted mixing across several prepared corpora.

The corpus is built in more than one job -- Latin-PD is 118 shards and hours of
CPU, the clean sources are a different shape entirely -- so the tokenizer and
packing stages have to read several corpus directories at once and sample them
in proportion, not concatenate them.

Proportion is the whole point. Latin-PD outweighs everything clean by roughly
30:1 on raw size; reading them end to end would train the tokenizer almost
entirely on OCR noise and put the clean text at the tail of the token stream
where a single-epoch run barely reaches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .prepare import read_shards


@dataclass(frozen=True)
class CorpusSpec:
    """One prepared corpus directory and the share of the mix it should get.

    WEIGHT IS PER DOCUMENT, NOT PER TOKEN. That distinction is not cosmetic
    here: Latin-PD documents are whole scanned books averaging ~800 KB, while
    the clean corpora average ~50 KB, so a 6:1 document weighting realises as
    roughly 1:2.7 by characters -- the opposite of what the number suggests.

    Use `by_chars=True` to weight by content instead, which divides the weight
    by the corpus's mean document size so the realised share of *text* matches
    the number asked for. Check the result with `mix_report` either way; weights
    are a request, not a guarantee.
    """

    path: str
    weight: float = 1.0
    #: Interpret `weight` as a share of characters rather than of documents.
    by_chars: bool = False
    #: Restart when exhausted, so a small corpus can hold its share of a long
    #: stream instead of running out early.
    repeat: bool = True
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or Path(self.path).name

    @classmethod
    def parse(cls, text: str) -> "CorpusSpec":
        """`path`, `path:weight`, or `path:weight:chars` for command lines."""
        by_chars = False
        if text.endswith(":chars"):
            text, by_chars = text[: -len(":chars")], True
        path, _, weight = text.rpartition(":")
        if path and weight:
            try:
                return cls(path=path, weight=float(weight), by_chars=by_chars)
            except ValueError:
                pass
        return cls(path=text, by_chars=by_chars)


def iter_mixed(specs: Sequence[CorpusSpec], split: str = "train",
               limit: int = 0) -> Iterator[tuple[str, str]]:
    """Yield `(corpus_label, text)` drawn from each corpus in proportion.

    Same credit scheduler as `corpora.interleave`: every live corpus accrues
    credit equal to its weight each round and the hungriest is served.

    Restart rule. A small corpus can only hold a 4:1 share against a corpus 30x
    its size by being read more than once, so `repeat` corpora restart when they
    run dry -- but only while the stream has a defined end, otherwise nothing
    ever terminates. With `limit` set (the normal case: a token budget), they
    restart freely. Without one, a repeating corpus restarts only while some
    non-repeating corpus is still producing, which is what bounds the stream.
    """
    if not specs:
        return
    effective = _effective_weights(specs, split)
    streams = {i: read_shards(s.path, split) for i, s in enumerate(specs)}
    credit = {i: 0.0 for i in range(len(specs))}
    live = set(streams)
    produced = 0

    while live:
        for i in live:
            credit[i] += effective[i]
        i = max(live, key=lambda k: credit[k])
        credit[i] -= sum(effective[k] for k in live)
        try:
            row = next(streams[i])
        except StopIteration:
            bounded = bool(limit) or any(
                not specs[k].repeat for k in live if k != i)
            if specs[i].repeat and bounded and len(live) > 1:
                streams[i] = read_shards(specs[i].path, split)
                try:
                    row = next(streams[i])
                except StopIteration:
                    live.discard(i)
                    continue
            else:
                live.discard(i)
                continue
        yield specs[i].label, row["text"]
        produced += 1
        if limit and produced >= limit:
            return


def _effective_weights(specs: Sequence[CorpusSpec], split: str,
                       probe: int = 60) -> list[float]:
    """Per-document weights, adjusting any `by_chars` spec for document size.

    A corpus whose documents are 16x larger needs 1/16th the document weight to
    contribute the same amount of text.
    """
    out = []
    for spec in specs:
        if not spec.by_chars:
            out.append(spec.weight)
            continue
        sizes = [len(r["text"]) for r, _ in zip(read_shards(spec.path, split), range(probe))]
        mean = (sum(sizes) / len(sizes)) if sizes else 1.0
        out.append(spec.weight / max(mean, 1.0))
    # Rescale so the numbers stay in a sane range for the credit scheduler.
    smallest = min((w for w in out if w > 0), default=1.0)
    return [w / smallest for w in out]


def mixed_texts(specs: Sequence[CorpusSpec], split: str = "train",
                limit: int = 0) -> Iterator[str]:
    for _, text in iter_mixed(specs, split, limit):
        yield text


def mix_report(specs: Sequence[CorpusSpec], split: str = "train",
               sample: int = 20_000) -> dict:
    """What the mix actually looks like, in documents and characters.

    Weights are a request, not a result: a corpus that runs out changes the
    realised proportions. Measure before committing a 20B-token run to them.
    """
    docs: dict[str, int] = {}
    chars: dict[str, int] = {}
    for label, text in iter_mixed(specs, split, limit=sample):
        docs[label] = docs.get(label, 0) + 1
        chars[label] = chars.get(label, 0) + len(text)
    total_chars = sum(chars.values()) or 1
    return {
        "sampled_documents": sum(docs.values()),
        "by_corpus": {
            label: {
                "documents": docs[label],
                "chars": chars[label],
                "share_of_chars": round(chars[label] / total_chars, 4),
            }
            for label in sorted(docs)
        },
    }
