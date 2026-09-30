"""Near-duplicate removal.

Latin-PD has 159k titles but far fewer works: the same edition of Cicero was
scanned by three libraries, and the same commentary was reprinted for two
centuries. Training on that is training on a handful of books many times over.

Two levels, as decided:

* document level -- MinHash + LSH over word shingles, which catches reprints
  and re-scans where the OCR differs slightly;
* paragraph level -- exact hashing, which catches the boilerplate (title pages,
  library plates, licence notices, running apparatus) shared across documents
  that are otherwise distinct.

Paragraph level is exact rather than fuzzy on purpose: fuzzy paragraph matching
over billions of paragraphs costs more than it buys, and the boilerplate this
targets is repeated verbatim.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

_MERSENNE = np.uint64((1 << 31) - 1)
_WORD = re.compile(r"[A-Z0-9]+")


def _hash32(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=4).digest(), "big") & 0x7FFFFFFF


def shingles(text: str, k: int = 5) -> np.ndarray:
    """Hashed word k-grams, deduplicated. Empty array if the text is too short."""
    words = _WORD.findall(text.upper())
    if len(words) < k:
        return np.empty(0, dtype=np.uint64)
    grams = {
        _hash32(" ".join(words[i : i + k]).encode("utf-8"))
        for i in range(len(words) - k + 1)
    }
    return np.fromiter(sorted(grams), dtype=np.uint64, count=len(grams))


class MinHasher:
    """Fixed, seeded permutations so signatures are reproducible across runs."""

    def __init__(self, num_perm: int = 128, seed: int = 0xC1A0D1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, int(_MERSENNE), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE), size=num_perm, dtype=np.uint64)

    def signature(self, text: str, k: int = 5, chunk: int = 4096) -> np.ndarray | None:
        s = shingles(text, k)
        if s.size == 0:
            return None
        # a*x + b mod (2^31 - 1); a, x < 2^31 so the product fits in uint64.
        # Chunked because a whole book can have millions of shingles and the
        # permutation matrix is num_perm wide.
        out = np.full(self.num_perm, _MERSENNE, dtype=np.uint64)
        for start in range(0, s.size, chunk):
            block = s[start : start + chunk]
            h = (self.a[:, None] * block[None, :] + self.b[:, None]) % _MERSENNE
            np.minimum(out, h.min(axis=1), out=out)
        return out.astype(np.uint32)


def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard similarity from two MinHash signatures."""
    return float((sig_a == sig_b).mean())


def _bands(num_perm: int, threshold: float, min_recall: float = 0.9) -> tuple[int, int]:
    """Pick (bands, rows) that CATCH pairs at `threshold`, not merely half of them.

    The earlier version chose the factorisation whose S-curve crosses 0.5 at the
    threshold. That is the wrong target: it means a pair sitting exactly at the
    stated threshold is a coin flip, and the deduper silently behaves like a
    much stricter one. With 128 permutations at threshold 0.8 it chose 8 bands
    of 16 rows, which catches 20.4% of true near-duplicates where the standard
    16 x 8 catches 94.7%.

    Now: take the largest number of rows that still reaches `min_recall` at the
    threshold. More rows means fewer candidate pairs to verify, so this is the
    cheapest banding that actually does the job.
    """
    best = (num_perm, 1)
    for r in range(1, num_perm + 1):
        if num_perm % r:
            continue
        bands = num_perm // r
        recall = 1.0 - (1.0 - threshold**r) ** bands
        if recall >= min_recall:
            best = (bands, r)   # loop ascends r, so this keeps the largest
    return best


def paragraph_hashes(text: str, min_paragraph_chars: int = 200) -> list[int]:
    """Hash of every paragraph long enough to be worth deduplicating.

    Index i corresponds to `text.split("\n\n")[i]`; short paragraphs get None
    so the positions stay aligned with the document.
    """
    out = []
    for para in text.split("\n\n"):
        if len(para) < min_paragraph_chars:
            out.append(0)
        else:
            out.append(_hash32(re.sub(r"\s+", " ", para).strip().encode("utf-8")))
    return out


@dataclass(frozen=True)
class Fingerprint:
    """Everything the global dedup pass needs, without the document text.

    A whole corpus of these is small -- 512 bytes of signature per document --
    so the expensive half (shingling, hashing) parallelises across shards and
    the decision half runs once, centrally, over a few dozen megabytes.
    """

    signature: np.ndarray | None
    paragraphs: tuple[int, ...]


def fingerprint(text: str, hasher: MinHasher, shingle_k: int = 5,
                min_paragraph_chars: int = 200) -> Fingerprint:
    return Fingerprint(
        signature=hasher.signature(text, shingle_k),
        paragraphs=tuple(paragraph_hashes(text, min_paragraph_chars)),
    )


@dataclass
class DedupStats:
    seen: int = 0
    kept: int = 0
    dup_docs: int = 0
    dup_paragraphs: int = 0
    empty: int = 0

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "duplicate_documents": self.dup_docs,
            "duplicate_paragraphs": self.dup_paragraphs,
            "unhashable": self.empty,
        }


@dataclass
class Deduper:
    """Streaming deduplicator. First occurrence wins.

    Memory is the limit on a single pass: ~1 KB of signature per kept document
    plus 8 bytes per distinct paragraph. For 100k documents that is trivial; if
    the corpus grows past a few million documents, shard by MinHash band and
    run one shard per process.
    """

    threshold: float = 0.8
    num_perm: int = 128
    shingle_k: int = 5
    #: Paragraphs shorter than this are never deduplicated -- short repeated
    #: lines ("CAPUT I") are legitimately common.
    min_paragraph_chars: int = 200
    dedup_paragraphs: bool = True

    hasher: MinHasher = field(init=False)
    stats: DedupStats = field(default_factory=DedupStats)

    def __post_init__(self):
        self.hasher = MinHasher(self.num_perm)
        self.n_bands, self.n_rows = _bands(self.num_perm, self.threshold)
        self._buckets: dict[bytes, list[int]] = {}
        self._sigs: list[np.ndarray] = []
        self._paragraphs: set[int] = set()

    # -- document level ----------------------------------------------------
    # KNOWN GAP: candidates are verified with the MinHash ESTIMATE, not the exact
    # shingle Jaccard. At 128 permutations the estimate has ~0.04 std near 0.8,
    # so a pair at true 0.74 can be accepted. Exact verification would need the
    # shingle sets, which the parallel reduce does not have (it holds only
    # fingerprints, by design). Measured on real Latin-PD: 404,550 pairs, none
    # above 0.5, so it costs nothing here. The knob, if a future corpus has mass
    # near the threshold, is num_perm=256 (halves the noise, ~1 KB per doc).
    def _is_duplicate(self, sig: np.ndarray) -> bool:
        keys = [
            bytes([b]) + sig[b * self.n_rows : (b + 1) * self.n_rows].tobytes()
            for b in range(self.n_bands)
        ]
        candidates = {i for k in keys for i in self._buckets.get(k, ())}
        for i in candidates:
            if jaccard(sig, self._sigs[i]) >= self.threshold:
                return True
        idx = len(self._sigs)
        self._sigs.append(sig)
        for k in keys:
            self._buckets.setdefault(k, []).append(idx)
        return False

    # -- paragraph level ---------------------------------------------------
    def _strip_repeated_paragraphs(self, text: str) -> str:
        out, dropped = [], 0
        for para in text.split("\n\n"):
            if len(para) < self.min_paragraph_chars:
                out.append(para)
                continue
            h = _hash32(re.sub(r"\s+", " ", para).strip().encode("utf-8"))
            if h in self._paragraphs:
                dropped += 1
                continue
            self._paragraphs.add(h)
            out.append(para)
        self.stats.dup_paragraphs += dropped
        return "\n\n".join(p for p in out if p.strip())

    def add(self, text: str) -> str | None:
        """Return the (possibly shortened) text to keep, or None to drop it."""
        self.stats.seen += 1
        sig = self.hasher.signature(text, self.shingle_k)
        if sig is None:
            self.stats.empty += 1
            return None
        if self._is_duplicate(sig):
            self.stats.dup_docs += 1
            return None
        if self.dedup_paragraphs:
            text = self._strip_repeated_paragraphs(text)
            if not text:
                self.stats.dup_docs += 1
                return None
        self.stats.kept += 1
        return text


class GlobalDedup:
    """The keep/drop decision, made centrally over fingerprints.

    First occurrence wins, so the result depends only on the order fingerprints
    are added -- feed them in a deterministic order (shard, then row) and the
    corpus is reproducible.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.n_bands, self.n_rows = _bands(num_perm, threshold)
        self._buckets: dict[bytes, list[int]] = {}
        self._sigs: list[np.ndarray] = []
        self._paragraphs: set[int] = set()
        self.stats = DedupStats()

    def _is_duplicate(self, sig: np.ndarray) -> bool:
        keys = [
            bytes([b]) + sig[b * self.n_rows : (b + 1) * self.n_rows].tobytes()
            for b in range(self.n_bands)
        ]
        for i in {i for k in keys for i in self._buckets.get(k, ())}:
            if jaccard(sig, self._sigs[i]) >= self.threshold:
                return True
        idx = len(self._sigs)
        self._sigs.append(sig)
        for k in keys:
            self._buckets.setdefault(k, []).append(idx)
        return False

    def decide(self, fp: Fingerprint) -> tuple[bool, set[int]]:
        """Return `(keep, paragraph_indices_to_drop)` for one document."""
        self.stats.seen += 1
        if fp.signature is None:
            self.stats.empty += 1
            return False, set()
        if self._is_duplicate(fp.signature):
            self.stats.dup_docs += 1
            return False, set()
        drop = set()
        for i, h in enumerate(fp.paragraphs):
            if not h:
                continue
            if h in self._paragraphs:
                drop.add(i)
            else:
                self._paragraphs.add(h)
        self.stats.dup_paragraphs += len(drop)
        self.stats.kept += 1
        return True, drop


def apply_paragraph_drops(text: str, drop: set[int]) -> str:
    if not drop:
        return text
    paras = text.split("\n\n")
    return "\n\n".join(p for i, p in enumerate(paras) if i not in drop and p.strip())
