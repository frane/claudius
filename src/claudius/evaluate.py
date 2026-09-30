"""Evaluation: bits-per-byte against a stated floor, and a memorisation check.

Three things are measured, all of them decided before training so that the
numbers cannot be chosen after seeing them:

1. Held-out bits-per-byte on a whole-author holdout. Per-token loss is not
   comparable across tokenizers; bits-per-byte is, which is what makes the
   vocabulary sweep decidable and what lets the 150M/400M/1B runs sit on one
   curve.
2. An interpolated character-bigram floor, fit on training text and reported
   next to the model. A language model that does not beat a bigram by a wide
   margin has learned nothing; stating the floor first stops it being a moving
   target.
3. A memorisation check: word n-gram windows of generated text searched
   verbatim against the training text, with a genuinely unseen author as a
   control. Borrowed from the Varro model card.

Normalised text is pure ASCII, so bytes and characters coincide -- but the code
counts UTF-8 bytes anyway so the numbers stay honest if that ever changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .normalize import ALPHABET, normalize

#: Standard openings, normalised. Continuations from these go in the model card.
INCIPITS = [
    normalize("Gallia est omnis divisa in partes tres"),
    normalize("Arma virumque cano, Troiae qui primus ab oris"),
    normalize("In principio creavit Deus caelum et terram"),
    normalize("Quousque tandem abutere, Catilina, patientia nostra"),
]

_WORD = re.compile(r"[A-Z0-9]+")


# -- floor -----------------------------------------------------------------
class InterpolatedBigram:
    """Character bigram interpolated with unigram and uniform.

    Deliberately simple and deliberately fit on the same training text the model
    sees, so the comparison is fair. Lambdas default to a fixed, stated split
    rather than being tuned on the holdout.
    """

    def __init__(self, alphabet: str = ALPHABET, lambdas: tuple[float, float, float] = (0.8, 0.19, 0.01)):
        self.chars = sorted(set(alphabet))
        self.index = {c: i for i, c in enumerate(self.chars)}
        self.n = len(self.chars)
        self.lambdas = lambdas
        self.bigram = np.zeros((self.n, self.n), dtype=np.float64)
        self.unigram = np.zeros(self.n, dtype=np.float64)
        self._fitted = False

    def _ids(self, text: str) -> np.ndarray:
        return np.fromiter(
            (self.index[c] for c in text if c in self.index), dtype=np.int64
        )

    def fit(self, texts: Iterable[str]) -> "InterpolatedBigram":
        for text in texts:
            ids = self._ids(text)
            if ids.size < 2:
                continue
            np.add.at(self.unigram, ids, 1.0)
            np.add.at(self.bigram, (ids[:-1], ids[1:]), 1.0)
        self._fitted = True
        return self

    def _probs(self) -> tuple[np.ndarray, np.ndarray]:
        l_bi, l_uni, l_unif = self.lambdas
        uni = self.unigram / max(self.unigram.sum(), 1.0)
        row_sums = self.bigram.sum(axis=1, keepdims=True)
        bi = np.divide(self.bigram, row_sums, out=np.zeros_like(self.bigram), where=row_sums > 0)
        # Rows never seen fall back entirely to the lower-order models.
        unseen = (row_sums.ravel() == 0)
        probs = l_bi * bi + l_uni * uni[None, :] + l_unif / self.n
        probs[unseen] = (l_uni + l_bi) * uni + l_unif / self.n
        return probs, l_uni * uni + l_unif / self.n + l_bi * uni

    def bits_per_byte(self, texts: Iterable[str]) -> float:
        if not self._fitted:
            raise RuntimeError("fit() the floor on training text first")
        probs, first = self._probs()
        total_bits = 0.0
        total_bytes = 0
        for text in texts:
            ids = self._ids(text)
            if ids.size == 0:
                continue
            p = np.empty(ids.size, dtype=np.float64)
            p[0] = first[ids[0]]
            if ids.size > 1:
                p[1:] = probs[ids[:-1], ids[1:]]
            total_bits += float(-np.log2(np.maximum(p, 1e-12)).sum())
            total_bytes += len(text.encode("utf-8"))
        if total_bytes == 0:
            return float("nan")
        return total_bits / total_bytes


# -- model bits-per-byte ---------------------------------------------------
@dataclass(frozen=True)
class BPBResult:
    bits_per_byte: float
    nats_per_token: float
    tokens: int
    bytes: int

    @property
    def perplexity(self) -> float:
        return math.exp(self.nats_per_token)


def bits_per_byte(model, tokenizer, texts: Sequence[str], seq_len: int = 2048, device=None) -> BPBResult:
    """Held-out bits-per-byte.

    Each text is encoded, cut into non-overlapping windows and scored. Total
    negative log-likelihood is divided by the total UTF-8 byte count of the
    *original* text, so the number is comparable across tokenizers and across
    model sizes.
    """
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors=None)["input_ids"]
            total_bytes += len(text.encode("utf-8"))
            for start in range(0, len(ids), seq_len):
                window = ids[start : start + seq_len]
                if len(window) < 2:
                    continue
                x = torch.tensor([window], device=device)
                out = model(input_ids=x, labels=x)
                n = len(window) - 1  # labels are shifted internally
                total_nll += float(out.loss) * n
                total_tokens += n
    if total_bytes == 0 or total_tokens == 0:
        return BPBResult(float("nan"), float("nan"), 0, 0)
    return BPBResult(
        bits_per_byte=total_nll / math.log(2) / total_bytes,
        nats_per_token=total_nll / total_tokens,
        tokens=total_tokens,
        bytes=total_bytes,
    )


# -- memorisation ----------------------------------------------------------
def _ngram_hashes(text: str, n: int) -> set[int]:
    words = _WORD.findall(text.upper())
    return {
        int.from_bytes(
            hashlib.blake2b(" ".join(words[i : i + n]).encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for i in range(len(words) - n + 1)
    }


@dataclass
class MemorisationResult:
    n: int
    windows: int
    verbatim: int
    control_windows: int
    control_verbatim: int

    @property
    def rate(self) -> float:
        return self.verbatim / max(self.windows, 1)

    @property
    def control_rate(self) -> float:
        return self.control_verbatim / max(self.control_windows, 1)

    def as_dict(self) -> dict:
        return {
            "ngram": self.n,
            "generated_windows": self.windows,
            "generated_verbatim": self.verbatim,
            "generated_rate": round(self.rate, 6),
            "control_windows": self.control_windows,
            "control_verbatim": self.control_verbatim,
            "control_rate": round(self.control_rate, 6),
        }


class MemorisationIndex:
    """Hashed word n-grams of the training text.

    Built from a sample, not the whole corpus: at n=13 the false-positive rate
    of a 64-bit hash is negligible and a sample keeps the index in memory. A hit
    means the model reproduced a training passage verbatim; the control tells
    you how many hits are just ordinary Latin phrasing.
    """

    def __init__(self, n: int = 13):
        self.n = n
        self._hashes: set[int] = set()

    def add(self, text: str) -> "MemorisationIndex":
        self._hashes |= _ngram_hashes(text, self.n)
        return self

    def add_all(self, texts: Iterable[str]) -> "MemorisationIndex":
        for t in texts:
            self.add(t)
        return self

    def __len__(self) -> int:
        return len(self._hashes)

    def _score(self, texts: Iterable[str]) -> tuple[int, int]:
        windows = hits = 0
        for text in texts:
            grams = _ngram_hashes(text, self.n)
            windows += len(grams)
            hits += len(grams & self._hashes)
        return windows, hits

    def check(self, generated: Iterable[str], control: Iterable[str]) -> MemorisationResult:
        """Compare generated text against a genuinely unseen author.

        `control` must be real Latin the model never saw. If the generated rate
        is not clearly above the control rate, the model is not regurgitating.
        """
        w, h = self._score(generated)
        cw, ch = self._score(control)
        return MemorisationResult(n=self.n, windows=w, verbatim=h, control_windows=cw, control_verbatim=ch)


# -- samples ---------------------------------------------------------------
def sample_incipits(model, tokenizer, prompts: Sequence[str] = tuple(INCIPITS),
                    max_new_tokens: int = 80, settings: dict | None = None) -> list[dict]:
    """Continue the standard openings with the anti-loop decoding settings.

    Plain sampling looped on 7.4% of 3-grams on the annealed 400M; these
    settings brought it to 0.0% (docs/decisions.md). Pass `settings=PLAIN`
    to see the raw model.
    """
    from .generate import ANTI_LOOP, generate

    out = []
    for prompt in prompts:
        prompt = normalize(prompt)
        out.append({"prompt": prompt,
                    "continuation": generate(model, tokenizer, prompt, max_new_tokens,
                                             settings or ANTI_LOOP)})
    return out
