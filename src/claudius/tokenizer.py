"""Byte-level BPE over normalised Latin, plus the vocabulary-size sweep.

Vocabulary size is a hyperparameter, not something BPE discovers. The procedure
decided in the brief: train one BPE to the largest size, truncate it to each
candidate size, measure fertility (tokens per word) on held-out text, and take
the elbow. Confirm the shortlist with held-out bits-per-byte from pilot runs --
`claudius eval` reports bits-per-byte precisely so that comparison is legal
across different vocabularies (per-token loss is not).

Truncation is exact, not an approximation: BPE merges are learned in order and
`tokenizers` assigns ids in that order, so the first N ids of a 64k tokenizer
are the vocabulary a run trained directly to N would have produced.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .normalize import LETTERS

EOS = "<|endoftext|>"
SPECIAL_TOKENS = [EOS]
DEFAULT_SWEEP = (4_096, 8_192, 16_384, 32_768, 65_536)
_WORD = re.compile(f"[{LETTERS}]+")


def classical_normalizer():
    """The classical orthography inside the tokenizer itself.

    Decompose and drop diacritics, uppercase, V -> U, J -> I, W -> UU. The same
    letters `normalize.normalize` produces, so plain Latin ("Salve, amice!")
    tokenizes exactly like the training text without a separate step. Special
    tokens are marked unnormalized and are not touched.
    """
    from tokenizers import Regex, normalizers

    steps = [normalizers.NFKD(), normalizers.StripAccents()]
    steps += [normalizers.Replace(c, c.upper()) for c in "abcdefghijklmnopqrstuvwxyz"]
    steps += [normalizers.Replace("V", "U"), normalizers.Replace("J", "I"),
              normalizers.Replace("W", "UU"), normalizers.Replace(Regex(r"[ \t]+"), " ")]
    return normalizers.Sequence(steps)


def add_normalizer(path: str | Path) -> Path:
    """Put the classical normalizer into an existing tokenizer.json."""
    from tokenizers import Tokenizer

    p = Path(path)
    tok = Tokenizer.from_file(str(p))
    tok.normalizer = classical_normalizer()
    tok.save(str(p))
    return p


def train_bpe(texts: Iterable[str], vocab_size: int = 65_536, min_frequency: int = 2):
    """Train a byte-level BPE on already-normalised text."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.normalizer = classical_normalizer()
    return tok


def _merge_parts(m) -> tuple[str, str]:
    if isinstance(m, (list, tuple)):
        return m[0], m[1]
    a, _, b = m.partition(" ")
    return a, b


def truncate_bpe(tok, target_vocab: int):
    """Return a new tokenizer using only the first `target_vocab` ids."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    state = json.loads(tok.to_str())
    vocab = state["model"]["vocab"]
    merges = state["model"]["merges"]
    kept = {t: i for t, i in vocab.items() if i < target_vocab}
    if len(kept) < target_vocab:
        raise ValueError(
            f"tokenizer has only {len(vocab)} tokens, cannot truncate to {target_vocab}"
        )
    kept_merges = []
    for m in merges:
        a, b = _merge_parts(m)
        if a in kept and b in kept and (a + b) in kept:
            kept_merges.append((a, b))
    out = Tokenizer(models.BPE(vocab=kept, merges=kept_merges, unk_token=None))
    out.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    out.decoder = decoders.ByteLevel()
    out.normalizer = classical_normalizer()
    return out


def wrap(tok):
    """Wrap a raw `tokenizers.Tokenizer` for use with transformers."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token=EOS,
        bos_token=EOS,
        pad_token=EOS,
        unk_token=None,
        model_max_length=10**12,
    )


@dataclass(frozen=True)
class Fertility:
    vocab_size: int
    tokens: int
    words: int
    bytes: int

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / max(self.words, 1)

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / max(self.tokens, 1)


def measure_fertility(tok, texts: Sequence[str], vocab_size: int | None = None) -> Fertility:
    """Tokens per word and bytes per token on held-out normalised text."""
    n_tokens = n_words = n_bytes = 0
    for text in texts:
        n_tokens += len(tok.encode(text).ids)
        n_words += len(_WORD.findall(text))
        n_bytes += len(text.encode("utf-8"))
    return Fertility(
        vocab_size=vocab_size if vocab_size is not None else tok.get_vocab_size(),
        tokens=n_tokens,
        words=n_words,
        bytes=n_bytes,
    )


def find_elbow(results: Sequence[Fertility], min_gain: float = 0.03) -> int | None:
    """The first vocabulary size past which doubling stops paying, or None.

    None means there is no elbow in the range measured -- every doubling still
    bought more than `min_gain`. That is the normal outcome for Latin, and it is
    not a detail to paper over: measured on clean text, doublings buy 9.9%, 8.3%,
    7.0% and 5.6% fewer tokens per word from 4k to 64k, never flattening. Heavy
    inflection gives Latin a genuinely large space of word forms, so fertility
    keeps improving as long as you keep buying vocabulary.

    A caller that silently treats None as "pick the largest" is not choosing a
    vocabulary size, it is deferring to the end of the sweep range.
    """
    ordered = sorted(results, key=lambda r: r.vocab_size)
    for cur, nxt in zip(ordered, ordered[1:]):
        gain = (cur.tokens_per_word - nxt.tokens_per_word) / cur.tokens_per_word
        if gain < min_gain:
            return cur.vocab_size
    return None


def pick_elbow(results: Sequence[Fertility], min_gain: float = 0.03) -> int:
    """`find_elbow`, falling back to the largest size measured.

    Kept so the pipeline always has a number, but read `find_elbow` first: when
    it returns None the fallback is arbitrary, and the real decision belongs to
    held-out bits-per-byte and the embedding-parameter cost.
    """
    elbow = find_elbow(results, min_gain)
    if elbow is not None:
        return elbow
    return max(r.vocab_size for r in results)


def sweep(
    train_texts: Sequence[str],
    holdout_texts: Sequence[str],
    sizes: Sequence[int] = DEFAULT_SWEEP,
    min_frequency: int = 2,
):
    """Train once at the largest size, evaluate every cutoff.

    Returns `(results, chosen, base)`. The trained tokenizer comes back so the
    caller can truncate it to the chosen size instead of training a second time
    -- BPE training is the expensive step, and on the full corpus doing it twice
    doubles the job.
    """
    sizes = sorted(sizes)
    base = train_bpe(train_texts, vocab_size=sizes[-1], min_frequency=min_frequency)
    actual = base.get_vocab_size()
    # BPE can run out of merges before reaching the requested ceiling on a small
    # sample. Clamp rather than fail: a corpus that yields fewer merges than the
    # largest candidate has answered the question by itself.
    candidates = [s for s in sizes if s <= actual]
    if actual not in candidates:
        candidates.append(actual)
    results = []
    for size in candidates:
        t = base if size == actual else truncate_bpe(base, size)
        results.append(measure_fertility(t, holdout_texts, vocab_size=size))
    return results, pick_elbow(results), base


def embedding_cost(vocab_size: int, hidden: int, params: int) -> str:
    """How much of a model the embedding table would be, for the sweep report."""
    emb = vocab_size * hidden
    return f"{emb / 1e6:.0f}M params ({100 * emb / params:.0f}% of the model)"


def format_sweep(results: Sequence[Fertility], chosen: int) -> str:
    lines = [f"{'vocab':>8}  {'tok/word':>9}  {'bytes/tok':>10}  {'vs prev':>8}"]
    prev = None
    for r in sorted(results, key=lambda x: x.vocab_size):
        delta = "" if prev is None else f"{(prev - r.tokens_per_word) / prev:+7.1%}"
        lines.append(
            f"{r.vocab_size:>8}  {r.tokens_per_word:>9.3f}  {r.bytes_per_token:>10.3f}  {delta:>8}"
        )
        prev = r.tokens_per_word
    if find_elbow(results) is None:
        lines.append(
            f"NO ELBOW: every doubling still bought >3% fewer tokens/word. "
            f"{chosen} is just the largest size measured, not a choice. "
            f"Decide on held-out bits-per-byte and embedding cost."
        )
    else:
        lines.append(f"elbow: {chosen}")
    return "\n".join(lines)


def tokenizer_card(results: Sequence[Fertility], chosen: int, corpus_note: str = "") -> dict:
    """The numbers the tokenizer card must report."""
    picked = next(r for r in results if r.vocab_size == chosen)
    elbow = find_elbow(results)
    return {
        "vocab_size": chosen,
        "elbow_found": elbow is not None,
        "elbow_note": (
            "no elbow in the measured range -- fertility keeps improving, so this "
            "size is the end of the sweep rather than a decision"
            if elbow is None else f"elbow at {elbow}"
        ),
        "algorithm": "byte-level BPE",
        "normalisation": "classical (uppercase, U for V, I for J, no diacritics)",
        "fertility_tokens_per_word": round(picked.tokens_per_word, 4),
        "bytes_per_token": round(picked.bytes_per_token, 4),
        "sweep": [
            {"vocab_size": r.vocab_size, "tokens_per_word": round(r.tokens_per_word, 4)}
            for r in sorted(results, key=lambda x: x.vocab_size)
        ],
        "holdout": "whole-author holdout, unseen during BPE training",
        "corpus": corpus_note,
    }


def save(tok, path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    wrap(tok).save_pretrained(str(path))
    return path


def load(path: str | Path):
    """Load a saved tokenizer from a local directory.

    Checked explicitly: `AutoTokenizer` treats an absent local path as a Hub
    repo id and reports it as a malformed repo name, which says nothing about
    the actual problem.
    """
    from transformers import AutoTokenizer

    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f"no tokenizer directory at {path} -- run `claudius tokenizer` first")
    if not (path / "tokenizer.json").exists():
        raise SystemExit(f"{path} has no tokenizer.json (is the run still in progress?)")
    return AutoTokenizer.from_pretrained(str(path))


def entropy_floor_bits(results: Sequence[Fertility]) -> float:
    """Convenience: log2(vocab) upper bound on bits per token, for sanity checks."""
    return math.log2(max(r.vocab_size for r in results))
