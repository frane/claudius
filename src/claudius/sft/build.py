"""Build the correction dataset from real Latin sentences.

Source sentences come from the prepared classical corpus -- genuine Latin, so
the "correct" side of every pair really is correct. They are split, filtered to
the length a learner would write, corrupted, and rendered.

THE DATASET IS NOT ALL CORRECTIONS. A model trained only on flawed sentence ->
correction learns that every student sentence is wrong, and will "correct"
perfectly good Latin. That is the fastest way to lose a learner's trust, and it
is invisible in a loss curve. So a fixed share of examples are correct sentences
the teacher simply affirms before continuing the conversation.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

from ..prepare import read_texts
from .errors import corrupt
from .template import LEVELS, STUDENT, TEACHER, Turn, _FOLLOW_UP, _PRAISE, correction_reply, render

# Split after terminal punctuation -- but NOT after a lone capital, because
# that is an abbreviated praenomen (M. Marcus, C. Gaius, P. Publius, L. Lucius)
# and Latin prose is full of them. Splitting there turns "P. Scipio" into a
# fragment ending "... EIUS P." and seeds garbage.
_SENT_END = re.compile(r"(?<!\s[A-Z]\.)(?<=[.!?])\s+")
_ABBREV_END = re.compile(r"(?:^|\s)[A-Z]\.$")

#: Multi-letter abbreviations that end in a full stop without ending a sentence.
#: Mostly editorial and citation apparatus -- CAP. is capitulum, LIB. liber --
#: plus the common titles. Normalised spelling (U for V).
_ABBREVIATIONS = frozenset({
    "CAP", "LIB", "UOL", "PAG", "CF", "ETC", "SC", "ID", "IB", "IBID", "UID",
    "UERS", "EP", "EPIST", "ED", "AD", "COS", "IMP", "TRIB", "PL", "SQ", "SQQ",
    "PRAEF", "COMM", "ANNOT", "FOL", "COL", "LIN", "N", "NR", "NUM",
})
_WORD = re.compile(r"[A-Z]+")


def extract_sentences(texts: Iterable[str], min_words: int = 4, max_words: int = 12
                      ) -> Iterator[str]:
    """Sentences a learner might plausibly write: short, complete, plain.

    Rejects fragments, verse lines with no terminal punctuation, anything with
    brackets or digits (editorial apparatus, line numbers, citations).
    """
    for text in texts:
        # Leading space so an abbreviation at the very start ("M. Tullius") has
        # the same whitespace-capital-dot shape the lookbehind tests for.
        for raw in _merge_abbreviations(_SENT_END.split(" " + " ".join(text.split()))):
            s = raw.strip()
            if not s or s[-1] not in ".!?":
                continue
            if _ABBREV_END.search(s):
                continue   # still ends on an abbreviation: a fragment
            if any(ch in s for ch in '[]()0123456789"'):
                continue
            n = len(_WORD.findall(s))
            if min_words <= n <= max_words:
                yield s


def _merge_abbreviations(pieces: list[str]) -> list[str]:
    """Rejoin pieces the splitter cut after a multi-letter abbreviation.

    A regex lookbehind cannot express a variable-length abbreviation list, so
    split first and repair after. "UIDE CAP." + "12 SUPRA." becomes one piece
    again -- and then the digits reject it as the citation it is.
    """
    out: list[str] = []
    for piece in pieces:
        if out:
            last = out[-1].rstrip().split()
            if last and last[-1].endswith(".") and last[-1][:-1] in _ABBREVIATIONS:
                out[-1] = out[-1].rstrip() + " " + piece.lstrip()
                continue
        out.append(piece)
    return out


def affirm_reply(rng: random.Random) -> str:
    """The student was right. Say so briefly, then keep talking."""
    return f"{rng.choice(_PRAISE)} {rng.choice(_FOLLOW_UP)}"


def build(sentences: Iterable[str], n: int, seed: int = 0,
          correct_share: float = 0.4, levels=LEVELS) -> list[dict]:
    """`n` examples, `correct_share` of them with no error to find."""
    rng = random.Random(seed)
    pool = list(sentences)
    if not pool:
        raise ValueError("no source sentences")
    out, attempts = [], 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        s = rng.choice(pool)
        level = rng.choice(levels)
        if rng.random() < correct_share:
            turns = [Turn(STUDENT, s), Turn(TEACHER, affirm_reply(rng))]
            out.append({"text": render(turns, level), "level": level,
                        "kind": "correct", "source": s})
            continue
        c = corrupt(s, rng)
        if c is None:
            continue
        turns = [Turn(STUDENT, c.corrupted), Turn(TEACHER, correction_reply(c, level, rng))]
        out.append({"text": render(turns, level), "level": level, "kind": c.kind,
                    "source": s, "wrong": c.wrong, "right": c.right})
    return out


def write(examples: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return path


def summary(examples: list[dict]) -> dict:
    kinds: dict[str, int] = {}
    levels: dict[str, int] = {}
    for ex in examples:
        kinds[ex["kind"]] = kinds.get(ex["kind"], 0) + 1
        levels[ex["level"]] = levels.get(ex["level"], 0) + 1
    return {"examples": len(examples), "by_kind": kinds, "by_level": levels}


def from_corpus(corpus_dir: str | Path, n: int, seed: int = 0, limit_docs: int = 0) -> list[dict]:
    return build(extract_sentences(read_texts(corpus_dir, "train", limit=limit_docs)), n, seed)


#: The Vulgate: simple narrative Latin in short verses, which is much closer to
#: what a learner writes than the literary prose of the classical corpus.
VULGATE_REPO = "professorf/latin-vulgate"


def vulgate_sentences(min_words: int = 4, max_words: int = 12, limit: int = 0) -> Iterator[str]:
    """Learner-register sentences from the Vulgate.

    Reads the `ltext` column ONLY. Every row also carries `etext`, an English
    translation, and the premise of the whole project is that the model has only
    ever seen Latin -- so the English column is never touched, not even to filter.

    Verses carry no terminal punctuation, so one is added.
    """
    from datasets import load_dataset

    from ..normalize import normalize

    ds = load_dataset(VULGATE_REPO, split="train", streaming=True)
    n = 0
    for row in ds:
        latin = (row.get("ltext") or "").strip()
        if not latin:
            continue
        s = normalize(latin)
        if not s:
            continue
        if s[-1] not in ".!?":
            s += "."
        words = len(_WORD.findall(s))
        if min_words <= words <= max_words:
            yield s
            n += 1
            if limit and n >= limit:
                return
