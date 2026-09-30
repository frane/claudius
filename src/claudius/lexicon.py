"""Misspelled non-words: NEMEN -> NOMEN, NOMUS -> NOMEN, CLAUDUS -> CLAUDIUS.

The other detectors only know wrong ENDINGS of real words; real beginners also
misspell the word itself. For every uncommon word in the student's sentence:

1. candidates are frequent dictionary words within two edits (a SymSpell-style
   deletion index, so lookup is fast);
2. the language model scores the sentence with each candidate in its place;
3. the best candidate is offered -- as a question, never a verdict -- only if
   it beats the student's word by more than `margin` nats.

The dictionary is out/latin_lexicon.json: word counts from clean Latin only
(the teacher dialogues, the classical core, Vicipaedia). The pretraining corpus
cannot serve: it is mostly medieval, so learner words are rare in it (ROSAM 70,
AMBULO 92) while OCR typos are not (NEMEN 136). Names are not in the dictionary
either (FRANE); the model gate is what lets them through, and its false-alarm
rate on names is measured in scripts/eval_lexicon.py.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

WORD = re.compile(r"[A-Z]+")
#: A word seen at least KNOWN times in clean Latin is never questioned; words
#: seen at least POOL times are the candidate fixes.
KNOWN = 3
POOL = 20
MAX_EDITS = 2


def _deletes(word: str, depth: int) -> set[str]:
    out, frontier = {word}, {word}
    for _ in range(depth):
        frontier = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        out |= frontier
    return out


def _edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (transposition counts as one edit)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


@lru_cache(maxsize=1)
def _index(path: str = "out/latin_lexicon.json") -> tuple[dict, dict, dict]:
    lex = json.load(open(path))
    pool = {w: c for w, c in lex.items() if c >= POOL and len(w) >= 2}
    index: dict[str, list[str]] = {}
    for w in pool:
        for d in _deletes(w, MAX_EDITS):
            index.setdefault(d, []).append(w)
    return lex, pool, index


def candidates(word: str, limit: int = 12) -> list[str]:
    """Dictionary words within two edits of `word`, closest and most frequent first."""
    _, pool, index = _index()
    found = {}
    for d in _deletes(word, MAX_EDITS):
        for w in index.get(d, ()):
            if w != word and w not in found:
                dist = _edit_distance(word, w)
                if dist <= MAX_EDITS:
                    found[w] = dist
    # Closest spelling first, then the more frequent word: PUELA -> PUELLA,
    # not the frequent but two-edit UEL.
    return sorted(found, key=lambda w: (found[w], -pool[w]))[:limit]


def is_common(word: str) -> bool:
    return _index()[0].get(word, 0) >= KNOWN


#: The word after these is a name, whatever it looks like: "NOMEN MEUM EST FRANE"
#: was "fixed" to FRATER before this guard.
NAME_SLOT = re.compile(r"(?:NOMEN(?: (?:MEUM|TUUM|EIUS|MIHI|TIBI|EI))? (?:EST|ERAT)|\bSUM|\bES|\bUOCOR|"
                       r"\bAPPELLOR|CUI NOMEN EST|NOMINE|\bMAGISTER|\bAMICUS|\bAMICA|\bFRATER|\bSOROR)[ ,]*$")


def suspects(sentence: str) -> list[str]:
    """Uncommon words worth checking (three letters or more), except in a name slot."""
    out = []
    for m in WORD.finditer(sentence):
        w = m.group()
        if len(w) >= 3 and not is_common(w) and not NAME_SLOT.search(sentence[:m.start()]):
            out.append(w)
    return out


def check(model, tokenizer, sentence: str, margin: float = 12.0, margin2: float | None = 20.0):
    """(wrong, right, gain, edits) for the most convincing misspelling fix, or None.

    A two-letter change must clear `margin2`: names are usually two edits away
    from a Latin word (FRANE/FRATER, PAUEL/PATER), a beginner's typo usually one.
    Defaults from scripts/eval_lexicon.py (chosen on dev: no false alarm on 120
    clean sentences or 44 name sentences). Test: 0/120 clean and 0/44 names
    flagged, 83/120 typos caught, 75 fixed exactly.
    """
    margin2 = margin if margin2 is None else margin2
    from .sft.scorer import sentence_logprob

    base = None
    best = None
    for w in suspects(sentence):
        cands = candidates(w)
        if not cands:
            continue
        if base is None:
            base = sentence_logprob(model, tokenizer, sentence)
        for c in cands:
            alt = re.sub(rf"(?<![A-Z]){w}(?![A-Z])", c, sentence, count=1)
            gain = sentence_logprob(model, tokenizer, alt) - base
            edits = _edit_distance(w, c)
            if gain > (margin if edits == 1 else margin2) and (best is None or gain > best[2]):
                best = (w, c, gain, edits)
    return best
