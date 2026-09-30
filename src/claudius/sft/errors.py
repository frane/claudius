"""Synthetic learner errors, for building correction pairs.

The buddy's core move is: student writes flawed Latin, buddy corrects it gently
and asks a follow-up. No corpus of real student errors exists, so errors are
manufactured from correct Latin -- and they have to be the errors students
actually make, not random noise. A model trained to fix typos learns nothing
useful; a model trained to fix wrong case endings learns the job.

Every corruption here is deterministic given a seed and returns the edit it
made, so a correction pair always knows exactly what was wrong. That matters
for the correctness guard: the morphological checker grades, and it can be
checked against the known answer.

Operates on NORMALISED text (uppercase, U for V, I for J), because that is the
only orthography the model has ever seen.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

_WORD = re.compile(r"[A-Z]+")

# Common endings a learner confuses. Each group is a set of forms that are
# individually valid Latin but wrong for a given slot -- which is the point: the
# corrupted sentence is made of real words, so the error is grammatical, not
# orthographic.
_CASE_SWAPS = [
    ("AM", "A"), ("AM", "AE"), ("A", "AM"), ("AE", "AM"),       # 1st decl acc/nom/gen
    ("UM", "US"), ("US", "UM"), ("UM", "O"), ("O", "UM"),       # 2nd decl
    ("EM", "IS"), ("IS", "EM"), ("EM", "E"), ("E", "EM"),       # 3rd decl
    ("AS", "AE"), ("OS", "I"), ("ES", "UM"),                    # plurals
]
_VERB_SWAPS = [
    ("AT", "ANT"), ("ANT", "AT"), ("ET", "ENT"), ("ENT", "ET"),  # 3sg <-> 3pl
    ("IT", "UNT"), ("UNT", "IT"), ("AT", "O"), ("ET", "EO"),
    ("AUIT", "ABAT"), ("ABAT", "AUIT"), ("EBAT", "IT"),          # perfect <-> imperfect
    ("IT", "ET"), ("ET", "IT"),                                  # conjugation slips
]
# Short function words whose ending a learner would never mangle.
_PROTECT = frozenset({
    "ET", "IN", "EST", "NON", "CUM", "AD", "UT", "SED", "DE", "EX", "SI", "AB",
    "QUI", "QUAE", "QUOD", "SUNT", "ESSE", "PER", "NEC", "AUT", "IAM", "ETIAM",
})


@dataclass(frozen=True)
class Corruption:
    kind: str          # "case" | "verb" | "order"
    original: str      # the correct sentence
    corrupted: str     # what the "student" wrote
    wrong: str         # the word as the student wrote it
    right: str         # the word as it should be
    position: int      # word index of the error


_TRAIL = re.compile(r"^([A-Z]+)([^A-Z]*)$")


def _words(sentence: str) -> list[str]:
    return sentence.split()


def _split(token: str) -> tuple[str, str] | None:
    """`AMAT.` -> (`AMAT`, `.`). Punctuation rides along, never gets corrupted.

    This is not a detail. Latin prose puts the verb last, so the verb is exactly
    the word most likely to carry a full stop -- a letters-only match skips it,
    and verb errors almost never get generated.
    """
    m = _TRAIL.match(token)
    return (m.group(1), m.group(2)) if m else None


def _swap_ending(token: str, swaps) -> str | None:
    parts = _split(token)
    if parts is None:
        return None
    word, trail = parts
    if word in _PROTECT or len(word) < 4:
        return None
    for old, new in swaps:
        if word.endswith(old) and len(word) - len(old) >= 2:
            return word[: -len(old)] + new + trail
    return None


def corrupt_case(sentence: str, rng: random.Random) -> Corruption | None:
    """Swap a noun-like ending for a wrong case: PUELLAM -> PUELLA."""
    return _corrupt_ending(sentence, rng, _CASE_SWAPS, "case")


def corrupt_verb(sentence: str, rng: random.Random) -> Corruption | None:
    """Swap a verb ending for the wrong person or tense: AMAT -> AMANT."""
    return _corrupt_ending(sentence, rng, _VERB_SWAPS, "verb")


def _corrupt_ending(sentence, rng, swaps, kind) -> Corruption | None:
    words = _words(sentence)
    candidates = [i for i, w in enumerate(words) if _swap_ending(w, swaps)]
    if not candidates:
        return None
    i = rng.choice(candidates)
    wrong_token = _swap_ending(words[i], swaps)
    corrupted = words[:i] + [wrong_token] + words[i + 1:]
    # Record the bare words, without punctuation, so the correction is clean.
    wrong = _split(wrong_token)[0]
    right = _split(words[i])[0]
    return Corruption(kind, sentence, " ".join(corrupted), wrong, right, i)


def corrupt_order(sentence: str, rng: random.Random) -> Corruption | None:
    """Move the verb out of final position, the classic learner-of-English slip.

    Latin prose puts the verb last far more often than English does, and
    learners import English order. This is a register error rather than a
    grammatical one -- the sentence still parses -- so the buddy should correct
    it more gently than a wrong ending.
    """
    words = _words(sentence)
    if len(words) < 4:
        return None
    last = words[-1].rstrip(".,;:!?")
    trailing = words[-1][len(last):]
    if not re.search(r"(T|NT|UR|NTUR|IT|UIT)$", last):
        return None
    moved = words[:1] + [last] + words[1:-1]
    moved[-1] = moved[-1] + trailing
    return Corruption("order", sentence, " ".join(moved), last, last, len(words) - 1)


CORRUPTIONS: dict[str, Callable[[str, random.Random], Corruption | None]] = {
    "case": corrupt_case,
    "verb": corrupt_verb,
    "order": corrupt_order,
}


def corrupt(sentence: str, rng: random.Random, kinds=("case", "verb", "order")) -> Corruption | None:
    """Apply one error of a randomly chosen kind; None if the sentence has no target."""
    order = list(kinds)
    rng.shuffle(order)
    for kind in order:
        c = CORRUPTIONS[kind](sentence, rng)
        if c is not None and c.corrupted != c.original:
            return c
    return None
