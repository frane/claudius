"""Corpus quality: what the released model's limitations trace back to.

Three defects in the 400M's output come straight from its training text:

- EDITION NUMBERS. Verse and section numbers printed by the source editions
  ("... AB AEUO 355") are learned as if they were Latin. `strip_edition_numbers`
  removes standalone Arabic numerals. Roman numerals are words and stay.
  The cost is neo-Latin dates written in digits (1644), which a Latin tutor
  does not need.
- FUSED WORDS. Text where spaces were lost (LUSITANORUMCUM) teaches the model to
  run words together. `fused_rate` estimates the share of long words that are
  two common words glued together; documents above a threshold are dropped.
- CHURCH REGISTER. 95% of the pretraining text is medieval and ecclesiastical,
  so unprompted output drifts to sermons. `church_rate` measures it, per 1,000
  words; the mix is rebalanced against it.

The measures are heuristics and are used as rates over many words, never to
judge a single word.
"""
from __future__ import annotations

import re

_NUMBER = re.compile(r"(?<![A-Z0-9])\d{1,4}(?![A-Z0-9])")
#: Section numbers glued to the next or previous word by the edition: "4SED", "EST.N. 4".
_GLUED_NUMBER = re.compile(r"(?<![A-Z0-9])\d+(?=[A-Z])|(?<=[A-Z])\d+(?![A-Z0-9])")
_SECTION_MARK = re.compile(r"(?<![A-Z])N\. ?(?=\d)")
_EMPTY_BRACKETS = re.compile(r"[\[(]\s*(?:\.\.\.|…)?\s*[\])]")
_SPACES = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r" +([,.;:!?])")
_WORD = re.compile(r"[A-Z]+")

#: Stems that mark church Latin. Classical authors say DEUS too, which is why
#: this is a rate compared across sources, not a filter on single words.
CHURCH_STEMS = ("DEU", "DEI", "DEO", "DOMIN", "CHRIST", "IESU", "ECCLESI", "SANCT",
                "EUANGELI", "APOSTOL", "EPISCOP", "PECCAT", "SPIRITU", "BAPTI",
                "PROPHET", "ANGEL", "DIABOL", "MARTYR", "PRESBYTER", "SACRAMENT")


_LINE_EDGE_NUMBER = re.compile(r"^\s*[\[(]?\d{1,4}[\])]?[.,:]?(?=\s|$)|(?<=\s)[\[(]?\d{1,4}[\])]?\s*$")
_BRACKET_NUMBER = re.compile(r"[\[(]\d{1,4}[\])]")
_SENTENCE = re.compile(r"(?<=[.;:?!])\s+")
_DIGIT = re.compile(r"\d")


def strip_edition_numbers(text: str) -> str:
    """Remove edition numbering; drop sentences that carry numbers as content.

    Edition numbers -- verse and section numbers at a line's start or end,
    bracketed ones, ones glued to a word (4SED), "N. 4" marks -- are not part of
    the Latin and are deleted. A number inside a sentence is content ("inter
    annos 27 et 14 a.C."): deleting it leaves broken Latin ("INTER ANNOS ET
    A.C.", 22,561 such holes in the first anneal data, and the model learned
    them), so the whole sentence goes instead.
    """
    text = _SECTION_MARK.sub(" ", text)
    text = _GLUED_NUMBER.sub(" ", text)
    text = _BRACKET_NUMBER.sub(" ", text)
    lines = []
    for line in text.split("\n"):
        prev = None
        while prev != line:                      # "12 ARMA ... CANO 355"
            prev, line = line, _LINE_EDGE_NUMBER.sub(" ", line)
        if _DIGIT.search(line):
            line = " ".join(s for s in _SENTENCE.split(line) if not _DIGIT.search(s))
        line = _EMPTY_BRACKETS.sub(" ", line)
        lines.append(_SPACE_BEFORE_PUNCT.sub(r"\1", _SPACES.sub(" ", line)).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def number_rate(text: str) -> float:
    """Standalone Arabic numerals per 1,000 words."""
    words = len(_WORD.findall(text)) or 1
    return 1000 * len(_NUMBER.findall(text)) / words


def church_rate(text: str) -> float:
    """Words with a church-Latin stem, per 1,000 words."""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    return 1000 * sum(w.startswith(CHURCH_STEMS) for w in words) / len(words)


def fused_rate(text: str, freq: dict, min_len: int = 9) -> float:
    """Share of long words (>= min_len letters) that look like two words glued.

    Uses the frequency splitter: a rare word that splits into two common ones.
    On clean text the splitter is wrong for about 0.2% of distinct long words
    (HILARISSIMUS), so only rates well above that mean real fusion.
    """
    from .segment import split_word

    long = [w for w in _WORD.findall(text) if len(w) >= min_len]
    if not long:
        return 0.0
    return sum(len(split_word(w, freq)) > 1 for w in long) / len(long)


_SENT = re.compile(r"(?<=[.;:?!])\s+")


def repair_fusions(text: str, model, tokenizer, gate: float = 2.0) -> str:
    """Split glued words (FRATERINUENIT) where a Latin model prefers the split.

    The model-gated splitter: 0 wrong splits in 3,000 clean sentences, about
    55% of fused pairs repaired. The frequency splitter alone cannot be used on
    corpus text: it "splits" rare scholastic words (CONSIGNIFICAT, PRAECOGNOSCERE).
    For a full retrain, run this over latin-literature (29 fused words per 1,000
    words, versus 6.6 in the classical core) before tokenizing.
    """
    from .tutor import split_fused

    out = []
    for line in text.split("\n"):
        out.append(" ".join(split_fused(model, tokenizer, s, gate) if len(s.split()) >= 3 else s
                            for s in _SENT.split(line)))
    return "\n".join(out)
