"""Split fused words (DEDECOREATQUE -> DEDECORE ATQUE) using corpus frequencies.

OCR drops spaces between words, and the model learned to write fused words from
it. Splitting needs a lexicon to be safe; the lexicon here is word frequency
from our own corpus. A word is split only when BOTH hold:

* it is itself rare -- a real word, even a long compound like RESPUBLICA, is
  common enough to be left alone;
* it breaks into two words that are each common.

Enclitics are never split off: -QUE, -NE, -UE attach to words legitimately
(DEDECOREQUE is correct Latin), so they are never accepted as the right half.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ENCLITICS = frozenset({"QUE", "NE", "UE", "CUM"})
#: Two-letter halves must be real function words. Without this, rare real words
#: split into junk: MITIGATOS -> MITIGAT OS, BALATRONIBUS -> BA LATRONIBUS.
SHORT_WORDS = frozenset({"ET", "IN", "AD", "EX", "UT", "AB", "DE", "SI", "NE",
                         "EO", "ID", "IS", "EA", "ME", "TE", "SE", "NI", "OB", "IAM"})
_WORD = re.compile(r"[A-Z]+")


@lru_cache(maxsize=1)
def _freq(path: str) -> dict:
    return json.loads(Path(path).read_text())


_STEM_CACHE: dict[int, frozenset] = {}


def _stems(freq: dict, min_freq: int = 20) -> frozenset:
    """Prefixes of common words with their last 1-4 letters removed.

    If a rare word's own stem is in here, it is an inflection of a common word
    (MITIGATOS next to MITIGATUS, MITIGATA), not two words run together.
    """
    key = id(freq)
    if key not in _STEM_CACHE:
        _STEM_CACHE[key] = frozenset(
            w[:-k] for w, c in freq.items() if c >= min_freq
            for k in (1, 2, 3, 4) if len(w) - k >= 5)
    return _STEM_CACHE[key]


def split_word(word: str, freq: dict, max_rare: int = 2, min_part: int = 150,
               min_len: int = 9) -> list[str]:
    """[word] unchanged, or [left, right] if it is a confident fusion."""
    if len(word) < min_len or freq.get(word, 0) > max_rare:
        return [word]
    stems = _stems(freq)
    if any(word[:-k] in stems for k in (2, 3, 4) if len(word) - k >= 5):
        return [word]   # an inflection of a common word, not a fusion
    best, best_score = None, 0
    for i in range(2, len(word) - 1):
        left, right = word[:i], word[i:]
        if right in ENCLITICS or len(right) < 2:
            continue
        if (len(left) == 2 and left not in SHORT_WORDS) or (len(right) == 2 and right not in SHORT_WORDS):
            continue
        fl, fr = freq.get(left, 0), freq.get(right, 0)
        if fl >= min_part and fr >= min_part:
            score = min(fl, fr)
            if score > best_score:
                best, best_score = [left, right], score
    return best or [word]


def segment(text: str, freq_path: str = "out/latin_wordfreq.json", **kw) -> str:
    freq = _freq(freq_path)
    return _WORD.sub(lambda m: " ".join(split_word(m.group(), freq, **kw)), text)
