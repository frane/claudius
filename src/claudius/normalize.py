"""Classical normalisation.

Everything that reaches the tokenizer -- corpus, prompts at inference, SFT data --
goes through `normalize()` first. The rules are deterministic and lossy on purpose:
the model should see one orthography, not four centuries of typographic fashion.

Target alphabet: the 23-letter classical Latin alphabet, uppercase, plus space,
Arabic digits and a small punctuation whitelist. J/V/W do not survive.
"""

from __future__ import annotations

import re
import unicodedata

#: The 23 letters a normalised document may contain.
LETTERS = "ABCDEFGHIKLMNOPQRSTUXYZ"
DIGITS = "0123456789"
#: Punctuation that survives normalisation. Everything else becomes a space.
PUNCT = ".,;:!?'\"()-[]"
ALPHABET = LETTERS + DIGITS + PUNCT + " \n"

# Ligatures and archaic letterforms, applied before case folding and before
# diacritic stripping (some of these carry combining marks of their own).
_LIGATURES = {
    "Æ": "AE", "æ": "AE",   # Æ æ
    "Œ": "OE", "œ": "OE",   # Œ œ
    "ſ": "S",                     # ſ long s
    "ꝛ": "RUM", "ꝝ": "RUM",  # ꝛ-style rum abbreviations seen in OCR
    "ẞ": "SS", "ß": "SS",    # ẞ ß (German intrusions)
    "ﬀ": "FF", "ﬁ": "FI", "ﬂ": "FL",
    "ﬃ": "FFI", "ﬄ": "FFL", "ﬅ": "ST", "ﬆ": "ST",
}

# Punctuation and whitespace variants folded onto the whitelist.
_PUNCT_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "―": "-", "‐": "-", "‑": "-",
    "‒": "-", "­": "-",
    "…": "...", "·": ".", "‧": ".",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "⁄": "/", "∕": "/",
}

# Scripts that are dropped outright rather than transliterated. The decision in
# the brief was: drop Greek, consistently. Everything else non-Latin is OCR noise
# or a foreign-language passage we do not want the model to learn from.
_FOREIGN_SCRIPT = re.compile(
    "["
    "Ͱ-Ͽἀ-῿"      # Greek and Coptic, Greek Extended
    "Ѐ-ԯ"                    # Cyrillic
    "֐-׿؀-ۿ"      # Hebrew, Arabic
    "Ⲁ-⳿"                    # Coptic
    "　-鿿가-힯"      # CJK, Hangul
    "]+"
)

# Line-break hyphenation: "propo-\nsitum" -> "propositum". OCR uses several
# characters for the break hyphen, and sometimes doubles it.
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑­¬=]{1,2}[ \t]*\n[ \t]*(?=\w)")

_COMBINING = re.compile(r"[̀-ͯ᪰-᫿⃐-⃰]")
_ALLOWED = re.compile("[^" + re.escape(LETTERS + DIGITS + PUNCT) + " \n]+")
_SPACES = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r" +([.,;:!?)\]])")
# OCR drops the space after punctuation ~4,000 times per million characters of
# Latin-PD ("CONUENIAT,CUIUS", "PROBANDUM,ATQUE"), and the 400M learned to write
# "RABIE,IN". Restore it. Letters only on the right, so decimals are untouched.
_MISSING_SPACE = re.compile(r"([,;:!?.])(?=[A-Z])")
_REPEAT_PUNCT = re.compile(r"([.,;:!?'\"()\[\]-])\1{3,}")

_LIG_RE = re.compile("|".join(map(re.escape, _LIGATURES)))
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_FOLD)))


def join_hyphenation(text: str) -> str:
    """Rejoin words broken across a line by a hyphen."""
    return _HYPHEN_BREAK.sub(r"\1", text)


def strip_diacritics(text: str) -> str:
    """Decompose, drop combining marks, recompose.

    Macrons, breves, apices and the accents that French and German printers left
    on Latin text all disappear. Vowel length is not represented in the classical
    alphabet and the corpus does not mark it consistently anyway.
    """
    return unicodedata.normalize("NFC", _COMBINING.sub("", unicodedata.normalize("NFD", text)))


def drop_foreign_scripts(text: str) -> str:
    """Replace Greek (and other non-Latin scripts) with a space."""
    return _FOREIGN_SCRIPT.sub(" ", text)


def normalize(text: str) -> str:
    """Apply the full classical normalisation pipeline.

    Idempotent: `normalize(normalize(x)) == normalize(x)`.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _LIG_RE.sub(lambda m: _LIGATURES[m.group()], text)
    text = _PUNCT_RE.sub(lambda m: _PUNCT_FOLD[m.group()], text)
    text = join_hyphenation(text)
    text = drop_foreign_scripts(text)
    text = strip_diacritics(text)
    text = text.upper()
    # The classical alphabet has no J, V or W.
    text = text.replace("J", "I").replace("V", "U").replace("W", "UU")
    text = _ALLOWED.sub(" ", text)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = _MISSING_SPACE.sub(r"\1 ", text)
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def fold_token(word: str) -> str:
    """Normalise a single word for lookup (stopword lists, vocabulary checks).

    Works on raw or already-normalised text, so the same stopword list can be
    used before and after `normalize()`.
    """
    word = strip_diacritics(unicodedata.normalize("NFKC", word)).upper()
    word = word.replace("J", "I").replace("V", "U").replace("W", "UU")
    return "".join(c for c in word if c in LETTERS)


def is_normalized(text: str) -> bool:
    """True if every character is in the target alphabet."""
    return all(c in ALPHABET for c in text)
