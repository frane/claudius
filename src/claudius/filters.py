"""Language identification and OCR line cleaning.

Latin-PD is raw OCR from scanned books. A "document" is a whole book, so it
contains title pages, library stamps, running heads, page numbers, French or
German editorial apparatus, Greek quotations and long stretches of gibberish
where the scan failed. Two jobs here:

* decide whether a document is Latin at all (stopword ratio, as decided);
* drop the lines inside a kept document that are clearly not text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .normalize import fold_token, normalize
from .quality import strip_edition_numbers

#: High-frequency Latin function words, in folded form (uppercase, U for V,
#: I for J) so the same list works before and after normalisation.
STOPWORDS = frozenset(
    fold_token(w)
    for w in (
        "et in est non cum ad ut quod qui sed de ex quae atque enim autem sunt "
        "esse per vel sive etiam nec hoc quam si ab id vero eius ipse tamen "
        "quia ita nam ergo quo cuius hunc haec his eorum inter ante post sub "
        "aut nisi quidem tunc omnia omnes res dei deus rex anno libro caput "
        "primo secundo dicitur potest oportet quibus illa ille illud unde"
    ).split()
)

#: Frequent function words of the languages that actually turn up in the corpus
#: -- French, German, English, Italian, Spanish -- restricted to words that are
#: *not* also Latin. The stopword ratio on its own is not enough: French shares
#: ET, EST and QUI with Latin and scores around 0.12, comfortably above the 0.06
#: threshold. A second, competing count settles it.
RIVAL_WORDS = frozenset(
    fold_token(w)
    for w in (
        # French
        "le la les des du une dans pour aux avec cette ceux elle ils nous vous "
        "leur tous ces eux pas dont notre votre entre etre sont etaient avait "
        "aussi tres apres sans chez"
        # German
        " der das den dem ein eine einer nicht mit auch sich auf von vor aber "
        "wird werden oder durch nach uber bei dass ist waren"
        # English
        " the and of to is was with that this from which have has but they there "
        "their would should been are were what when"
        # Italian / Spanish
        " il lo gli della del che nel sono anche come piu los las por con para "
        "pero mas muy este esta una"
    ).split()
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")


def sample_text(text: str, cfg: FilterConfig) -> str:
    """The slice of a document the language scores are computed on.

    Taken from throughout the body rather than from the head. On Latin-PD the
    head is front matter: Migne's Patrologia Latina scores 0.05 on its first
    20k characters -- a garbled title page -- and far higher on its body.
    Scoring the head rejects some of the most valuable books in the corpus.
    """
    n = len(text)
    if n <= cfg.sample_chars:
        return text
    lo = int(n * cfg.sample_skip_head)
    hi = n - int(n * cfg.sample_skip_tail)
    span = hi - lo
    if span <= cfg.sample_chars:
        return text[lo:hi]
    per = max(1, cfg.sample_chars // cfg.sample_slices)
    step = span // cfg.sample_slices
    return "\n".join(text[lo + i * step : lo + i * step + per] for i in range(cfg.sample_slices))



@dataclass(frozen=True)
class FilterConfig:
    """Thresholds for the document filter.

    `min_latin` is the stopword ratio decided in the brief. Verify it on the
    pilot by eyeballing accepted and rejected documents -- see
    `claudius filter-report`.
    """

    min_latin: float = 0.06
    min_chars: int = 2_000
    #: Total characters scored per document.
    sample_chars: int = 20_000
    #: Spread over this many slices rather than taken from the head. A scanned
    #: book opens with a title page, an imprimatur and a library plate; the
    #: first 20k characters of it are not the text.
    sample_slices: int = 5
    #: Fractions of the document skipped at each end -- front matter and the
    #: index or colophon.
    sample_skip_head: float = 0.05
    sample_skip_tail: float = 0.05
    min_words_for_score: int = 50
    #: A line must be at least this fraction letters to survive.
    min_line_letter_ratio: float = 0.6
    min_line_chars: int = 3
    #: A document with more Greek than this (before normalisation) is a Greek
    #: text with a Latin translation, not a Latin text with Greek quotations.
    max_greek_ratio: float = 0.15
    #: A Latin book may carry a French preface or German footnotes; a French book
    #: quoting Latin must still be rejected. Both bounds have to hold.
    max_rival: float = 0.05
    #: Latin must out-score the best rival language by this factor.
    min_latin_over_rival: float = 1.5
    #: Blocks shorter than this are never judged for language -- a five-word
    #: line carries no evidence either way.
    min_block_words: int = 8


def latin_score(text: str, cfg: FilterConfig = FilterConfig()) -> float:
    """Fraction of word tokens that are Latin function words.

    Roughly 0.10-0.20 for real Latin prose, under 0.02 for French, German or
    English, and near zero for OCR noise.
    """
    words = _WORD_RE.findall(sample_text(text, cfg))
    if len(words) < cfg.min_words_for_score:
        return 0.0
    hits = sum(1 for w in words if fold_token(w) in STOPWORDS)
    return hits / len(words)


def rival_score(text: str, cfg: FilterConfig = FilterConfig()) -> float:
    """Fraction of word tokens that are function words of a *rival* language."""
    words = _WORD_RE.findall(sample_text(text, cfg))
    if len(words) < cfg.min_words_for_score:
        return 0.0
    hits = sum(1 for w in words if fold_token(w) in RIVAL_WORDS)
    return hits / len(words)


def is_latin_scores(latin: float, rival: float, cfg: FilterConfig = FilterConfig()) -> bool:
    """The decision rule, given already-computed scores."""
    return (
        latin >= cfg.min_latin
        and rival <= cfg.max_rival
        and latin >= cfg.min_latin_over_rival * rival
    )


def is_latin(text: str, cfg: FilterConfig = FilterConfig()) -> bool:
    """The full language decision: Latin enough, and not some other language."""
    return is_latin_scores(latin_score(text, cfg), rival_score(text, cfg), cfg)


def _block_is_foreign(block: str, cfg: FilterConfig) -> bool:
    """True if a paragraph looks like French, German or English prose.

    Decided by a head-to-head count rather than a ratio, because a paragraph is
    too short for a stable ratio. Two rival function words and more of them than
    Latin ones is enough: a Latin paragraph essentially never looks like that.
    """
    words = _WORD_RE.findall(block)
    if len(words) < cfg.min_block_words:
        return False
    folded = [fold_token(w) for w in words]
    rival = sum(1 for w in folded if w in RIVAL_WORDS)
    latin = sum(1 for w in folded if w in STOPWORDS)
    return rival >= 2 and rival > latin


def drop_foreign_blocks(text: str, cfg: FilterConfig = FilterConfig()) -> str:
    """Remove the French preface and the German footnotes, keep the Latin.

    Many Latin-PD volumes are Latin texts inside a modern editorial frame. The
    document-level filter would either keep the frame (leaking non-Latin into
    pretraining, the one thing the model must never see) or throw the volume
    away. Cutting at paragraph level keeps the text and drops the frame.
    """
    blocks = text.split("\n\n") if "\n\n" in text else text.split("\n")
    sep = "\n\n" if "\n\n" in text else "\n"
    kept = [b for b in blocks if not _block_is_foreign(b, cfg)]
    return sep.join(kept).strip()


def greek_ratio(text: str, cfg: FilterConfig = FilterConfig()) -> float:
    sample = sample_text(text, cfg)
    if not sample:
        return 0.0
    return len(_GREEK_RE.findall(sample)) / len(sample)


def line_is_text(line: str, cfg: FilterConfig = FilterConfig()) -> bool:
    """Reject running heads, page numbers and OCR rubble."""
    s = line.strip()
    if len(s) < cfg.min_line_chars:
        return False
    letters = sum(c.isalpha() for c in s)
    return letters / len(s) >= cfg.min_line_letter_ratio


def clean_lines(text: str, cfg: FilterConfig = FilterConfig()) -> str:
    """Drop non-text lines, keeping blank lines as paragraph separators."""
    out = []
    for line in text.split("\n"):
        if not line.strip():
            out.append("")
        elif line_is_text(line, cfg):
            out.append(line.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def prepare_document(text: str, cfg: FilterConfig = FilterConfig()) -> str | None:
    """Full per-document pipeline: filter, clean, normalise.

    Returns the normalised text, or None if the document is rejected. The
    Order matters. OCR rubble goes first, then foreign-language paragraphs, then
    the document-level language decision -- which is made on the *raw* text,
    because diacritics and casing are signal for "this is French" and
    normalisation would destroy them. Normalisation is last.
    """
    if not text or len(text) < cfg.min_chars:
        return None
    body = drop_foreign_blocks(clean_lines(text, cfg), cfg)
    if len(body) < cfg.min_chars:
        return None
    if greek_ratio(body, cfg) > cfg.max_greek_ratio:
        return None
    if not is_latin(body, cfg):
        return None
    # Edition line/section numbers are not Latin; the 400M learned to print them.
    cleaned = strip_edition_numbers(normalize(body))
    if len(cleaned) < cfg.min_chars:
        return None
    return cleaned


def filter_report(
    docs: Iterable[tuple[str, str]], cfg: FilterConfig = FilterConfig(), limit: int = 20
) -> dict:
    """Score a sample of `(identifier, text)` pairs without filtering them.

    Use this to eyeball the threshold on the pilot: it returns the documents
    closest to the cutoff from both sides, which is where the mistakes live. It
    also reports how much text the paragraph-level foreign cut removes from the
    documents that are kept -- if that number is large, the document-level
    thresholds are being propped up by the paragraph cut and both deserve a look.
    """
    scored = []
    for ident, text in docs:
        cleaned = clean_lines(text, cfg)
        body = drop_foreign_blocks(cleaned, cfg)
        cut = 1.0 - len(body) / max(len(cleaned), 1)
        scored.append(
            (latin_score(body, cfg), rival_score(body, cfg), greek_ratio(body, cfg),
             cut, ident, body[:400])
        )
    scored.sort(key=lambda r: r[0])
    below, above = [], []
    for row in scored:
        (above if is_latin_scores(row[0], row[1], cfg) else below).append(row)

    cuts = sorted(r[3] for r in above)
    def _pct(xs, q):
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) if xs else 0.0

    return {
        "n": len(scored),
        "kept": len(above),
        "rejected": len(below),
        "accept_rate": round(len(above) / max(len(scored), 1), 4),
        "threshold": cfg.min_latin,
        "max_rival": cfg.max_rival,
        "paragraph_cut_of_accepted": {
            "mean": round(sum(cuts) / len(cuts), 4) if cuts else 0.0,
            "median": _pct(cuts, 0.5),
            "p90": _pct(cuts, 0.9),
            "docs_cut_over_5pct": sum(1 for c in cuts if c > 0.05),
        },
        "just_rejected": below[-limit:],
        "just_accepted": above[:limit],
    }
