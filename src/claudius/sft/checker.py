"""The deterministic correctness guard.

The brief made this non-optional: the language model holds the conversation, and
something deterministic grades it, because a small model that mis-corrects a
student is worse than no tutor.

Built on LatinCy (spaCy models for Latin). One measured fact governs the whole
design: LatinCy CANNOT parse our normalised orthography as-is. On uppercase text
it tags every word NUM -- it reads PUELLA ROSAM PULCHRAM AMAT as Roman numerals --
and the lemmas come back as garbage like `aMAT`. Lowercased, the same text
parses perfectly: case, number, gender and dependencies all correct, and U-for-V
("uir bonus uenit") is no problem. So everything is lowercased before parsing.

Checks only agreement, which is where the synthetic learner errors live and where
a dependency parse gives a reliable answer:

* adjective <-> noun: an `amod` must match its head in case, number and gender;
* subject <-> verb: an `nsubj` and its finite verb must match in number.

It reports disagreement; it does not rewrite. Deciding which of two disagreeing
words is the wrong one is exactly the step that would need to be right every
time, and the parse cannot always tell.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MODEL = "la_core_web_sm"
MODEL_ENV = "CLAUDIUS_CHECKER_MODEL"


@dataclass(frozen=True)
class Disagreement:
    rule: str        # "adjective-noun" | "subject-verb"
    feature: str     # "Case" | "Number" | "Gender"
    word: str        # the dependent, as written (uppercase)
    head: str        # what it should agree with
    word_value: str
    head_value: str

    def describe(self) -> str:
        return (f"{self.rule}: {self.word} ({self.feature}={self.word_value}) "
                f"vs {self.head} ({self.feature}={self.head_value})")


@lru_cache(maxsize=1)
def _nlp():
    import os

    import spacy

    return spacy.load(os.environ.get(MODEL_ENV, MODEL))


def _feat(token, name: str) -> str | None:
    vals = token.morph.get(name)
    return vals[0] if vals else None


#: Features checked for adjective-noun agreement. GENDER IS DELIBERATELY ABSENT.
#: Measured on 300 correct Vulgate verses, gender produced 16 of 41 false
#: positives: Latin does not mark gender on most third-declension endings, so the
#: parser guesses (DIE SEPTIMO: DIES tagged feminine). The synthetic learner
#: errors never corrupt gender, so dropping it costs essentially no detection.
AGREEMENT_FEATURES = ("Case", "Number")


def _unreliable(tok) -> bool:
    """Tokens whose morphology the parser cannot be trusted on.

    Proper names (MOSES tagged plural, EDOM tagged a verb), relative pronouns
    (QUAE mistaken for a subject), and numerals (NONAGINTA). Each was a
    measured source of false positives on correct text.
    """
    if tok.pos_ in ("PROPN", "NUM"):
        return True
    if tok.pos_ == "PRON" and "Rel" in tok.morph.get("PronType"):
        return True
    return False


def check(sentence: str) -> list[Disagreement]:
    """Agreement errors in a normalised sentence. Empty list means none found."""
    doc = _nlp()(sentence.lower())
    out: list[Disagreement] = []
    for tok in doc:
        head = tok.head
        if head is tok or _unreliable(tok) or _unreliable(head):
            continue
        if tok.dep_ == "amod" and tok.pos_ == "ADJ" and head.pos_ == "NOUN":
            for feat in AGREEMENT_FEATURES:
                a, b = _feat(tok, feat), _feat(head, feat)
                # Only a conflict when BOTH are known. A missing feature is the
                # parser being unsure, and guessing is how a checker lies.
                if a and b and a != b:
                    out.append(Disagreement("adjective-noun", feat, tok.text.upper(),
                                            head.text.upper(), a, b))
        if tok.dep_ == "nsubj" and head.pos_ in ("VERB", "AUX"):
            if _feat(head, "VerbForm") not in (None, "Fin"):
                continue   # infinitives and participles carry no subject agreement
            # A Latin subject is nominative. When the parser labels a word nsubj
            # but tags it accusative, it has contradicted itself -- measured on
            # correct text, that is QUAE and MOSES, both nsubj|Case=Acc -- and a
            # self-contradictory parse cannot be the basis for telling a
            # student they are wrong.
            if _feat(tok, "Case") != "Nom":
                continue
            a, b = _feat(tok, "Number"), _feat(head, "Number")
            if a and b and a != b:
                out.append(Disagreement("subject-verb", "Number", tok.text.upper(),
                                        head.text.upper(), a, b))
    return out


def is_clean(sentence: str) -> bool:
    return not check(sentence)
