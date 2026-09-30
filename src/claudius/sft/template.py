"""The conversation format, and deterministic correction replies.

Roles are plain Latin, not special tokens. Adding tokens would change the
vocabulary size, and the base model's embedding table is sized to the 32k
tokenizer -- so new tokens mean resizing the checkpoint and training fresh
embeddings for them from nothing. `DISCIPULUS:` and `MAGISTER:` cost nothing,
tokenize with the vocabulary the model already has, and are what a Latin
classroom dialogue would actually say.

Everything is normalised, because it is the only orthography the model knows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..normalize import normalize
from .errors import Corruption

#: Proficiency levels, per the brief.
LEVELS = ("tiro", "medius", "peritus")

STUDENT = "DISCIPULUS"
TEACHER = "MAGISTER"
EOS = "<|endoftext|>"

#: The system prompt, in Latin: identical at training and inference, because a
#: 400M model follows a prompt only as a pattern it was trained on. The level
#: word is the only switch; the teacher turns in the data match its difficulty.
_SYSTEM = ("Tu es Claudius, magister linguae Latinae. Discipulus est {level}. "
           "Responde breviter et simpliciter. Si errat, corrige et regulam explica. "
           "Semper interroga.")
SYSTEM = {lv: normalize(_SYSTEM.format(level=lv)) for lv in ("tiro", "medius", "peritus")}


@dataclass(frozen=True)
class Turn:
    role: str   # STUDENT or TEACHER
    text: str


def render(turns: list[Turn], level: str = "tiro", *, add_eos: bool = True) -> str:
    """One training example as a single normalised string."""
    if level not in SYSTEM:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    lines = [f"GRADUS: {level.upper()}", SYSTEM[level], ""]
    for t in turns:
        if t.role not in (STUDENT, TEACHER):
            raise ValueError(f"unknown role {t.role!r}")
        lines.append(f"{t.role}: {normalize(t.text)}")
    text = "\n".join(lines)
    return text + ("\n" + EOS if add_eos else "")


def prompt_for_reply(turns: list[Turn], level: str = "tiro") -> str:
    """The prefix a model continues at inference: ends with `MAGISTER:`."""
    return render(turns, level, add_eos=False) + f"\n{TEACHER}:"


# -- correction replies -----------------------------------------------------
# Deterministic, templated replies. They exist to seed SFT with replies whose
# correctness is guaranteed by construction; a model's own corrections are only
# trustworthy after the morphological checker has graded them.

_PRAISE = [normalize(p) for p in ("Bene!", "Optime!", "Recte fere!", "Paene recte!")]
_FIX = {
    "case": [
        'Sed dicendum est "{right}", non "{wrong}".',
        'Attende: "{right}" rectum est, non "{wrong}".',
    ],
    "verb": [
        'Sed verbum est "{right}", non "{wrong}".',
        'Cura verbum: "{right}", non "{wrong}".',
    ],
    # No praise word here: correction_reply already opens with one, and
    # "BENE! BENE INTELLEGO" reads like a stammer.
    "order": [
        "Intellego te, sed Latine verbum saepe in fine ponitur.",
        "Latine autem verbum in fine melius sonat.",
    ],
}
#: Explanations must be TRUE FOR EVERY ERROR OF THE CATEGORY, not just typical.
#:
#: A specific explanation is more instructive and is sometimes a lie. "Noun and
#: adjective must agree in case" is right for PULCHRAM -> PULCHRA and wrong for
#: ROSAM -> ROSA, which is a case-ROLE error (the object must be accusative).
#: "Verb must agree with subject in number" is right for AMAT -> AMANT and wrong
#: for AMAUIT -> AMABAT, which is a tense error. The swap tables mix both kinds.
#:
#: The brief is explicit that a tutor which mis-corrects is worse than none, so
#: these say only what is guaranteed: the ending is wrong. Even "the case is
#: wrong" is not guaranteed, because the swap tables are string edits and -AM
#: -> -A hits verbs (CURREBAM) as well as nouns (ROSAM).
#: Specific diagnosis belongs to the morphological checker, which actually knows.
_WHY = {
    "case": normalize("Terminatio huius verbi non recta est."),
    "verb": normalize("Terminatio huius verbi non recta est."),
    "order": normalize("Ordo verborum Latine liberior est, sed verbum finem amat."),
}
_FOLLOW_UP = [normalize(q) for q in (
    "Quid heri fecisti?", "Quem librum legis?", "Ubi habitas?",
    "Quid tibi placet?", "Quis est amicus tuus?", "Quid cras facies?",
    "Potesne sententiam aliam scribere?", "Quid in schola discis?",
)]


def correction_reply(c: Corruption, level: str, rng: random.Random) -> str:
    """A gentle correction: praise, the fix, an explanation above tiro, a question.

    Tiro gets the fix and a question, nothing more -- a beginner told *why* in
    Latin grammatical terms learns nothing from the explanation. Medius and
    peritus get the reason.
    """
    parts = [rng.choice(_PRAISE)]
    parts.append(normalize(rng.choice(_FIX[c.kind]).format(right=c.right, wrong=c.wrong)))
    if level != "tiro":
        parts.append(_WHY[c.kind])
    parts.append(rng.choice(_FOLLOW_UP))
    return " ".join(parts)


def correction_example(c: Corruption, level: str, rng: random.Random) -> str:
    """A complete SFT example: student errs, teacher corrects."""
    return render([Turn(STUDENT, c.corrupted), Turn(TEACHER, correction_reply(c, level, rng))], level)
