"""The tutor: detectors and checker decide on corrections, the model converses.

A tutor that mis-corrects is worse than none, so the conversation model never
writes a correction on its own. Instead two detectors look at each student
sentence:

- the learned detector: the same model, trained on a CORRIGE/RECTE task with
  claude-written learner errors (sft/correct.py). Good at realistic and
  multi-slot errors (VENIT -> VENERUNT, ALBAM -> ALBUS).
- the swap scorer: every single-ending swap scored by the model. Reliable on
  single-ending errors.

Firm correction ("dicendum est X", with the rule) only when both propose the
same fix and the LatinCy checker objects to the word. Either one alone gives a
question ("num X dicere voluisti?"). The fix is written into the start of the
teacher's reply, and the model continues the conversation from there, which is
what it was trained to do (the prefix was context, never a target, in SFT).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import re

from .generate import clean_output, repeat_rate
from .normalize import normalize
from .sft.errors import _CASE_SWAPS, _VERB_SWAPS, _split, _swap_ending
from .sft.template import _PRAISE, _WHY, TEACHER, Turn, prompt_for_reply


#: Decoding for replies. Tuned on eval/tutor_messages.json (scripts/tune_tutor.py).
#: Grid of temperature 0.5-0.7 x penalty 1.1-1.2 x best-of 1/3 on 51 messages,
#: judged by claude -p: no loops anywhere, off-topic 15% / 17% / 20% at 0.5 /
#: 0.6 / 0.7 (best-of-1 means). Best-of-3 was worse in all 9 pairs (22-34%) and
#: 3x slower. Differences are near the judge's noise: it rated one identical
#: reply both ways across runs.
REPLY = dict(do_sample=True, temperature=0.5, top_p=0.9, repetition_penalty=1.15,
             no_repeat_ngram_size=4)
BEST_OF = 1
MAX_NEW_TOKENS = 60
ECHO_RETRIES = 3


@dataclass
class Finding:
    wrong: str
    right: str
    margin: float      # nats by which the fix beats the student's sentence
    confirmed: bool    # the morphological checker also objects
    rule: str = ""     # the agreement rule broken, in Latin, when the checker knows it


_NUMBER = {"Sing": "singularis", "Plur": "pluralis"}
_FEATURE = {"Case": "casu", "Number": "numero", "Gender": "genere"}


def rule_text(d, wrong: str, right: str) -> str:
    """The rule behind a checker disagreement, stated so it is true for the fix.

    Only what the parse says: the subject's number, which features must agree.
    Words are named as the student wrote them.
    """
    if d.rule == "subject-verb" and d.word_value in _NUMBER:
        if wrong == d.head:      # the verb was fixed to match the subject
            return (f"{d.word} numeri {_NUMBER[d.word_value]} est, "
                    f"ergo et verbum: {right}.")
        if d.head_value in _NUMBER:  # the subject was fixed to match the verb
            return (f"Verbum {d.head} numeri {_NUMBER[d.head_value]} est, "
                    f"ergo et subiectum: {right}.")
    if d.rule == "adjective-noun" and d.feature in _FEATURE:
        return (f"Adiectivum {d.word} cum nomine {d.head} "
                f"{_FEATURE[d.feature]} consentire debet: {right}.")
    return ""


def candidates(sentence: str) -> list[tuple[str, str, str]]:
    """(alternative sentence, wrong word, right word) for every ending swap."""
    words = sentence.split()
    out = {}
    for i, w in enumerate(words):
        for swaps in (_CASE_SWAPS, _VERB_SWAPS):
            for pair in swaps:
                alt = _swap_ending(w, [pair])
                if alt and alt != w:
                    sent = " ".join(words[:i] + [alt] + words[i + 1:])
                    out[sent] = (_split(w)[0], _split(alt)[0])
    return [(s, wr, rt) for s, (wr, rt) in out.items()]


def detect(model, tokenizer, sentence: str) -> Finding | None:
    """Both detectors combined (scripts/eval_combined.py, docs/decisions.md).

    Firm: the learned model (RECTE line) and the swap scorer propose the same
    fix, and the checker objects to the changed word. Hint: either detector
    flags on its own, at its own thresholds. Measured against each alone: firm
    false alarms 0/1000 on learner errors and 1/300 on held-out Vulgate, and
    more errors caught than either (321 vs 291 / 89 of 1000; 124 and 121 of
    200 on A and B).
    """
    from .sft.checker import check
    from .sft.correct import propose
    from .sft.scorer import sentence_logprob

    disagreements = check(sentence)
    flagged = {w for d in disagreements for w in (d.word, d.head)}
    base = sentence_logprob(model, tokenizer, sentence)
    gains = {(w, r): sentence_logprob(model, tokenizer, alt) - base
             for alt, w, r in candidates(sentence)}

    learned = None
    c = propose(model, tokenizer, sentence, samples=0)
    if c is not None:
        learned = (" ".join(w for w, _ in c.changes), " ".join(r for _, r in c.changes))
        hit = any(w in flagged for w, _ in c.changes)
        if hit and gains.get(learned, float("-inf")) > 0:
            rule = ""
            if len(c.changes) == 1:
                d = next(d for d in disagreements if learned[0] in (d.word, d.head))
                rule = normalize(rule_text(d, *learned))
            return Finding(*learned, c.margin, True, rule)
        if (hit and c.margin > LEARNED_FIRM) or c.margin >= LEARNED_HINT:
            return Finding(*learned, c.margin, False)
        # Short learner messages: the model's own fix is reliable at a lower
        # threshold there (scripts/eval_short.py), and a fix that turns a
        # non-dictionary word into a close real word (NEMEN -> NOMEN) has two
        # signals behind it. Name slots are never questioned (lexicon.suspects).
        from . import lexicon
        if len(sentence.split()) <= SHORT_WORDS:
            suspect = set(lexicon.suspects(sentence))
            respelled = all(w in suspect and r in lexicon.candidates(w) for w, r in c.changes)
            if c.margin > SHORT_HINT or (respelled and c.margin > 0):
                return Finding(*learned, c.margin, False)
    # The swap scorer alone, at its own thresholds -- but only ever as a
    # question: the detectors did not agree.
    firm = [(g, w, r) for (w, r), g in gains.items() if w in flagged and g > SWAP_FIRM]
    hint = [(g, w, r) for (w, r), g in gains.items() if g >= SWAP_HINT]
    if firm or hint:
        g, w, r = max(firm or hint)
        return Finding(w, r, g, False)
    # Misspelled non-words (NEMEN, MAGISTR), which ending-based detectors never
    # propose. Always a question: a name or a rare real word must not be "fixed".
    from . import lexicon
    m = lexicon.check(model, tokenizer, sentence)
    if m is not None:
        return Finding(m[0], m[1], m[2], False)
    return None


#: Learned-detector thresholds (log-prob margin, nats), picked on
#: out/sft/correct_dev.jsonl for fbandov/claudius-tutor-v3.
LEARNED_FIRM, LEARNED_HINT = 0.0, 4.0
#: Short learner messages (<= SHORT_WORDS words): the learned fix is a hint above
#: SHORT_HINT nats. On 240 clean learner sentences 0 flagged, names 1/88.
SHORT_WORDS, SHORT_HINT = 12, 2.0
#: Swap-scorer thresholds (gain, nats), from the held-out sweeps.
SWAP_FIRM, SWAP_HINT = 2.5, 8.0


def detect_swaps(model, tokenizer, sentence: str, hint_margin: float = 8.0,
                 confirm_margin: float = 2.5) -> Finding | None:
    """The previous detector, kept only to compare against. Remove once the
    learned one wins on held-out data (docs/decisions.md).

    Confirmed: the checker objects to a word, and the model prefers a fix of
    that same word (or the word it should agree with) by > `confirm_margin`.
    Unconfirmed: the model alone prefers a fix by >= `hint_margin`.
    """
    from .sft.checker import check
    from .sft.scorer import sentence_logprob

    disagreements = check(sentence)
    flagged = {w for d in disagreements for w in (d.word, d.head)}
    base = sentence_logprob(model, tokenizer, sentence)
    best_confirmed = best_hint = None
    for alt, wrong, right in candidates(sentence):
        gain = sentence_logprob(model, tokenizer, alt) - base
        if wrong in flagged and gain > confirm_margin:
            if best_confirmed is None or gain > best_confirmed.margin:
                best_confirmed = Finding(wrong, right, gain, True)
        if gain >= hint_margin and (best_hint is None or gain > best_hint.margin):
            best_hint = Finding(wrong, right, gain, False)
    if best_confirmed is not None:
        d = next(d for d in disagreements if best_confirmed.wrong in (d.word, d.head))
        best_confirmed.rule = normalize(rule_text(d, best_confirmed.wrong, best_confirmed.right))
    return best_confirmed or best_hint


def correction_text(f: Finding, level: str, rng: random.Random) -> str:
    if f.confirmed:
        parts = [rng.choice(_PRAISE), normalize(f'Sed dicendum est "{f.right}", non "{f.wrong}".')]
        parts.append(f.rule or _WHY["case"])
        return " ".join(parts)
    return normalize(f'Num "{f.right}" dicere voluisti?')


_FIRST_QUESTION = re.compile(r"^(.*?\?)")
_CONTENT = re.compile(r"[A-Z]{3,}")
_FUNCTION = {w[:5] for w in ("TUUS", "TUA", "TUUM", "MEUS", "MEA", "MEUM", "EGO", "TU", "EST", "SUM", "ET",
                             "NE", "QUOQUE", "HABEO", "HABES", "ETIAM", "NONNE", "NUM")}


def echoes(student: str, reply: str) -> bool:
    """The reply's closing question just restates the student's statement.

    "EGO CANEM HABEO." -> "TU CANEM HABES?" asks for what was just said. v7 did
    this on 6% of plain statements (v6: 1%), learned from the messy-input data,
    where restating a broken sentence is right. Asking a question back is fine
    ("QUID AGIS?" -> "ET TU, QUID AGIS?"), so only statements count.
    """
    if student.rstrip().endswith("?"):
        return False
    q = re.split(r"(?<=[.!;])\s+", reply)[-1]
    cq = {w[:5] for w in _CONTENT.findall(q)} - _FUNCTION
    cs = {w[:5] for w in _CONTENT.findall(student)} - _FUNCTION
    return q.endswith("?") and len(cq) >= 1 and len(cq & cs) / len(cq) >= 0.75
_STEM = re.compile(r"[A-Z]{4,}")


def _stems(text: str) -> set[str]:
    return {w[:4] for w in _STEM.findall(text)}


def split_fused(model, tokenizer, text: str, gate: float = 2.0) -> str:
    """Split words the model ran together (TUSALUS), only if the model agrees.

    The frequency splitter alone is wrong on clean text (HILARISSIMUS ->
    HILARIS SIMUS). Gated by the model preferring the split by > `gate` nats:
    0 false splits in 3,000 clean sentences, 55% of fused pairs repaired.
    """
    from .segment import _freq, split_word
    from .sft.scorer import sentence_logprob

    freq = _freq("out/latin_wordfreq.json")
    words = text.split()
    for i, w in enumerate(words):
        core = re.match(r"[A-Z]+", w)
        if not core:
            continue
        parts = split_word(core.group(), freq, min_len=5)
        if len(parts) < 2:
            continue
        alt = words[:i] + [" ".join(parts) + w[core.end():]] + words[i + 1:]
        if sentence_logprob(model, tokenizer, " ".join(alt)) - \
                sentence_logprob(model, tokenizer, " ".join(words)) > gate:
            words = " ".join(alt).split()
    return " ".join(words)


def trim(text: str) -> str:
    """One teacher turn: up to and including the first question."""
    text = clean_output(text.split("\n")[0]).strip()
    m = _FIRST_QUESTION.match(text)
    return m.group(1).strip() if m else text


def score(candidate: str, student: str) -> float:
    """Pick among sampled replies: grammatical, asks something, on topic, no loop."""
    from .sft.checker import check

    s = 0.0
    s += 2.0 if not check(candidate) else 0.0
    s += 1.0 if candidate.endswith("?") else 0.0
    s += min(len(_stems(candidate) & _stems(student)), 2) * 0.5
    s -= 3.0 * repeat_rate(candidate)
    return s


def respond(model, tokenizer, turns: list[Turn], level: str = "tiro", *,
            max_new_tokens: int = MAX_NEW_TOKENS, settings: dict | None = None,
            best_of: int = BEST_OF, seed: int | None = None) -> dict:
    """The teacher's next turn, with the correction kept apart from the talk.

    {"correction": None | {"kind": "firm" | "hint", "wrong", "right", "rule",
     "text"}, "text": the conversational part, "full": both as one turn}.
    `full` is what goes back into the history: the model was trained on turns
    that start with the correction.
    """
    import torch

    turns = [Turn(t.role, normalize(t.text)) for t in turns]
    finding = None
    if turns[-1].role != TEACHER:
        finding = detect(model, tokenizer, turns[-1].text)
    prefix = correction_text(finding, level, random.Random(seed)) if finding else ""

    prompt = prompt_for_reply(turns, level) + (f" {prefix}" if prefix else "")
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id,
                             num_return_sequences=best_of, **(settings or REPLY))
    student = turns[-1].text
    cands = [trim(tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True)) for o in out]
    text = max(cands, key=lambda c: score(c, student))
    for _ in range(ECHO_RETRIES):             # sample again rather than echo the student
        if not echoes(student, text):
            break
        with torch.no_grad():
            again = model.generate(ids, max_new_tokens=max_new_tokens,
                                   pad_token_id=tokenizer.eos_token_id, **(settings or REPLY))
        text = trim(tokenizer.decode(again[0][ids.shape[1]:], skip_special_tokens=True))
    text = split_fused(model, tokenizer, text)
    correction = None
    if finding is not None:
        correction = {"kind": "firm" if finding.confirmed else "hint",
                      "wrong": finding.wrong, "right": finding.right,
                      "rule": finding.rule, "text": prefix}
    return {"correction": correction, "text": text,
            "full": f"{prefix} {text}".strip() if prefix else text}


def reply(model, tokenizer, turns: list[Turn], level: str = "tiro", **kw) -> tuple[str, Finding | None]:
    """The teacher's next turn as one string, and what (if anything) was detected."""
    r = respond(model, tokenizer, turns, level, **kw)
    c = r["correction"]
    finding = None if c is None else Finding(c["wrong"], c["right"], 0.0,
                                             c["kind"] == "firm", c["rule"])
    return r["full"], finding
