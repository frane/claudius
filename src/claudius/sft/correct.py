"""The learned error detector: a correction task trained into the tutor model.

    CORRIGE: EGO PUELLA UIDEO.
    RECTE: EGO PUELLAM UIDEO.<eos>

Trained on learner errors written by `claude -p` (scripts/gen_errors.py) whose
fix is the original, correct sentence, plus correct sentences copied unchanged
(hard negatives: Hebrew names, rare forms), so "no change" is the usual answer.
Loss is on the RECTE line only.

It replaces the hand-listed ending swaps, which could only propose fixes that
were in the table (no VENIT -> VENERUNT, no ALBAM -> ALBUS) and scored them with
a language model that barely prefers DEFENDUNT over DEFENDIT.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .template import EOS

PROMPT = "CORRIGE: {sentence}\nRECTE:"
_WORD = re.compile(r"[A-Z]+")


def pieces(wrong: str, right: str) -> list[tuple[str, bool]]:
    return [(PROMPT.format(sentence=wrong), False), (f" {right}\n", True), (EOS, True)]


def diff_words(a: str, b: str) -> list[tuple[str, str]] | None:
    """(word in a, word in b) pairs that differ, or None if the shape differs."""
    wa, wb = _WORD.findall(a), _WORD.findall(b)
    if len(wa) != len(wb):
        return None
    return [(x, y) for x, y in zip(wa, wb) if x != y]


def examples(errors_path: str | Path, pool_path: str | Path, negatives: int,
             exclude: set[str], seed: int = 0) -> list[dict]:
    """Correction examples: every error pair, its original copied, extra negatives.

    `negatives` extra correct sentences are drawn per call; a different `seed`
    draws different ones, so a second pass adds new hard negatives rather than
    repeating the first pass's.
    """
    rng = random.Random(seed)
    out, used = [], set()
    for line in open(errors_path, encoding="utf-8"):
        e = json.loads(line)
        if e["right"] in exclude:
            continue
        out.append({"pieces": pieces(e["wrong"], e["right"])})
        out.append({"pieces": pieces(e["right"], e["right"])})
        used.add(e["right"])
    pool = [json.loads(l)["text"] for l in open(pool_path, encoding="utf-8")]
    rng.shuffle(pool)
    extra = [s for s in pool if s not in used and s not in exclude][:negatives]
    out += [{"pieces": pieces(s, s)} for s in extra]
    rng.shuffle(out)
    return out


@dataclass
class Correction:
    wrong: str
    right: str
    margin: float           # nats: log p(fixed) - log p(unchanged) after RECTE:
    changes: list           # [(student word, fixed word)]
    agreement: float = 1.0  # share of sampled fixes identical to the greedy one


def _continuation_logprob(model, tokenizer, prompt: str, text: str) -> float:
    import torch

    device = next(model.parameters()).device
    p = tokenizer(prompt, add_special_tokens=False).input_ids
    t = tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    ids = torch.tensor([p + t], device=device)
    with torch.no_grad():
        logits = model(ids).logits[0, len(p) - 1:-1].float()
    lp = torch.log_softmax(logits, -1)
    return lp[torch.arange(len(t)), torch.tensor(t, device=device)].sum().item()


def propose(model, tokenizer, sentence: str, max_new_tokens: int = 64,
            samples: int = 8, temperature: float = 1.0) -> Correction | None:
    """The model's fixed sentence and its confidence, or None if it changes nothing.

    Confidence is agreement: the share of `samples` sampled RECTE lines that are
    exactly the greedy fix. A single log-prob margin did not separate right
    fixes from changes to correct sentences (both mostly 0-2 nats on dev); a fix
    the model reproduces when sampling is one it actually believes.
    """
    import torch

    prompt = PROMPT.format(sentence=sentence)
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    def decode(o):
        return tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip()

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    fixed = decode(out[0])
    d = diff_words(sentence, fixed)
    if not d or len(d) > 2:     # unchanged, or a rewrite rather than a fix
        return None
    agreement = 1.0
    if samples:
        with torch.no_grad():
            outs = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=temperature, top_p=1.0,
                                  num_return_sequences=samples,
                                  pad_token_id=tokenizer.eos_token_id)
        agreement = sum(decode(o) == fixed for o in outs) / samples
    margin = (_continuation_logprob(model, tokenizer, prompt, f" {fixed}\n")
              - _continuation_logprob(model, tokenizer, prompt, f" {sentence}\n"))
    return Correction(sentence, fixed, margin, d, agreement)
