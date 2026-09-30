"""Grammaticality from the language model itself.

The dependency-parse checker is a good veto and a poor detector: it re-analyses
an erroneous sentence to make it grammatical, and so hides the error. A language
model trained on billions of tokens of Latin should do the opposite -- a wrong
ending is exactly the kind of thing it has learned to find improbable.

So: score a sentence by its total log-likelihood, and compare a student's
sentence against candidate corrections. If the correction scores clearly higher,
that is evidence of an error. Combined with the deterministic checker as an
independent confirmation, it gives a tutor that corrects only when two unrelated
signals agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    student: str
    candidate: str
    student_logprob: float
    candidate_logprob: float

    @property
    def margin(self) -> float:
        """Nats by which the candidate beats the student's version."""
        return self.candidate_logprob - self.student_logprob

    def prefers_candidate(self, threshold: float = 0.0) -> bool:
        return self.margin > threshold


def sentence_logprob(model, tokenizer, sentence: str) -> float:
    """Total log-probability of the sentence's tokens, in nats.

    Summed, not averaged: two versions of one sentence differ by a single word,
    so they have nearly the same token count and the sums are directly
    comparable. Averaging would reward a wrong ending that happens to split
    into more, individually likely, pieces.
    """
    import torch

    device = next(model.parameters()).device
    ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < 2:
        return 0.0
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    target = ids[0, 1:]
    return float(logp.gather(1, target.unsqueeze(1)).sum())


def compare(model, tokenizer, student: str, candidate: str) -> Comparison:
    return Comparison(student, candidate,
                      sentence_logprob(model, tokenizer, student),
                      sentence_logprob(model, tokenizer, candidate))


def pairwise_accuracy(model, tokenizer, pairs) -> dict:
    """How often the model prefers the correct sentence over its corruption."""
    wins = 0
    margins = []
    for correct, corrupted in pairs:
        c = compare(model, tokenizer, corrupted, correct)
        margins.append(c.margin)
        wins += c.prefers_candidate()
    n = len(margins) or 1
    return {
        "pairs": len(margins),
        "prefers_correct": wins,
        "accuracy": wins / n,
        "median_margin_nats": sorted(margins)[len(margins) // 2] if margins else math.nan,
    }
