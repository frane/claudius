from claudius.sft import scorer
from claudius.sft.scorer import Comparison, pairwise_accuracy


def test_margin_is_candidate_minus_student():
    c = Comparison("PUELLA ROSA AMAT.", "PUELLA ROSAM AMAT.", -20.0, -16.5)
    assert c.margin == 3.5
    assert c.prefers_candidate()
    assert not c.prefers_candidate(threshold=4.0)


def test_pairwise_accuracy_counts_preferences(monkeypatch):
    """Correct sentences score higher in 3 of 4 pairs -> 75%."""
    scores = {"GOOD1": -10, "BAD1": -15, "GOOD2": -10, "BAD2": -12,
              "GOOD3": -10, "BAD3": -11, "GOOD4": -10, "BAD4": -8}
    monkeypatch.setattr(scorer, "sentence_logprob", lambda m, t, s: scores[s])
    pairs = [("GOOD1", "BAD1"), ("GOOD2", "BAD2"), ("GOOD3", "BAD3"), ("GOOD4", "BAD4")]
    r = pairwise_accuracy(None, None, pairs)
    assert r["pairs"] == 4
    assert r["prefers_correct"] == 3
    assert r["accuracy"] == 0.75


def test_scores_are_summed_not_averaged(monkeypatch):
    """Averaging would reward a wrong ending that splits into more likely pieces.

    Versions of one sentence differ by a single word and have nearly the same
    token count, so summed log-probabilities compare directly.
    """
    import inspect

    src = inspect.getsource(scorer.sentence_logprob)
    assert ".sum()" in src
    assert ".mean()" not in src
