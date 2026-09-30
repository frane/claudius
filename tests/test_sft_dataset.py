import random

import pytest

pytest.importorskip("tokenizers")

from claudius.sft.dataset import IGNORE, segments, tokenize_example, weave_correction  # noqa: E402
from claudius.train import _collate  # noqa: E402

TURNS = [("DISCIPULUS", "PUELLA ROSAM PULCHRAM AMAT."),
         ("MAGISTER", "BENE. QUID TIBI PLACET?"),
         ("DISCIPULUS", "MAGISTER DISCIPULOS BONOS DOCET."),
         ("MAGISTER", "ITA EST. QUID DISCIS?")]


class FakeTok:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 250 + 1 for c in text]}


def test_only_teacher_text_is_trainable():
    ids, labels = tokenize_example(FakeTok(), segments(TURNS, "tiro"))
    assert len(ids) == len(labels)
    pieces = segments(TURNS, "tiro")
    student_text = "".join(t for t, trainable in pieces if not trainable and "DISCIPULUS" in t)
    assert "PUELLA" in student_text, "student turn must be present as context"
    n_trainable = sum(l != IGNORE for l in labels)
    teacher_chars = sum(len(t) for t, tr in pieces if tr and t != "<|endoftext|>")
    assert n_trainable == teacher_chars + 1, "teacher text plus the final EOS"


def test_corrupted_student_turn_is_never_a_target():
    rng = random.Random(3)
    woven, meta = weave_correction(TURNS, "medius", rng)
    assert meta is not None
    ids, labels = tokenize_example(FakeTok(), segments(woven, "medius"))
    pieces = segments(woven, "medius")
    # Find the corrupted student line and check none of its tokens are labelled.
    pos = 0
    for text, trainable in pieces:
        n = 1 if text == "<|endoftext|>" else len(text)
        if meta["wrong"] in text and text.startswith("DISCIPULUS"):
            assert all(l == IGNORE for l in labels[pos:pos + n])
        pos += n


def test_woven_correction_keeps_the_teachers_real_reply():
    woven, meta = weave_correction(TURNS, "tiro", random.Random(3))
    i = next(k for k, (r, t) in enumerate(woven) if r == "DISCIPULUS" and meta["wrong"] in t)
    reply = woven[i + 1][1]
    assert f'"{meta["right"]}"' in reply
    assert reply.endswith(TURNS[i + 1][1]), "the original reply continues the conversation"


def test_correction_prefix_is_context_not_target():
    woven, meta = weave_correction(TURNS, "medius", random.Random(3))
    pieces = segments(woven, "medius", meta)
    prefix_pieces = [(t, tr) for t, tr in pieces if meta["prefix"] in t]
    assert prefix_pieces == [(f" {meta['prefix']}", False)], "the fix is never learned"
    reply = TURNS[meta["turn"]][1]
    assert (f" {reply}\n", True) in pieces, "the conversation after the fix is learned"


def test_collate_uses_labels_when_present():
    batch = _collate([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}])
    assert batch["labels"].tolist() == [[-100, 6, 7]]
    batch = _collate([{"input_ids": [5, 6, 7]}])
    assert batch["labels"].tolist() == [[5, 6, 7]]


def test_persona_contradictions_are_detected():
    from claudius.sft.persona import contradicts

    assert contradicts([("DISCIPULUS", "ET TU?"), ("MAGISTER", "FRATER MEUS ROMAE HABITAT.")])
    assert not contradicts([("DISCIPULUS", "FRATER MEUS ROMAE HABITAT.")]), "the student may have brothers"
    assert not contradicts([("MAGISTER", "SOROR MEA CLAUDIA MEDICA EST.")])
