"""Assemble and tokenize the tutor fine-tuning set, with loss only on the teacher.

Two decisions shape this file.

LOSS IS MASKED TO THE TEACHER'S TURNS. Student turns include deliberately
corrupted Latin. Training on them would teach the tutor to write the very errors
it exists to catch, so student tokens are context only: label -100.

CORRECTIONS ARE WOVEN INTO REAL CONVERSATIONS. One student turn in a dialogue is
corrupted and the teacher's original reply is prefixed with the fix ("Bene! Sed
dicendum est 'X', non 'Y'."), so the conversation still flows. The correction
text is deterministic and guaranteed right; only the conversation is
model-written.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..normalize import normalize
from .errors import corrupt
from .template import EOS, STUDENT, SYSTEM, TEACHER, _FIX, _PRAISE, _WHY

IGNORE = -100


def correction_prefix(c, level: str, rng: random.Random) -> str:
    parts = [rng.choice(_PRAISE),
             normalize(rng.choice(_FIX[c.kind]).format(right=c.right, wrong=c.wrong))]
    if level != "tiro":
        parts.append(_WHY[c.kind])
    return " ".join(parts)


def weave_correction(turns: list, level: str, rng: random.Random) -> tuple[list, dict | None]:
    """Corrupt one student turn; prefix the following teacher reply with the fix."""
    idx = [i for i, (r, _) in enumerate(turns) if r == STUDENT and i + 1 < len(turns)]
    rng.shuffle(idx)
    for i in idx:
        c = corrupt(turns[i][1], rng, kinds=("case", "verb"))
        if c is None:
            continue
        out = list(turns)
        out[i] = (STUDENT, c.corrupted)
        prefix = correction_prefix(c, level, rng)
        out[i + 1] = (TEACHER, prefix + " " + turns[i + 1][1])
        return out, {"wrong": c.wrong, "right": c.right, "kind": c.kind,
                     "turn": i + 1, "prefix": prefix}
    return turns, None


def segments(turns: list, level: str, correction: dict | None = None) -> list[tuple[str, bool]]:
    """(text, trainable) pieces in render order. Only teacher text is trainable.

    A woven correction prefix is context, not a target. A 400M model cannot tell
    right Latin from wrong reliably (it "corrected" FORUM to FORUM), so it must
    never learn to open a reply with a correction on its own. The detector and
    checker decide; the wrapper writes the fix into the reply and the model
    learns only to carry the conversation on from there.
    """
    out = [(f"GRADUS: {level.upper()}\n{SYSTEM[level]}\n\n", False)]
    for n, (role, text) in enumerate(turns):
        if role == TEACHER:
            out.append((f"{TEACHER}:", False))
            prefix = correction["prefix"] if correction and correction.get("turn") == n else None
            if prefix and text.startswith(prefix + " "):
                out.append((f" {prefix}", False))
                text = text[len(prefix) + 1:]
            out.append((f" {text}\n", True))
        else:
            out.append((f"{STUDENT}: {text}\n", False))
    out.append((EOS, True))    # learn to stop after the teacher's turn
    return out


def tokenize_example(tok, pieces) -> tuple[list[int], list[int]]:
    ids, labels = [], []
    for text, trainable in pieces:
        if text == EOS:
            p = [tok.eos_token_id]
        else:
            p = tok(text, add_special_tokens=False)["input_ids"]
        ids.extend(p)
        labels.extend(p if trainable else [IGNORE] * len(p))
    return ids, labels


def build(dialogue_path: str | Path, correction_share: float = 0.5, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for line in open(dialogue_path, encoding="utf-8"):
        d = json.loads(line)
        turns = [tuple(t) for t in d["turns"]]
        meta = None
        if rng.random() < correction_share:
            turns, meta = weave_correction(turns, d["level"], rng)
        out.append({"level": d["level"], "turns": turns, "correction": meta,
                    "engine": d.get("engine"), "topic": d.get("topic")})
    return out


def pack(tok, examples: list[dict], seq_len: int = 1024) -> tuple[list, list]:
    """Greedy packing without splitting an example; pad with EOS, labels IGNORE."""
    seqs, labs = [], []
    cur_i, cur_l = [], []
    for ex in examples:
        pieces = ex.get("pieces") or segments(ex["turns"], ex["level"], ex.get("correction"))
        i, l = tokenize_example(tok, pieces)
        if len(i) > seq_len:
            i, l = i[:seq_len], l[:seq_len]
        if len(cur_i) + len(i) > seq_len:
            pad = seq_len - len(cur_i)
            seqs.append(cur_i + [tok.eos_token_id] * pad)
            labs.append(cur_l + [IGNORE] * pad)
            cur_i, cur_l = [], []
        cur_i += i; cur_l += l
    if cur_i:
        pad = seq_len - len(cur_i)
        seqs.append(cur_i + [tok.eos_token_id] * pad)
        labs.append(cur_l + [IGNORE] * pad)
    return seqs, labs


def write(seqs, labs, out_dir: str | Path, split: str) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = Path(out_dir) / split
    d.mkdir(parents=True, exist_ok=True)
    n = len(seqs[0]) if seqs else 0
    t = pa.table({"input_ids": pa.array(seqs, type=pa.list_(pa.uint16(), n)),
                  "labels": pa.array(labs, type=pa.list_(pa.int32(), n))})
    p = d / f"{split}-00000.parquet"
    pq.write_table(t, p, compression="zstd")
    return p
