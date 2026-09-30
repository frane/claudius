"""Compare the learned detector with the ending-swap one, on data neither saw.

    python scripts/eval_detector.py --model out/tutor/final

1. Thresholds for the learned detector are picked on out/sft/correct_dev.jsonl
   (held out from training): the lowest firm and hint agreement levels whose false-alarm
   rate on dev negatives is <= --max-fa.
2. Both detectors are then scored once on held-out Vulgate sets A and B
   (300 correct / 200 corrupted each, never in any training pool) and on the
   51 tutor test messages.
"""
from __future__ import annotations

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from claudius import tutor
from claudius.sft.build import vulgate_sentences
from claudius.sft.checker import check
from claudius.sft.correct import propose
from claudius.sft.errors import corrupt


def heldout(seed: int, limit: int, lo: int) -> tuple[list[str], list]:
    sents = list(vulgate_sentences(min_words=4, max_words=9, limit=limit))
    rng = random.Random(seed)
    rng.shuffle(sents)
    correct = sents[lo:lo + 300]
    errors = []
    for s in sents[lo + 300:]:
        c = corrupt(s, rng, kinds=("case", "verb"))
        if c:
            errors.append(c)
        if len(errors) >= 200:
            break
    return correct, errors


def raw(model, tok, sentence):
    """Everything thresholds need, computed once: (margin, checker hit, fix)."""
    c = propose(model, tok, sentence)
    if c is None:
        return None
    ds = check(sentence)
    hit = any(w in (d.word, d.head) for d in ds for w, _ in c.changes)
    return {"margin": c.margin, "agreement": c.agreement, "hit": hit, "right": c.right,
            "changes": [list(x) for x in c.changes]}


def verdict(r, firm, hint):
    """Firm: checker objects to a changed word and agreement >= firm. Hint: agreement >= hint."""
    if r is None:
        return None
    if r["hit"] and r["agreement"] >= firm:
        return "firm"
    if r["agreement"] >= hint:
        return "hint"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/tutor/final")
    ap.add_argument("--dev", default="out/sft/correct_dev.jsonl")
    ap.add_argument("--max-fa", type=float, default=0.005)
    ap.add_argument("--skip-swaps", action="store_true", help="old detector already measured")
    ap.add_argument("--out", default="out/detector_eval.json")
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(device).eval()

    # 1. thresholds on dev
    dev = [json.loads(l) for l in open(a.dev)]
    dev_raw = [(d, raw(model, tok, d["sentence"])) for d in dev]
    neg = [r for d, r in dev_raw if d["sentence"] == d["right"]]
    pos = [(d, r) for d, r in dev_raw if d["sentence"] != d["right"]]
    grid = [x / 8 for x in range(0, 9)] + [1.01]     # agreement, 8 samples; 1.01 = never
    firm = next(t for t in grid if sum(r is not None and r["hit"] and r["agreement"] >= t for r in neg) <= a.max_fa * len(neg))
    hint = next(t for t in grid if sum(r is not None and r["agreement"] >= t for r in neg) <= a.max_fa * len(neg))
    dev_caught = sum(verdict(r, firm, hint) is not None for _, r in pos)
    dev_right = sum(verdict(r, firm, hint) is not None and r["right"] == d["right"] for d, r in pos)
    print(f"dev: firm agreement >= {firm}, hint >= {hint} | caught {dev_caught}/{len(pos)}, "
          f"right sentence {dev_right}", flush=True)

    report = {"thresholds": {"firm": firm, "hint": hint}, "sets": {}}
    # 2. held-out sets, both detectors
    for name, (seed, limit, lo) in {"A": (11, 900, 0), "B": (23, 2000, 1000)}.items():
        correct, errors = heldout(seed, limit, lo)
        row = {}
        for det in ("learned",) if a.skip_swaps else ("learned", "swaps"):
            if det == "learned":
                vc = [verdict(raw(model, tok, s), firm, hint) for s in correct]
                ve = []
                for c in errors:
                    r = raw(model, tok, c.corrupted)
                    ve.append((verdict(r, firm, hint), r is not None and r["right"] == c.original))
            else:
                fc = [tutor.detect_swaps(model, tok, s) for s in correct]
                vc = [None if f is None else ("firm" if f.confirmed else "hint") for f in fc]
                ve = []
                for c in errors:
                    f = tutor.detect_swaps(model, tok, c.corrupted)
                    ve.append((None if f is None else ("firm" if f.confirmed else "hint"),
                               f is not None and (f.wrong, f.right) == (c.wrong, c.right)))
            row[det] = {
                "firm_fa": sum(v == "firm" for v in vc), "hint_fa": sum(v == "hint" for v in vc),
                "firm_caught": sum(v == "firm" for v, _ in ve), "caught": sum(v is not None for v, _ in ve),
                "right_fix": sum(v is not None and ok for v, ok in ve), "n_correct": len(correct), "n_errors": len(errors),
            }
            print(name, det, json.dumps(row[det]), flush=True)
        report["sets"][name] = row

    # 3. tutor test messages
    msgs = json.load(open("eval/tutor_messages.json"))
    rows = []
    for m in msgs:
        r = raw(model, tok, m["text"])
        v = verdict(r, firm, hint)
        rows.append({"id": m["id"], "text": m["text"], "verdict": v,
                     "fix": r and r["changes"], "expected": m.get("error")})
        print(f'{m["id"]:8} {str(v):5} {r and r["changes"]} expected {m.get("error")}', flush=True)
    report["messages"] = rows
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
