"""Learned detector, swap detector, and their combination, side by side.

    python scripts/eval_combined.py --model out/tutor-v3/final

Combination (Frane's spec):
  firm  = both detectors propose the same fix AND the checker objects to it
  hint  = either detector flags on its own (at its own thresholds)
Every signal is computed once per sentence; all three verdicts come from it.
Sets: dev learner errors (claude-written, never trained on), held-out Vulgate
A/B (swap-generated), the 51 tutor messages.
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from claudius.sft.checker import check
from claudius.sft.correct import propose
from claudius.sft.scorer import sentence_logprob
from claudius.tutor import candidates

sys_path = __import__("sys").path
sys_path.insert(0, "scripts")
from eval_detector import heldout  # noqa: E402

# Thresholds: learned (v3, margin in nats, picked on dev), swaps (as shipped).
LEARNED_FIRM, LEARNED_HINT = 0.0, 4.0
SWAP_FIRM, SWAP_HINT = 2.5, 8.0


def signals(model, tok, sentence: str) -> dict:
    ds = check(sentence)
    flagged = {w for d in ds for w in (d.word, d.head)}
    c = propose(model, tok, sentence, samples=0)
    learned = None
    if c is not None and len(c.changes) == 1:
        learned = {"fix": list(c.changes[0]), "margin": c.margin,
                   "hit": c.changes[0][0] in flagged}
    elif c is not None:
        learned = {"fix": [" ".join(w for w, _ in c.changes), " ".join(r for _, r in c.changes)],
                   "margin": c.margin, "hit": any(w in flagged for w, _ in c.changes)}
    base = sentence_logprob(model, tok, sentence)
    swaps = [(w, r, sentence_logprob(model, tok, alt) - base) for alt, w, r in candidates(sentence)]
    return {"flagged": sorted(flagged), "learned": learned, "swaps": swaps}


def v_learned(s):
    l = s["learned"]
    if l is None:
        return None, None
    if l["hit"] and l["margin"] > LEARNED_FIRM:
        return "firm", tuple(l["fix"])
    if l["margin"] >= LEARNED_HINT:
        return "hint", tuple(l["fix"])
    return None, None


def v_swaps(s):
    flagged = set(s["flagged"])
    firm = [x for x in s["swaps"] if x[0] in flagged and x[2] > SWAP_FIRM]
    if firm:
        w, r, _ = max(firm, key=lambda x: x[2])
        return "firm", (w, r)
    hint = [x for x in s["swaps"] if x[2] >= SWAP_HINT]
    if hint:
        w, r, _ = max(hint, key=lambda x: x[2])
        return "hint", (w, r)
    return None, None


def v_combined(s):
    l = s["learned"]
    if l is not None and l["hit"]:
        fix = tuple(l["fix"])
        # the swap detector must prefer the very same fix
        if any((w, r) == fix and g > 0 for w, r, g in s["swaps"]):
            return "firm", fix
    for v in (v_learned(s), v_swaps(s)):
        if v[0] is not None:
            return "hint", v[1]
    return None, None


DETECTORS = {"learned": v_learned, "swaps": v_swaps, "combined": v_combined}


def score(items, sig):
    """items: (sentence, expected fix or None)."""
    out = {}
    for name, fn in DETECTORS.items():
        fa_f = fa_h = caught = firm = right = n_err = n_ok = 0
        for (sent, exp), s in zip(items, sig):
            v, fix = fn(s)
            if exp is None:
                n_ok += 1
                fa_f += v == "firm"
                fa_h += v == "hint"
            else:
                n_err += 1
                caught += v is not None
                firm += v == "firm"
                right += v is not None and fix == tuple(exp)
        out[name] = {"firm_fa": fa_f, "hint_fa": fa_h, "n_correct": n_ok,
                     "caught": caught, "firm_caught": firm, "right_fix": right, "n_errors": n_err}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/tutor-v3/final")
    ap.add_argument("--dev", default="out/sft/correct_dev.jsonl")
    ap.add_argument("--out", default="out/combined_eval.json")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(device).eval()

    from claudius.sft.correct import diff_words
    sets = {}
    dev = [json.loads(l) for l in open(a.dev)]
    sets["dev"] = [(d["sentence"], None if d["sentence"] == d["right"] else
                    (diff_words(d["sentence"], d["right"]) or [[None, None]])[0]) for d in dev]
    for name, args in {"A": (11, 900, 0), "B": (23, 2000, 1000)}.items():
        correct, errors = heldout(*args)
        sets[name] = [(s, None) for s in correct] + [(c.corrupted, (c.wrong, c.right)) for c in errors]
    msgs = json.load(open("eval/tutor_messages.json"))
    sets["messages"] = [(m["text"], m.get("error")) for m in msgs]

    report = {}
    for name, items in sets.items():
        items = [(s, tuple(e) if e else None) for s, e in items]
        sig = [signals(model, tok, s) for s, _ in items]
        report[name] = score(items, sig)
        for det, r in report[name].items():
            print(name, det, json.dumps(r), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
