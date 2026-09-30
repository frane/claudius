"""The released correction mode as a Hugging Face user gets it: any change is a flag.

    python scripts/eval_correct_mode.py out/publish/claudius-latin-400m-chat

Greedy CORRIGE/RECTE with the repetition penalties off, no second detector and
no checker. Sets: dev learner errors (claude-written, never trained on),
held-out Vulgate A/B, the 51 tutor messages.
"""
from __future__ import annotations

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts")
from eval_detector import heldout  # noqa: E402

from claudius.normalize import normalize  # noqa: E402


def main():
    path = sys.argv[1]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()

    def correct(s):
        ids = tok(f"CORRIGE: {s}\nRECTE:", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=0,
                                 max_new_tokens=64, eos_token_id=[tok.eos_token_id, 199])
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    sets = {}
    dev = [json.loads(l) for l in open("out/sft/correct_dev.jsonl")]
    sets["dev"] = [(d["sentence"], d["right"]) for d in dev]
    for name, args in {"A": (11, 900, 0), "B": (23, 2000, 1000)}.items():
        correct_s, errors = heldout(*args)
        sets[name] = [(s, s) for s in correct_s] + [(c.corrupted, c.original) for c in errors]
    msgs = json.load(open("eval/tutor_messages.json"))
    sets["messages"] = [(m["text"], m["text"] if not m.get("error") else
                         normalize(m["text"]).replace(m["error"][0], m["error"][1], 1)) for m in msgs]
    report = {}
    for name, items in sets.items():
        fa = n_ok = caught = exact = n_err = 0
        for s, right in items:
            s, right = normalize(s), normalize(right)
            fixed = correct(s)
            changed = fixed != s
            if s == right:
                n_ok += 1; fa += changed
            else:
                n_err += 1; caught += changed; exact += fixed == right
        report[name] = {"false_alarms": fa, "n_correct": n_ok, "flagged": caught,
                        "exact_fix": exact, "n_errors": n_err}
        print(name, json.dumps(report[name]), flush=True)
    json.dump(report, open("out/correct_mode_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
