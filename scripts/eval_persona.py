"""Regression set for perspective and persona (eval/tutor_dialogues.json).

    python scripts/eval_persona.py out/tutor-v5/final [--samples 3]

Each multi-turn case is answered by the tutor (claudius.tutor.respond) several
times with different seeds; claude -p grades each reply against the persona
sheet for:
  perspective  ego/tu and meus/tuus the right way round; "ET TU?" answered
               about the teacher himself
  persona      no contradiction of the persona sheet
  names        the student's people referred to correctly (names, plural/singular)
  sense        the reply answers what a broken or misspelled message most likely
               meant ("TU NOMUS?" = what is your name?), no nonsense or digits
"""
from __future__ import annotations

import argparse
import json
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from claudius.sft.persona import sheet
from claudius.sft.template import Turn
from claudius.tutor import respond

JUDGE = """You grade replies of a Latin tutor chatbot, MAGISTER Claudius. His fixed persona:
{sheet}

For each numbered case you see the conversation and the MAGISTER's new reply. Grade the reply ONLY:
- "perspective": true if he uses EGO/MEUS/MIHI for himself and TU/TUUS/TIBI for the student correctly, and if asked about himself ("ET TU?", "HABESNE...?", "UBI HABITAS?") he answers about HIMSELF. False if he swaps them, echoes the student's question back as if it were his own, or answers about himself with the student's facts.
- "persona": true if nothing he says about himself contradicts the persona (saying nothing about himself is fine). False if it contradicts it (e.g. claims brothers, another home, a cat).
- "names": true if people the student mentioned are referred to correctly (right names, plural for several, singular for one); true if no people are involved.
- "sense": true if the reply is sensible Latin that responds to what the student most likely meant (for broken or misspelled input, understanding the intent, e.g. "TU NOMUS?" = "what is your name?"); false for nonsense, digits, or ignoring the message.
Reply with ONLY a JSON list, one object per case, in order: {{"perspective": bool, "persona": bool, "names": bool, "sense": bool}}.

{cases}"""


def judge(items):
    text = "\n\n".join(
        f"{i+1}.\n" + "\n".join(f"{r}: {t}" for r, t in it["history"]) + f"\nNEW MAGISTER REPLY: {it['reply']}"
        for i, it in enumerate(items))
    r = subprocess.run(["claude", "-p", JUDGE.format(sheet=sheet(), cases=text), "--output-format", "text",
                        "--effort", "low", "--strict-mcp-config", "--settings", '{"disableAllHooks": true}',
                        "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Agent"],
                       capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    out = r.stdout[r.stdout.index("["): r.stdout.rindex("]") + 1]
    v = json.loads(out)
    assert len(v) == len(items), (len(v), len(items))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(device).eval()
    cases = json.load(open("eval/tutor_dialogues.json"))
    items = []
    for c in cases:
        for seed in range(a.samples):
            r = respond(model, tok, [Turn(role, text) for role, text in c["history"]], c["level"], seed=seed)
            items.append({"id": c["id"], "history": c["history"], "reply": r["full"]})
    verdicts = judge(items)
    for it, v in zip(items, verdicts):
        it.update(v)
    n = len(items)
    keys = ("perspective", "persona", "names", "sense")
    summary = {k: sum(it[k] for it in items) / n for k in keys}
    summary["all"] = sum(all(it[k] for k in keys) for it in items) / n
    print(json.dumps({k: round(v, 3) for k, v in summary.items()}), f"({n} replies)")
    for it in items:
        bad = [k for k in keys if not it[k]]
        if bad:
            print(f'  x {it["id"]:17} {it["reply"]}  {bad}')
    if a.out:
        json.dump({"summary": summary, "items": items}, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
