"""Grid-search the tutor's decoding on a fixed set of student messages.

    python scripts/tune_tutor.py --model out/tutor/final --out out/tune.json [--judge]

Per setting it reports, over eval/tutor_messages.json:
  loop       reply repeats a 3-gram (repeat_rate > 0)
  no_q       reply does not end with a question
  grammar    the checker objects to the reply's own Latin
  off_topic  --judge only: `claude -p` says the reply ignores the student
  identity   identity questions answered with CLAUDIUS
plus detector numbers (the same for every setting, detection is deterministic):
false alarms on correct messages, caught errors and right fixes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from claudius import tutor
from claudius.generate import repeat_rate
from claudius.sft.checker import check
from claudius.sft.template import STUDENT, Turn

JUDGE = """You are grading replies of a Latin tutor chatbot. For each numbered pair, answer whether the MAGISTER reply responds sensibly and on topic to what the DISCIPULUS wrote (a correction followed by a relevant question counts as on topic). Ignore small grammar slips. Reply with ONLY a JSON list of true/false, one per pair, in order.

{pairs}"""


def judge(pairs: list[tuple[str, str]]) -> list[bool] | None:
    text = "\n".join(f"{i+1}. DISCIPULUS: {s}\n   MAGISTER: {r}" for i, (s, r) in enumerate(pairs))
    r = subprocess.run(["claude", "-p", JUDGE.format(pairs=text), "--output-format", "text",
                        "--effort", "low", "--strict-mcp-config",
                        "--settings", '{"disableAllHooks": true}',
                        "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Agent"],
                       capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    try:
        out = r.stdout[r.stdout.index("["): r.stdout.rindex("]") + 1]
        v = json.loads(out)
        return v if len(v) == len(pairs) else None
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/tutor-v3/final")
    ap.add_argument("--messages", default="eval/tutor_messages.json")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--penalties", type=float, nargs="+", default=[1.1, 1.15, 1.2])
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--best-of", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", default="out/tune.json")
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(device).eval()
    msgs = json.load(open(a.messages))

    # Detection does not depend on decoding: compute it once per message.
    real_detect = tutor.detect
    tutor.detect = lru_cache(maxsize=None)(lambda m, t, s: real_detect(m, t, s))

    results = []
    for temp, pen, k in itertools.product(a.temperatures, a.penalties, a.best_of):
        settings = dict(do_sample=True, temperature=temp, top_p=a.top_p,
                        repetition_penalty=pen, no_repeat_ngram_size=4)
        rows = []
        for m in msgs:
            answer, f = tutor.reply(model, tok, [Turn(STUDENT, m["text"])], m["level"],
                                    settings=settings, best_of=k, seed=0)
            body = answer
            if f is not None:   # judge only the model's own words
                body = answer[len(tutor.correction_text(f, m["level"], __import__("random").Random(0))):].strip()
            rows.append({"id": m["id"], "student": m["text"], "reply": answer,
                         "loop": repeat_rate(body) > 0, "no_q": not body.endswith("?"),
                         "grammar": bool(check(body)) if body else False,
                         "identity_ok": ("CLAUDI" in answer) if m.get("identity") else None,
                         "flag": None if f is None else [f.wrong, f.right, f.confirmed],
                         "error": m.get("error")})
        if a.judge:
            verdict = judge([(r["student"], r["reply"]) for r in rows])
            for r, v in zip(rows, verdict or [None] * len(rows)):
                r["on_topic"] = v
        n = len(rows)
        ids = [r for r in rows if r["identity_ok"] is not None]
        summary = {
            "temperature": temp, "repetition_penalty": pen, "best_of": k,
            "loop": sum(r["loop"] for r in rows) / n,
            "no_question": sum(r["no_q"] for r in rows) / n,
            "grammar_objections": sum(r["grammar"] for r in rows) / n,
            "identity": sum(r["identity_ok"] for r in ids) / max(len(ids), 1),
        }
        if a.judge and all(r.get("on_topic") is not None for r in rows):
            summary["off_topic"] = sum(not r["on_topic"] for r in rows) / n
        print(json.dumps(summary), flush=True)
        results.append({"summary": summary, "rows": rows})

    rows = results[0]["rows"]
    correct = [r for r in rows if not r["error"]]
    errs = [r for r in rows if r["error"]]
    det = {"false_alarms": sum(r["flag"] is not None for r in correct), "correct_messages": len(correct),
           "caught": sum(r["flag"] is not None for r in errs), "errors": len(errs),
           "right_fix": sum(r["flag"] is not None and r["flag"][:2] == r["error"] for r in errs)}
    print("detector:", json.dumps(det))
    json.dump({"detector": det, "grid": results}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
