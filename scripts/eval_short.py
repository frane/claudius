"""The model's own fix (CORRIGE/RECTE) on SHORT learner messages: how low can
the hint threshold go before correct sentences and names get flagged?

    python scripts/eval_short.py out/tutor-v6/final

Sets as in eval_lexicon.py (clean learner sentences, sentences with modern
names, sentences with an injected typo) plus the 51 tutor messages and the
held-out Vulgate correct sentences (A), all restricted to <= 12 words.
"""
import json
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts")
from eval_lexicon import NAMES, TEMPLATES, typo  # noqa: E402
from eval_detector import heldout  # noqa: E402

from claudius.normalize import normalize  # noqa: E402
from claudius.sft.correct import propose  # noqa: E402


def main():
    p = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32).to("mps").eval()
    rng = random.Random(5)
    sents = [t for l in open("out/sft/dialogues_v2.jsonl") for r, t in json.loads(l)["turns"]
             if r == "DISCIPULUS" and 2 <= len(t.split()) <= 12]
    rng.shuffle(sents)
    clean, typos = sents[:240], []
    for s in sents[240:]:
        ws = re.findall(r"[A-Z]{4,}", s)
        if not ws: continue
        w = rng.choice(ws); t = typo(w, rng)
        if t: typos.append((re.sub(rf"(?<![A-Z]){w}(?![A-Z])", t, s, count=1), s))
        if len(typos) >= 240: break
    names = [normalize(tp.format(n=n)) for n in NAMES for tp in rng.sample(TEMPLATES, 4)]
    vulg = [s for s in heldout(11, 900, 0)[0] if len(s.split()) <= 12]
    msgs = json.load(open("eval/tutor_messages.json"))
    msg_ok = [normalize(m["text"]) for m in msgs if not m.get("error")]
    msg_err = [(normalize(m["text"]), normalize(m["text"]).replace(m["error"][0], m["error"][1], 1)) for m in msgs if m.get("error")]

    def margin(s):
        c = propose(model, tok, s, samples=0)
        return None if c is None else (c.margin, c.right)

    ok_sets = {"clean learner": clean, "names": names, "Vulgate correct": vulg, "messages correct": msg_ok}
    err_sets = {"typos": typos, "messages errors": msg_err}
    m_ok = {k: [margin(s) for s in v] for k, v in ok_sets.items()}
    m_err = {k: [(margin(s), right) for s, right in v] for k, v in err_sets.items()}
    for tau in (-99, 0, 0.5, 1, 2, 4):
        parts = [f"{k} FA {sum(x is not None and x[0] > tau for x in v)}/{len(v)}" for k, v in m_ok.items()]
        parts += [f"{k} caught {sum(x is not None and x[0] > tau for x, _ in v)}/{len(v)} exact {sum(x is not None and x[0] > tau and x[1] == r for x, r in v)}"
                  for k, v in m_err.items()]
        print(f"tau {tau:>4}: " + " | ".join(parts), flush=True)


if __name__ == "__main__":
    main()
