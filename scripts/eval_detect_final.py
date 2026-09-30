"""The shipped tutor.detect on FRESH data (seed 11, not used to pick thresholds).

    python scripts/eval_detect_final.py out/tutor-v6/final
"""
import json
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts")
from eval_lexicon import NAMES, TEMPLATES, typo  # noqa: E402

from claudius.normalize import normalize  # noqa: E402
from claudius.tutor import detect  # noqa: E402

p = sys.argv[1]
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32).to("mps").eval()
rng = random.Random(11)
sents = [t for l in open("out/sft/dialogues_v2.jsonl") for r, t in json.loads(l)["turns"]
         if r == "DISCIPULUS" and 2 <= len(t.split()) <= 12]
rng.shuffle(sents)
sents = sents[5000:]          # away from the slices the other evals used
clean, typos = sents[:200], []
for s in sents[200:]:
    ws = re.findall(r"[A-Z]{4,}", s)
    if not ws: continue
    w = rng.choice(ws); t = typo(w, rng)
    if t: typos.append((re.sub(rf"(?<![A-Z]){w}(?![A-Z])", t, s, count=1), w))
    if len(typos) >= 200: break
extra = ["NOMEN MIHI EST {n}.", "EGO ET {n} AMICI SUMUS.", "{n} CANEM HABET.", "UIDISTINE {n}?"]
names = [normalize(tp.format(n=n)) for n in NAMES for tp in rng.sample(TEMPLATES + extra, 3)]
frane = ["SIC FRANE, ET TU?", "TU NOMUS?", "TUUS NEMEN EST?", "TU CLAUDIUS?", "IG?", "NOMEN MEUM EST FRANE.",
         "SALUE, MAGISTR!", "PUELA ROSAM AMAT.", "EGO HABET DUOS FRATRES, STIPE ET DARIO."]
res = lambda s: detect(model, tok, s)
fc = [s for s in clean if res(s)]
fn = [(s, res(s)) for s in names]; fn = [(s, f) for s, f in fn if f]
tc = [(s, w, res(s)) for s, w in typos]
print(f"clean learner flagged {len(fc)}/{len(clean)} | names flagged {len(fn)}/{len(names)} | "
      f"typos caught {sum(f is not None for _, _, f in tc)}/{len(tc)}, exact {sum(f is not None and f.right == w for _, w, f in tc)}")
for s in fc: print("  CLEAN FA:", s, res(s).wrong, "->", res(s).right)
for s, f in fn: print("  NAME FA:", s, f.wrong, "->", f.right)
for s in frane:
    f = res(s); print(f"  {s:42} -> {None if f is None else (f.wrong, f.right, 'firm' if f.confirmed else 'hint')}")
