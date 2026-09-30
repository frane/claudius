"""The released model's output defects, measured the same way for every model.

    python scripts/measure_output.py out/400m/final out/400m-annealed/final ...

For each model: continuations of 40 fixed prompts (classical, everyday, modern),
3 samples each with the anti-loop settings, scored per 1,000 words for
  fused    words the judge model would split by > 2 nats (TUSALUS -> TU SALUS);
           one fixed judge model for all, so the scores are comparable
  numbers  Arabic numerals (edition line and section numbers)
  church   church-Latin words (register drift)
  loops    share of continuations that repeat a 3-gram
"""
from __future__ import annotations

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from claudius.generate import ANTI_LOOP, repeat_rate
from claudius.normalize import normalize
from claudius.quality import church_rate, number_rate
from claudius.tutor import split_fused

PROMPTS = [
    # classical
    "Gallia est omnis divisa in partes tres", "Arma virumque cano", "Quousque tandem abutere",
    "Omnia vincit amor", "Tityre, tu patulae recubans", "Iam ver egelidos refert tepores",
    "Nunc est bibendum", "Vivamus, mea Lesbia", "Urbem Romam a principio reges habuere",
    "Sed de his hactenus. Nunc ad",
    # everyday
    "Hodie mane surrexi et", "Mater mea in culina", "Amicus meus Marcus", "Post cenam in hortum",
    "Canis noster parvus est et", "In schola hodie didicimus", "Frater meus librum legit",
    "Caelum hodie serenum est", "Cras ad forum ibimus", "Puella rosas in horto",
    # modern / expository
    "Roma est caput Italiae", "Urbs Berolinum", "Lingua Latina", "Mare Mediterraneum",
    "Astronomia est scientia", "Musica est ars", "Sol est stella", "Europa est continens",
    "Ars culinaria", "Ludus pedifollis",
    # narrative
    "Olim rex quidam", "Erat vir quidam in urbe", "Nauta in portu", "Milites castra posuerunt",
    "Senex et puer", "Ubi primum sol ortus est", "Epistula ad amicum:", "Fabula de lupo et agno",
    "Mercator ex Graecia", "Iter longum erat",
]


def main():
    judge_path, models = sys.argv[1], sys.argv[2:]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    jtok = AutoTokenizer.from_pretrained(judge_path)
    judge = AutoModelForCausalLM.from_pretrained(judge_path, dtype=torch.float32).to(device).eval()
    out = {}
    for path in models:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
        texts, loops = [], 0
        for i, p in enumerate(PROMPTS):
            ids = tok(normalize(p), return_tensors="pt").input_ids.to(device)
            torch.manual_seed(i)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=80, num_return_sequences=3,
                                     pad_token_id=tok.eos_token_id, **ANTI_LOOP)
            for g in gen:
                t = tok.decode(g[ids.shape[1]:], skip_special_tokens=True)
                loops += repeat_rate(t) > 0.1
                texts.append(t)
        words = sum(len(t.split()) for t in texts)
        fused = 0
        for t in texts:
            for line in t.split("\n"):
                if len(line.split()) >= 3:
                    fused += len(split_fused(judge, jtok, line).split()) - len(line.split())
        joined = "\n".join(texts)
        out[path] = {"words": words, "fused_per_1k": round(1000 * fused / words, 1),
                     "numbers_per_1k": round(number_rate(joined), 1),
                     "church_per_1k": round(church_rate(joined), 1),
                     "loops": round(loops / len(texts), 3),
                     "samples": texts[:3] + texts[30:33] + texts[60:63]}
        print(path, json.dumps({k: v for k, v in out[path].items() if k != "samples"}), flush=True)
        del model
    json.dump(out, open("out/measure_output.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
