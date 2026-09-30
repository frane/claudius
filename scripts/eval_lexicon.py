"""Misspelling detector: catches, false alarms on clean Latin and on names.

    python scripts/eval_lexicon.py out/tutor-v6/final

Sentences are student turns from the clean dialogues (never misspelled). A typo
(one substitution, deletion, insertion or transposition) is injected into one
word of four letters or more, only where the result is not a dictionary word.
Names: sentences built around modern first names, which must not be "fixed".
Dev half picks the margin (<= 1% false alarms on clean + names), test half reports.
"""
import json
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from claudius import lexicon
from claudius.normalize import normalize

NAMES = ["FRANE", "STIPE", "DARIO", "LUKA", "ANNA", "MARKO", "IVANA", "PETRA", "IOSIP", "MIA", "NIKOLA",
         "KEUIN", "LAURA", "SUEN", "OLGA", "TOMAS", "BIANCA", "HANNA", "PAUEL", "EMMA", "NOAH", "ELIF"]
TEMPLATES = ["NOMEN MEUM EST {n}.", "SUM {n}, DISCIPULUS TUUS.", "{n} FRATER MEUS EST.",
             "AMICA MEA {n} ROMAE HABITAT.", "SALUE, MAGISTER! EGO SUM {n}.", "HODIE CUM {n} IN HORTO LUSI.",
             "SOROR MEA, CUI NOMEN EST {n}, MEDICA EST.", "{n} ET EGO LINGUAM LATINAM DISCIMUS."]
AB = "ABCDEFGHILMNOPQRSTUX"


def typo(word, rng):
    for _ in range(50):
        i = rng.randrange(len(word)); op = rng.choice("sdit")
        if op == "s": w = word[:i] + rng.choice(AB) + word[i + 1:]
        elif op == "d": w = word[:i] + word[i + 1:]
        elif op == "i": w = word[:i] + rng.choice(AB) + word[i:]
        else:
            if i >= len(word) - 1: continue
            w = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if w != word and len(w) >= 3 and not lexicon.is_common(w):
            return w
    return None


def main():
    p = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32).to("mps").eval()
    rng = random.Random(5)
    sents = [t for l in open("out/sft/dialogues_v2.jsonl") for r, t in json.loads(l)["turns"]
             if r == "DISCIPULUS" and 3 <= len(t.split()) <= 12]
    rng.shuffle(sents)
    clean, errs = [], []
    for s in sents:
        if len(clean) < 240:
            clean.append(s); continue
        ws = [w for w in re.findall(r"[A-Z]{4,}", s)]
        if not ws: continue
        w = rng.choice(ws); t = typo(w, rng)
        if t:
            errs.append((re.sub(rf"(?<![A-Z]){w}(?![A-Z])", t, s, count=1), t, w))
        if len(errs) >= 240: break
    names = [normalize(tp.format(n=n)) for n in NAMES for tp in rng.sample(TEMPLATES, 4)]
    rng.shuffle(names)

    def gains(sentence):
        """Every candidate's (wrong, right, gain, edits), so any margins can be read off."""
        from claudius.sft.scorer import sentence_logprob
        out = []
        base = None
        for w in lexicon.suspects(sentence):
            for c in lexicon.candidates(w):
                if base is None:
                    base = sentence_logprob(model, tok, sentence)
                alt = re.sub(rf"(?<![A-Z]){w}(?![A-Z])", c, sentence, count=1)
                out.append((w, c, sentence_logprob(model, tok, alt) - base, lexicon._edit_distance(w, c)))
        return out

    def best(cands, m1, m2):
        ok = [x for x in cands if x[2] > (m1 if x[3] == 1 else m2)]
        return max(ok, key=lambda x: x[2]) if ok else None

    data = {k: [(x, gains(x[0] if isinstance(x, tuple) else x)) for x in v]
            for k, v in (("clean", clean), ("names", names), ("errors", errs))}
    half = lambda xs, h: xs[: len(xs) // 2] if h == "dev" else xs[len(xs) // 2:]

    def rates(split, m1, m2):
        b = lambda g: best(g, m1, m2)
        fa_c = sum(b(g) is not None for _, g in half(data["clean"], split))
        fa_n = sum(b(g) is not None for _, g in half(data["names"], split))
        e = half(data["errors"], split)
        caught = sum(b(g) is not None for _, g in e)
        exact = sum(b(g) is not None and b(g)[0] == t and b(g)[1] == w for (s, t, w), g in e)
        return fa_c, len(half(data["clean"], split)), fa_n, len(half(data["names"], split)), caught, exact, len(e)

    grid = [(m1, m2) for m1 in range(0, 21, 2) for m2 in range(m1, 41, 4)]
    ok = [(rates("dev", *mm), mm) for mm in grid]
    ok = [(r, mm) for r, mm in ok if r[0] == 0 and r[2] == 0]      # no false alarm on dev at all
    r, (m1, m2) = max(ok, key=lambda x: (x[0][4], -x[1][0])) if ok else (None, (20, 40))
    print(f"dev: margins 1-edit {m1} / 2-edit {m2}: clean FA {r[0]}/{r[1]} names FA {r[2]}/{r[3]} caught {r[4]}/{r[6]}")
    fc, nc, fn, nn, c, x, ne = rates("test", m1, m2)
    print(f"TEST: clean FA {fc}/{nc}  names FA {fn}/{nn}  caught {c}/{ne}  exact fix {x}/{ne}")
    for (s, t, w), g in half(data["errors"], "test")[:8]:
        print("   ", s, "->", (best(g, m1, m2) or (None, None))[:2], "| truth", w)
    for split in ("dev", "test"):
        for s, g in half(data["names"], split):
            if best(g, m1, m2): print(f"   NAME FA ({split}):", s, best(g, m1, m2))


if __name__ == "__main__":
    main()
