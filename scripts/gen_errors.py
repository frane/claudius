"""Turn clean Latin sentences into realistic learner errors with `claude -p`.

    python scripts/gen_errors.py --calls 1200 --per-call 25 --workers 6

The fix is verified by construction: it is the original sentence, which comes
from the Vulgate, the classical corpus or a checked dialogue. Claude only writes
the wrong version, and a pair is kept only if exactly 1-2 words changed, the
word count is the same, and the changed words are not mere spelling variants.
Each sentence gets an assigned error type so the mix covers what the
ending-swap detector missed (subject-verb number, multi-slot forms like
VENERUNT -> VENIT, case after prepositions, adjective agreement, tense/person).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_dialogues import call_claude  # noqa: E402

ERROR_TYPES = {
    "subject-verb": "make the verb disagree in number with its subject (e.g. MILITES ... DEFENDIT, PUELLA ... AMANT)",
    "person": "put the verb in the wrong person for its subject (e.g. EGO ... LEGIT, TU ... CANTO, NOS ... LABORAT)",
    "tense": "use a wrong tense that clashes with the context (e.g. HERI ... AMBULO, CRAS ... UENIT)",
    "irregular": "use a wrong form of an irregular or strong verb, changing more than the ending (e.g. UENERUNT -> UENIT, LEGO -> LEGIT, FUIT -> ERAT, IUIT -> IBAT)",
    "preposition": "use the wrong case after a preposition (e.g. IN URBE -> IN URBEM when no motion, CUM PATRE -> CUM PATREM, AD FORUM -> AD FORO)",
    "object": "put the direct object in the nominative, or the subject in the accusative (e.g. PUELLAM UIDEO -> PUELLA UIDEO)",
    "adjective": "break adjective-noun agreement in gender, number or case (e.g. EQUUS ALBUS -> EQUUS ALBA, PUERI BONI -> PUERI BONUS)",
    "declension": "use a wrong-declension or wrong-case ending on a noun (e.g. CANEM -> CANIS, SERUUM -> SERUUS, REGEM -> REGUM)",
}

PROMPT = """You help build a Latin grammar checker. Below are correct Latin sentences, each with an ERROR TYPE. For each one, write the sentence as a learner might write it, with exactly ONE realistic mistake of that type.

Rules:
- Change only one word (two only if the mistake needs it). Keep every other word, the word order and punctuation identical.
- The result must be clearly wrong Latin, not an acceptable alternative (no mere spelling variants, no other valid word order or tense that still makes sense).
- If a sentence has no place for that error type, output it with a different type from the list instead.
- Output ONLY lines of the form "N: sentence", one per input, same numbering. No explanations.

{items}"""

_WORD = re.compile(r"[A-Z]+")


def diff_words(a: str, b: str) -> list[tuple[str, str]] | None:
    """Changed (original, wrong) word pairs, or None if the shape differs."""
    wa, wb = _WORD.findall(a), _WORD.findall(b)
    if len(wa) != len(wb):
        return None
    return [(x, y) for x, y in zip(wa, wb) if x != y]


def valid(orig: str, wrong: str, normalize) -> tuple[bool, list]:
    d = diff_words(orig, wrong)
    if not d or len(d) > 2:
        return False, []
    for x, y in d:
        # U/V and I/J are the same letter after normalisation: a spelling variant.
        if normalize(x) == normalize(y):
            return False, []
    return True, d


def parse(raw: str) -> dict[int, str]:
    out = {}
    for line in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)]\s*(.+)", line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def job(spec):
    idx, items = spec
    text = "\n".join(f"{i+1}: [{t}] {s}" for i, (s, t) in enumerate(items))
    prompt = PROMPT.format(items=text)
    for attempt in range(3):
        try:
            return spec, call_claude(prompt)
        except Exception as e:
            time.sleep(20 * (attempt + 1))
            err = str(e)
    return spec, "ERROR: " + err


def main():
    from claudius.normalize import normalize

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="out/sft/clean_pool.jsonl")
    ap.add_argument("--out", default="out/sft/errors.jsonl")
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--per-call", type=int, default=25)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="pool offset per source, to resume")
    a = ap.parse_args()

    pool = [json.loads(l) for l in open(a.pool)]
    by_src = {}
    for p in pool:
        by_src.setdefault(p["src"], []).append(p["text"])
    # Learner-register first: Vulgate and dialogue student turns 40% each,
    # classical prose 20%.
    rng = random.Random(a.seed)
    weights = {"vulgate": 0.4, "student": 0.4, "classical": 0.2}
    cursors = {s: a.skip for s in by_src}
    types = list(ERROR_TYPES)
    specs = []
    for c in range(a.calls):
        items = []
        for _ in range(a.per_call):
            src = rng.choices(list(weights), weights=list(weights.values()))[0]
            s = by_src[src][cursors[src] % len(by_src[src])]
            cursors[src] += 1
            items.append((s, rng.choice(types)))
        specs.append((c, items))

    kept = rejected = errors = 0
    out = Path(a.out)
    with open(out, "a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
        for (c, items), raw in ex.map(job, specs):
            if raw.startswith("ERROR"):
                errors += 1
                print(f"error: {raw[:120]}", flush=True)
                continue
            got = parse(raw)
            for i, (orig, etype) in enumerate(items):
                wrong = got.get(i + 1)
                if not wrong:
                    rejected += 1
                    continue
                wrong = normalize(wrong)
                ok, d = valid(orig, wrong, normalize)
                if not ok:
                    rejected += 1
                    continue
                kept += 1
                fh.write(json.dumps({"right": orig, "wrong": wrong, "type": etype,
                                     "changes": d}, ensure_ascii=False) + "\n")
            print(f"call {c}: kept={kept} rejected={rejected} errors={errors}", flush=True)
    print(f"DONE kept={kept} rejected={rejected} errors={errors} -> {out}")


if __name__ == "__main__":
    main()
