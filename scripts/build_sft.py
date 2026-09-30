"""Pack the tutor SFT set from generated dialogues.

    python scripts/build_sft.py --passes 3 --out out/sft/packed

Dialogues are deduplicated and split 95/5 by dialogue. Each training pass
re-weaves corrections with a new seed, so repeated dialogues carry different
planted errors. Output: <out>/{train,val}/*.parquet plus the tokenizer.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile

import numpy as np
from transformers import AutoTokenizer

from claudius.sft import dataset as d


def _jsonl(lines: list[str]) -> str:
    f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8")
    f.writelines(lines)
    f.close()
    return f.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", nargs="+",
                    default=["out/sft/dialogues.jsonl", "out/sft/dialogues_v2.jsonl",
                             "out/sft/dialogues_persona.jsonl", "out/sft/dialogues_messy.jsonl"])
    ap.add_argument("--no-weave", nargs="*", default=["out/sft/dialogues_messy.jsonl"],
                    help="dialogue files that get no planted corrections: their student "
                         "turns are already broken, and the teacher recasts them")
    ap.add_argument("--tokenizer", default="out/mix-tokenizer/tokenizer")
    ap.add_argument("--out", default="out/sft/packed")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--correction-share", type=float, default=0.5)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--errors", default=None,
                    help="learner-error pairs (scripts/gen_errors.py) for the correction task")
    ap.add_argument("--pool", default="out/sft/clean_pool.jsonl")
    ap.add_argument("--correction-passes", type=int, default=1)
    ap.add_argument("--negative-ratio", type=float, default=0.5,
                    help="extra correct sentences per error pair, per pass")
    ap.add_argument("--dev-errors", type=int, default=1000,
                    help="error pairs held out to tune detector thresholds")
    a = ap.parse_args()

    from claudius.sft.persona import contradicts

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    seen, rows, plain = set(), [], set()
    for path in a.dialogues:
        for line in open(path, encoding="utf-8"):
            if path in a.no_weave:
                plain.add(line)
            turns = json.loads(line)["turns"]
            key = json.dumps(turns)
            if contradicts(turns):
                continue
            if key not in seen:
                seen.add(key)
                rows.append(line)
    random.Random(42).shuffle(rows)
    n_val = len(rows) // 20

    def build(lines, seed):
        woven = d.build(_jsonl([l for l in lines if l not in plain]), a.correction_share, seed=seed)
        kept = [l for l in lines if l in plain]
        return woven + (d.build(_jsonl(kept), 0.0, seed=seed) if kept else [])

    val = build(rows[:n_val], 100)
    train = []
    for seed in range(a.passes):
        ex = build(rows[n_val:], seed)
        random.Random(seed).shuffle(ex)
        train += ex

    if a.errors:
        from claudius.sft import correct

        errs = [json.loads(l) for l in open(a.errors, encoding="utf-8")]
        random.Random(7).shuffle(errs)
        dev = errs[:a.dev_errors]
        dev_right = {e["right"] for e in dev}
        pool = [json.loads(l)["text"] for l in open(a.pool, encoding="utf-8")]
        random.Random(8).shuffle(pool)
        err_right = {e["right"] for e in errs}
        dev_neg = [s for s in pool if s not in err_right][:a.dev_errors]
        with open("out/sft/correct_dev.jsonl", "w", encoding="utf-8") as f:
            for e in dev:
                f.write(json.dumps({"sentence": e["wrong"], "right": e["right"]}, ensure_ascii=False) + "\n")
            for s in dev_neg:
                f.write(json.dumps({"sentence": s, "right": s}, ensure_ascii=False) + "\n")
        n_train = len(errs) - len(dev)
        cex = []
        for p in range(a.correction_passes):
            ex = correct.examples(a.errors, a.pool, negatives=int(n_train * a.negative_ratio),
                                  exclude=dev_right | set(dev_neg), seed=p)
            if p == 0:
                n_val_c = len(ex) // 50
                val += ex[:n_val_c]
                ex = ex[n_val_c:]
            cex += ex
        train += cex
        random.Random(9).shuffle(train)
        print(f"correction task: {len(cex)} examples ({n_train} error pairs), dev -> out/sft/correct_dev.jsonl")

    shutil.rmtree(a.out, ignore_errors=True)
    for split, ex in (("train", train), ("val", val)):
        seqs, labs = d.pack(tok, ex, a.seq_len)
        path = d.write(seqs, labs, a.out, split)
        trained = int(sum((np.array(l) != d.IGNORE).sum() for l in labs))
        print(f"{split}: {len(ex)} examples, {len(seqs)} sequences, {trained} trained tokens -> {path}")
    shutil.copytree(a.tokenizer, f"{a.out}/tokenizer")


if __name__ == "__main__":
    main()
