"""Quality of each browser variant, same inputs for all (ONNX Runtime, CPU).

    .venv-onnx/bin/python scripts/eval_onnx.py out/onnx/web

- bits per byte on held-out text (40 documents from the whole-author validation
  split, first 512 tokens each)
- correction mode on the 51 tutor messages: errors fixed exactly, correct
  messages changed
- agreement with the fp32 variant: share of greedy chat replies and corrections
  that are word-for-word identical
"""
import gzip
import json
import math
import sys
import time

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

VARIANTS = ["model.onnx", "model_fp16.onnx", "model_quantized.onnx", "model_q4.onnx", "model_q4f16.onnx"]
CHAT = ["SALVE, MAGISTER! QUIS ES?", "HODIE PLUIT.", "CANEM PARUUM HABEO. ET TU?",
        "QUOT MILITES IN LEGIONE ERANT?", "LIBROS DE BELLO TROIANO LEGERE MIHI PLACET.",
        "PATER MEUS AGRICOLA EST.", "QUID AGIS?", "UBI HABITAS, MAGISTER?"]
SYSTEM = ("GRADUS: TIRO\nTU ES CLAUDIUS, MAGISTER LINGUAE LATINAE. DISCIPULUS EST TIRO. RESPONDE BREUITER "
          "ET SIMPLICITER. SI ERRAT, CORRIGE ET REGULAM EXPLICA. SEMPER INTERROGA.\n\n")


def main():
    d = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(d)
    docs = []
    for l in gzip.open("out/corpus-clean/val-00000.jsonl.gz", "rt"):
        docs.append(json.loads(l)["text"])
        if len(docs) >= 40:
            break
    msgs = json.load(open("eval/tutor_messages.json"))
    ref = {}
    for v in VARIANTS:
        m = ORTModelForCausalLM.from_pretrained(d, subfolder="onnx", file_name=v, use_cache=True, use_io_binding=False)
        t0 = time.time()
        nll = nbytes = 0.0
        for text in docs:
            ids = tok(text, return_tensors="pt").input_ids[:, :512]
            logits = m(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].double().numpy()
            ids = ids.numpy()
            logp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits.max(-1, keepdims=True)
            nll -= logp[np.arange(ids.shape[1] - 1), ids[0, 1:]].sum()
            nbytes += len(tok.decode(ids[0, 1:]).encode())
        bpb = nll / math.log(2) / nbytes

        def gen(prompt, n=40):
            ids = tok(prompt, return_tensors="pt").input_ids
            out = m.generate(ids, do_sample=False, max_new_tokens=n, repetition_penalty=1.0,
                             no_repeat_ngram_size=0, eos_token_id=[tok.eos_token_id, 199])
            return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

        fixed = fa = 0
        outs = []
        for msg in msgs:
            r = gen(f"CORRIGE: {msg['text']}\nRECTE:", 48)
            outs.append(r)
            if msg.get("error"):
                fixed += r == msg["text"].replace(msg["error"][0], msg["error"][1], 1)
            else:
                fa += r != msg["text"]
        chats = [gen(SYSTEM + f"DISCIPULUS: {c}\nMAGISTER:") for c in CHAT]
        outs += chats
        if v == "model.onnx":
            ref = outs
        agree = sum(a == b for a, b in zip(outs, ref)) / len(ref)
        print(json.dumps({"variant": v, "bits_per_byte": round(bpb, 4), "fixed_exact": f"{fixed}/21",
                          "correct_changed": f"{fa}/30", "same_as_fp32": round(agree, 3),
                          "seconds": round(time.time() - t0)}), flush=True)
        print("   ", chats[0], "|", chats[2], flush=True)


if __name__ == "__main__":
    main()
