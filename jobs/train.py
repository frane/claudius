# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4", "transformers>=4.45", "datasets>=3.0", "tokenizers>=0.20",
#     "accelerate>=0.34", "huggingface_hub>=0.25", "pyarrow>=15", "numpy>=1.26",
#     "trackio",
# ]
# ///
"""Pretraining on HF Jobs (GPU). Nothing here loads pretrained weights.

    hf jobs uv run --flavor h200 --secrets HF_TOKEN --timeout 24h \
      jobs/train.py --data-repo fbandov/claudius-tokens --size 1b \
                    --hub-repo fbandov/claudius-1b --trackio

Checkpoints push to the Hub every --save-steps; on timeout, relaunch with
--resume and it continues from the last one.
"""
import argparse
import pathlib
import os, sys

CODE_REPO = os.environ.get("CLAUDIUS_CODE_REPO", "fbandov/claudius-code")
WHEEL = os.environ.get("CLAUDIUS_WHEEL", "claudius-0.1.0-py3-none-any.whl")


def install_claudius(target: str = "/tmp/claudius-pkg"):
    """Make the package importable, from a PRIVATE Hub repo, using HF_TOKEN.

    Not a uv dependency, because a dependency URL carries no credentials. And
    not `pip install`, because a uv-managed environment has no pip in it --
    `python -m pip` fails with "No module named pip". A wheel is a zip and this
    package is pure Python, so unpacking it onto sys.path needs no installer at
    all and cannot fail on a missing one.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CODE_REPO, WHEEL, repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))
    zipfile.ZipFile(path).extractall(target)
    sys.path.insert(0, target)
    import claudius

    print(f"[setup] claudius {claudius.__version__} from {CODE_REPO} -> {target}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-repo", default=None)
    p.add_argument("--data", default="packed")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--size", default="1b", choices=["150m", "400m", "1b", "test"])
    p.add_argument("--out", default="model")
    p.add_argument("--hub-repo", default=None)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=2000)
    p.add_argument("--eval-sequences", type=int, default=512)
    p.add_argument("--trackio", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default=None,
                   help="continue from these weights (annealing); a local path or Hub repo")
    p.add_argument("--eval-corpus-repo", default=None,
                   help="prepared corpus to evaluate against when training ends")
    p.add_argument("--eval-val-docs", type=int, default=150)
    p.add_argument("--eval-train-docs", type=int, default=1500)
    p.add_argument("--no-eval", action="store_true")
    a = p.parse_args()

    install_claudius()

    if a.data_repo:
        from huggingface_hub import snapshot_download

        a.data = snapshot_download(a.data_repo, repo_type="dataset",
                                   local_dir=a.data, token=os.environ.get("HF_TOKEN"))
    tokenizer = a.tokenizer or os.path.join(a.data, "tokenizer")

    from claudius.train import TrainConfig, restore_hub_checkpoint, run

    # --resume needs the checkpoint LOCALLY; a fresh container has none.
    if a.resume and a.hub_repo:
        if restore_hub_checkpoint(a.hub_repo, a.out, os.environ.get("HF_TOKEN")) is None:
            print(f"[resume] no checkpoint on {a.hub_repo}; starting fresh", flush=True)
            a.resume = False


    cfg = TrainConfig(size=a.size, data_dir=a.data, tokenizer_dir=tokenizer, out_dir=a.out,
                      epochs=a.epochs, per_device_batch=a.batch, grad_accum=a.accum, lr=a.lr,
                      max_steps=a.max_steps, save_steps=a.save_steps,
                      eval_steps=a.eval_steps, eval_sequences=a.eval_sequences,
                      hub_repo=a.hub_repo,
                      trackio=a.trackio, resume=a.resume, init_from=a.init_from,
                      run_name=f"claudius-{a.size}")
    final = run(cfg)
    print("final model:", final, flush=True)

    # Evaluate on the GPU that just trained it. Otherwise the model has to be
    # pulled back down -- 4 GB for the 1B -- to compute numbers this machine can
    # produce in seconds, and the run ends without ever being scored.
    if a.no_eval:
        return 0
    try:
        import json as _json

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM

        from claudius.evaluate import (
            InterpolatedBigram, MemorisationIndex, bits_per_byte, sample_incipits,
        )
        from claudius.prepare import read_texts
        from claudius.tokenizer import load as load_tok

        corpus = a.eval_corpus_repo
        corpus_dir = (snapshot_download(corpus, repo_type="dataset",
                                        local_dir="eval-corpus",
                                        token=os.environ.get("HF_TOKEN"))
                      if corpus else a.data)
        val = list(read_texts(corpus_dir, "val", limit=a.eval_val_docs))
        train_sample = list(read_texts(corpus_dir, "train", limit=a.eval_train_docs))
        if not val:
            print("[eval] no validation documents; skipping", flush=True)
            return 0

        tok = load_tok(final)
        model = AutoModelForCausalLM.from_pretrained(final)
        if torch.cuda.is_available():
            model = model.cuda()

        report = {"val_documents": len(val)}
        floor = InterpolatedBigram().fit(train_sample)
        report["bigram_floor_bits_per_byte"] = round(floor.bits_per_byte(val), 4)
        res = bits_per_byte(model, tok, val, seq_len=cfg.seq_len)
        report["model_bits_per_byte"] = round(res.bits_per_byte, 4)
        report["model_perplexity_per_token"] = round(res.perplexity, 3)
        report["improvement_over_floor"] = round(
            report["bigram_floor_bits_per_byte"] - res.bits_per_byte, 4)
        samples = sample_incipits(model, tok, max_new_tokens=60)
        report["samples"] = samples
        index = MemorisationIndex(n=13).add_all(train_sample)
        # Continuations only: every incipit is itself a verbatim quotation,
        # so windows spanning the prompt 'match' on any model trained on it.
        mem = index.check([s["continuation"] for s in samples],
                          val[:20])
        report["memorisation"] = mem.as_dict()
        print("[eval] " + _json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        pathlib.Path(final, "eval.json").write_text(
            _json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        # A failed evaluation must not discard a finished training run.
        print(f"[eval] failed: {exc!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
