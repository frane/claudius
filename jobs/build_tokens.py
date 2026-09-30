# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=3.0", "tokenizers>=0.20", "transformers>=4.45",
#     "huggingface_hub>=0.25", "pyarrow>=15", "numpy>=1.26",
# ]
# ///
"""Tokenizer sweep and packing on HF Jobs (CPU).

NOT named tokenize.py. A job script sits first on sys.path, so a file called
`tokenize.py` shadows the standard library's `tokenize`, which `linecache` and
`inspect` import internally. The job dies with
`AttributeError: module 'tokenize' has no attribute 'open'` from deep inside an
unrelated dependency, nowhere near anything that looks like this file.

    hf jobs uv run --flavor cpu-performance --secrets HF_TOKEN --timeout 12h \
      jobs/build_tokens.py --corpus-repo fbandov/claudius-corpus --push fbandov/claudius-tokens
"""
import argparse, json
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
    p.add_argument("--corpus-repo", nargs="+", default=None,
                   help="prepared corpus repos, each optionally repo:weight")
    p.add_argument("--corpus", nargs="+", default=["corpus"])
    p.add_argument("--tokenizer-out", default="tokenizer")
    p.add_argument("--out", default="packed")
    p.add_argument("--vocab-size", type=int, default=None, help="omit to run the sweep")
    p.add_argument("--tokenizer-docs", type=int, default=20000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--push", default=None)
    a = p.parse_args()

    install_claudius()

    if a.corpus_repo:
        from huggingface_hub import snapshot_download

        specs = []
        for item in a.corpus_repo:
            repo, _, weight = item.rpartition(":")
            if not repo:
                repo, weight = item, "1"
            local = os.path.join("corpora", repo.replace("/", "__"))
            path = snapshot_download(repo, repo_type="dataset", local_dir=local,
                                     allow_patterns=["*.jsonl.gz", "stats.json"],
                                     token=os.environ.get("HF_TOKEN"))
            specs.append(f"{path}:{weight}")
            print(f"[setup] {repo} (weight {weight}) -> {path}", flush=True)
        a.corpus = specs

    from claudius.pipeline import stage_pack, stage_tokenizer
    from claudius.prepare import push

    card = stage_tokenizer(a.corpus, a.tokenizer_out, vocab_size=a.vocab_size,
                           train_docs=a.tokenizer_docs)
    print(json.dumps(card, indent=2), flush=True)
    summary = stage_pack(a.corpus, a.tokenizer_out, a.out,
                         seq_len=a.seq_len, max_tokens=a.max_tokens)
    print(json.dumps(summary, indent=2), flush=True)
    if a.push:
        import shutil

        shutil.copytree(a.tokenizer_out, os.path.join(a.out, "tokenizer"), dirs_exist_ok=True)
        push(a.out, a.push, private=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
