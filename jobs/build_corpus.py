# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=3.0",
#     "tokenizers>=0.20",
#     "transformers>=4.45",
#     "huggingface_hub>=0.25",
#     "pyarrow>=15",
#     "numpy>=1.26",
# ]
# ///
"""Build the full corpus on HF Jobs, using every vCPU the flavor has.

    hf jobs uv run --flavor cpu-performance --secrets HF_TOKEN --timeout 6h \
      jobs/build_corpus.py --push fbandov/claudius-corpus

The package is installed at runtime from a PRIVATE Hub repo rather than being
declared as a uv dependency: a uv dependency URL carries no credentials, and
the code repo is private.

Resumable. If the job hits its timeout, relaunch the same command: finished
shards carry a DONE marker and are skipped. Pass --resume-from to pick the
part directory back up from a Hub snapshot.
"""
import argparse
import json
import os
import sys

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
    p.add_argument("--source", nargs="+", default=["latin-pd"])
    p.add_argument("--out", default="corpus")
    p.add_argument("--shards", type=int, default=0, help="0 = every shard")
    p.add_argument("--workers", type=int, default=0, help="0 = all vCPUs but two")
    p.add_argument("--min-latin", type=float, default=0.06)
    p.add_argument("--push", default=None, help="dataset repo to upload the corpus to")
    p.add_argument("--push-parts", action="store_true", default=None,
                   help="upload fingerprints after the map stage so a timed-out "
                        "job can resume (default: on when --push is set)")
    p.add_argument("--no-push-parts", dest="push_parts", action="store_false",
                   help="skip the resume upload")
    p.add_argument("--map-only", action="store_true")
    p.add_argument("--keep-parts", action="store_true",
                   help="keep intermediate parts (default: freed as the reduce consumes them)")
    p.add_argument("--keep-cache", action="store_true",
                   help="keep downloaded parquet shards (default: delete once processed)")
    a = p.parse_args()

    install_claudius()

    from claudius.cli import _sources, _spread
    from claudius.filters import FilterConfig
    from claudius.parallel import run_map, run_reduce
    from claudius.prepare import push

    sources = _spread(_sources(a.source), a.shards)
    cfg = FilterConfig(min_latin=a.min_latin)

    run_map(sources, a.out, cfg, workers=a.workers, drop_cache=not a.keep_cache)

    # Job storage is ephemeral: when the container exits, parts/ and every DONE
    # marker go with it, so the in-container resume logic cannot survive a
    # timeout. Uploading the fingerprints after the map stage is what makes a
    # relaunch cheap -- without it, a timeout throws away every CPU-hour spent.
    if a.push_parts is None:
        a.push_parts = bool(a.push)
    if a.push_parts and a.push:
        print("[push] uploading parts so a relaunch can resume", flush=True)
        push(a.out, a.push + "-parts", private=True, include_parts=True)
    if a.map_only:
        return 0

    stats = run_reduce(a.out, drop_parts=not a.keep_parts)
    print(json.dumps(stats.as_dict(), indent=2), flush=True)
    if a.push:
        print("[push] uploading corpus to", a.push, flush=True)
        push(a.out, a.push, private=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
