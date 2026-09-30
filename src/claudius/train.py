"""Pretraining. Same recipe at every size.

Hyperparameters are the ones decided in the brief: AdamW with beta2 = 0.95,
weight decay 0.1, peak lr 3e-4 on a cosine schedule with ~1% warmup, 0.5M tokens
per optimizer step, bf16, gradient checkpointing at 1B.

Nothing here loads pretrained weights. The model is initialised from scratch and
has only ever seen Latin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .model import SEQ_LEN, SIZES, param_count, training_flops
from .pipeline import load_packed
from .tokenizer import load as load_tokenizer

#: Tokens per optimizer step. Kept constant across sizes so the three runs are
#: comparable and the scaling curve means something.
TOKENS_PER_STEP = 512 * 1024


@dataclass
class TrainConfig:
    size: str = "400m"
    data_dir: str = "out/packed"
    tokenizer_dir: str = "out/tokenizer"
    out_dir: str = "out/model"
    seq_len: int = SEQ_LEN
    epochs: float = 1.0
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_ratio: float = 0.01
    per_device_batch: int = 8
    grad_accum: int = 0          # 0 = derive from TOKENS_PER_STEP
    max_steps: int = -1
    #: Each save also PUSHES a full checkpoint to the Hub -- weights plus AdamW
    #: state, 4.1 GB for the 400M. At every 800 steps that is 70 GB over a run.
    #: 2000 still bounds a crash to ~5 GPU-hours of lost work.
    save_steps: int = 2000
    eval_steps: int = 2000
    #: Sequences used for the periodic loss check. The packed validation split
    #: is 427M tokens -- 208,575 sequences -- and evaluating all of it takes ~45
    #: minutes at batch 24. At the default cadence that was 27 evaluations, 20
    #: GPU-hours, on a 37-hour run: a 55% cost overrun for a number that a few
    #: hundred sequences estimates just as well.
    eval_sequences: int = 512
    log_steps: int = 10
    seed: int = 1
    bf16: bool = True
    gradient_checkpointing: bool | None = None  # None = on for 1b
    hub_repo: str | None = None
    private: bool = True
    trackio: bool = False
    run_name: str = "claudius"
    resume: bool = False
    #: Continue from existing weights instead of initialising from scratch.
    #: This is how the annealing phase works: take the finished base model, give
    #: it a small decaying learning rate, and finish on the classical subset.
    #: The last percent of training sets register disproportionately, which is
    #: the only lever that works when clean classical Latin is 21M words against
    #: a 15B-word corpus and cannot be upweighted into dominance.
    init_from: str | None = None


def derive_schedule(cfg: TrainConfig, n_sequences: int, world_size: int = 1) -> dict:
    """Work out accumulation and step count before spending GPU hours."""
    tokens_per_micro = cfg.per_device_batch * cfg.seq_len * world_size
    accum = cfg.grad_accum or max(1, TOKENS_PER_STEP // tokens_per_micro)
    tokens_per_step = tokens_per_micro * accum
    epoch_tokens = int(n_sequences * cfg.seq_len * cfg.epochs)
    steps = cfg.max_steps if cfg.max_steps > 0 else max(1, epoch_tokens // tokens_per_step)
    # What the run actually trains on. With max_steps set this is far less than
    # an epoch, and budgeting the epoch instead overstates the cost -- by 34x on
    # a 300-step smoke run. `plan` is the tool used to decide whether to spend
    # hundreds of dollars, so it reports the tokens that will really be seen.
    trained_tokens = steps * tokens_per_step
    return {
        "grad_accum": accum,
        "tokens_per_step": tokens_per_step,
        "epoch_tokens": epoch_tokens,
        "trained_tokens": trained_tokens,
        "epochs_covered": round(trained_tokens / max(epoch_tokens, 1), 4),
        "steps": steps,
        "warmup_steps": max(1, int(steps * cfg.warmup_ratio)),
    }


#: Peak bf16 throughput, TFLOP/s.
PEAK_TFLOPS = {"a10g": 125e12, "l4": 121e12, "a100": 312e12, "h200": 990e12}

#: Measured, not assumed. A 110M model at batch 8 x 2048, grad_accum 32, on
#: a10g-small: 16.4 s/step at 524,288 tokens/step = ~32k tokens/s, which is
#: 6 * 110e6 * 32e3 = 2.1e13 FLOP/s against the A10G's 125 TFLOP/s peak.
#: Textbook estimates of 40-50% do not survive contact with a small model on a
#: single mid-range GPU, and budgeting at 40% understates cost by 2.3x.
MEASURED_MFU = {"a10g-small-150m": 0.17}
DEFAULT_MFU = 0.17


def plan(cfg: TrainConfig, n_sequences: int, vocab_size: int, world_size: int = 1,
         mfu: float = DEFAULT_MFU, gpu: str = "a100") -> dict:
    """A dry-run budget: parameters, tokens, FLOPs, GPU-hours.

    `mfu` defaults to the value measured on the pilot rather than a textbook
    40%. This is the tool used to decide whether to spend hundreds of dollars,
    so it should be pessimistic by default and corrected upward by measurement,
    not the reverse.
    """
    sched = derive_schedule(cfg, n_sequences, world_size)
    params = param_count(SIZES[cfg.size], vocab_size)
    flops = training_flops(params, sched["trained_tokens"])
    peak = PEAK_TFLOPS.get(gpu, PEAK_TFLOPS["a100"])
    hours = flops / (mfu * peak) / 3600
    return {
        **sched,
        "size": cfg.size,
        "params": params,
        "tokens_per_param": sched["trained_tokens"] / params,
        "flops": flops,
        "gpu": gpu,
        "assumed_mfu": mfu,
        "gpu_hours": round(hours, 1),
        "gpu_hours_at_40pct_mfu": round(flops / (0.40 * peak) / 3600, 1),
    }


def restore_hub_checkpoint(hub_repo: str, out_dir: str | Path, token: str | None = None) -> int | None:
    """Pull `last-checkpoint/` from the Hub into `out_dir/checkpoint-<step>`.

    The Trainer's `resume_from_checkpoint=True` looks for checkpoint-* folders
    LOCALLY. A fresh job container has none, so without this a relaunch with
    --resume raises "no valid checkpoint" and a 37-hour run restarts from zero.
    Returns the restored step, or None if the Hub has no checkpoint.
    """
    import shutil

    from huggingface_hub import snapshot_download

    pulled = snapshot_download(hub_repo, allow_patterns=["last-checkpoint/*"],
                               local_dir=str(Path(out_dir) / "_hub_pull"), token=token)
    src = Path(pulled) / "last-checkpoint"
    state = src / "trainer_state.json"
    if not state.exists():
        return None
    step = json.loads(state.read_text())["global_step"]
    dst = Path(out_dir) / f"checkpoint-{step}"
    if not dst.exists():
        shutil.move(str(src), str(dst))
    shutil.rmtree(Path(out_dir) / "_hub_pull", ignore_errors=True)
    print(f"[resume] restored checkpoint-{step} from {hub_repo}", flush=True)
    return step


def _collate(features):
    """Causal-LM batch. Uses a dataset's own `labels` column when it has one.

    Pretraining data has only input_ids, so every token is a target. The tutor
    fine-tuning set carries labels with -100 on student turns: those turns
    contain deliberately corrupted Latin, and training on them would teach the
    tutor to write the errors it is meant to catch.
    """
    import torch

    ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    if "labels" in features[0]:
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    else:
        labels = ids.clone()
    return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}


def run(cfg: TrainConfig):
    import torch
    from transformers import Trainer, TrainingArguments

    from .model import build_model

    tok = load_tokenizer(cfg.tokenizer_dir)
    train_ds = load_packed(cfg.data_dir, "train")
    try:
        eval_ds = load_packed(cfg.data_dir, "val")
        if cfg.eval_sequences and len(eval_ds) > cfg.eval_sequences:
            eval_ds = eval_ds.select(range(cfg.eval_sequences))
            print(f"[eval] using {cfg.eval_sequences} of the validation sequences "
                  f"for the periodic loss check", flush=True)
    except SystemExit:
        eval_ds = None

    world = max(1, torch.cuda.device_count())
    sched = derive_schedule(cfg, len(train_ds), world)
    budget = plan(cfg, len(train_ds), len(tok), world)
    print("[plan] " + json.dumps(budget, indent=2), flush=True)

    if cfg.init_from:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(cfg.init_from)
        if model.config.vocab_size != len(tok):
            raise SystemExit(
                f"tokenizer mismatch: checkpoint has vocab {model.config.vocab_size}, "
                f"tokenizer has {len(tok)}. Annealing must use the base model's tokenizer."
            )
        model.config.use_cache = False
        print(f"[init] continuing from {cfg.init_from}", flush=True)
    else:
        model = build_model(cfg.size, len(tok), cfg.seq_len, eos_id=tok.eos_token_id)
    grad_ckpt = cfg.gradient_checkpointing
    if grad_ckpt is None:
        grad_ckpt = cfg.size == "1b"

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=sched["steps"],
        per_device_train_batch_size=cfg.per_device_batch,
        per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=sched["grad_accum"],
        learning_rate=cfg.lr,
        # Cosine to a FLOOR of min_lr_ratio x peak, as Chinchilla did. This field
        # existed for the first runs but was never wired in: they used plain
        # "cosine", which decays to zero (the 150M ended at 2.7e-9). Both are
        # standard, but a config that silently ignores a declared setting is
        # worse than either.
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": cfg.min_lr_ratio},
        warmup_steps=sched["warmup_steps"],
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.beta1,
        adam_beta2=cfg.beta2,
        max_grad_norm=cfg.grad_clip,
        bf16=cfg.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=grad_ckpt,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.eval_steps,
        seed=cfg.seed,
        dataloader_num_workers=4,
        report_to=["trackio"] if cfg.trackio else [],
        run_name=cfg.run_name,
        push_to_hub=bool(cfg.hub_repo),
        hub_model_id=cfg.hub_repo,
        hub_private_repo=cfg.private,
        hub_strategy="checkpoint",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate,
        processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=cfg.resume or None)

    final = out_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    (final / "training_plan.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
    if cfg.hub_repo:
        trainer.push_to_hub(commit_message=f"claudius-{cfg.size} final")
    return str(final)


def loss_to_bits_per_byte(nats_per_token: float, tokens_per_word: float, chars_per_word: float = 6.2) -> float:
    """Convert a training loss into bits-per-byte for a quick sanity check.

    Only approximate -- `claudius eval` measures it properly -- but useful while
    a run is in flight to see whether it is anywhere near the bigram floor.
    """
    return nats_per_token / math.log(2) * tokens_per_word / chars_per_word
