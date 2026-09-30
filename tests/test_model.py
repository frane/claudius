import pathlib
import pytest

from claudius.model import SIZES, param_count, training_flops


@pytest.mark.parametrize(
    "size,expected",
    [("150m", 109_529_856), ("400m", 341_099_520), ("1b", 1_034_512_384)],
)
def test_param_counts_match_the_brief(size, expected):
    assert param_count(SIZES[size], 32_000) == expected


def test_untied_embeddings_cost_one_more_matrix():
    s = SIZES["400m"]
    assert param_count(s, 32_000, tied_embeddings=False) - param_count(s, 32_000) == 32_000 * s.hidden


def test_head_dim_divides():
    for s in SIZES.values():
        assert s.hidden % s.heads == 0
        assert s.heads % s.kv_heads == 0


def test_sizes_are_ordered():
    counts = [param_count(SIZES[n], 32_000) for n in ("test", "150m", "400m", "1b")]
    assert counts == sorted(counts)


def test_flops_budget_matches_the_brief():
    # 1B params on 10B tokens, the number the GPU budget was built on.
    flops = training_flops(1_034_512_384, 10_000_000_000)
    assert 5.5e19 < flops < 6.5e19


def test_plan_budgets_what_the_run_actually_trains():
    """max_steps caps the run; the budget must not quote a full epoch."""
    from claudius.train import TrainConfig, plan

    seqs = 330_622  # the real packed clean corpus
    full = plan(TrainConfig(size="150m", per_device_batch=4, grad_accum=8), seqs, 32_768)
    capped = plan(TrainConfig(size="150m", per_device_batch=4, grad_accum=8,
                              max_steps=300), seqs, 32_768)

    assert capped["trained_tokens"] == 300 * capped["tokens_per_step"]
    assert capped["trained_tokens"] < full["trained_tokens"] / 10
    assert capped["flops"] < full["flops"] / 10
    assert capped["epochs_covered"] < 0.05
    assert 0.99 <= full["epochs_covered"] <= 1.0


def test_plan_1b_budget_is_unchanged():
    """The headline number the GPU spend rests on."""
    from claudius.train import TrainConfig, plan

    seqs = 20_000_000_000 // 2048
    p = plan(TrainConfig(size="1b"), seqs, 32_000)
    assert 1.2e20 < p["flops"] < 1.3e20
    assert 0.99 <= p["epochs_covered"] <= 1.0


def test_plan_defaults_to_measured_mfu_not_textbook():
    """A budgeting tool should be pessimistic by default.

    40% MFU is the textbook figure; 17% is what a 110M model actually achieved
    on an A10G. Budgeting the 1B at 40% understates it by 2.35x -- $436 against
    $1,026 on an H200.
    """
    from claudius.train import DEFAULT_MFU, TrainConfig, plan

    assert DEFAULT_MFU == 0.17
    seqs = 20_000_000_000 // 2048
    p = plan(TrainConfig(size="1b"), seqs, 32_768, gpu="h200")
    assert p["assumed_mfu"] == 0.17
    assert 200 < p["gpu_hours"] < 210
    assert 85 < p["gpu_hours_at_40pct_mfu"] < 90
    assert p["gpu_hours"] > p["gpu_hours_at_40pct_mfu"] * 2


def test_plan_gpu_choice_changes_the_budget():
    from claudius.train import TrainConfig, plan

    seqs = 20_000_000_000 // 2048
    cfg = TrainConfig(size="1b")
    h200 = plan(cfg, seqs, 32_768, gpu="h200")["gpu_hours"]
    a100 = plan(cfg, seqs, 32_768, gpu="a100")["gpu_hours"]
    assert a100 > h200 * 2.5, "an H200 is much faster than an A100"


def test_min_lr_ratio_is_actually_used():
    """The field was declared but never wired in; runs decayed to zero.

    The 150M ended at learning rate 2.7e-9 despite min_lr_ratio = 0.1.
    """
    import inspect

    from claudius import train

    src = inspect.getsource(train.run)
    assert "cosine_with_min_lr" in src
    assert "cfg.min_lr_ratio" in src


def test_scheduler_floors_at_min_lr_ratio():
    import torch
    from transformers import get_scheduler

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
    sch = get_scheduler("cosine_with_min_lr", opt, num_warmup_steps=10,
                        num_training_steps=500,
                        scheduler_specific_kwargs={"min_lr_rate": 0.1})
    for _ in range(500):
        opt.step()
        sch.step()
    assert abs(opt.param_groups[0]["lr"] - 3e-5) < 1e-7


def test_validation_split_is_capped_when_packing():
    """An uncapped val split is invisible until it bills.

    The 8B mix produced 427M validation tokens; a run evaluating all of it spent
    45 minutes per check, 20 GPU-hours over the run.
    """
    import inspect

    from claudius import pipeline

    src = inspect.getsource(pipeline.stage_pack)
    assert "max_val_tokens" in src
    assert 'budget = max_tokens if split == "train" else max_val_tokens' in src
    assert inspect.signature(pipeline.stage_pack).parameters["max_val_tokens"].default == 20_000_000


def test_checkpoint_pushes_are_not_every_800_steps():
    """Each save pushes weights AND optimizer state: 4.1 GB for the 400M."""
    from claudius.train import TrainConfig

    assert TrainConfig().save_steps >= 2000


def test_eval_is_bounded_in_both_size_and_cadence():
    from claudius.train import TrainConfig

    cfg = TrainConfig()
    assert cfg.eval_sequences == 512
    assert cfg.eval_steps >= 2000


def test_restore_hub_checkpoint_places_it_where_the_trainer_looks(tmp_path, monkeypatch):
    """The Trainer resumes from out_dir/checkpoint-<step>; the Hub has last-checkpoint/."""
    import json

    import huggingface_hub

    from claudius.train import restore_hub_checkpoint

    def fake_snapshot(repo, allow_patterns=None, local_dir=None, token=None):
        d = pathlib.Path(local_dir) / "last-checkpoint"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text(json.dumps({"global_step": 4800}))
        (d / "model.safetensors").write_bytes(b"weights")
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    step = restore_hub_checkpoint("fbandov/claudius-400m", tmp_path)
    assert step == 4800
    assert (tmp_path / "checkpoint-4800" / "model.safetensors").read_bytes() == b"weights"
    assert not (tmp_path / "_hub_pull").exists()


def test_restore_hub_checkpoint_reports_nothing_to_restore(tmp_path, monkeypatch):
    import huggingface_hub

    from claudius.train import restore_hub_checkpoint

    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, allow_patterns=None, local_dir=None, token=None:
                        (pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True) or local_dir))
    assert restore_hub_checkpoint("fbandov/nothing", tmp_path) is None
