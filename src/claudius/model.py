"""Model shapes and construction.

Three dense models on one recipe, one tokenizer, one data mix:

* 150M -- drills and the browser (WebGPU);
* 400M -- the on-device default, ~250 MB at 4-bit;
* 1B    -- the flagship and the evaluation reference.

Same recipe across all three so the three points form a scaling curve worth
publishing. MoE was rejected: it cuts compute, not memory, and memory is the
on-device constraint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SEQ_LEN = 2048


@dataclass(frozen=True)
class Shape:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    inter: int

    @property
    def head_dim(self) -> int:
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        return self.hidden // self.heads


#: TinyLlama shape at 1B; the smaller two keep the same aspect ratio family.
SIZES: dict[str, Shape] = {
    "test": Shape(hidden=128, layers=2, heads=2, kv_heads=2, inter=384),
    "150m": Shape(hidden=768, layers=12, heads=12, kv_heads=12, inter=2048),
    "400m": Shape(hidden=1024, layers=24, heads=16, kv_heads=16, inter=2816),
    "1b": Shape(hidden=2048, layers=22, heads=32, kv_heads=4, inter=5632),
}


def param_count(shape: Shape, vocab_size: int, tied_embeddings: bool = True) -> int:
    """Exact parameter count for the Llama architecture we instantiate.

    Analytic so it can be checked without torch installed -- useful when sizing
    a run before paying for the GPU.
    """
    h, kv = shape.hidden, shape.kv_heads * shape.head_dim
    attn = h * h + h * kv + h * kv + h * h
    mlp = 3 * h * shape.inter          # gate, up, down
    norms = 2 * h                       # input + post-attention RMSNorm
    per_layer = attn + mlp + norms
    embeddings = vocab_size * h * (1 if tied_embeddings else 2)
    return shape.layers * per_layer + embeddings + h  # + final norm


def training_flops(n_params: int, n_tokens: int) -> float:
    """The usual 6ND estimate, for budgeting GPU hours."""
    return 6.0 * n_params * n_tokens


def describe(vocab_size: int = 32_000) -> str:
    rows = [f"{'size':>6}  {'params':>12}  {'hidden':>6}  {'layers':>6}  {'heads':>5}  {'kv':>3}  {'inter':>5}"]
    for name, s in SIZES.items():
        rows.append(
            f"{name:>6}  {param_count(s, vocab_size):>12,}  {s.hidden:>6}  {s.layers:>6}  "
            f"{s.heads:>5}  {s.kv_heads:>3}  {s.inter:>5}"
        )
    return "\n".join(rows)


def build_config(size: str, vocab_size: int, seq_len: int = SEQ_LEN):
    from transformers import LlamaConfig

    s = SIZES[size]
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=s.hidden,
        num_hidden_layers=s.layers,
        num_attention_heads=s.heads,
        num_key_value_heads=s.kv_heads,
        intermediate_size=s.inter,
        max_position_embeddings=seq_len,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
    )


def build_model(size: str, vocab_size: int, seq_len: int = SEQ_LEN, eos_id: int = 0):
    """A randomly initialised Llama. Nothing is loaded from any pretrained model."""
    from transformers import LlamaForCausalLM

    cfg = build_config(size, vocab_size, seq_len)
    cfg.bos_token_id = eos_id
    cfg.eos_token_id = eos_id
    cfg.pad_token_id = eos_id
    model = LlamaForCausalLM(cfg)
    model.config.use_cache = False
    return model


def shape_dict() -> dict:
    return {name: asdict(s) for name, s in SIZES.items()}
