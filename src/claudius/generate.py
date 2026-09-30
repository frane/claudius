"""Generation settings and output clean-up.

The base model loops ("QUID SIBI UULT? QUID SIBI UULT?") and fuses words after
punctuation ("RABIE,IN"). Neither needs retraining to fix at the point of use:
looping is a decoding problem (greedy-ish sampling keeps re-choosing the most
likely continuation), and missing spaces after punctuation are restored by the
same rule the normaliser applies to input.
"""

from __future__ import annotations

import re

from .normalize import _MISSING_SPACE

#: Decoding that suppresses loops. Measured before/after in docs/decisions.md.
ANTI_LOOP = dict(
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
)
#: What evaluate.sample_incipits used before: sampling, no loop suppression.
PLAIN = dict(do_sample=True, temperature=0.8, top_p=0.95)

_WORD = re.compile(r"[A-Z]+")


def clean_output(text: str) -> str:
    """Restore the space the model drops after punctuation."""
    return _MISSING_SPACE.sub(r"\1 ", text)


def roman(text: str) -> str:
    """Display form: U is written V everywhere (SALVE, VERBVM, VVLT).

    Display only. Normalised text and the tokenizer keep U; on normalised text
    U <-> V is a 1:1 swap, so this needs no retraining and loses nothing.
    """
    return text.replace("U", "V")


def repeat_rate(text: str, n: int = 3) -> float:
    """Fraction of word n-grams that already occurred earlier in the text.

    0.0 means no phrase repeats; a looping continuation scores high.
    """
    words = _WORD.findall(text)
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams)


def fused_rate(text: str) -> float:
    """Punctuation directly followed by a letter, per 100 words."""
    words = len(_WORD.findall(text)) or 1
    return 100 * len(re.findall(r"[,;:!?.][A-Z]", text)) / words


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 80,
             settings: dict | None = None, clean: bool = True) -> str:
    import torch

    from .normalize import normalize

    prompt = normalize(prompt)
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id,
                             **(settings or ANTI_LOOP))
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return clean_output(text) if clean else text
