"""Packing documents into fixed-length training sequences."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from .model import SEQ_LEN


def pack(
    token_lists: Iterable[Sequence[int]], eos_id: int, seq_len: int = SEQ_LEN
) -> Iterator[list[int]]:
    """Concatenate documents with EOS between them and cut into fixed blocks.

    The tail shorter than `seq_len` is dropped rather than padded: padding wastes
    a fraction of every step for no gain when there is more data than compute.
    """
    buf: list[int] = []
    for ids in token_lists:
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            del buf[:seq_len]


def count_tokens(token_lists: Iterable[Sequence[int]]) -> int:
    return sum(len(x) for x in token_lists)
