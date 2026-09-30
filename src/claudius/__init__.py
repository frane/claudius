"""Claudius -- a Latin-only LLM pretrained from scratch.

The model sees the 23-letter classical alphabet and nothing else: no English, no
modern orthography, no pretrained weights. See README.md for the recipe and
docs/decisions.md for why each choice was made.
"""

__version__ = "0.1.0"

from .normalize import normalize  # noqa: F401
