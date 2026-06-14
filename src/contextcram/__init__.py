"""contextcram — fit anything into an LLM context window.

A tiny, zero-dependency, priority-aware token-budget packer.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .packer import (
    BudgetExceeded,
    FittedItem,
    Item,
    Packer,
    PackResult,
    Strategy,
)
from .tokenizers import (
    CallableTokenizer,
    HeuristicTokenizer,
    Tokenizer,
    tiktoken_tokenizer,
)

try:
    __version__ = version("contextcram")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "Packer",
    "PackResult",
    "FittedItem",
    "Item",
    "Strategy",
    "BudgetExceeded",
    "Tokenizer",
    "HeuristicTokenizer",
    "CallableTokenizer",
    "tiktoken_tokenizer",
    "__version__",
]
