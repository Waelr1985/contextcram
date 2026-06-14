"""Token counting and truncation backends for contextpacker.

The packer only needs two operations from a tokenizer: count the tokens in a
string, and truncate a string to at most N tokens (keeping the head). Anything
implementing the :class:`Tokenizer` protocol works, so you can plug in tiktoken,
a Hugging Face tokenizer, or the zero-dependency heuristic shipped here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal interface the packer needs from a tokenizer."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return a prefix of ``text`` that is at most ``max_tokens`` tokens."""
        ...


class HeuristicTokenizer:
    """Dependency-free token estimator based on characters-per-token.

    Defaults to 4 characters per token, the common rule of thumb for English
    text with GPT/Claude-style BPE tokenizers. It is good enough for budgeting;
    swap in a real tokenizer when you need exact counts.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        max_chars = int(max_tokens * self.chars_per_token)
        return text[:max_chars]


class CallableTokenizer:
    """Adapt a plain ``count`` function into a full :class:`Tokenizer`.

    Handy for wrapping ``lambda s: len(enc.encode(s))`` from tiktoken or any
    other library. Truncation starts from a proportional character estimate and
    then shrinks until the count fits, so it stays exact regardless of the
    underlying tokenizer.
    """

    def __init__(self, count_fn: Callable[[str], int]) -> None:
        self._count = count_fn

    def count(self, text: str) -> int:
        return self._count(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        total = self._count(text)
        if total <= max_tokens:
            return text
        # Estimate a starting cut from average char/token density, then back off
        # until we are within budget.
        approx_chars = max(1, int(len(text) * max_tokens / max(1, total)))
        cut = text[:approx_chars]
        while cut and self._count(cut) > max_tokens:
            cut = cut[: max(0, int(len(cut) * 0.9))]
        return cut


def tiktoken_tokenizer(model: str = "gpt-4o") -> CallableTokenizer:
    """Build a :class:`CallableTokenizer` backed by tiktoken.

    Requires the optional ``tiktoken`` dependency (``pip install
    contextpacker[tiktoken]``). Falls back to the ``cl100k_base`` encoding for
    unknown models.
    """
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return CallableTokenizer(lambda text: len(enc.encode(text)))
