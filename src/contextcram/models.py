"""Model context-window registry.

A small, best-effort lookup of total context-window sizes (in tokens) for common
models, so you can write ``Packer(model="gpt-4o")`` instead of hard-coding a
number. Values are approximate and may drift as providers change them — pass an
explicit ``budget=`` (or call :func:`register_model`) whenever you need an exact
or newer figure.
"""

from __future__ import annotations

#: Approximate total context-window sizes (tokens), keyed by normalized name.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic Claude
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # Google Gemini
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
}


def _normalize(name: str) -> str:
    return name.strip().lower()


def register_model(name: str, context_window: int) -> None:
    """Add or override a model's context-window size in the registry."""
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    MODEL_CONTEXT_WINDOWS[_normalize(name)] = context_window


def context_window_for(model: str) -> int:
    """Return the registered context-window size for ``model``.

    Raises ``ValueError`` for unknown models, pointing at the explicit-budget
    and :func:`register_model` escape hatches.
    """
    try:
        return MODEL_CONTEXT_WINDOWS[_normalize(model)]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; pass budget=<int> explicitly or register "
            f"it first with register_model({model!r}, <tokens>)"
        ) from None
