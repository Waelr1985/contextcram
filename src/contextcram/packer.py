"""Priority-aware token-budget packing.

:class:`Packer` assembles a set of text items into a single context that fits
within a token budget. Required items are always kept; optional items are fitted
in priority order, and items that do not fit are truncated, trimmed, or dropped
according to their per-item strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .tokenizers import HeuristicTokenizer, Tokenizer

Content = str | list[str]


class BudgetExceeded(Exception):  # noqa: N818  (reads better than ...Error in `except`)
    """Raised when required items alone do not fit within the budget."""


class Strategy(str, Enum):
    """What to do with an optional item that does not fully fit."""

    DROP = "drop"  # include the item whole, or not at all
    TRUNCATE = "truncate"  # cut from the end, keeping the head
    TRUNCATE_HEAD = "truncate_head"  # cut from the start, keeping the tail
    TRIM = "trim"  # for list content: drop oldest segments first


#: Named priority levels. Higher numbers are kept first. ``"required"`` is a
#: sentinel that flags an item as never-droppable (see :meth:`Packer.add`).
PRIORITY_LEVELS = {
    "required": 1_000_000,
    "high": 30,
    "medium": 20,
    "low": 10,
}


@dataclass
class Item:
    """An item queued for packing."""

    content: Content
    priority: int
    required: bool
    strategy: Strategy
    sep: str
    name: str | None
    order: int


@dataclass
class FittedItem:
    """An item as it ended up in the packed result."""

    item: Item
    text: str
    tokens: int
    truncated: bool

    @property
    def name(self) -> str | None:
        return self.item.name


@dataclass
class PackResult:
    """The outcome of :meth:`Packer.fit`."""

    text: str
    items: list[FittedItem]
    used_tokens: int
    budget: int
    dropped: list[Item] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.budget - self.used_tokens

    @property
    def dropped_names(self) -> list[str | None]:
        return [it.name for it in self.dropped]

    def __str__(self) -> str:
        return self.text


class Packer:
    """Pack prioritized text items into a token budget.

    Example::

        from contextcram import Packer

        packer = Packer(budget=8000)
        packer.add(system_prompt, priority="required")
        packer.add(history, priority="high", strategy="trim")
        packer.add(docs, priority="medium", strategy="drop")
        result = packer.fit()
        print(result.text, result.used_tokens, result.dropped_names)
    """

    def __init__(
        self,
        budget: int,
        tokenizer: Tokenizer | None = None,
        joiner: str = "\n\n",
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = budget
        self.tokenizer: Tokenizer = tokenizer or HeuristicTokenizer()
        self.joiner = joiner
        self._items: list[Item] = []
        self._counter = 0

    def add(
        self,
        content: Content,
        *,
        priority: str | int = "medium",
        strategy: str | Strategy = "truncate",
        sep: str = "\n",
        name: str | None = None,
    ) -> Packer:
        """Queue an item. Returns ``self`` so calls can be chained.

        ``content`` is a string, or a list of strings (segments) for ``trim``.
        ``priority`` is ``"required"``, ``"high"``, ``"medium"``, ``"low"`` or an
        integer (higher is kept first). ``strategy`` decides what happens when the
        item does not fully fit: ``"drop"``, ``"truncate"``, ``"truncate_head"``
        or ``"trim"``.
        """
        required = False
        if isinstance(priority, str):
            if priority not in PRIORITY_LEVELS:
                raise ValueError(
                    f"unknown priority {priority!r}; use one of {sorted(PRIORITY_LEVELS)} or an int"
                )
            required = priority == "required"
            resolved_priority = PRIORITY_LEVELS[priority]
        else:
            resolved_priority = int(priority)

        strat = strategy if isinstance(strategy, Strategy) else Strategy(strategy)

        self._items.append(
            Item(
                content=content,
                priority=resolved_priority,
                required=required,
                strategy=strat,
                sep=sep,
                name=name,
                order=self._counter,
            )
        )
        self._counter += 1
        return self

    def fit(self) -> PackResult:
        """Pack the queued items and return the assembled :class:`PackResult`."""
        tok = self.tokenizer
        fitted: dict[int, FittedItem] = {}
        dropped: list[Item] = []

        required = [it for it in self._items if it.required]
        optional = [it for it in self._items if not it.required]

        used = 0
        for it in required:
            text = self._render(it)
            n = tok.count(text)
            used += n
            fitted[it.order] = FittedItem(it, text, n, truncated=False)

        if used > self.budget:
            raise BudgetExceeded(f"required items need {used} tokens but budget is {self.budget}")

        remaining = self.budget - used

        # Higher priority first; ties keep insertion order for determinism.
        for it in sorted(optional, key=lambda x: (-x.priority, x.order)):
            text = self._render(it)
            n = tok.count(text)
            if n <= remaining:
                fitted[it.order] = FittedItem(it, text, n, truncated=False)
                remaining -= n
                continue

            if it.strategy is Strategy.DROP or remaining <= 0:
                dropped.append(it)
                continue

            if it.strategy is Strategy.TRIM and isinstance(it.content, list):
                trimmed = self._trim_segments(it, remaining)
            elif it.strategy is Strategy.TRUNCATE_HEAD:
                trimmed = self._truncate_head(text, remaining)
            else:  # TRUNCATE, or TRIM on non-list content
                trimmed = tok.truncate(text, remaining)

            ntok = tok.count(trimmed)
            if trimmed and ntok > 0:
                fitted[it.order] = FittedItem(it, trimmed, ntok, truncated=True)
                remaining -= ntok
            else:
                dropped.append(it)

        ordered = [fitted[o] for o in sorted(fitted)]
        text = self.joiner.join(f.text for f in ordered)
        return PackResult(
            text=text,
            items=ordered,
            used_tokens=self.budget - remaining,
            budget=self.budget,
            dropped=dropped,
        )

    # -- internals ---------------------------------------------------------

    def _render(self, item: Item) -> str:
        if isinstance(item.content, list):
            return item.sep.join(item.content)
        return item.content

    def _trim_segments(self, item: Item, max_tokens: int) -> str:
        """Keep the most recent segments (end of the list) that fit."""
        segments = item.content if isinstance(item.content, list) else [item.content]
        kept: list[str] = []
        for seg in reversed(segments):
            candidate = item.sep.join([seg, *kept]) if kept else seg
            if self.tokenizer.count(candidate) > max_tokens:
                break
            kept.insert(0, seg)
        return item.sep.join(kept)

    def _truncate_head(self, text: str, max_tokens: int) -> str:
        """Keep as much of the END of ``text`` as fits (binary search)."""
        tok = self.tokenizer
        if tok.count(text) <= max_tokens:
            return text
        lo, hi, best = 0, len(text), ""
        while lo < hi:
            mid = (lo + hi) // 2
            suffix = text[mid:]
            if tok.count(suffix) <= max_tokens:
                best = suffix
                hi = mid
            else:
                lo = mid + 1
        return best
