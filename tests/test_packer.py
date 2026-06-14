"""Tests for contextcram.Packer."""

from __future__ import annotations

import pytest

from contextcram import BudgetExceeded, HeuristicTokenizer, Packer, Strategy


def tok(chars_per_token: float = 1.0) -> HeuristicTokenizer:
    # 1 char == 1 token makes budgets easy to reason about in tests.
    return HeuristicTokenizer(chars_per_token=chars_per_token)


def test_everything_fits_preserves_insertion_order():
    p = Packer(budget=100, tokenizer=tok())
    p.add("aaa", priority="low", name="a")
    p.add("bbb", priority="high", name="b")
    result = p.fit()
    # Output is in insertion order, not priority order.
    assert result.text == "aaa\n\nbbb"
    assert result.used_tokens == 6
    assert result.remaining == 94
    assert result.dropped == []


def test_chained_add_returns_self():
    p = Packer(budget=100, tokenizer=tok())
    assert p.add("x").add("y") is p


def test_required_always_kept_optional_dropped_when_over_budget():
    p = Packer(budget=5, tokenizer=tok())
    p.add("SYS..", priority="required", name="sys")  # 5 tokens, fills budget
    p.add("docs", priority="low", strategy="drop", name="docs")
    result = p.fit()
    assert "SYS.." in result.text
    assert result.dropped_names == ["docs"]
    assert result.used_tokens == 5


def test_required_over_budget_raises():
    p = Packer(budget=3, tokenizer=tok())
    p.add("toolong", priority="required")
    with pytest.raises(BudgetExceeded):
        p.fit()


def test_priority_order_decides_who_gets_dropped():
    p = Packer(budget=4, tokenizer=tok())
    p.add("LLLL", priority="low", strategy="drop", name="low")
    p.add("HHHH", priority="high", strategy="drop", name="high")
    result = p.fit()
    # Only one 4-token item fits; the higher priority one wins.
    assert "HHHH" in result.text
    assert result.dropped_names == ["low"]


def test_truncate_keeps_head():
    p = Packer(budget=3, tokenizer=tok())
    p.add("abcdef", priority="medium", strategy="truncate")
    result = p.fit()
    assert result.text == "abc"
    assert result.items[0].truncated is True
    assert result.used_tokens == 3


def test_truncate_head_keeps_tail():
    p = Packer(budget=3, tokenizer=tok())
    p.add("abcdef", priority="medium", strategy=Strategy.TRUNCATE_HEAD)
    result = p.fit()
    assert result.text == "def"
    assert result.items[0].truncated is True


def test_trim_drops_oldest_segments():
    p = Packer(budget=7, tokenizer=tok())
    # sep="\n" -> joining "msg2"+"\n"+"msg3" = 9 tokens; only the newest fits.
    p.add(["msg1", "msg2", "msg3"], priority="high", strategy="trim", sep="\n")
    result = p.fit()
    assert result.text == "msg3"
    assert result.items[0].truncated is True


def test_integer_priorities():
    p = Packer(budget=4, tokenizer=tok())
    p.add("AAAA", priority=1, strategy="drop", name="a")
    p.add("BBBB", priority=99, strategy="drop", name="b")
    result = p.fit()
    assert "BBBB" in result.text
    assert result.dropped_names == ["a"]


def test_unknown_priority_and_strategy_raise():
    p = Packer(budget=10, tokenizer=tok())
    with pytest.raises(ValueError):
        p.add("x", priority="urgent")
    with pytest.raises(ValueError):
        p.add("x", strategy="nuke")


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        Packer(budget=-1)


def test_str_returns_packed_text():
    p = Packer(budget=100, tokenizer=tok())
    p.add("hello")
    assert str(p.fit()) == "hello"


def test_default_heuristic_tokenizer_used():
    # No tokenizer passed: defaults to chars/4. "12345678" -> 2 tokens.
    p = Packer(budget=100)
    result = p.add("12345678").fit()
    assert result.used_tokens == 2
