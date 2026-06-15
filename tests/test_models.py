"""Tests for model-aware budgets and token reservation."""

from __future__ import annotations

import pytest

from contextcram import (
    BudgetExceeded,
    HeuristicTokenizer,
    Packer,
    context_window_for,
    register_model,
)


def tok() -> HeuristicTokenizer:
    return HeuristicTokenizer(chars_per_token=1.0)  # 1 char == 1 token


def test_model_sets_budget_from_registry():
    p = Packer(model="gpt-4o")
    assert p.budget == 128_000
    assert p.full_budget == 128_000
    assert p.reserve == 0


def test_model_lookup_is_case_insensitive():
    assert Packer(model="GPT-4O").budget == 128_000


def test_reserve_reduces_effective_budget():
    p = Packer(model="gpt-4o", reserve=8_000)
    assert p.full_budget == 128_000
    assert p.budget == 120_000  # effective = full - reserve
    assert p.reserve == 8_000


def test_explicit_budget_overrides_model():
    p = Packer(budget=50, model="gpt-4o")
    assert p.budget == 50


def test_reserve_is_reflected_in_result():
    p = Packer(budget=10, reserve=4, tokenizer=tok())
    p.add("abcdef", priority="low", strategy="drop", name="doc")  # 6 tokens
    result = p.fit()
    assert result.budget == 6  # 10 - 4 reserved
    assert result.reserved == 4
    assert result.text == "abcdef"  # fits exactly in the effective budget
    assert result.remaining == 0


def test_reserve_can_force_a_drop():
    p = Packer(budget=10, reserve=6, tokenizer=tok())  # effective 4
    p.add("abcdef", priority="low", strategy="drop", name="doc")  # 6 > 4
    result = p.fit()
    assert result.dropped_names == ["doc"]


def test_reserve_can_make_required_exceed_budget():
    p = Packer(budget=10, reserve=6, tokenizer=tok())  # effective 4
    p.add("toolong", priority="required")  # 7 > 4
    with pytest.raises(BudgetExceeded):
        p.fit()


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        Packer(model="totally-made-up-model")


def test_no_budget_and_no_model_raises():
    with pytest.raises(ValueError, match="budget"):
        Packer()


def test_negative_reserve_raises():
    with pytest.raises(ValueError):
        Packer(budget=100, reserve=-1)


def test_register_custom_model():
    register_model("my-private-llm", 5_000)
    assert context_window_for("my-private-llm") == 5_000
    assert Packer(model="my-private-llm").budget == 5_000


def test_register_model_rejects_nonpositive():
    with pytest.raises(ValueError):
        register_model("bad", 0)
