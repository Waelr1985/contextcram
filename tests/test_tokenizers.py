"""Tests for contextcram tokenizers."""

from __future__ import annotations

import pytest

from contextcram import CallableTokenizer, HeuristicTokenizer


def test_heuristic_count_rounds_up():
    t = HeuristicTokenizer(chars_per_token=4)
    assert t.count("") == 0
    assert t.count("abcd") == 1
    assert t.count("abcde") == 2  # 5 chars / 4 -> ceil = 2


def test_heuristic_truncate():
    t = HeuristicTokenizer(chars_per_token=4)
    assert t.truncate("abcdefgh", 1) == "abcd"
    assert t.truncate("abcdefgh", 0) == ""


def test_heuristic_rejects_bad_ratio():
    with pytest.raises(ValueError):
        HeuristicTokenizer(chars_per_token=0)


def test_callable_tokenizer_counts_words():
    t = CallableTokenizer(lambda s: len(s.split()))
    assert t.count("one two three") == 3


def test_callable_tokenizer_truncate_fits_budget():
    # Token == word; truncating "a b c d e" to 2 tokens must yield <= 2 words.
    t = CallableTokenizer(lambda s: len(s.split()))
    out = t.truncate("a b c d e", 2)
    assert len(out.split()) <= 2
    assert out.startswith("a")


def test_callable_tokenizer_no_truncation_when_within_budget():
    t = CallableTokenizer(lambda s: len(s.split()))
    assert t.truncate("a b", 10) == "a b"
