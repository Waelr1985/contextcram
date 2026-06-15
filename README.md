# contextcram

[![PyPI version](https://img.shields.io/pypi/v/contextcram.svg)](https://pypi.org/project/contextcram/)
[![Python versions](https://img.shields.io/pypi/pyversions/contextcram.svg)](https://pypi.org/project/contextcram/)
[![CI](https://github.com/Waelr1985/contextcram/actions/workflows/ci.yml/badge.svg)](https://github.com/Waelr1985/contextcram/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Fit anything into an LLM context window.** A tiny, zero-dependency, priority-aware
token-budget packer for RAG pipelines and agents.

Every RAG or agent app has the same problem: you have too much stuff — a system
prompt, chat history, retrieved documents, tool output — and a fixed token
budget. `contextcram` packs it all in *by priority*, truncating, trimming, or
dropping the least important pieces so the important ones always make it.

```python
from contextcram import Packer

packer = Packer(budget=8000)  # token budget

packer.add(system_prompt, priority="required")                 # never dropped
packer.add(chat_history, priority="high", strategy="trim")     # drop oldest turns
packer.add(retrieved_docs, priority="medium", strategy="drop") # all-or-nothing
packer.add(tool_output, priority="low", strategy="truncate")   # cut to fit

result = packer.fit()
print(result.text)            # assembled, in-budget context
print(result.used_tokens)     # e.g. 7840
print(result.dropped_names)   # what didn't make the cut
```

## Why

- **Zero dependencies.** Pure Python. Works out of the box with a fast
  characters-per-token heuristic; plug in `tiktoken` or any tokenizer when you
  need exact counts.
- **Framework-agnostic.** Use it with LangChain, LlamaIndex, the raw provider
  SDKs, or nothing at all.
- **Priority-aware.** You decide what survives a tight budget, not a blind
  truncate at the end.
- **Observable.** Every result tells you what was kept, truncated, and dropped.

## Installation

```bash
pip install contextcram
# optional: exact token counts via tiktoken
pip install "contextcram[tiktoken]"
```

## Strategies

When an optional item doesn't fully fit, its `strategy` decides what happens:

| Strategy         | Behavior                                              |
| ---------------- | ----------------------------------------------------- |
| `drop`           | Include the item whole, or not at all                 |
| `truncate`       | Cut from the end, keeping the head (default)          |
| `truncate_head`  | Cut from the start, keeping the tail                  |
| `trim`           | For list content: drop oldest segments first          |

`required` items are always kept; if they alone exceed the budget, a
`BudgetExceeded` error is raised.

## Model-aware budgets

Skip the magic number — set the budget from the model, and reserve room for the
response in one go:

```python
from contextcram import Packer

# 128k window for gpt-4o, holding back 2k tokens for the model's reply
packer = Packer(model="gpt-4o", reserve=2000)
print(packer.full_budget)  # 128000
print(packer.budget)       # 126000  (effective budget you pack into)
```

`reserve` is the easy way to avoid the classic "prompt fit, but no room left to
answer" failure. Unknown model? Pass `budget=` explicitly or register it:

```python
from contextcram import register_model

register_model("my-internal-llm", 32000)
packer = Packer(model="my-internal-llm", reserve=1000)
```

## Exact token counts

```python
from contextcram import Packer, tiktoken_tokenizer

packer = Packer(budget=8000, tokenizer=tiktoken_tokenizer("gpt-4o"))
```

Or wrap any tokenizer with `CallableTokenizer(lambda s: len(my_encode(s)))`.

## Priorities

Use the named levels `"required"`, `"high"`, `"medium"`, `"low"`, or pass any
integer (higher is kept first):

```python
packer.add(text, priority=42, strategy="truncate")
```

## Alternatives

Priority-based context assembly isn't a new idea, and depending on your needs
one of these may fit better — `contextcram` deliberately trades features for
simplicity and zero dependencies:

| Library | Approach | When to prefer it over `contextcram` |
| ------- | -------- | ------------------------------------ |
| [Priompt](https://github.com/anysphere/priompt) / [PriomptiPy](https://pypi.org/project/priompt/) | Component/JSX-style priority rendering | You want fine-grained, composable prompt components and don't mind a learning curve |
| [Prompt Poet](https://pypi.org/project/prompt-poet/) | YAML + Jinja2 templating with cache-aware, priority truncation | You need templating and production GPU prefix-cache optimization |
| [LLMLingua](https://github.com/microsoft/LLMLingua) | Model-based prompt *compression* | You want to shrink text rather than drop/truncate whole pieces |

**Choose `contextcram` when** you want a tiny, zero-dependency, framework-agnostic
helper with a 3-line API (`Packer(...).add(...).fit()`) that does one thing —
fit prioritized pieces into a budget — and gets out of your way.

## Development

```bash
git clone https://github.com/Waelr1985/contextcram.git
cd contextcram
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

## License

MIT
