# contextpacker

[![PyPI version](https://img.shields.io/pypi/v/contextpacker.svg)](https://pypi.org/project/contextpacker/)
[![Python versions](https://img.shields.io/pypi/pyversions/contextpacker.svg)](https://pypi.org/project/contextpacker/)
[![CI](https://github.com/waelr1985/contextpacker/actions/workflows/ci.yml/badge.svg)](https://github.com/waelr1985/contextpacker/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Fit anything into an LLM context window.** A tiny, zero-dependency, priority-aware
token-budget packer for RAG pipelines and agents.

Every RAG or agent app has the same problem: you have too much stuff — a system
prompt, chat history, retrieved documents, tool output — and a fixed token
budget. `contextpacker` packs it all in *by priority*, truncating, trimming, or
dropping the least important pieces so the important ones always make it.

```python
from contextpacker import Packer

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
pip install contextpacker
# optional: exact token counts via tiktoken
pip install "contextpacker[tiktoken]"
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

## Exact token counts

```python
from contextpacker import Packer, tiktoken_tokenizer

packer = Packer(budget=8000, tokenizer=tiktoken_tokenizer("gpt-4o"))
```

Or wrap any tokenizer with `CallableTokenizer(lambda s: len(my_encode(s)))`.

## Priorities

Use the named levels `"required"`, `"high"`, `"medium"`, `"low"`, or pass any
integer (higher is kept first):

```python
packer.add(text, priority=42, strategy="truncate")
```

## Development

```bash
git clone https://github.com/waelr1985/contextpacker.git
cd contextpacker
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

## License

MIT
