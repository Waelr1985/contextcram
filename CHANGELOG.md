# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-15

### Added

- `Packer(model="gpt-4o")` — set the budget from a built-in model
  context-window registry instead of hard-coding a number.
- `reserve=` — hold back tokens (e.g. for the model's response); the effective
  packing budget becomes `budget - reserve`.
- `register_model()`, `context_window_for()`, and `MODEL_CONTEXT_WINDOWS` for
  customizing or inspecting the registry.
- `PackResult.reserved` and `Packer.full_budget` for transparency.

### Changed

- `Packer(budget=...)` is now optional when `model=` is given. An explicit
  `budget` still wins over `model`.

## [0.1.0] - 2026-06-14

### Added

- `Packer` with priority-aware token-budget packing.
- Per-item strategies: `drop`, `truncate`, `truncate_head`, `trim`.
- `required` items that are never dropped (raises `BudgetExceeded` if they
  alone exceed the budget).
- Zero-dependency `HeuristicTokenizer` (characters-per-token estimate).
- `CallableTokenizer` and `tiktoken_tokenizer` for exact counts.
- `PackResult` with `used_tokens`, `remaining`, `dropped`, and per-item detail.

[Unreleased]: https://github.com/Waelr1985/contextcram/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Waelr1985/contextcram/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Waelr1985/contextcram/releases/tag/v0.1.0
