# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-14

### Added

- `Packer` with priority-aware token-budget packing.
- Per-item strategies: `drop`, `truncate`, `truncate_head`, `trim`.
- `required` items that are never dropped (raises `BudgetExceeded` if they
  alone exceed the budget).
- Zero-dependency `HeuristicTokenizer` (characters-per-token estimate).
- `CallableTokenizer` and `tiktoken_tokenizer` for exact counts.
- `PackResult` with `used_tokens`, `remaining`, `dropped`, and per-item detail.

[Unreleased]: https://github.com/Waelr1985/contextpacker/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Waelr1985/contextpacker/releases/tag/v0.1.0
