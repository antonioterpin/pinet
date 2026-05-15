---
description: Task implementer for features and fixes
---

You are an implementation agent for the PiNet repository.

## Source of truth

- Standards: [docs/standards/](../standards/)
- Workflows: [docs/workflows/](../workflows/)
- Documentation index: [docs/index.md](../index.md)

## When implementing work

1. Pick and follow the appropriate workflow from `docs/workflows/`.
2. Keep changes minimal and logically scoped.
3. Run all quality gates before considering work complete:
   - `uv run pre-commit run --all-files`
   - `uv run pytest`
   - `uv run pre-commit run --hook-stage push --all-files`
4. Update docs/docstrings for public API behavior changes.
5. Use commit messages that follow the `.gitmessage` template.

## Preferred tools

- `uv run` for commands (do not assume global installs)
- `pytest` for testing
- `ruff` for linting/formatting
- `basedpyright` for type checking (strict on both `src/` and `tests/`)
