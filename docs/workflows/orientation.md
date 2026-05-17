---
description: First-time orientation before making changes
---

Use when working on the project for the first time, or returning after a long absence. Goal: build a mental model of the system before touching code.

1. **Read the project README.**
   [`README.md`](../../README.md) at the repo root explains the scope and positioning of Pinet: a differentiable orthogonal projection layer for training hard-constrained neural networks.

2. **Read the contributing guide.**
   [`CONTRIBUTING.md`](../../CONTRIBUTING.md) covers development setup, branch workflow, PR preparation, and commit style.

3. **Scan the repository layout.**
   - `src/pinet/` - library code (public API lives here).
   - `src/test/` - test suite (mirrors `src/pinet/`; also type-checked).
   - `src/benchmarks/` - reproducible benchmarks with configs.
   - `src/tools/` - developer/CLI tools.
   - `docs/` - agent-driven development documentation (this folder).

4. **Review the standards.**
   Skim [`docs/standards/`](../standards/) before writing code -- in particular [`typing-docstrings.md`](../standards/typing-docstrings.md), [`code-clarity.md`](../standards/code-clarity.md), and [`linting-formatting.md`](../standards/linting-formatting.md).

5. **Confirm the environment works.**
   Run the test suite:
   ```bash
   uv run pytest
   ```
   All tests should pass before you write a single line.

6. **Pick the right workflow for your task.**
   | Task | Workflow |
   |---|---|
   | New feature | [`feature.md`](feature.md) |
   | Bug fix | [`bugfix.md`](bugfix.md) |
   | Refactor | [`refactor.md`](refactor.md) |
   | External API exploration | [`api-validation.md`](api-validation.md) |
   | Documentation only | [`docs.md`](docs.md) |

**Done criteria:**
- You can describe what the top-level modules under `src/pinet/` (`constraints`, `solver`, ...) do without looking them up.
- You know which module owns the code you intend to change.
- `uv run pytest` passes.
