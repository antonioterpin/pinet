---
description: External API validation and integration
---

Use when interacting with a new external library API or uncertain behavior.

1. **Validate first in a scratch script** (not committed):
   - Write a small script to confirm the API, edge cases, and expected behavior.
   - Capture outputs, performance notes, and pitfalls.
2. Summarize findings in 1-5 bullets (for yourself) and decide the minimal integration.
3. **Write failing unit tests** in the test suite that define the desired API and expected behavior (based on your scratch script findings).
// turbo
4. Run tests to confirm they fail:
   ```bash
   uv run pytest
   ```
5. Implement the **minimal core change** in `src/pinet/...` to make the tests pass.
   - Do not over-generalize or add multiple input modes prematurely.
// turbo
6. Run formatting/lint gate:
   ```bash
   uv run pre-commit run --all-files
   ```
// turbo
7. Run all tests (including the new ones):
   ```bash
   uv run pytest
   ```
// turbo
8. Run typecheck gate:
   ```bash
   uv run pre-commit run --hook-stage push --all-files
   ```
9. Delete the scratch script. Encode conclusions:
   - small local rationale -> comment (WHY)
   - API behavior -> docstring
   - important design decision -> `docs/` entry (if applicable)

**Done criteria:**
- `uv run pre-commit run --all-files` passes
- `uv run pytest` passes
- `uv run pre-commit run --hook-stage push --all-files` passes
- No temporary scripts committed
- Public APIs are documented (without docstring type hints)
