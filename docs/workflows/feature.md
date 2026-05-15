---
description: Standard feature implementation
---

Use for ordinary feature work that doesn't require external API exploration.

1. Define the desired API and behavior by **writing failing unit tests** (and integration tests if needed).
// turbo
2. Run tests to confirm they fail:
   ```bash
   uv run pytest
   ```
3. Identify the ownership module under `src/pinet/` and implement the **minimal change** to make the tests pass.
// turbo
4. Run formatting/lint gate:
   ```bash
   uv run pre-commit run --all-files
   ```
// turbo
5. Run tests to confirm they pass:
   ```bash
   uv run pytest
   ```
// turbo
6. Run typecheck gate:
   ```bash
   uv run pre-commit run --hook-stage push --all-files
   ```
7. Add/adjust docstrings and documentation as needed.
   - Docstrings describe behavior; type info stays in signatures.

**Done criteria:**
- `uv run pre-commit run --all-files` passes
- `uv run pytest` passes
- `uv run pre-commit run --hook-stage push --all-files` passes
- Docs are updated for public surface changes
