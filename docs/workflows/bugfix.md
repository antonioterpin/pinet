---
description: Bugfix or regression workflow
---

Use when fixing a bug or preventing a known regression.

1. Write a failing test that reproduces the issue (minimal reproduction).
2. Implement the fix in the correct module under `src/pinet/`.
// turbo
3. Run all checks:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   uv run pre-commit run --hook-stage push --all-files
   ```
4. If the fix changes behavior, update docstrings and (if needed) `docs/`.

**Done criteria:**
- The test fails on old code, passes after the fix
- All checks pass
