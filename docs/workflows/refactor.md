---
description: Code refactoring without behavior change
---

Use when improving structure, naming, or internal APIs without changing outputs.

1. Identify the refactor scope and keep it small and coherent.
2. Ensure existing tests cover the behavior.
   - If coverage is weak, add characterization tests first.
3. Apply the refactor in small steps (prefer multiple local commits).
// turbo
4. Run all checks:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   uv run pre-commit run --hook-stage push --all-files
   ```
5. Update docstrings only if public APIs changed.

**Done criteria:**
- Tests provide confidence behavior didn't change
- All gates pass
- No new "convenience" API generalization added during refactor
