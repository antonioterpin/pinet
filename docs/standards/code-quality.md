# Code quality gates

All modified code must pass the following quality gates before a task is considered complete.

## Required checks

- **Ruff and Ruff-format must pass** on all modified files.
- **Pydoclint must pass** on all modified files.
- **BasedPyright must pass** on both `src/` and `tests/`.

## Running quality gates

Before finishing a task, **run both**:
```bash
uv run pre-commit run --all-files
uv run pre-commit run --hook-stage push --all-files
```

**A task is NOT complete unless these checks pass.**

All linting and formatting rules live in `pyproject.toml`. Do not override rule selection via CLI flags in pre-commit or scripts. Pre-commit controls *when* tools run, not *what* rules apply.
