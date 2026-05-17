# Pinet development

All documentation is centralized in `docs/`. Claude will discover what it needs based on your task.

## Essential commands

```bash
# Test
uv run pytest

# Lint and format
uv run pre-commit run --all-files

# Type check
uv run pre-commit run --hook-stage push --all-files

# Add dependency
uv add <package>              # Runtime
uv add --dev <package>        # Dev dependency
```

## Workflows

When starting a task, refer to the appropriate workflow from `docs/workflows/`:

- **Feature**: see `docs/workflows/feature.md`
- **Bugfix**: see `docs/workflows/bugfix.md`
- **Refactor**: see `docs/workflows/refactor.md`
- **API Validation**: see `docs/workflows/api-validation.md`
- **Documentation**: see `docs/workflows/docs.md`

## Standards

All code must follow standards in `docs/standards/`:

- Always use `uv run` for commands (never assume global installs)
- Run all quality gates before finishing (pre-commit hooks must pass)
- Write tests first (TDD approach)
- Google-style docstrings, types in signatures only
- Prefer JAX for ML/numerical code

## Key project-specific rules

- **Never push to remote** - commits are pushed centrally
- **Commit messages**: follow the `.gitmessage` template (Conventional Commits)
- **Allowed uppercase vars**: `H`, `W`, `B`, `N`, `T` (tensor dimensions)
- **Strict typing**: `src/` AND `tests/` both type-check under basedpyright
- **No global imports in pyproject.toml** - use `uv add` commands

## Documentation

For comprehensive documentation, see `docs/index.md`
