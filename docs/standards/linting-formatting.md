# Linting & formatting policy

## Configuration

- **All linting and formatting rules live in `pyproject.toml`.**
  - Do not override rule selection via CLI flags in pre-commit or scripts.
  - Pre-commit controls *when* tools run, not *what* rules apply.

## Ruff as single source of truth

- **Ruff is the single source of truth** for:
  - unused variables/imports
  - import ordering
  - modern typing syntax
  - docstring style
  - low-noise bug patterns

## Allowed exceptions

- Allowed uppercase local variable names for tensor dimensions:
  - `H`, `W`, `B`, `N`, `T`

## Import organization

- **Imports must be at the top of the file** by default.
  - Local imports (inside functions/methods) are allowed **only** to avoid circular dependencies or for specific performance/lazy-loading reasons.
  - If a local import is used, it **must** be documented with a comment above it explaining why it's not at the top-level.

### Never guard imports with try/except

Wrap an import in a `try/except` block **only when explicitly instructed** to do so.

**Rationale:** Guarded imports hide missing dependencies until the code path that uses the symbol is executed, turning a clear `ModuleNotFoundError` at startup into a cryptic `AttributeError` deep inside a call stack. If a dependency is optional, the optionality must be expressed in the project's packaging (e.g. an extras group in `pyproject.toml`), not silenced in source.

```python
# Wrong
try:
    from canlib import canlib
except (ImportError, OSError):
    canlib = None  # type: ignore[assignment]

# Right
from canlib import canlib
```
