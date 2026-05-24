# Testing policy

## Test framework

- **Always use `pytest`** for testing.

## Test structure

- Tests live **flat in `src/test/`**, one file per area, named
  `test_<area>.py` (e.g. `test_box.py`, `test_project.py`). A future
  reorganization into a tree mirroring `src/pinet/` is tracked in
  [issue #130](https://github.com/antonioterpin/pinet/issues/130).
- **Prefer small, focused files**, one concern or layer per file. Some
  legacy files are still large; split them when you touch them.
- Each test file carries a **module-level docstring** explaining what
  behavior it verifies.
- Tests are **type-checked** under basedpyright, same as `src/`. Fixtures and helpers must carry accurate type hints.

## Test design

- **Functionality-driven tests only.** Test behavior and contracts, not
  implementation details. Drop tests that just verify a constructor sets an
  attribute or that a type check fires.
- Use **multiple small unit tests** rather than one large test.
- Add **integration tests** when behavior spans multiple components.

## Docs-first workflow

When adding tests to a new area:

1. **Write the test file** with a module docstring explaining the tested behavior.
2. Each test function gets a **one-line docstring** saying what contract it verifies.
3. Implement the tests, run them, iterate.

## Testing best practices

- Prefer `pytest.mark.parametrize` over repeated tests.
- **Always check `conftest.py`** for existing fixtures before adding new ones.
- **Always include a descriptive message in `assert` statements** to explain
  what exactly failed and why (e.g., `assert x == y, f"Expected {y}, got {x}"`).
- Use `pytest.mark.skip(reason=...)` for tests that need adaptation, with a
  clear reason explaining what changed.

## Running tests

```bash
# All tests
uv run pytest

# A specific file
uv run pytest src/test/test_box.py

# With coverage
uv run pytest --cov=src/pinet --cov-report=term-missing
```
