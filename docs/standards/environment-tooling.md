# Environment & tooling

## Using uv

- **Always use `uv run`** to execute tools and scripts.
  - Never assume global installs.
  - Examples:
    ```bash
    uv run python
    uv run pytest
    uv run ruff
    ```

## Managing dependencies

- **Add dependencies with uv (don't edit `pyproject.toml` by hand unless necessary).**
  - Runtime dependency:
    ```bash
    uv add <package>
    ```
  - Dev dependency (linters/formatters/type checkers/test tooling):
    ```bash
    uv add --dev <package>
    ```

- **If you add a new external dependency (new import), you must:**
  1. Add it with `uv add` / `uv add --dev`
  2. Ensure the lockfile is updated (run `uv lock` if needed)
  3. Commit both `pyproject.toml` and the lockfile
