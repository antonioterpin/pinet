# Typing & docstrings

## Type information placement

- **Type information belongs in function signatures**, not in docstrings.
- **Do not include type hints in `Args:` or `Returns:` sections.**
- Use **Google-style docstrings** with descriptions only.

## Example

```python
def resize(img: jax.Array, *, H: int, W: int) -> jax.Array:
    """Resize an image.

    Args:
        img: Input image.
        H: Output height.
        W: Output width.

    Returns:
        Resized image.
    """
```

## Strict typing scope

- **Both `src/` and `tests/` are type-checked** by basedpyright.
- Tests must have accurate type hints on fixtures, helpers, and public test functions.

## Type ignore & lint suppressions

- **Never use `type: ignore` comments** without explicitly telling the user first.
- **Never use `# noqa` for specific rules** without explicitly telling the user first.
- Both are code debt and should only be used with awareness and acknowledgment.

## Documentation requirements

- All **public functions and classes must be documented**.
- Private helpers may omit docstrings if their intent is obvious.
- **All files must have a file-level docstring at the top**, including tests.

## Shell command examples in docstrings

- **Never use doctest syntax (`>>>` / `...`) for shell commands.** `>>>` denotes a Python REPL prompt; using it for shell commands is semantically wrong, breaks copy-paste, and misleads tooling.
- Use a Sphinx **`.. code-block:: console`** directive with `$` as the shell prompt. This renders correctly in Sphinx and keeps commands copyable as a contiguous block.
- Use `r"""..."""` (raw docstring) when the block contains backslash line continuations, otherwise `\<newline>` is interpreted as a string-literal line continuation and collapses the lines.

### Example

```python
r"""Run a training job.

Usage:

.. code-block:: console

    $ uv run python -m <mod> \
        --config configs/toy_QP.yaml --seed 0
    $ uv run python -m <mod> \
        --config configs/toy_QP.yaml --seed 1
"""
```
