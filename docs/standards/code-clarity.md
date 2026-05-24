# Comments & code clarity

## Comment philosophy

- **Do not pollute the code with comments explaining reasoning step-by-step.**
- This applies to inline comments; docstrings are an exception.
- Comments should explain **why** something is done, not:
  - what the code does (the code should be readable)
  - how it does it (that should be clear from implementation)

## Good comment example

```python
# Keep gradient magnitudes comparable across resolutions to prevent training instability.
```

## American English

- All prose in this repository uses **American English** spelling.
- This rule applies to code (comments, docstrings, identifiers, log/error messages) and markdown documentation.
- Prefer `-ize` / `-ization` over `-ise` / `-isation` (`optimize`, `normalize`, `serialize`).
- Prefer `-or` over `-our` (`color`, `behavior`, `favor`).
- Prefer `-er` over `-re` (`center`, `meter`).
- Examples:
  - correct: `normalize`, `optimization`, `behavior`, `serialized`, `center`
  - wrong: `normalise`, `optimisation`, `behaviour`, `serialised`, `centre`

## Avoid unnecessary special characters

- Prefer characters that are easily available on a US keyboard. Do not reach for fancy Unicode glyphs when a plain equivalent exists.
- Standard punctuation (`"`, `'`, etc.) is fine.
- **Do not use raw Unicode math/Greek/unit symbols** (`pi`, `sigma`, `mu`, `+/-`, `~=`, `<=`, etc.) from the keyboard, even in docstrings. For math, use **LaTeX with ReadTheDocs convention** instead -- it renders properly in the docs:
  - In docstrings (Sphinx/reST): ``:math:`\mu\mathrm{s}` ``, ``:math:`2\pi f t` ``, ``:math:`\pm 1` ``.
  - In markdown (MyST/ReadTheDocs): `$\mu\mathrm{s}$`, `$2\pi f t$`, `$\pm 1$`.
  - In runtime-facing strings (logs, prints, CLI output) where LaTeX will not render: use plain ASCII (`us`, `2*pi*f*t`, `+/-`).
- Do not write half-measures like `\mus` outside a math environment -- wrap LaTeX in the proper role/delimiters, or fall back to plain ASCII.
- This rule applies to all code, comments, docstrings, log/error messages, and markdown documentation.
- Examples of substitutions to make:
  - Use `...` instead of the single ellipsis character.
  - Use `"` and `'` instead of curly/smart quotes.
  - Use `-` or `--` instead of en/em dashes.
  - Use `->` instead of arrow glyphs.

## Header and title capitalization

- In markdown documentation **and in docstrings**, **only capitalize the first word** of headers, titles, and docstring section headings.
- This rule applies to all Markdown files in the repository (including `docs/`, `README.md`, PR/issue templates) and to all Python docstrings (including the first-line summary and any in-docstring section headings).
- Keep proper nouns, library names, and acronyms as needed (for example, `GitHub`, `API`, `JAX`, `NumPy`, `Pinet`, `CVXPY`, `QP`, `SOC`, `ADMM`).
- Examples:
  - correct: `## Projection layer` (not `## Projection Layer`)
  - correct: `## Constraint types` (not `## Constraint Types`)
  - correct: `## Type ignore & lint suppressions` (not `## Type Ignore & Lint Suppressions`)
  - correct: `# Benchmark toy QP` (keep `QP` acronym, lowercase the rest)
  - correct: `"""Orthogonal projection onto the feasible set."""` (not `Orthogonal Projection`)

## Markdown link paths

- For links to files or folders in this repository, use **relative paths**.
- This rule applies to markdown documentation in the repository (for example `docs/`, `README.md`, `CONTRIBUTING.md`).
- Do not use absolute local paths such as `file:///...`, `/home/...`, or `C:\...`.
- Do not use repository-internal raw GitHub links (`https://raw.githubusercontent.com/...`) when the target file exists in this repo; use relative links so IDE navigation works.
- Examples:
  - correct: `[Code clarity](../standards/code-clarity.md)`
  - correct: `[Contributing](CONTRIBUTING.md)`
  - wrong: `[Code clarity](file:///home/user/project/docs/standards/code-clarity.md)`
  - wrong: `[Contributing](https://raw.githubusercontent.com/org/repo/main/CONTRIBUTING.md)`
