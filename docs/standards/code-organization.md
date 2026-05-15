# Code organization & readability

## Code style

- Prefer **clear, explicit code** over clever or compact code.
- Avoid introducing new abstractions unless they reduce complexity.
- Match existing patterns in the codebase before inventing new ones.

## Repository layout

- `src/pinet/` - library code (the `pinet` package).
- `src/test/` - test suite (mirrors `src/pinet/` structure).
- `src/benchmarks/` - reproducible benchmarks and configs.
- `src/tools/` - auxiliary CLI and developer tools.
- `docs/` - project documentation (single source of truth for agent-driven workflows).

## Ownership

- Public API lives under `src/pinet/`. Anything exposed there has stability expectations and must have tests under `src/test/`.
- Internal helpers stay private (module-level underscore prefix or in a non-exported submodule).
- Benchmarks and tools import from `pinet` but do not reach into private submodules.
