# Machine learning & numerical code

## Framework preferences

- **Prefer JAX** for machine learning and numerical computation.
- Avoid mixing ML frameworks unless there is a clear, documented reason.

## Code style

- Write code with **functional style and explicit data flow** where possible.
- Keep functions testable both with and without `jax.jit`. When a function is expected to be jittable, include a test that confirms it can be traced and compiled.

## Randomness & reproducibility

When randomness is involved:
- make it explicit
- set seeds
- keep tests deterministic
