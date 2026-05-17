# Naming conventions

Single source of truth for identifier naming across the library, tests, tools, and benchmarks. Whenever this document and the code disagree, fix the code.

## Identifiers in code

| Kind | Convention | Examples |
|------|------------|----------|
| Matrix | lowercase + `_mat` suffix | `a_mat`, `c_mat`, `q_mat`, `g_mat`, `a_mat_pinv`, `scaled_a_mat` |
| Vector or offset | bare lowercase letter | `a`, `b`, `f`, `h`, `p`, `lb`, `ub`, `xhat` |
| Scalar | bare lowercase letter | `alpha`, `sigma`, `omega`, `tol` |
| Count or row dimension | `n_*` or `m_*` | `n_eq`, `n_ineq`, `n_a`, `n_c`, `n_a_soc_1`, `dim`, `dim_lifted` |
| Is per-instance | `var_*` | `var_a_mat`, `var_b` |
| RNG key | `k*` prefix, matrix variant uses the full matrix name | `key`, `ka`, `kb`, `kf`, `kx`, `ka_mat`, `kc_mat` |
| Boolean mask | `mask_*` | `mask_u`, `mask_t`, `mask_box` |
| Private attribute | leading `_` | `_a_mat`, `_a`, `_f`, `_b`, `_nl_type`, `_dim`, `_rng_key` |

`EqualityConstraintsSpecification.a_mat` (matrix) and `NonLinearSpecification.a` (offset vector) are intentionally distinct attributes on different classes. The `_mat` suffix is what tells them apart at the call site.

## Uppercase letters

Allowed only for:

| What | Examples | Why |
|------|----------|-----|
| Tensor-dimension single letters | `H`, `W`, `B`, `N`, `T` | Image dims and batch / sequence sizes per `.claude/CLAUDE.md` |
| Class names | `EqualityConstraint`, `NonLinearSpecification`, `Project` | PEP 8 |
| Type aliases | `BatchedPrimal`, `BatchedRHS`, `ArrayLike`, `NLMatrix`, `ScalarLike` | PEP 8 |
| Module-level constants | `EXPECTED_BOUND_NDIM`, `PROJECTION_DEFAULT_SIGMA`, `EQUILIBRATION_DEFAULT_TOL` | PEP 8 |
| Enum or type-tag values | `SOCType`, `L2NormType`, `NonLinearConstraintType` | PEP 8 |

Not allowed: single-letter uppercase Python identifiers for matrices. `A`, `C`, `G`, `Q` as code identifiers are out. Use `a_mat`, `c_mat`, `g_mat`, `q_mat` instead.

Math notation in docstrings and comments may use uppercase letters when reproducing a published equation, e.g. ``g(A x + a) <= f x + b`` in prose. That is fine because it is prose, not a Python identifier.

## jaxtyping axis names

These live inside `Float[Array, "..."]` and `Real[ArrayLike, "..."]` annotation strings and are a separate namespace from Python identifiers. Axis names may be short and uppercase because they are annotation tokens, not runtime values.

| Axis | Meaning |
|------|---------|
| `B` or `#B` | Batch dimension; `#B` marks it broadcast-compatible (size-1 OK) |
| `n` | Primal dimension (number of decision variables) |
| `m` | Rows in an equality or inequality matrix |
| `n_ineq` | Inequality-constraint rows (slack count) |
| `d_lifted` | Dimension of the lifted problem |
| `d`, `n_r`, `n_c`, `m_scale`, `n_scale` | Role-specific axes used in narrow spots |

Use existing axis names when you can; introduce a new one only when the role does not match any of the above.

## Dataset keys (`.npz`)

On-disk keys mirror the Python identifier of the field they store, with one transitional exception:

| Field | Current key | Legacy keys still accepted in loaders |
|-------|-------------|----------------------------------------|
| Equality matrix | `a_mat` | `a_dyn` (intermediate), `A` / `As` (main / v0.1.0) |
| Inequality matrix | `g_mat` | `G` |
| Quadratic-objective matrix | `q_mat` | `Q` |
| Input features | `x_data` | `X` |
| Optimal solutions | `y_star` | `Ystar` |
| Time horizon | `horizon` | `T` |

The legacy fallbacks live in `_pick(...)` calls inside `src/benchmarks/QP/load_qp.py` and `src/benchmarks/toy_MPC/load_toy_mpc.py`. Removing them, together with regenerating the in-repo datasets so the `a_dyn` fallback can be dropped, is tracked in issue #112.

## What `_mat` is not for

- Counts, even when they describe rows of a matrix. Write `n_a`, not `n_a_mat`.
- Vectors. Write `b`, not `b_mat`.
- Scalars. Write `alpha`, not `alpha_mat`.

## Related standards

- [code-clarity.md](code-clarity.md) -- naming intent, comments, prose conventions, American English, ASCII-only.
- [typing-docstrings.md](typing-docstrings.md) -- Google-style docstrings, types in signatures.
- [api-design.md](api-design.md) -- public API contracts and simplicity.
