"""Modified ruiz equilibration."""

import jax.numpy as jnp

from .dataclasses import EquilibrationParams

EXPECTED_MATRIX_NDIM = 2


def ruiz_equilibration(
    a: jnp.ndarray, params: EquilibrationParams
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform modified Ruiz equilibration on matrix a.

    Ruiz equilibration iteratively scales the rows and columns of a so that
    all rows have equal norms and all columns have equals norms.

    TODO: Add equilibration for joint constraints.

    Args:
        a: Input matrix with shape (n_r, n_c).
        params: Parameters for equilibration.

    Returns:
        scaled_a: Equilibrated matrix.
        d_r: Row scaling factors
            such that scaled_a = diag(d_r) @ a @ diag(d_c).
        d_c: Column scaling factors.
    """
    # Ruiz equilibration is defined here only for plain 2D matrices.
    assert a.ndim == EXPECTED_MATRIX_NDIM, (
        "Input matrix to equilibration must be 2-dimensional."
    )

    scaled_a = a
    d_r = jnp.ones(a.shape[0])
    d_c = jnp.ones(a.shape[1])
    # Keep track of best criterion
    best_criterion = 1.0
    d_r_best = d_r
    d_c_best = d_c
    # Initialize column scaling
    alpha = (
        (a.shape[0] / a.shape[1]) ** (1 / (2 * params.ord)) if params.col_scaling else 1.0
    )

    for _ in range(params.max_iter):
        if params.update_mode == "Gauss":
            # Scale rows
            row_norms = jnp.linalg.norm(scaled_a, axis=1, ord=params.ord)
            # Avoid division by zero.
            row_factors = jnp.where(row_norms > 0, jnp.sqrt(row_norms), 1.0)
            # Update row scaling factors.
            d_r = d_r / row_factors
            # Scale rows.
            scaled_a = scaled_a / row_factors[:, None]

            # Scale columns
            col_norms = jnp.linalg.norm(scaled_a, axis=0, ord=params.ord)
            col_factors = alpha * jnp.where(col_norms > 0, jnp.sqrt(col_norms), 1.0)
            d_c = d_c / col_factors
            scaled_a = scaled_a / col_factors[None, :]
        else:
            # Scale rows
            row_norms = jnp.linalg.norm(scaled_a, axis=1, ord=params.ord)
            row_factors = jnp.where(row_norms > 0, jnp.sqrt(row_norms), 1.0)
            # Scale columns
            col_norms = jnp.linalg.norm(scaled_a, axis=0, ord=params.ord)
            col_factors = alpha * jnp.where(col_norms > 0, jnp.sqrt(col_norms), 1.0)
            # Update
            d_r = d_r / row_factors
            d_c = d_c / col_factors
            scaled_a = scaled_a / row_factors[:, None]
            scaled_a = scaled_a / col_factors[None, :]

        # Check convergence: after scaling, row and column norms should be close to 1.
        new_row_norms = jnp.linalg.norm(scaled_a, axis=1, ord=params.ord)
        new_col_norms = jnp.linalg.norm(scaled_a, axis=0, ord=params.ord)
        term_criterion = jnp.maximum(
            1 - jnp.min(new_row_norms) / jnp.max(new_row_norms),
            1 - jnp.min(new_col_norms) / jnp.max(new_col_norms),
        )

        # Best termination criterion so far
        if term_criterion < best_criterion:
            best_criterion = term_criterion
            d_r_best = d_r
            d_c_best = d_c

        if term_criterion < params.tol:
            break

    # Get the best scaled matrix
    scaled_a_best = a * d_r_best[:, None]
    scaled_a_best = scaled_a_best * d_c_best[None, :]

    # Safeguard
    if params.safeguard:
        cond_a = jnp.linalg.cond(a)
        cond_scaled_a = jnp.linalg.cond(scaled_a_best)
        if cond_scaled_a > cond_a:
            scaled_a_best = a
            d_r_best = jnp.ones(a.shape[0])
            d_c_best = jnp.ones(a.shape[1])

    return scaled_a_best, d_r_best, d_c_best
