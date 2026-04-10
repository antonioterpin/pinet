"""Modified ruiz equilibration."""

import jax.numpy as jnp

from .dataclasses import EquilibrationParams

EXPECTED_MATRIX_NDIM = 2


def ruiz_equilibration(
    a_dyn: jnp.ndarray, params: EquilibrationParams
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform modified Ruiz equilibration on matrix a_dyn.

    Ruiz equilibration iteratively scales the rows and columns of a_dyn so that
    all rows have equal norms and all columns have equals norms.

    TODO: Add equilibration for joint constraints.

    Args:
        a_dyn: Input matrix with shape (n_r, n_c).
        params: Parameters for equilibration.

    Returns:
        scaled_a_dyn: Equilibrated matrix.
        d_r: Row scaling factors
            such that scaled_a_dyn = diag(d_r) @ a_dyn @ diag(d_c).
        d_c: Column scaling factors.
    """
    assert a_dyn.ndim == EXPECTED_MATRIX_NDIM, (
        "Input matrix to equilibration must be 2-dimensional."
    )

    scaled_a_dyn = a_dyn
    d_r = jnp.ones(a_dyn.shape[0])
    d_c = jnp.ones(a_dyn.shape[1])
    # Keep track of best criterion
    best_criterion = 1.0
    d_r_best = d_r
    d_c_best = d_c
    # Initialize column scaling
    alpha = (
        (a_dyn.shape[0] / a_dyn.shape[1]) ** (1 / (2 * params.ord))
        if params.col_scaling
        else 1.0
    )

    for _ in range(params.max_iter):
        if params.update_mode == "Gauss":
            # Scale rows
            row_norms = jnp.linalg.norm(scaled_a_dyn, axis=1, ord=params.ord)
            # Avoid division by zero.
            row_factors = jnp.where(row_norms > 0, jnp.sqrt(row_norms), 1.0)
            # Update row scaling factors.
            d_r = d_r / row_factors
            # Scale rows.
            scaled_a_dyn = scaled_a_dyn / row_factors[:, None]

            # Scale columns
            col_norms = jnp.linalg.norm(scaled_a_dyn, axis=0, ord=params.ord)
            col_factors = alpha * jnp.where(col_norms > 0, jnp.sqrt(col_norms), 1.0)
            d_c = d_c / col_factors
            scaled_a_dyn = scaled_a_dyn / col_factors[None, :]
        else:
            # Scale rows
            row_norms = jnp.linalg.norm(scaled_a_dyn, axis=1, ord=params.ord)
            row_factors = jnp.where(row_norms > 0, jnp.sqrt(row_norms), 1.0)
            # Scale columns
            col_norms = jnp.linalg.norm(scaled_a_dyn, axis=0, ord=params.ord)
            col_factors = alpha * jnp.where(col_norms > 0, jnp.sqrt(col_norms), 1.0)
            # Update
            d_r = d_r / row_factors
            d_c = d_c / col_factors
            scaled_a_dyn = scaled_a_dyn / row_factors[:, None]
            scaled_a_dyn = scaled_a_dyn / col_factors[None, :]

        # Check convergence: after scaling, row and column norms should be close to 1.
        new_row_norms = jnp.linalg.norm(scaled_a_dyn, axis=1, ord=params.ord)
        new_col_norms = jnp.linalg.norm(scaled_a_dyn, axis=0, ord=params.ord)
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
    scaled_a_dyn_best = a_dyn * d_r_best[:, None]
    scaled_a_dyn_best = scaled_a_dyn_best * d_c_best[None, :]

    # Safeguard
    if params.safeguard:
        cond_a_dyn = jnp.linalg.cond(a_dyn)
        cond_scaled_a_dyn = jnp.linalg.cond(scaled_a_dyn_best)
        if cond_scaled_a_dyn > cond_a_dyn:
            scaled_a_dyn_best = a_dyn
            d_r_best = jnp.ones(a_dyn.shape[0])
            d_c_best = jnp.ones(a_dyn.shape[1])

    return scaled_a_dyn_best, d_r_best, d_c_best
