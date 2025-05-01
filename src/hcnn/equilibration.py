"""Modified ruiz equilibration."""

import jax.numpy as jnp


# TODO: Add equilibration for joint constraints.
def ruiz_equilibration(
    A: jnp.ndarray,
    max_iter: int = 0,
    tol: float = 1e-3,
    ord: float = 2.0,
    col_scaling: bool = False,
    update_mode: str = "Gauss",
    safeguard: bool = False,
):
    """Perform modified Ruiz equilibration on matrix A.

    Ruiz equilibration iteratively scales the rows and columns of A so that
    all rows have equal norms and all columns have equals norms.

    Args:
        A (jnp.ndarray): Input matrix with shape (n_r, n_c).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        ord (float): Order of the norm used.
        col_scaling (bool): Use a scaling when computing the column norms.
        update_mode (str): Valid options are "Gauss" and "Jacobi".
            "Jacobi" means compute both row and column norms and update.
            "Gauss" means compute row, update, compute column, update.
        safeguard (bool): Check if the condition nubmer of A has decreased.

    Returns:
        scaled_A (jnp.ndarray): Equilibrated matrix.
        d_r (jnp.ndarray): Row scaling factors
            such that scaled_A = diag(d_r) @ A @ diag(d_c).
        d_c (jnp.ndarray): Column scaling factors.
    """
    assert A.ndim == 2, "Input matrix to equilibration must be 2-dimensional."
    assert update_mode in [
        "Gauss",
        "Jacobi",
    ], 'update_mode must be either "Gauss" or "Jacobi"'
    scaled_A = A
    d_r = jnp.ones(A.shape[0])
    d_c = jnp.ones(A.shape[1])
    # Keep track of best criterion
    best_criterion = 1.0
    d_r_best = d_r
    d_c_best = d_c
    # Initialize column scaling
    if col_scaling:
        alpha = (A.shape[0] / A.shape[1]) ** (1 / (2 * ord))
    else:
        alpha = 1.0

    for ii in range(max_iter):
        if update_mode == "Gauss":
            # Scale rows
            row_norms = jnp.linalg.norm(scaled_A, axis=1, ord=ord)
            # Avoid division by zero.
            row_factors = jnp.where(row_norms > 0, jnp.sqrt(row_norms), 1.0)
            # Update row scaling factors.
            d_r = d_r / row_factors
            # Scale rows.
            scaled_A = scaled_A / row_factors[:, None]

            # Scale columns
            col_norms = jnp.linalg.norm(scaled_A, axis=0, ord=ord)
            col_factors = alpha * jnp.where(col_norms > 0, jnp.sqrt(col_norms), 1.0)
            d_c = d_c / col_factors
            scaled_A = scaled_A / col_factors[None, :]
        else:
            # Scale rows
            row_norms = jnp.linalg.norm(scaled_A, axis=1, ord=ord)
            row_factors = jnp.where(row_norms > 0, jnp.sqrt(row_norms), 1.0)
            # Scale columns
            col_norms = jnp.linalg.norm(scaled_A, axis=0, ord=ord)
            col_factors = alpha * jnp.where(col_norms > 0, jnp.sqrt(col_norms), 1.0)
            # Update
            d_r = d_r / row_factors
            d_c = d_c / col_factors
            scaled_A = scaled_A / row_factors[:, None]
            scaled_A = scaled_A / col_factors[None, :]

        # Check convergence: after scaling, row and column norms should be close to 1.
        new_row_norms = jnp.linalg.norm(scaled_A, axis=1, ord=ord)
        new_col_norms = jnp.linalg.norm(scaled_A, axis=0, ord=ord)
        term_criterion = jnp.maximum(
            1 - jnp.min(new_row_norms) / jnp.max(new_row_norms),
            1 - jnp.min(new_col_norms) / jnp.max(new_col_norms),
        )

        # Best termination criterion so far
        if term_criterion < best_criterion:
            best_criterion = term_criterion
            d_r_best = d_r
            d_c_best = d_c

        if term_criterion < tol:
            break

    # Get the best scaled matrix
    scaled_A_best = A * d_r_best[:, None]
    scaled_A_best = scaled_A_best * d_c_best[None, :]

    # Safeguard
    if safeguard:
        cond_A = jnp.linalg.cond(A)
        cond_scaled_A = jnp.linalg.cond(scaled_A_best)
        if cond_scaled_A > cond_A:
            scaled_A_best = scaled_A
            d_r_best = jnp.ones(A.shape[0])
            d_c_best = jnp.ones(A.shape[1])

    return scaled_A_best, d_r_best, d_c_best
