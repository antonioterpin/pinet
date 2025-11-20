"""Projection layer implentation for SOC constraints."""

import cvxpy as cp
import jax.numpy as jnp
from cvxpylayers.jax import CvxpyLayer
from jax import custom_vjp as _custom_vjp
from jax import jit, lax, vjp
from jax.scipy.sparse.linalg import bicgstab


def _project_soc(z: jnp.ndarray) -> jnp.ndarray:
    """Project onto the second-order cone (SOC) constraint.

    Args:
        z (jnp.ndarray):
            Input array of shape (B, m + 1, 1) where the last column is the SOC radius.

    Returns:
        jnp.ndarray:
            Projected array of the same shape as `z`, satisfying the SOC constraint.
    """
    eps = 1e-12
    u, t = z[:, :-1], z[:, -1:]
    norm_u = jnp.linalg.norm(u, axis=1, keepdims=True)

    proj1 = z
    proj2 = jnp.zeros_like(z)
    proj3 = (
        (t + norm_u)
        / 2
        * jnp.concatenate((u / (norm_u + eps), jnp.ones_like(t)), axis=1)
    )

    when1 = norm_u <= t
    when2 = norm_u <= -t

    return jnp.where(when1, proj1, jnp.where(when2, proj2, proj3))


project_soc = jit(_project_soc)


def build_project_pinet(
    A: jnp.ndarray,
    sigma: float,
    omega: float,
    n: int,
    n_iter_forward: int,
    n_iter_backward: int,
    use_custom_vjp: bool,
):
    """Build the iteration and result retrieval step for pinet.

    Args:
        A (jnp.ndarray): Equality constraint matrix.
        sigma (float): ADMM parameter sigma.
        omega (float): ADMM parameter omega.
        n (int): Dimension of the primal variables.
        n_iter_forward (int): Number of forward iterations.
        n_iter_backward (int): Number of backward iterations.
        use_custom_vjp (bool): Whether to use custom VJP for the projection.

    Returns:
        Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
        ]:
            Projection function that takes in the governing sequence,
            raw input, and right-hand side, and returns the projected
            value and final governing sequence.
    """
    if use_custom_vjp:
        custom_vjp = _custom_vjp
    else:

        def custom_vjp(f):
            """No op."""
            return f

    A_inv = jnp.linalg.pinv(A)

    def project_pinv_vb(xs: jnp.ndarray, b: jnp.ndarray):
        """Project onto the pseudo-inverse of the augmented matrix A.

        Args:
            xs (jnp.ndarray):
                Input array of shape (B, m + n, 1) where the first n columns
                are the primal variables and the last m columns are residuals.

            b (jnp.ndarray):
                Right-hand side of the equality constraints, shape (B, m, 1).
        """
        return xs - A_inv @ (A @ xs - b)

    def step_iteration(yraw: jnp.ndarray, sk: jnp.ndarray, b: jnp.ndarray):
        zk = project_pinv_vb(sk, b)
        reflect = 2 * zk - sk
        toproj = (reflect - 2 * sigma * yraw) / (1 + 2 * sigma)
        tk1 = toproj[:, :n]
        tk2 = project_soc(toproj[:, n:])
        tk = jnp.concatenate((tk1, tk2), axis=1)
        return sk + omega * (tk - zk)

    def step_final(s, b):
        """Retrieve the final result from the forward step.

        Args:
            s (jnp.ndarray): Governing sequence, shape (B, m + n, 1).
            b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).

        Returns:
            jnp.ndarray: Final projected value, shape (B, m + n, 1).
        """
        return project_pinv_vb(s, b)

    @custom_vjp
    def project(
        s0: jnp.ndarray,
        yraw: jnp.ndarray,
        b: jnp.ndarray,
    ):
        """Project the raw input onto the feasible set defined by the constraints.

        Args:
            s0 (jnp.ndarray): Initial governing sequence, shape (B, m + n, 1).
            yraw (jnp.ndarray): Raw input array, shape (B, m + n, 1).
            b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).

        Returns:
            tuple:
                - zk1 (jnp.ndarray): Projected value, shape (B, m + n, 1).
                - sk (jnp.ndarray): Final governing sequence, shape (B, m + n, 1).
        """
        sk = s0
        sk, _ = lax.scan(
            lambda sk, _: (
                step_iteration(yraw.reshape((yraw.shape[0], yraw.shape[1], 1)), sk, b),
                None,
            ),
            sk,
            xs=None,
            length=n_iter_forward,
        )

        # NOTE: There is no auxiliary variable in this case
        zk1 = step_final(sk, b).reshape(yraw.shape)

        # return values and residuals
        return zk1, sk

    def _project_fwd(s0: jnp.ndarray, yraw: jnp.ndarray, b: jnp.ndarray):
        """Forward pass of the projection function.

        Args:
            s0 (jnp.ndarray): Initial governing sequence, shape (B, m + n, 1).
            yraw (jnp.ndarray): Raw input array, shape (B, m + n, 1).
            b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).

        Returns:
            - tuple:
                - zk1 (jnp.ndarray): Projected value, shape (B, m + n, 1).
                - sk (jnp.ndarray): Final governing sequence, shape (B, m + n, 1).
            - tuple:
                - (sk, yraw, b): Residuals for the backward pass.
        """
        zk1, sk = project(s0, yraw, b)
        return (zk1, sk), (sk, yraw.reshape((yraw.shape[0], yraw.shape[1], 1)), b)

    def _project_bwd(residuals: tuple, cotangent: tuple):
        """Backward pass of the projection function.

        Args:
            residuals (tuple): Residuals from the forward pass, containing:
                - sk (jnp.ndarray): Governing sequence, shape (B, m + n, 1).
                - yraw (jnp.ndarray): Raw input array, shape (B, m + n, 1).
                - b (jnp.ndarray):
                    Right-hand side of the equality constraints, shape (B, m, 1).

            cotangent (tuple): Cotangent vector from the backward pass, containing:
                - cotangent_zk1 (jnp.ndarray):
                    Cotangent vector for the projected value, shape (B, m + n, 1).
                - cotangent_sk (jnp.ndarray):
                    Cotangent vector for the governing sequence, shape (B, m + n, 1).

        Returns:
            tuple:
                - None: Placeholder for the vjp wrt to sk (the DRA governing sequence).
                - thevjp (jnp.ndarray):
                    The vjp wrt to yraw, shape (B, m + n, 1).
                - None: Placeholder for the vjp wrt to b
                    (the right-hand side of the equality constraints).
        """
        sk, yraw, b = residuals
        cotangent_zk1, _ = cotangent

        # Compute the vjp of the iteration step wrt to the DRA governing sequence
        _, iteration_vjp = vjp(lambda xx: step_iteration(yraw, xx, b), sk)
        # Compute the vjp of the iteration step wrt to the value to be projected
        _, iteration_vjp2 = vjp(lambda xx: step_iteration(xx, sk, b), yraw)
        # Compute the vjp of the final step wrt to DRA governing sequence
        _, equality_vjp = vjp(lambda xx: step_final(xx, b), sk)

        cotangent_eq_6 = equality_vjp(cotangent_zk1)[0]

        def Aop(xx):
            return xx - iteration_vjp(xx)[0]

        cotangent_eq_7 = bicgstab(Aop, cotangent_eq_6, maxiter=n_iter_backward)[0]

        thevjp = iteration_vjp2(cotangent_eq_7)[0]

        # We only care about the vjp wrt to yraw
        # So, we return None for the vjp wrt to sk (the DRA governing sequence)
        # and None for the vjp wrt to b (the right-hand side of the equality constraints)
        return (None, thevjp, None)

    if use_custom_vjp:
        project.defvjp(_project_fwd, _project_bwd)

    return jit(project)


def build_project_cvxpy(Aaug: jnp.ndarray, n: int, m: int, eps=1e-3):
    """Build the projection layer with cvxpyalayers.

    Args:
        Aaug (jnp.ndarray): Augmented equality constraint matrix.
        n (int): Dimension of the primal variables.
        m (int): Dimension of the slack variables.
        eps (float): Solver tolerance.

    Returns:
        Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
        ]:
            Projection function that takes in the governing sequence,
            raw input, and right-hand side, and returns the projected
            value and final governing sequence.
    """
    x_var = cp.Variable(n)
    s_var = cp.Variable(m)
    xproj_par = cp.Parameter(n)
    sproj_par = cp.Parameter(m)
    b_par = cp.Parameter(m)
    constraints = [
        Aaug @ cp.hstack([x_var, s_var]) == b_par,
        cp.SOC(s_var[-1], s_var[:-1]),
    ]
    problem_cvxpy = cp.Problem(
        objective=cp.Minimize(
            cp.sum_squares(x_var - xproj_par) + cp.sum_squares(s_var - sproj_par)
        ),
        constraints=constraints,
    )
    assert problem_cvxpy.is_dpp()

    cvxpylayer = CvxpyLayer(
        problem_cvxpy,
        parameters=[xproj_par, sproj_par, b_par],
        variables=[x_var, s_var],
    )

    def project(_, xx, bb):
        projection = cvxpylayer(
            xx[:, :n, :].reshape(-1, n),
            xx[:, n:, :].reshape(-1, m),
            bb.reshape(-1, m),
            solver_args={
                "solve_method": "CLARABEL",
                "tol_gap_abs": eps,
                "tol_gap_rel": eps,
                "tol_feas": eps,
                "verbose": False,
            },
        )
        return jnp.concatenate(projection, axis=1)[..., None], _

    return project


def build_projection_layer(
    A: jnp.ndarray,
    n: int,
    m: int,
    hyperparameters: dict,
    method: str,
):
    """Build the projection layer based on the specified method.

    Args:
        A (jnp.ndarray): Equality constraint matrix.
        n (int): Dimension of the primal variables.
        m (int): Dimension of the slack variables.
        hyperparameters (dict): Hyperparameters for the projection layer.
        method (str): Projection method to use ("pinet" or "cvxpylayers").

    Returns:
        Callable:
            Projection function that takes in the governing sequence,
            raw input, and right-hand side, and returns the projected
            value and final governing sequence.
    """
    if method == "pinet":
        return build_project_pinet(
            A=A,
            n=n,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter_forward=hyperparameters["n_iter_train"],
            n_iter_backward=hyperparameters["n_iter_bwd"],
            use_custom_vjp=hyperparameters["use_custom_vjp"],
        )
    elif method == "cvxpylayers":
        return build_project_cvxpy(Aaug=A, n=n, m=m, eps=hyperparameters["cvxpy_tol"])
    else:
        raise ValueError(f"Unknown projection method: {method}")
