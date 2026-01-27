"""Abstract class for non-linear constraints."""

from typing import Optional

import jax.numpy as jnp

from pinet.constraints.base import Constraint
from pinet.constraints.non_linear_types import (
    L2NormType,
    NonLinearConstraintType,
    SOCType,
)


class NonLinearConstraint(Constraint):
    """Abstract class for non-linear constraints.

    This class describes constraints of the form:
    g(A @ y + a) <= f @ y + b,
    where g is a convex function.
    """

    def __init__(
        self,
        nl_type: NonLinearConstraintType,
        A: Optional[jnp.ndarray] = None,
        a: Optional[jnp.ndarray] = None,
        f: Optional[jnp.ndarray] = None,
        b: Optional[jnp.ndarray] = None,
        dim: Optional[int] = None,
        var_a: Optional[bool] = False,
        var_b: Optional[bool] = False,
    ) -> None:
        """Initialize the non-linear constraint.

        Args:
            A (Optional[jnp.ndarray]): Matrix for linear transformation before g.
                Shape (batch_size, constraint_dim, variable_dim).
            a (Optional[jnp.ndarray]): Offset vector before non-linear function g.
                Shape (batch_size, constraint_dim, 1).
            f (Optional[jnp.ndarray]): Linear coefficients on the right-hand side.
                Shape (batch_size, 1, variable_dim).
            b (Optional[jnp.ndarray]): Constant offset on the right-hand side.
                Shape (batch_size, 1, 1).
            dim (Optional[int]): Variable dimension if A and f are not provided.
            var_a (Optional[bool]): Whether 'a' is a variable. Defaults to False.
            var_b (Optional[bool]): Whether 'b' is a variable. Defaults to False.
            nl_type (NonLinearConstraintType): Type of non-linear constraint.
        """
        # Check for dimension
        if (A is None) and (f is None) and (dim is None):
            raise ValueError(
                "At least one of A, f, or dim must be provided to "
                "determine variable dimension."
            )
        else:
            if A is not None:
                self._dim = A.shape[-1]
            elif f is not None:
                self._dim = f.shape[-1]
            else:
                self._dim = dim
        # Make sure A matrix exists
        if A is None:
            A = jnp.eye(self.dim).reshape(1, -1, -1)

        # Validate nl_type
        if not isinstance(nl_type, NonLinearConstraintType):
            raise ValueError(
                f"nl_type must be a NonLinearConstraintType instance, "
                f"got {type(nl_type)}"
            )

        # Parse nl_type: if L2Norm with RHS (f is present), convert to SOC
        if isinstance(nl_type, L2NormType) and f is not None:
            nl_type = SOCType()

        self._A = A
        self._a = a
        self._f = f
        self._b = b
        self.var_a = var_a
        self.var_b = var_b
        self._nl_type = nl_type

        # Validate batch size consistency
        batch_sizes = []
        if A is not None:
            batch_sizes.append(A.shape[0])
        if a is not None:
            batch_sizes.append(a.shape[0])
        if f is not None:
            batch_sizes.append(f.shape[0])
        if b is not None:
            batch_sizes.append(b.shape[0])

        if batch_sizes:
            non_one_sizes = [size for size in batch_sizes if size != 1]
            if len(set(non_one_sizes)) > 1:
                raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")

        # Validate dimension consistency
        if A is not None and a is not None:
            if A.shape[1] != a.shape[1]:
                raise ValueError(
                    f"A and a must have same constraint dimension: "
                    f"{A.shape[1]} vs {a.shape[1]}"
                )

        if A is not None and f is not None:
            if A.shape[2] != f.shape[2]:
                raise ValueError(
                    f"A and f must have same variable dimension: "
                    f"{A.shape[2]} vs {f.shape[2]}"
                )

        if f is not None and b is not None:
            if f.shape[1] != b.shape[1]:
                raise ValueError(
                    f"f and b must have same constraint dimension: "
                    f"{f.shape[1]} vs {b.shape[1]}"
                )

        # Validate that b is scalar if provided
        if b is not None:
            if b.shape[1] != 1:
                raise ValueError(
                    f"b must be scalar (shape should be (batch_size, 1, 1)): "
                    f"got {b.shape}"
                )

    @property
    def A(self) -> Optional[jnp.ndarray]:
        """Matrix for linear transformation before non-linear function g.

        Returns:
            Optional[jnp.ndarray]: Shape (batch_size, constraint_dim, variable_dim)
        """
        return self._A

    @property
    def a(self) -> Optional[jnp.ndarray]:
        """Offset vector before non-linear function g.

        Returns:
            Optional[jnp.ndarray]: Shape (batch_size, constraint_dim, 1)
        """
        return self._a

    @property
    def f(self) -> Optional[jnp.ndarray]:
        """Linear coefficients on the right-hand side.

        Returns:
            Optional[jnp.ndarray]: Shape (batch_size, 1, variable_dim)
        """
        return self._f

    @property
    def b(self) -> Optional[jnp.ndarray]:
        """Constant offset on the right-hand side.

        Returns:
            Optional[jnp.ndarray]: Shape (batch_size, 1, 1)
        """
        return self._b

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return 1

    @property
    def nl_type(self) -> NonLinearConstraintType:
        """Return the type of non-linear constraint.

        Returns:
            NonLinearConstraintType: The type of non-linear constraint.
        """
        return self._nl_type
