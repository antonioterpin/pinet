"""Shared jaxtyping aliases for precise shape and dtype annotations.

Axis-name convention used across the library:

* ``B``            — batch dimension. Prefixed with ``#`` in the aliases
  so a size-1 batch broadcasts against a full batch (the library's own
  batch-compatibility asserts enforce "equal OR one of them is 1").
* ``n``            — primal dimension (number of decision variables).
* ``m``            — number of rows in an equality / inequality matrix.
* ``n_ineq``       — number of affine-inequality rows (slack-variable count).
* ``d_lifted``     — lifted primal dimension (``n`` + auxiliary / slack variables).
* ``n_r`` / ``n_c`` — row / column dimensions of an unbatched equilibration
  matrix.

Each alias below names a common shape so call sites stay short and docstrings
no longer need to carry shape/dtype prose.
"""

import numpy as np
from jaxtyping import Array, Bool, Float, Real

# Library entry points routinely receive ``numpy.ndarray`` from user code
# (CVXPY, manually constructed test fixtures, etc.) and JAX silently
# promotes them at the first arithmetic op. ``ArrayLike`` widens
# jaxtyping's first-slot type so beartype accepts either backend without
# forcing callers to ``jnp.asarray`` everywhere.
ArrayLike = Array | np.ndarray

# The ``#B`` prefix marks the batch axis as broadcast-compatible, so a
# function that accepts both a per-instance tensor (``B=10``) and a
# shared tensor (``B=1``) passes jaxtyping's check. The semantic
# "sizes must match or one must be 1" invariant is enforced by the
# ``__init__`` asserts in the constraint classes.
#
# ``Real`` (instead of ``Float``) covers both floating and integer dtypes so
# callers can pass ``jnp.array([-1, 1])`` (int64) without a manual ``.astype``.
# JAX promotes these on arithmetic, so the library never cares about the
# distinction once the array enters a computation.
BatchedPrimal = Real[ArrayLike, "#B n 1"]
BatchedRHS = Real[ArrayLike, "#B m 1"]
BatchedEqMatrix = Real[ArrayLike, "#B m n"]
BatchedEqPinv = Real[ArrayLike, "#B n m"]
BatchedIneqMatrix = Real[ArrayLike, "#B n_ineq n"]
BatchedIneqBound = Real[ArrayLike, "#B n_ineq 1"]
BatchedScalar = Real[ArrayLike, "#B 1 1"]
BatchedLifted = Real[ArrayLike, "#B d_lifted 1"]

# Distinct axis names for row vs column scaling — jaxtyping would otherwise
# tie them to the same length when a function accepts both (e.g. the ADMM
# ``_project_general`` path).
RowScaling = Real[ArrayLike, "#B m_scale 1"]
ColScaling = Real[ArrayLike, "#B n_scale 1"]

Mask1D = Bool[ArrayLike, "d"]

Matrix2D = Real[ArrayLike, "n_r n_c"]
Vector1D = Real[ArrayLike, "d"]

NLMatrix = Real[ArrayLike, "1 m n"]
NLScalarRHS = Real[ArrayLike, "#B 1 1"]
NLLinearRHS = Real[ArrayLike, "1 1 n"]

# Scalar-like: Python ``float`` or a 0-dim JAX array (what ``jnp.array(1.0)``
# or ``jnp.float64(...)`` produce). Library knobs like ``sigma`` / ``omega``
# accept both forms at runtime.
ScalarLike = float | Float[Array, ""]
