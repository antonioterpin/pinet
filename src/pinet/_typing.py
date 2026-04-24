"""Shared jaxtyping aliases for precise shape and dtype annotations.

Axis-name convention used across the library:

* ``B``            — batch dimension.
* ``n``            — primal dimension (number of decision variables).
* ``m``            — number of rows in an equality / inequality matrix.
* ``n_ineq``       — number of affine-inequality rows (slack-variable count).
* ``d_lifted``     — lifted primal dimension (``n`` + auxiliary / slack variables).

Each alias below names a common shape so call sites stay short and docstrings
no longer need to carry shape/dtype prose.
"""

from jaxtyping import Array, Bool, Float

BatchedPrimal = Float[Array, "B n 1"]
BatchedRHS = Float[Array, "B m 1"]
BatchedEqMatrix = Float[Array, "B m n"]
BatchedEqPinv = Float[Array, "B n m"]
BatchedIneqMatrix = Float[Array, "B n_ineq n"]
BatchedIneqBound = Float[Array, "B n_ineq 1"]
BatchedScalar = Float[Array, "B 1 1"]
BatchedLifted = Float[Array, "B d_lifted 1"]

ScalingVector = Float[Array, "B d 1"]

Mask1D = Bool[Array, "d"]

Matrix2D = Float[Array, "n_r n_c"]
Vector1D = Float[Array, "d"]

NLMatrix = Float[Array, "1 m n"]
NLScalarRHS = Float[Array, "B 1 1"]
NLLinearRHS = Float[Array, "1 1 n"]
