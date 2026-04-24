"""Non-linear constraint types."""

from enum import Enum


class NonLinearConstraintType(Enum):
    """Enumeration of supported non-linear constraint types.

    Attributes:
        L2_NORM: Second-order cone constraint on the L2 norm.
        SOC: Second-order cone constraint.
    """

    L2_NORM = "l2_norm"
    SOC = "soc"


# Aliases for backward compatibility
L2NormType = NonLinearConstraintType.L2_NORM
SOCType = NonLinearConstraintType.SOC
