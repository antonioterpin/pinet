"""Non-linear constraint types."""

from enum import Enum


class NonLinearConstraintType(Enum):
    """Enumeration of supported non-linear constraint types."""

    L2_NORM = "l2_norm"
    SOC = "soc"


# Aliases for backward compatibility
L2NormType = NonLinearConstraintType.L2_NORM
SOCType = NonLinearConstraintType.SOC
