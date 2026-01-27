"""Non-linear constraint types."""

from abc import ABC


class NonLinearConstraintType(ABC):
    """Base class for non-linear constraint types."""


class L2NormType(NonLinearConstraintType):
    """L2 norm constraint type."""


class SOCType(NonLinearConstraintType):
    """Second-order cone (SOC) constraint type."""
