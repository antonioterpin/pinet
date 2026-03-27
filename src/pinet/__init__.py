"""Hard constraint neural network package."""

from .constants import (
    EQUILIBRATION_DEFAULT_COL_SCALING,
    EQUILIBRATION_DEFAULT_MAX_ITER,
    EQUILIBRATION_DEFAULT_ORD,
    EQUILIBRATION_DEFAULT_SAFEGUARD,
    EQUILIBRATION_DEFAULT_TOL,
    EQUILIBRATION_DEFAULT_UPDATE_MODE,
    PROJECTION_DEFAULT_CHECK_EVERY,
    PROJECTION_DEFAULT_CHECK_REDUCTION,
    PROJECTION_DEFAULT_MAX_ITER,
    PROJECTION_DEFAULT_OMEGA,
    PROJECTION_DEFAULT_SIGMA,
    PROJECTION_DEFAULT_TOL,
    SOC_CONSTRAINT_EPSILON,
)
from .constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    CartesianConstraint,
    ConstraintParser,
    EqualityConstraint,
    L2NormType,
    NonLinearConstraint,
    NonLinearConstraintType,
    SocConstraint,
    SOCType,
)
from .dataclasses import (
    BoxConstraintSpecification,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    NonLinearSpecification,
    ProjectionInstance,
    SocConstraintSpecification,
)
from .equilibration import ruiz_equilibration
from .project import Project
from .solver import build_iteration_step

__all__ = [
    # Constants
    "SOC_CONSTRAINT_EPSILON",
    "EQUILIBRATION_DEFAULT_TOL",
    "EQUILIBRATION_DEFAULT_MAX_ITER",
    "EQUILIBRATION_DEFAULT_ORD",
    "EQUILIBRATION_DEFAULT_COL_SCALING",
    "EQUILIBRATION_DEFAULT_UPDATE_MODE",
    "EQUILIBRATION_DEFAULT_SAFEGUARD",
    "PROJECTION_DEFAULT_SIGMA",
    "PROJECTION_DEFAULT_OMEGA",
    "PROJECTION_DEFAULT_CHECK_EVERY",
    "PROJECTION_DEFAULT_TOL",
    "PROJECTION_DEFAULT_MAX_ITER",
    "PROJECTION_DEFAULT_CHECK_REDUCTION",
    # Classes and functions
    "EqualityConstraint",
    "AffineInequalityConstraint",
    "BoxConstraint",
    "SocConstraint",
    "CartesianConstraint",
    "ConstraintParser",
    "ruiz_equilibration",
    "Project",
    "build_iteration_step",
    "ProjectionInstance",
    "EqualityConstraintsSpecification",
    "EquilibrationParams",
    "BoxConstraintSpecification",
    "SocConstraintSpecification",
    "NonLinearConstraint",
    "NonLinearConstraintType",
    "NonLinearSpecification",
    "L2NormType",
    "SOCType",
]
