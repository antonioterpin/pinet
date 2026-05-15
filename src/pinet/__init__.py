"""Hard constraint neural network package.

Setting ``PINET_RUNTIME_CHECK=1`` before importing ``pinet`` installs the
jaxtyping import hook, which wraps every function and class defined under
``pinet.*`` with ``beartype.beartype`` so the shape/dtype annotations in
``pinet._typing`` are enforced at trace time — a mismatched argument
surfaces at the call site instead of deep inside a jitted function.

The hook is **opt-in** for now. With the current annotations enabled the
hook fires on broadcast-compatible (but not axis-equal) calls and on the
negative-path validation tests that deliberately pass malformed input to
check for a ``ValueError``. Flipping the default to on requires widening
the annotations and updating those tests; tracked as a follow-up of #99.
Beartype checks are O(1) on the hot path, so overhead lives on the first
call / trace path.
"""

import contextlib
import os

from jaxtyping import install_import_hook

_RUNTIME_CHECK = os.environ.get("PINET_RUNTIME_CHECK", "0") == "1"

# The hook must wrap every submodule imported below so their annotations
# are beartype-checked. Exiting the context removes the hook, so later
# user-level ``import pinet.x`` calls are not instrumented twice.
_hook = (
    install_import_hook("pinet", "beartype.beartype")
    if _RUNTIME_CHECK
    else contextlib.nullcontext()
)

with _hook:
    from .constants import Constants
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
    "AffineInequalityConstraint",
    "BoxConstraint",
    "BoxConstraintSpecification",
    "CartesianConstraint",
    "Constants",
    "ConstraintParser",
    "EqualityConstraint",
    "EqualityConstraintsSpecification",
    "EquilibrationParams",
    "L2NormType",
    "NonLinearConstraint",
    "NonLinearConstraintType",
    "NonLinearSpecification",
    "Project",
    "ProjectionInstance",
    "SOCType",
    "SocConstraint",
    "SocConstraintSpecification",
    "build_iteration_step",
    "ruiz_equilibration",
]
