"""Hard constraint neural network package.

The jaxtyping import hook wraps every function and class defined under
``pinet.*`` with ``beartype.beartype`` so the shape/dtype annotations in
``pinet._typing`` are enforced at trace time — a mismatched argument
surfaces at the call site instead of deep inside a jitted function.

The hook is **on by default**. Set ``PINET_RUNTIME_CHECK=0`` before
importing ``pinet`` to disable it (useful for latency-sensitive
production runs); beartype checks are O(1) on the hot path, so the
overhead lives on the first call / trace path and disappears once jit
compiles the wrapped function.
"""

import contextlib
import os

from jaxtyping import install_import_hook

# Pre-load the configured beartype decorator before the import hook
# resolves its dotted path. Importing ``_typecheck_config`` eagerly here
# sidesteps the circular import that would otherwise happen when
# ``install_import_hook`` tries to look up ``pinet._typecheck_config``
# while ``pinet/__init__.py`` itself is still executing.
from . import _typecheck_config as _typecheck_config

_RUNTIME_CHECK = os.environ.get("PINET_RUNTIME_CHECK", "1") != "0"

# ``pinet._typecheck_config.beartype`` is beartype pre-configured with
# PEP 484's numeric tower (int-as-float). Pass the dotted path so
# jaxtyping imports it once (the import is already a no-op thanks to the
# line above) and then uses the already-parameterised decorator as-is.
_hook = (
    install_import_hook("pinet", "pinet._typecheck_config.beartype")
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
