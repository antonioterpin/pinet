"""Patch cvxpylayers to work with JAX >= 0.6.

Temporary shim for packages that still import jax.core.Primitive on JAX >= 0.6.
NOTE: We can remove this once cvxpylayers has been released with the patch.
"""

import sys
import types
from typing import Any, cast

try:
    import jax

    jax_module = cast(Any, jax)
    jax_extend = getattr(jax_module, "extend", None)

    # Only do this if jax.extend exists (JAX >= 0.6)
    if jax_extend is not None:
        # Build a fake module "jax.core" that mirrors jax.extend.core
        shim = types.ModuleType("jax.core")
        shim.__dict__.update(jax_extend.core.__dict__)
        sys.modules["jax.core"] = shim
        jax_module.core = shim
except Exception:
    pass
