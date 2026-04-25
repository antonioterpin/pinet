"""Beartype wrapper with PEP 484 numeric tower enabled.

Used as the typechecker for jaxtyping's import hook. Beartype defaults to
strict typing — ``ord=1`` is rejected where the annotation says ``float``
— but PEP 484 explicitly allows ``int`` wherever ``float`` is annotated.
Every static type-checker (pyright, mypy) already follows PEP 484 here,
so the library keeps idiomatic ``float`` annotations without forcing
callers to write ``float(1)`` at every site.

The module is imported by ``jaxtyping.install_import_hook`` before the
hook's finder is installed, so there is no circular-import risk even
though this file lives under ``pinet.*``.
"""

from beartype import BeartypeConf
from beartype import beartype as _beartype

# Parameterized decorator — calling ``beartype(func)`` returns the
# wrapped function, which is the shape jaxtyping's import hook expects.
beartype = _beartype(conf=BeartypeConf(is_pep484_tower=True))
