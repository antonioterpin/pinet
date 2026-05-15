"""Smoke test for the ``PINET_RUNTIME_CHECK`` import hook.

Verifies that setting ``PINET_RUNTIME_CHECK=1`` before importing ``pinet``
activates the jaxtyping import hook with beartype, so a shape/dtype
violation at a public API boundary raises a ``TypeCheckError`` at the
call site instead of surfacing deep inside a jitted function.

The test spawns a subprocess so the env var is set before ``pinet`` is
imported — the parent test process already imported ``pinet`` without
the hook.
"""

import subprocess
import sys
import textwrap


def test_import_hook_catches_wrong_rank_when_enabled() -> None:
    """With ``PINET_RUNTIME_CHECK=1``, constructing a ``ProjectionInstance``
    with a 1D ``x`` raises ``TypeCheckError`` instead of silently
    constructing an object that fails later.
    """
    script = textwrap.dedent(
        """
        import os, sys
        os.environ["PINET_RUNTIME_CHECK"] = "1"
        import jax.numpy as jnp
        import pinet
        try:
            pinet.ProjectionInstance(x=jnp.zeros((5,)))
        except Exception as exc:
            # Expected: TypeCheckError from the beartype wrapper.
            print(type(exc).__name__, flush=True)
            sys.exit(0)
        sys.exit(1)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        "Expected a TypeCheckError with hook enabled, got: "
        f"{proc.stdout!r} / {proc.stderr!r}"
    )
    assert "TypeCheckError" in proc.stdout, (
        f"Expected TypeCheckError, got: {proc.stdout!r}"
    )


def test_default_off_does_not_break_existing_imports() -> None:
    """Without ``PINET_RUNTIME_CHECK``, a malformed ``ProjectionInstance``
    constructs without a beartype check (the existing ``validate()``
    methods are still the primary gate).
    """
    script = textwrap.dedent(
        """
        import os, sys
        os.environ.pop("PINET_RUNTIME_CHECK", None)
        import jax.numpy as jnp
        import pinet
        # Constructs without a TypeCheckError even with a 1D x.
        inst = pinet.ProjectionInstance(x=jnp.zeros((5,)))
        print(type(inst).__name__, flush=True)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ProjectionInstance" in proc.stdout
