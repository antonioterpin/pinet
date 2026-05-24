"""Smoke test for the ``PINET_RUNTIME_CHECK`` import hook.

The hook is on by default — every function/class under ``pinet.*`` is
wrapped with ``beartype.beartype`` so shape/dtype mismatches surface as
``TypeCheckError`` at the call site instead of deep inside a jitted
function. ``PINET_RUNTIME_CHECK=0`` disables it.

The tests spawn subprocesses so the env var is set before ``pinet`` is
imported — the parent test process already imported ``pinet`` once and
the hook can't be retroactively (un)installed.
"""

import subprocess
import sys
import textwrap


def test_import_hook_catches_wrong_rank_by_default() -> None:
    """The default-on hook rejects a 1D ``x`` to ``ProjectionInstance``
    with ``TypeCheckError`` at construction.
    """
    script = textwrap.dedent(
        """
        import os, sys
        os.environ.pop("PINET_RUNTIME_CHECK", None)
        import jax.numpy as jnp
        import pinet
        try:
            pinet.ProjectionInstance(x=jnp.zeros((5,)))
        except Exception as exc:
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


def test_disable_via_env_var() -> None:
    """``PINET_RUNTIME_CHECK=0`` disables the hook so a malformed
    ``ProjectionInstance`` constructs silently — the existing
    ``validate()`` methods remain the primary gate.
    """
    script = textwrap.dedent(
        """
        import os, sys
        os.environ["PINET_RUNTIME_CHECK"] = "0"
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
