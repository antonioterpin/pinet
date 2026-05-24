# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |
| < 0.1   | no        |

Only the latest released minor line of `pinet-hcnn` receives security
fixes. Update this table when the release version changes.

## Reporting a vulnerability

Please report security issues privately to **antonio.terpin@gmail.com**.

Include where possible:

- a minimal reproduction (inputs / constraint specification and the
  sequence of calls);
- the `pinet-hcnn` version (`pip show pinet-hcnn`) and the JAX / jaxlib
  versions;
- whether a CPU-only or GPU (CUDA) backend is required to trigger it.

The maintainer will acknowledge receipt within 7 days and aim to ship a
fix and a coordinated disclosure within 30 days for confirmed issues.
Public discussion happens after a fix is released.

## Scope

pinet is a numerical library (a JAX output layer that projects onto convex
constraint sets). It exposes no network services and opens no sockets, so
the attack surface is small:

- **Dependencies.** The most likely source of a security issue is a
  vulnerable transitive dependency. These are monitored by the scheduled
  `pip-audit` workflow and bumped via Dependabot; reports of an unpatched
  advisory affecting the runtime tree are in scope.
- **Untrusted artifacts.** Loading untrusted datasets, model checkpoints,
  or pickled objects (e.g. via `h5py`, `torch`, or `wandb`) inherits the
  security properties of those libraries and is the caller's
  responsibility — it is outside this policy unless pinet itself
  mishandles the data in a way that escalates beyond the underlying loader.
- **Numerical behavior.** Incorrect or non-converging projections are
  correctness bugs, not security vulnerabilities; please file them as
  regular issues.

Crashes or memory-safety problems reachable purely through pinet's public
Python API with well-formed array inputs are in scope.
