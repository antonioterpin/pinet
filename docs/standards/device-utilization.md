# Device utilization

## GPU management

- When running on a multi-GPU host, run `nvidia-smi` first to check availability.
- Unless specified differently by the user, run on the first free GPU.
- Make sure to kill running processes you've started and don't leave them hanging at the end of your work.

## JAX device placement

- Do not hard-code device assignment in library code. Let the caller decide via
  `jax.default_device(...)` or environment flags (`JAX_PLATFORMS`, `CUDA_VISIBLE_DEVICES`).
- Tests that must run on CPU should set the device explicitly and restore it on teardown.
