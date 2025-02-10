"""Generate simple QP problem.

Using the approach from the DC3 paper:
https://arxiv.org/pdf/2104.12225
"""

import os

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Save dataset flag
SAVE_DATASET = False

# Parameters setup
SEED = 42
NUM_VAR = 100
NUM_INEQ = 50
NUM_EQ = 50
NUM_EXAMPLES = 10000

# Setup keys
key = jax.random.PRNGKey(SEED)
key = jax.random.split(key, 5)

# Generate matrices
Q = jnp.expand_dims(
    jnp.diag(jax.random.uniform(key[0], shape=(NUM_VAR,), minval=0.0, maxval=1.0)),
    axis=0,
)
p = jax.random.uniform(key[1], shape=(1, NUM_VAR, 1), minval=0.0, maxval=1.0)
A = jax.random.normal(key[2], shape=(1, NUM_EQ, NUM_VAR))
X = jax.random.uniform(key[3], shape=(NUM_EXAMPLES, NUM_EQ, 1), minval=-1.0, maxval=1.0)
G = jax.random.normal(key[4], shape=(1, NUM_INEQ, NUM_VAR))
h = jnp.expand_dims(jnp.sum(jnp.abs(G @ jnp.linalg.pinv(A[0])), axis=1), axis=2)

if SAVE_DATASET:
    # Create the datasets directory if it doesn't exist
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")

    # Define the filename and save the dataset
    filename = os.path.join(
        datasets_dir,
        (
            f"SimpleQP_seed{SEED}_var{NUM_VAR}_ineq{NUM_INEQ}"
            f"_eq{NUM_EQ}_examples{NUM_EXAMPLES}.npz"
        ),
    )
    jnp.savez(filename, Q=Q, p=p, A=A, X=X, G=G, h=h)
