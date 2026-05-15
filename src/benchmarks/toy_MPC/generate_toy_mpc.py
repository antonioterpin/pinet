"""Generate toy MPC problem data."""

import pathlib
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

# Enable saving the dataset
save_results = True
# Define parameters
# We are planning in 2D
base_dim = 2
# Horizon
horizon = 20
# State and input dimension
dimx = base_dim * (horizon + 1)
dimu = base_dim * horizon
# Objective tradeoff
alpha = 1.0
# One instance of initial point
x0 = jnp.array([0, 4])
# Target point
xhat = jnp.array([3, -12])
# The decision variable looks like
# [x_0, x_1, ..., x_T, u_0, u_1, ..., u_{T-1}] = [x; u]
# Setup equality constraints
# They look like [A_x, A_u] [x; u] == [x_0; 0]
a_x = jnp.diag(jnp.ones(horizon + 1)) - jnp.diag(jnp.ones(horizon), k=-1)
a_x = jnp.kron(a_x, jnp.eye(base_dim))
a_u = jnp.diag(jnp.ones(horizon), k=-1)[:, :-1]
a_u = jnp.kron(a_u, jnp.eye(base_dim))
a_mat = jnp.concatenate((a_x, a_u), axis=1)
# Setup Inequality constraints
# -boundx <= x <= boundx
# -boundu <= u <= boundu
boundx = 10
boundu = 1
lbx = -boundx * jnp.ones((dimx, 1))
ubx = boundx * jnp.ones((dimx, 1))
lbu = -boundu * jnp.ones((dimu, 1))
ubu = boundu * jnp.ones((dimu, 1))
# Solve one instance of the MPC problem
z = cp.Variable(dimx + dimu)
xinit = cp.Parameter(base_dim)
constraints = [
    a_mat @ z == cp.hstack([xinit, jnp.zeros(dimx - base_dim)]),
    z[:dimx] >= lbx,
    z[:dimx] <= ubx,
    z[dimx:] >= lbu,
    z[dimx:] <= ubu,
]
objective = cp.Minimize(
    cp.sum_squares(z[:dimx] - jnp.tile(xhat, horizon + 1))
    + alpha * cp.sum_squares(z[dimx:])
)
problem = cp.Problem(objective, cast(list[CvxpyConstraint], constraints))
# Setup problem parameter
xinit.value = np.array(x0)
problem.solve()

# Solve many problems
seed = 42
NUM_EXAMPLES = 10000
# Randomly generate dataset initial conditions
key = jax.random.PRNGKey(seed)
x0set = jax.random.uniform(
    key, shape=(NUM_EXAMPLES, base_dim), minval=-boundx, maxval=boundx
)
objectives = jnp.zeros(NUM_EXAMPLES)
y_star = jnp.zeros((NUM_EXAMPLES, dimx + dimu))
print(f"Solving {NUM_EXAMPLES} problem instances")
for idx in tqdm(range(NUM_EXAMPLES)):
    xinit.value = np.array(x0set[idx])
    problem.solve()
    objectives = objectives.at[idx].set(problem.value)
    y_star = y_star.at[idx, :].set(jnp.array(z.value))
# Generate matrices with appropriate shape
a_mat = a_mat.reshape((1, a_mat.shape[0], a_mat.shape[1]))
lbxs = lbx.reshape(1, -1, 1)
ubxs = ubx.reshape(1, -1, 1)
lbus = lbu.reshape(1, -1, 1)
ubus = ubu.reshape(1, -1, 1)
x0sets = x0set.reshape((NUM_EXAMPLES, base_dim, 1))
xhat = xhat.reshape((base_dim, 1))
if save_results:
    datasets_path = pathlib.Path(__file__).parent.resolve() / "datasets"
    filename = f"toy_MPC_seed{seed}_examples{NUM_EXAMPLES}.npz"
    path = datasets_path / filename
    jnp.savez(
        path,
        a_mat=a_mat,
        lbxs=lbxs,
        ubxs=ubxs,
        lbus=lbus,
        ubus=ubus,
        x0sets=x0sets,
        xhat=xhat,
        objectives=objectives,
        y_star=y_star,
        horizon=horizon,
        base_dim=base_dim,
        alpha=alpha,
    )
