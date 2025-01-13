"""Standalone code for preliminary evaluation of iterated projection idea."""

# %%
import os
import time

import cvxpy as cp
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import scipy

jax.config.update("jax_enable_x64", True)


os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
# %% Problem sizes
# Vector dimension
N = 100
# Number of equality constraints
neq = 50
# Number of inequality constraints
nineq = 100
# %% Generate random problem data
# For initial simplicity, we will consider the intersection of
# a linear subspace with semi-box constraints above zero

# Equality LHS
# (generate in sparse form to control density)
sparse_matrix = scipy.sparse.random(
    neq, N, density=0.10, format="csr", data_rvs=np.random.random
)
A = np.array(sparse_matrix.todense())

# Equality RHS
b = np.zeros(neq)

# Inequality LHS
C = np.eye(N)

# Inequality RHS
d = np.ones(N)
# %% Generate random infeasible point
infeas = 1 * np.random.rand(N)
# %% Solve projection as QP
# Solver tolerance
tol = 1e-4
# Projected point
projqp = cp.Variable(N)
# Constraints
constraintsqp = [A @ projqp == b]
constraintsqp += [C @ projqp <= d]
# Objective
objectiveqp = cp.Minimize(cp.sum_squares(projqp - infeas))
# Define problem
problemqp = cp.Problem(objective=objectiveqp, constraints=constraintsqp)
# Solve with specified tolerance
problemqp.solve(solver=cp.OSQP, ignore_dpp=True, eps_abs=tol, eps_rel=tol, verbose=True)
# Extract objective function value and solution time
print(objectiveqp.value)
print(problemqp.solver_stats.solve_time)
# %% Define projection functions
# Function that projects onto a subspace
# Flag to choose between pseudo-inverting and factorizing
factorize = True
if not (factorize):
    # Compute pseudo-inverse
    Ap = jnp.linalg.pinv(A)

    @jax.jit
    def project_subspace(A, b, Ap, infeas):
        """Projects onto the subspace A @ x = b. Uses the pseudoinverse of A."""
        return infeas - Ap @ (A @ infeas - b)

    project_subspace(A, b, Ap, infeas)
else:
    # Compute Gram matrix for the subspace projection
    Agram = A @ A.T
    # Compute Cholesky factorization (pretty efficient for PSD matrices)
    cfac = jax.scipy.linalg.cho_factor(Agram, lower=False)[0]

    @jax.jit
    def project_subspace(A, b, cfact, infeas):
        """Projects onto the subspace A @ x = b. Uses a factorization of A."""
        return infeas - A.T @ jax.scipy.linalg.cho_solve((cfact, False), A @ infeas - b)

    project_subspace(A, b, cfac, infeas)


# Function that projects onto a box
@jax.jit
def project_box(infeas, d):
    """Projects on a semi-infinite box."""
    return jnp.clip(infeas, max=d)


project_box(infeas, d)

# This includes the factorization time
# (not sure if this is done in existing benchmarks though)
start = time.time()
if not (factorize):
    Ap = jnp.linalg.pinv(A)
else:
    Agram = A @ A.T
    cfac = cfac = jax.scipy.linalg.cho_factor(Agram, lower=False)[0]

# Initialize iterates
xk = infeas.copy()
pk = jnp.zeros(N)
qk = jnp.zeros(N)
yk = jnp.zeros(N)
# Set number of iterations
niter = 3
# AAMR
alpha = 0.9
beta = 0.8
# Timing without the factorization/pseudo-inversion
start_in = time.time()
for ii in range(niter):
    if not (factorize):
        qk = 2 * beta * (project_subspace(A, b, Ap, xk + infeas) - infeas) - xk
    else:
        qk = 2 * beta * (project_subspace(A, b, cfac, xk + infeas) - infeas) - xk
    pk = 2 * beta * (project_box(qk + infeas, d) - infeas) - qk
    xk = (1 - alpha) * xk + alpha * pk

    # Different order of the box and subspace projections
    # qk = 2 * beta * (project_box(xk  + infeas, d) - infeas) - xk
    # pk = 2 * beta * (project_subspace(A, b, cfac, qk + infeas)  - infeas) - qk
    # xk = (1-alpha) * xk + alpha * pk

# Get actual projected point from auxiliary iterates
if not (factorize):
    xk = project_subspace(A, b, Ap, xk + infeas)
else:
    xk = project_subspace(A, b, cfac, xk + infeas)

end = time.time()
print("Pure iteration time: {} [ms]".format((end - start_in) * 1e3))
print("Factorization+iteration time: {} [ms]".format((end - start) * 1e3))
# %%
print("================ Iterative ================")
print("Distance: \t {0}".format(np.linalg.norm(projqp.value - xk, ord=np.inf)))
print("Time: \t\t {0} [ms]".format(1e3 * (end - start)))
print("Cost (norm): \t {0}".format(np.linalg.norm(xk - infeas)))
print("Equality: \t {0}".format(np.abs(A @ xk - b).max()))
print("Inequality: \t {0}".format(np.maximum(C @ xk - d, 0).max()))
# %%
print("================ Solver ================")
print("Time: \t\t {0} [ms]".format(1e3 * problemqp.solver_stats.solve_time))
print("Cost (norm): \t {0}".format(np.linalg.norm(projqp.value - infeas)))
print("Equality: \t {0}".format(np.abs(A @ projqp.value - b).max()))
print("Inequality: \t {0}".format(np.maximum(C @ projqp.value - d, 0).max()))
# %% Computing the VJP of the subspace projection
# VJP on the linear system solver
if not (factorize):

    def psub_wrapper(x):
        """Wrapper for project subspace."""
        return project_subspace(A, b, Ap, x)

else:

    def psub_wrapper(x):
        """Wrapper for project subspace."""
        return project_subspace(A, b, cfac, x)


primals, psub_vjp = jax.vjp(psub_wrapper, infeas)
psub_vjp_jit = jax.jit(psub_vjp)

# Test cotangent vector
cotvec = np.random.rand(N)

# Compute with VJP
vjp_jax = psub_vjp_jit(cotvec)

# Closed-form computation of VJP
if not (factorize):
    vjp_exact_2 = jax.jit(lambda x, y: project_subspace(A, y, Ap, x))
else:
    vjp_exact_2 = jax.jit(lambda x, y: project_subspace(A, y, cfac, x))
vjp_exact_2(cotvec, 0)

# Another closed-form computation of VJP
if not (factorize):
    vjp_exact = project_subspace(A, 0, Ap, cotvec)
else:
    vjp_exact = project_subspace(A, 0, cfac, cotvec)


# %% Computing the VJP of the semi-box
def box_wrapper(x):
    """Wrapper for project box."""
    return project_box(x, d)


primals, box_vjp = jax.vjp(box_wrapper, infeas)
box_vjp_jit = jax.jit(box_vjp)

# Test cotangent vector
cotvec = np.random.rand(N)

# Compute with VJP
vjp_box_jax = box_vjp_jit(cotvec)

# Closed-form computation of VJP
mask = infeas < d
box_vjp_man = jax.jit(lambda mask, cot: mask * cot)
vjp_closed = box_vjp_man(mask, cotvec)
# %% Do the whole loop with manual (seems tedious but could be made more efficient)

# %% Do the whole loop with autodiff
# Define a function runing a single iteration
if not (factorize):

    @jax.jit
    def proj_iter(A, b, Ap, xk, infeas, d):
        """Implements an algorithm to compute the projection using pseudoinverse."""
        qk = 2 * beta * (project_subspace(A, b, Ap, xk + infeas) - infeas) - xk
        pk = 2 * beta * (project_box(qk + infeas, d) - infeas) - qk
        xk = (1 - alpha) * xk + alpha * pk
        return xk

    proj_iter(A, b, Ap, xk, infeas, d)
else:

    @jax.jit
    def proj_iter(A, b, Ap, xk, infeas, d):
        """Implements an algorithm to compute the projection using factorization."""
        qk = 2 * beta * (project_subspace(A, b, cfac, xk + infeas) - infeas) - xk
        pk = 2 * beta * (project_box(qk + infeas, d) - infeas) - qk
        xk = (1 - alpha) * xk + alpha * pk
        return xk

    proj_iter(A, b, cfac, xk, infeas, d)

# omegak = the main iterate
omegak = infeas.copy()
start = time.time()
niter = 20
for ii in range(niter):
    if not (factorize):
        omegak = proj_iter(A, b, Ap, omegak, infeas, d)
    else:
        omegak = proj_iter(A, b, cfac, omegak, infeas, d)

if not (factorize):
    xk = project_subspace(A, b, Ap, omegak + infeas)
else:
    xk = project_subspace(A, b, cfac, omegak + infeas)
end = time.time()
print(end - start)
# %% Vmapping `proj_iter` to see how it deals with batches
proj_iter_mapped = jax.jit(jax.vmap(proj_iter, in_axes=[None, None, None, 0, 0, None]))
batch_size = 250

# Create artificial batch of iterates
omegak = infeas.copy()
omegak_batch = jnp.repeat(omegak[jnp.newaxis, :], batch_size, axis=0)

# Create artificial batch of points to be projected
infeas_batch = jnp.repeat(infeas[jnp.newaxis, :], batch_size, axis=0)
if not (factorize):
    proj_iter_mapped(A, b, Ap, omegak_batch, infeas_batch, d)
else:
    proj_iter_mapped(A, b, cfac, omegak_batch, infeas_batch, d)
# %%
# VJP of the iteration with respect to iterate
if not (factorize):
    projection, iteration_vjp = jax.vjp(
        lambda xx: proj_iter(A, b, Ap, xx, infeas, d), omegak
    )
else:
    projection, iteration_vjp = jax.vjp(
        lambda xx: proj_iter(A, b, cfac, xx, infeas, d), omegak
    )
iteration_vjp_jit = jax.jit(iteration_vjp)

# VJP of the iteration with respect to point-to-projected
if not (factorize):
    projection_again, iteration_vjp2 = jax.vjp(
        lambda to_proj: proj_iter(A, b, Ap, omegak, to_proj, d), infeas
    )
else:
    projection_again, iteration_vjp2 = jax.vjp(
        lambda to_proj: proj_iter(A, b, cfac, omegak, to_proj, d), infeas
    )

# Number of iterations for overall VJP calculation
niter = 5
# The iterate for VJP calculation
vjp_iter = jnp.zeros(N)
if not (factorize):
    corr_cotvec_1 = project_subspace(A, 0, Ap, cotvec)
    # This should be the same as corr_cotvec_1
    corr_cotvec = jax.grad(
        lambda xx: jnp.dot(project_subspace(A, b, Ap, xx + infeas), cotvec)
    )(omegak)
else:
    corr_cotvec_1 = project_subspace(A, 0, cfac, cotvec)
    # This should be the same as corr_cotvec_1
    corr_cotvec = jax.grad(
        lambda xx: jnp.dot(project_subspace(A, b, cfac, xx + infeas), cotvec)
    )(omegak)

jvp_iter = iteration_vjp_jit(vjp_iter)[0] + corr_cotvec
start = time.time()
for ii in range(niter):
    jvp_iter = iteration_vjp_jit(jvp_iter)[0] + corr_cotvec
thejvp = iteration_vjp2(jvp_iter)[0] + corr_cotvec
end = time.time()
print(end - start)
print(thejvp)
# %% Compare the result with using jaxopt to solve the QP and differentiate through it
Q = jnp.eye(N)

qp = jaxopt.OSQP()


@jax.jit
def qp_proj_solver(infeas):
    """Solve projection on polytope as a QP."""
    sol = qp.run(params_obj=(Q, -infeas), params_eq=(A, b), params_ineq=(C, d)).params

    # Scalarize the output to compute the vjp
    loss = jnp.dot(cotvec, sol.primal)

    return loss


loss = qp_proj_solver(infeas)

start = time.time()
loss = qp_proj_solver(infeas)
end = time.time()
print(end - start)

# Compute the VJP with the cotvec
jaxopt_grad = jax.jit(lambda infeas: jax.grad(qp_proj_solver)(infeas))
jaxopt_grad(infeas)
start = time.time()
jaxopt_grad(infeas)
end = time.time()
print(end - start)

# %% Check the result using the VJP function
_, exact_project_vjp = jax.vjp(
    lambda xx: qp.run(
        params_obj=(Q, -xx), params_eq=(A, b), params_ineq=(C, d)
    ).params.primal,
    infeas,
)
exact_project_vjp_jit = jax.jit(exact_project_vjp)
exact_project_vjp_jit(cotvec)
