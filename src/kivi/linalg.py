"""Useful linear algebra functions. Mostly projections with custom derivatives via adjoints [1] and various implementations of solvers.

[1]: https://arxiv.org/abs/1905.00578
"""

import collections
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from kivi import cg


SolveInfo = collections.namedtuple("SolveInfo", ("residuals"))


def prod_tr(mat_a, mat_b):
    """
    Computes tr(mat_a @ mat_b) without computing all entries of
    mat_a @ mat_b.
    """
    tr = jnp.sum(jnp.sum(mat_a * mat_b.T, axis=-1), axis=-1)
    return tr


def nystroem_pp(A_samples, iso_samples):
    """The Nyström++ trace estimator [1].

    Args:
      A_samples: A @ iso_samples
      iso_samples: i.i.d. samples from a standard Gaussian

    Reference:
      [1] Persson, David, Alice Cortinovis, and Daniel Kressner.
      "Improved variants of the Hutch++ algorithm for trace
      estimation." SIAM Journal on Matrix Analysis and Applications
      43.3 (2022): 1162-1185.
    """
    num_samples = iso_samples.shape[0]
    Om, Psi = jnp.split(iso_samples.T, 2, axis=-1)
    X, Y = jnp.split(A_samples.T, 2, axis=-1)

    # Rewrites X @ pinv(Om.T @ X) @ X.T as X @ Q @ R^-1 @ X.T
    Q, R = jnp.linalg.qr(X.T @ Om)
    R_inv_XT = jsp.linalg.solve_triangular(R, X.T, trans="T")
    X_pinv_XT = X @ Q @ R_inv_XT
    fst_tr = jnp.linalg.trace(X_pinv_XT)
    snd_tr = prod_tr(Psi.T, Y)
    trd_tr = prod_tr(X_pinv_XT, Psi @ Psi.T)
    tr = fst_tr + 2 / num_samples * (snd_tr - trd_tr)
    return tr


def projection_dense(use_custom_vjp=True):
    """Construct a function that projects a vector onto the kernel of
    a (dense) matrix.

    This is the baseline implementation that `projection_matfree`
    improves upon. You probably want to use that instead.
    """

    def _projsolve(A, c):
        _, m = A.shape
        c_1, c_2 = jnp.split(c, [m])
        AA_t = A @ A.T

        proj_c_1 = A @ c_1
        proj_c_1 = jnp.linalg.solve(AA_t, proj_c_1)
        inv_c_2 = jnp.linalg.solve(AA_t, c_2)

        x_2 = proj_c_1 - inv_c_2  # image
        proj_c_1 = c_1 - A.T @ proj_c_1
        inv_c_2 = A.T @ inv_c_2
        x_1 = proj_c_1 + inv_c_2  # kernel
        return jnp.concatenate((x_1, x_2)), SolveInfo(residuals=jnp.nan)

    def projfun(A, v):
        c = jnp.concatenate((v, jnp.zeros(A.shape[0])))
        return _projsolve(A, c)

    if not use_custom_vjp:
        return projfun

    projfun_vjp = jax.custom_vjp(projfun)

    def projfun_fwd(A, v):
        out, info = projfun(A, v)
        return (out, info), {"A": A, "v": v, "z": out}

    def projfun_rev(cache, vjp_incoming):
        A, v, z = cache["A"], cache["v"], cache["z"]
        xi, _info = _projsolve(A, vjp_incoming[0])

        full_grad_B = -jnp.outer(xi, z)
        full_grad_c = xi

        grad_A = (full_grad_B + full_grad_B.T)[-A.shape[0] :, : A.shape[1]]
        grad_v = full_grad_c[: v.shape[0]]

        return grad_A, grad_v

    projfun_vjp.defvjp(projfun_fwd, projfun_rev)

    return projfun_vjp


def projection_matfree(
    matvec, vecmat, *, num_rows, num_cols, solve_normaleq, use_custom_vjp=False
):
    """
    Constructs and returns a function like `projection_dense` that
    solves the same projection problem, but without materializing the
    matrix of interest via `matvec` and `vecmat`.
    """

    def _projsolve(param, c):
        # cols/rows according to A's cols/rows
        c_cols, c_rows = jnp.split(c, [num_cols])

        Ac_cols = matvec(param, c_cols)
        rhs = Ac_cols - c_rows
        lagrange_multiplier, info = solve_normaleq(matvec, vecmat, param, rhs)

        # If test breaks, reevaluate sign in front of inv_c_rows
        c_projection = c_cols - vecmat(param, lagrange_multiplier)
        return jnp.concatenate((c_projection, lagrange_multiplier)), info

    def projfun(param, v):
        c = jnp.concatenate((v, jnp.zeros(num_rows)))
        return _projsolve(param, c)

    # If we are satisfied with JAX's autodiff, stop here
    if not use_custom_vjp:
        return projfun

    projfun_vjp = jax.custom_vjp(projfun)

    def projfun_fwd(param, v):
        out, info = projfun(param, v)
        return (out, info), {"param": param, "z": out}

    def projfun_rev(cache, vjp_incoming):
        param = cache["param"]
        z = cache["z"]
        xi, _stats = _projsolve(param, vjp_incoming[0])

        def fn_1(p):
            v_1 = xi[-num_rows:]
            v_2 = z[:num_cols]
            Av_2 = matvec(p, v_2)
            return -jnp.dot(v_1, Av_2)

        def fn_2(p):
            v_1 = z[-num_rows:]
            v_2 = xi[:num_cols]
            Av_2 = matvec(p, v_2)
            return -jnp.dot(v_1, Av_2)

        grad_1 = jax.grad(fn_1)(param)
        grad_2 = jax.grad(fn_2)(param)

        grad_v = xi[:num_cols]
        grad_param = grad_1 + grad_2
        return grad_param, grad_v

    projfun_vjp.defvjp(projfun_fwd, projfun_rev)

    return projfun_vjp


def normalized_residuals(solution, rhs):
    return jnp.linalg.norm(rhs - solution) / jnp.linalg.norm(rhs)


def solve_normaleq_qr_of_jac():
    def solve(matvec, vecmat, param, rhs):
        del matvec  # unused argument

        Jt = jax.jacfwd(lambda v: vecmat(param, v))(rhs)
        Lt = jnp.linalg.qr(Jt, mode="r")

        cho_factor = (Lt, False)
        solution = jsp.linalg.cho_solve(cho_factor, rhs)
        return solution, SolveInfo(
            residuals=normalized_residuals(Jt.T @ (Jt @ solution), rhs)
        )

    return solve


def dense_solver_fn_cholesky(symmetrize=False, eps=1e-6):
    """
    Solver based on Cholesky decomposition. Allows manipulation of the
    input matrix either by forcing symmetrisation (`symmetrize`
    argument) or adding a small constant to the diagonal entries via
    the `eps` argument.

    GGN matrices tend to be ill-conditioned. Using this solver can
    cause numerical instability.
    """

    def solver_fn_cho(mat, rhs):
        D = mat.shape[-1]
        noise = eps * jnp.eye(D)
        mat = mat + noise

        if symmetrize:
            mat = (mat + mat.T) / 2

        cho_factor = jsp.linalg.cho_factor(mat)
        return jsp.linalg.cho_solve(cho_factor, rhs)

    return solver_fn_cho


def dense_solver_fn_lu():
    return jnp.linalg.solve


def dense_solver_fn_eig(eps):
    """
    Uses eigendecomposition to solve the linear system. When not
    numerically unstable, it is very useful for ignoring noisy
    eigenvalues that should be zero. All eigenvalues smaller than
    `eps` are dropped.
    """

    def solve(mat, rhs):
        eigvals, eigvecs = jnp.linalg.eigh(mat)
        eigvals = jnp.where(eigvals < eps, -1.0, eigvals)
        out = eigvecs.T @ rhs
        inv_eigvals = jnp.reciprocal(eigvals)
        inv_eigvals = jnp.where(inv_eigvals < eps, 0.0, inv_eigvals)
        out = jnp.diag(inv_eigvals) @ out
        return eigvecs @ out

    return solve


def solve_normaleq_materialize(solver_fn):
    """Solve the normal equation by calling a dense solver."""

    def solve(matvec, vecmat, param, rhs):
        def AAt_mv(s):
            Atv = vecmat(param, s)
            return matvec(param, Atv)

        # Since AAt is linear, the Jacobian at _any_
        # value will materialize the matrix.
        # For shape reasons, we mimic the RHS vector.
        # But any vector of the correct shape would do.
        irrelevant_value = jnp.zeros_like(rhs)
        AA_t = jax.jacfwd(AAt_mv)(irrelevant_value)
        solution = solver_fn(AA_t, rhs)
        return solution, SolveInfo(
            residuals=normalized_residuals(AAt_mv(solution), rhs)
        )

    return solve


def solve_normaleq_cg(tol=1e-5, atol=1e-12, maxiter=10):
    """Solve the normal equation by conjugate gradients."""

    def solve(matvec, vecmat, param, rhs):
        def AAt_mv(s):
            Atv = vecmat(param, s)
            return matvec(param, Atv)

        solution, _ = jsp.sparse.linalg.cg(
            AAt_mv, rhs, tol=tol, atol=atol, maxiter=maxiter
        )
        return solution, SolveInfo(
            residuals=normalized_residuals(AAt_mv(solution), rhs)
        )

    return solve


def solve_normaleq_cg_adaptive_step(**kwargs):
    """Solve the normal equation by conjugate gradients."""

    cg_solve = cg.cg_adaptive(**kwargs)

    def solve(matvec, vecmat, param, rhs):
        def AAt_mv(s):
            Atv = vecmat(param, s)
            return matvec(param, Atv)

        solution, _ = cg_solve(AAt_mv, rhs)
        return solution, SolveInfo(
            residuals=normalized_residuals(AAt_mv(solution), rhs)
        )

    return solve


def solve_normaleq_cg_fixed_step_reortho(maxiter=10):
    """
    Solve the normal equation by conjugate gradients (CG). This
    version performs reorthogonalization in every CG step, ensuring a
    more accurate solution at the cost of more computation time.
    """

    cg_solve = cg.cg_fixed_step_reortho(maxiter)

    def solve(matvec, vecmat, param, rhs):
        def AAt_mv(s):
            Atv = vecmat(param, s)
            return matvec(param, Atv)

        solution, _ = cg_solve(AAt_mv, rhs)
        return solution, SolveInfo(
            residuals=normalized_residuals(AAt_mv(solution), rhs)
        )

    return solve


def ggn_apply_fn_at_x(batch_apply_fn, unflatten_fn, x):
    """
    This is used to build the GGN matrix at the (mini-batch of) points
    `x`.

    Args:
      batch_apply_fn: Forward pass function that takes the model
        parameters and (mini-batch) data
      unflatten_fn: Takes a parameter vector and builds the original
        model parameter structure
      x: Mini-batch of data to evaluate on

    Returns:
      A function that takes a parameter vector (point in parameter
      space) and returns the model outputs as a vector of length $N
      \\times O$, where $O$ is the number of model outputs and $N$ is
      the batch size.
    """

    def apply_fn_at_x(param_vec):
        params = unflatten_fn(param_vec)
        return batch_apply_fn(params, x).reshape(-1)

    return apply_fn_at_x


def loss_apply_fn_at_x(loss_fn, batch_apply_fn, unflatten_fn, x, y):
    """
    This is used to build the pseudo GGN (based on loss Jacobian) at
    the mini-batch of points `x`, using the losses at `y`.

    Args:
      loss_fn: Loss function that takes the outputs computed by
        `batch_apply_fn` and compares them to each `y`
      batch_apply_fn: Forward pass function that takes the model
        parameters and (mini-batch) data
      unflatten_fn: Takes a parameter vector and builds the original
        model parameter structure
      x: Mini-batch of data to evaluate on
      y: Mini-batch of labels/targets to evaluate on

    Returns:
      A function that takes a parameter vector (point in parameter
      space) and returns the loss values as a vector of length $N$,
      where $N$ is the batch size.
    """

    def apply_fn_at_x(param_vec):
        params = unflatten_fn(param_vec)
        preds = batch_apply_fn(params, x)
        return jax.vmap(loss_fn)(preds, y)

    return apply_fn_at_x


def projection_kernel(
    apply_fn: Callable,
    unflatten_fn: Callable,
    solve_normaleq: Callable,
    loss_fn: Callable = None,
    use_custom_vjp=True,
):
    """
    Prepares and returns a `Callable` that can be used to project
    parameter vectors onto the kernel (null) space of the parametrised
    function `apply_fn` (usually a neural network).

    If `loss_fn` is provided, it uses the loss (computed using the
    function outputs) Jacobian instead of the function Jacobian,
    projecting into that kernel (null) space instead.

    This scales better for models with large output dimensions at the
    cost of shifting the requirement of maintaining model outputs at
    training data to maintaining the loss values at those data points.
    For more details, refer to [1; Sec. 4.1].

    [1] https://arxiv.org/abs/2410.16901

    """
    batch_apply_fn = jax.vmap(apply_fn, in_axes=(None, 0))

    def make_projection_fn(param_vec, x, y=None):
        if loss_fn is None:
            apply_fn_at_x = ggn_apply_fn_at_x(batch_apply_fn, unflatten_fn, x)
        else:
            apply_fn_at_x = loss_apply_fn_at_x(
                loss_fn, batch_apply_fn, unflatten_fn, x, y
            )

        def matvec(p, s):
            _, jvp_x = jax.jvp(apply_fn_at_x, (p,), (s,))
            return jvp_x

        def vecmat(p, s):
            _, vjp = jax.vjp(apply_fn_at_x, p)
            return vjp(s)[0]

        D = len(param_vec)
        out_dummy = jax.eval_shape(apply_fn_at_x, param_vec)
        projection_fn = projection_matfree(
            matvec,
            vecmat,
            num_rows=out_dummy.size,
            num_cols=D,
            solve_normaleq=solve_normaleq,
            use_custom_vjp=use_custom_vjp,
        )

        def project_onto_kernel(vec):
            fx, stats = projection_fn(param_vec, vec)
            return fx[:D], stats

        return project_onto_kernel

    return make_projection_fn
