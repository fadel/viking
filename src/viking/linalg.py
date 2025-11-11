"""
Useful linear algebra functions. Mostly projections with custom
derivatives via [adjoints]_ and various implementations of solvers.

.. [adjoints] https://arxiv.org/abs/1905.00578
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matfree.lstsq

from viking import cg


class SolveInfo(NamedTuple):
    residuals: float


def _project_normaleq(
    matvec, vecmat, *, num_rows, num_cols, solve_normaleq, custom_vjp=False
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

    def project(param, v):
        c = jnp.concatenate((v, jnp.zeros(num_rows)))
        return _projsolve(param, c)

    # If we are satisfied with JAX's autodiff, stop here
    if not custom_vjp:
        return project

    project_vjp = jax.custom_vjp(project)

    def project_fwd(param, v):
        out, info = project(param, v)
        return (out, info), {"param": param, "z": out}

    def project_rev(cache, vjp_incoming):
        param = cache["param"]
        z = cache["z"]
        xi, _info = _projsolve(param, vjp_incoming[0])

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

    project_vjp.defvjp(project_fwd, project_rev)

    return project_vjp


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


def ggn_apply_fn_at_x(batch_apply_fn: Callable, unflatten_fn: Callable, x: jax.Array):
    """
    This is used to build the GGN matrix at the (mini-batch of) points
    ``x``.

    Args:
      batch_apply_fn: Forward pass function that takes the model
        parameters and (mini-batch) data
      unflatten_fn: Takes a parameter vector and builds the original
        model parameter structure
      x: Mini-batch of data to evaluate on

    Returns:
      A function that takes a parameter vector (point in parameter
      space) and returns the model outputs as a vector of length
      :math:`N \\times O`, where :math:`O` the number of model outputs
      and :math:`N` is the batch size.
    """

    def apply_fn_at_x(param_vec):
        params = unflatten_fn(param_vec)
        return batch_apply_fn(params, x).reshape(-1)

    return apply_fn_at_x


def loss_apply_fn_at_x(
    loss_fn: Callable,
    batch_apply_fn: Callable,
    unflatten_fn: Callable,
    x: jax.Array,
    y: jax.Array,
):
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
      space) and returns the loss values as a vector of length
      :math:`N`, where :math:`N` is the batch size.
    """

    def apply_fn_at_x(param_vec):
        params = unflatten_fn(param_vec)
        outputs = batch_apply_fn(params, x)
        return jax.vmap(loss_fn)(outputs, y)

    return apply_fn_at_x


def _prepare_matvec_vecmat(apply_fn, unflatten_fn, x, loss_fn=None, y=None):
    batch_apply_fn = jax.vmap(apply_fn, in_axes=(None, 0))

    if loss_fn is None:
        apply_fn_at_x = ggn_apply_fn_at_x(batch_apply_fn, unflatten_fn, x)
    else:
        apply_fn_at_x = loss_apply_fn_at_x(loss_fn, batch_apply_fn, unflatten_fn, x, y)

    def matvec(p, s):
        _, jvp_x = jax.jvp(apply_fn_at_x, (p,), (s,))
        return jvp_x

    def vecmat(p, s):
        _, vjp = jax.vjp(apply_fn_at_x, p)
        return vjp(s)[0]

    return apply_fn_at_x, matvec, vecmat


def projection_kernel_normaleq(
    solve_normaleq: Callable = solve_normaleq_cg_fixed_step_reortho(maxiter=10),
    custom_vjp: bool = True,
) -> Callable:
    """
    Projects onto the kernel (null) space of the GGN matrix using an
    explicit normal equations formulation.

    .. caution::

      GGN matrices tend to be ill-conditioned. Carefully choose a
      solver (via ``solve_normaleq``) and always experiment with
      alternative ones (and their options) if you experience issues.
      The default choice for this argument behaves well for many
      problems and model sizes.

      For additional diagnosis, you can use information from the
      solver (see :class:`SolveInfo`) and, for example, ensure
      kernel and image samples are orthogonal (see
      :class:`viking.vi.ELBOInfo`).

    Args:
      solve_normaleq: One of the ``solve_normaleq_`` variants.

    Returns:
      A function to be passed on to the ``projection_fn`` argument of
      :func:`viking.vi.make_posterior`.
    """

    def projection_kernel(
        apply_fn: Callable,
        unflatten_fn: Callable,
        loss_fn: Callable = None,
        custom_vjp: bool = True,
    ):
        def make_projection_fn(param_vec, x, y=None):
            apply_fn_at_x, matvec, vecmat = _prepare_matvec_vecmat(
                apply_fn, unflatten_fn, x, loss_fn, y
            )

            D = len(param_vec)
            out_dummy = jax.eval_shape(apply_fn_at_x, param_vec)
            projection_fn = _project_normaleq(
                matvec,
                vecmat,
                num_rows=out_dummy.size,
                num_cols=D,
                solve_normaleq=solve_normaleq,
                custom_vjp=custom_vjp,
            )

            def project_onto_kernel(vec):
                proj_vec_augmented, info = projection_fn(param_vec, vec)
                proj_vec = proj_vec_augmented[:D]
                return proj_vec, info

            return project_onto_kernel

        return make_projection_fn

    return projection_kernel


def projection_kernel_lsmr(
    **lsmr_kwargs,
):
    """
    Projects onto the kernel (null) space of the GGN matrix using the
    LSMR solver from [matfree]_ (:func:`matfree.lstsq.lsmr`).

    Args:
      lsmr_kwargs: These are passed unchanged onto the initializer of
                   the LSMR implementation. Refer to the matfree
                   documentation for details.

    Returns:
      A function to be passed on to the ``projection_fn`` argument of
      :func:`viking.vi.make_posterior`.

    References:
      D. C.-L. Fong and M. A. Saunders, "LSMR: An iterative algorithm
      for sparse least-squares problems", SIAM J. Sci. Comput., vol.
      33, pp. 2950-2971, 2011.

    .. [matfree] https://pnkraemer.github.io/matfree/
    """

    def projection_kernel(
        apply_fn: Callable,
        unflatten_fn: Callable,
        loss_fn: Callable = None,
        custom_vjp: bool = True,
    ):
        def make_projection_fn(param_vec, x, y=None):
            apply_fn_at_x, matvec, vecmat = _prepare_matvec_vecmat(
                apply_fn, unflatten_fn, x, loss_fn, y
            )
            projection_fn = matfree.lstsq.lsmr(
                custom_vjp=custom_vjp, atol=1e-8, btol=1e-8, **lsmr_kwargs
            )

            def fake_vecmat(v):
                return matvec(param_vec, v)

            def project_onto_kernel(vec):
                lagr, info = projection_fn(fake_vecmat, vec)
                proj_vec = vec - vecmat(param_vec, lagr)
                return proj_vec, info

            return project_onto_kernel

        return make_projection_fn

    return projection_kernel
