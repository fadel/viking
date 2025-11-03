import jax
import jax.numpy as jnp
import pytest_cases

from viking.linalg import (
    SolveInfo,
    dense_solver_fn_cholesky,
    dense_solver_fn_eig,
    dense_solver_fn_lu,
    _project_normaleq,
    solve_normaleq_cg,
    solve_normaleq_cg_fixed_step_reortho,
    solve_normaleq_materialize,
    solve_normaleq_qr_of_jac,
)


def _projection_dense(custom_vjp=True):
    """Construct a function that projects a vector onto the kernel of
    a (dense) matrix.

    This is the baseline implementation that tests whether `_project_normaleq`
    is correct.
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

    if not custom_vjp:
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


def test_projection_dense(m=2, n=5):
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, (m, n))
    v = jax.random.normal(key2, (n,))

    projfun = _projection_dense()
    x, _ = projfun(A, v)
    residual = A @ x[:n]
    assert jnp.allclose(residual, 0, atol=1e-4)


def test_projection_dense_vjp(m=2, n=5):
    projfun_true = _projection_dense(custom_vjp=False)
    projfun_custom = _projection_dense(custom_vjp=True)

    # Random points to evaluate the two VJPs
    key = jax.random.PRNGKey(1)
    key, key_A, key_v = jax.random.split(key, 3)
    A = jax.random.normal(key_A, (m, n))
    v = jax.random.normal(key_v, (n,))
    primals_out_true, vjp_fn_true, _ = jax.vjp(projfun_true, A, v, has_aux=True)
    primals_out_custom, vjp_fn_custom, _ = jax.vjp(projfun_custom, A, v, has_aux=True)

    # Random cotangent vectors (of shape like output of the fn)
    _key, key_w = jax.random.split(key)
    w = jax.random.normal(key_w, (m + n,))
    A_dot_true, v_dot_true = vjp_fn_true(w)
    A_dot_custom, v_dot_custom = vjp_fn_custom(w)

    tols = {"atol": 1e-4, "rtol": 1e-4}
    assert jnp.allclose(primals_out_true, primals_out_custom)
    assert jnp.allclose(v_dot_true, v_dot_custom, **tols)
    assert jnp.allclose(A_dot_true, A_dot_custom, **tols)


def case_solver_fn_eig():
    return dense_solver_fn_eig(eps=1e-8)


def case_solver_fn_lu():
    return dense_solver_fn_lu()


@pytest_cases.parametrize("symmetrize", [True, False])
@pytest_cases.parametrize("eps", [0.0, 1e-8])
def case_solver_fn_cholesky(symmetrize, eps):
    return dense_solver_fn_cholesky(symmetrize=symmetrize, eps=eps)


@pytest_cases.parametrize_with_cases("solver_fn", cases=".", prefix="case_solver_fn_")
def case_normaleq_materialize(solver_fn):
    return solve_normaleq_materialize(solver_fn=solver_fn)


def case_normaleq_cg():
    return solve_normaleq_cg()


def case_normaleq_cg_fixed_step_reortho():
    return solve_normaleq_cg_fixed_step_reortho()


def case_normaleq_qr_of_jac():
    return solve_normaleq_qr_of_jac()


@pytest_cases.parametrize_with_cases("solve_ne", cases=".", prefix="case_normaleq_")
def test_project_normaleq(solve_ne, m=2, n=5):
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, (m, n))
    v = jax.random.normal(key2, (n,))
    projfun = _projection_dense()
    x, _ = projfun(A, v)

    def matvec(p, w):
        return p @ w

    def vecmat(p, w):
        return p.T @ w

    projfun = _project_normaleq(
        matvec, vecmat, num_rows=m, num_cols=n, solve_normaleq=solve_ne
    )

    x_matfree, _ = projfun(A, v)
    assert jnp.allclose(x, x_matfree)


@pytest_cases.parametrize_with_cases(
    "solve_normaleq", cases=".", prefix="case_normaleq_"
)
def test_project_normaleq_vjp(solve_normaleq, m=2, n=5):
    key = jax.random.PRNGKey(0)
    key, key_A, key_v = jax.random.split(key, 3)
    A = jax.random.normal(key_A, (m, n))
    v = jax.random.normal(key_v, (n,))
    projfun = _projection_dense()
    primals_out, vjp_fn, _ = jax.vjp(projfun, A, v, has_aux=True)

    def matvec(p, w):
        return p @ w

    def vecmat(p, w):
        return p.T @ w

    projfun_normaleq = _project_normaleq(
        matvec,
        vecmat,
        num_rows=m,
        num_cols=n,
        solve_normaleq=solve_normaleq,
        custom_vjp=True,
    )
    primals_out_normaleq, vjp_fn_normaleq, _ = jax.vjp(
        projfun_normaleq, A, v, has_aux=True
    )

    # Random cotangent vectors (of shape like output of the fn)
    _key, key_w = jax.random.split(key)
    w = jax.random.normal(key_w, (m + n,))
    A_dot, v_dot = vjp_fn(w)
    A_dot_normaleq, v_dot_normaleq = vjp_fn_normaleq(w)
    assert jnp.allclose(primals_out, primals_out_normaleq, rtol=1e-4)
    assert jnp.allclose(v_dot, v_dot_normaleq, rtol=1e-4)
    assert jnp.allclose(A_dot, A_dot_normaleq, rtol=1e-3)
