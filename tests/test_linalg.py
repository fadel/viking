import jax
import jax.numpy as jnp
import pytest_cases

from kivi.linalg import (
    projection_dense,
    projection_matfree,
    solve_normaleq_cg,
    solve_normaleq_cg_fixed_step_reortho,
    solve_normaleq_materialize,
    solve_normaleq_qr_of_jac,
    solver_fn_cholesky,
    solver_fn_eig,
    solver_fn_lu,
)


def test_projection_dense(m=2, n=5):
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, (m, n))
    v = jax.random.normal(key2, (n,))

    projfun = projection_dense()
    x, _ = projfun(A, v)
    residual = A @ x[:n]
    assert jnp.allclose(residual, 0, atol=1e-4)


def test_projection_dense_vjp(m=2, n=5):
    projfun_true = projection_dense(use_custom_vjp=False)
    projfun_custom = projection_dense(use_custom_vjp=True)

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
    return solver_fn_eig(eps=1e-8)


def case_solver_fn_lu():
    return solver_fn_lu()


@pytest_cases.parametrize("symmetrize", [True, False])
@pytest_cases.parametrize("eps", [0.0, 1e-8])
def case_solver_fn_cholesky(symmetrize, eps):
    return solver_fn_cholesky(symmetrize=symmetrize, eps=eps)


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
def test_projection_matfree(solve_ne, m=2, n=5):
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.normal(key1, (m, n))
    v = jax.random.normal(key2, (n,))
    projfun = projection_dense()
    x, _ = projfun(A, v)

    def matvec(p, w):
        return p @ w

    def vecmat(p, w):
        return p.T @ w

    projfun_matfree = projection_matfree(
        matvec, vecmat, num_rows=m, num_cols=n, solve_normaleq=solve_ne
    )

    x_matfree, _ = projfun_matfree(A, v)
    assert jnp.allclose(x, x_matfree)


@pytest_cases.parametrize_with_cases(
    "solve_normaleq", cases=".", prefix="case_normaleq_"
)
def test_projection_matfree_vjp(solve_normaleq, m=2, n=5):
    key = jax.random.PRNGKey(0)
    key, key_A, key_v = jax.random.split(key, 3)
    A = jax.random.normal(key_A, (m, n))
    v = jax.random.normal(key_v, (n,))
    projfun = projection_dense()
    primals_out, vjp_fn, _ = jax.vjp(projfun, A, v, has_aux=True)

    def matvec(p, w):
        return p @ w

    def vecmat(p, w):
        return p.T @ w

    projfun_matfree = projection_matfree(
        matvec,
        vecmat,
        num_rows=m,
        num_cols=n,
        solve_normaleq=solve_normaleq,
        use_custom_vjp=True,
    )
    primals_out_matfree, vjp_fn_matfree, _ = jax.vjp(
        projfun_matfree, A, v, has_aux=True
    )

    # Random cotangent vectors (of shape like output of the fn)
    _key, key_w = jax.random.split(key)
    w = jax.random.normal(key_w, (m + n,))
    A_dot, v_dot = vjp_fn(w)
    A_dot_matfree, v_dot_matfree = vjp_fn_matfree(w)
    assert jnp.allclose(primals_out, primals_out_matfree, rtol=1e-4)
    assert jnp.allclose(v_dot, v_dot_matfree, rtol=1e-4)
    assert jnp.allclose(A_dot, A_dot_matfree, rtol=1e-3)
