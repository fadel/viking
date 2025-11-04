from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest_cases

from viking import vi
from viking.linalg import (
    projection_kernel_lsmr,
    projection_kernel_normaleq,
)


class ModelParams(NamedTuple):
    w1: jax.Array
    w2: jax.Array
    bias: jax.Array


def make_mlp():
    def init_fn(*, key):
        k_w1, k_w2, k_bias = jax.random.split(key, num=3)
        return ModelParams(
            jax.random.normal(k_w1, shape=(1, 1)),
            jax.random.normal(k_w2, shape=(1,)),
            jax.random.normal(k_bias, shape=()),
        )

    def apply_fn(params: ModelParams, x_single):
        x = jnp.broadcast_to(x_single, (1,))
        x = params.w1 @ x
        x = jax.nn.relu(x)
        x = params.w2 @ x + params.bias
        return x

    return init_fn, apply_fn


def squared_loss(y_hat, y):
    return jnp.square(y_hat - y)


@pytest_cases.parametrize(
    "projection_fn", (projection_kernel_normaleq, projection_kernel_lsmr)
)
def test_posterior(projection_fn, n=5, num_samples=3):
    key = jax.random.PRNGKey(0)
    key_mlp, key_x, key_sample = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (n,))

    init_fn, apply_fn = make_mlp()
    params = init_fn(key=key_mlp)
    y_hat = jax.vmap(apply_fn, in_axes=(None, 0))(params, x)

    @jax.jit
    def y_tilde_from_posterior(log_scale_image):
        posterior = vi.make_posterior(
            apply_fn,
            params,
            log_precision=0.0,
            log_scale_image=log_scale_image,
            projection_fn=projection_fn(),
            flatten_fn=jax.flatten_util.ravel_pytree,
            is_linearized=True,
        )

        samples = vi.sample(posterior, num_samples, x, key=key_sample)
        y_tilde = jax.vmap(
            vi.predict_from_samples, in_axes=(None, None, 0), out_axes=1
        )(posterior, samples, x)
        return y_tilde

    # Essentially no image component: predictions should be the same
    y_tilde = y_tilde_from_posterior(log_scale_image=-16.0)
    assert jnp.allclose(y_hat[None, ...], y_tilde), "Not in kernel"

    # With image component: predictions should be different
    y_tilde = y_tilde_from_posterior(log_scale_image=-2.0)
    assert not jnp.allclose(y_hat[None, ...], y_tilde).item()


@pytest_cases.parametrize(
    "projection_fn", (projection_kernel_normaleq, projection_kernel_lsmr)
)
@pytest_cases.parametrize("is_linearized", (True, False))
def test_loss_posterior(projection_fn, is_linearized, n=5, num_samples=3):
    key = jax.random.PRNGKey(0)
    key_mlp, key_x, key_y, key_sample = jax.random.split(key, 4)
    x = jax.random.normal(key_x, (n,))
    y = jax.random.normal(key_x, (n,)) * 0.1

    init_fn, apply_fn = make_mlp()
    params = init_fn(key=key_mlp)
    y_hat = jax.vmap(apply_fn, in_axes=(None, 0))(params, x)
    losses = squared_loss(y_hat, y)

    @jax.jit
    def losses_from_posterior(log_scale_image):
        posterior = vi.make_posterior(
            apply_fn,
            params,
            # `log_precision` controls the bound on the variance of
            # predictive posterior losses. See Lemma 4.3 by Miani et
            # al. (2025).
            #
            # Miani, M., Roy, H., & Hauberg, S. (2024). Bayes without
            # Underfitting: Fully Correlated Deep Learning Posteriors
            # via Alternating Projections. arXiv preprint
            # arXiv:2410.16901.
            log_precision=4.0,
            log_scale_image=log_scale_image,
            projection_fn=projection_fn(),
            flatten_fn=jax.flatten_util.ravel_pytree,
            loss_fn=squared_loss,
            is_linearized=is_linearized,
        )

        samples = vi.sample(posterior, num_samples, x, y, key=key_sample)
        y_tilde = jax.vmap(
            vi.predict_from_samples, in_axes=(None, None, 0), out_axes=1
        )(posterior, samples, x)
        losses_tilde = squared_loss(y_tilde, jnp.reshape(y, (1,) + y.shape))
        return losses_tilde

    # Essentially no image component: losses should be the same
    losses_tilde = losses_from_posterior(log_scale_image=-16.0)
    # We are quite liberal with the tolerances, since the bound on the
    # variance is relatively loose
    assert jnp.allclose(losses[None, ...], losses_tilde, rtol=1e-1, atol=1e-1)

    # With image component: losses should be different
    losses_tilde = losses_from_posterior(log_scale_image=-2.0)
    assert not jnp.allclose(losses[None, ...], losses_tilde).item()
