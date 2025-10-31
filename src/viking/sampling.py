from typing import Callable

import jax
import jax.numpy as jnp
import scipy as sp
from tqdm import tqdm


def make_output_sampler(
    apply_fn: Callable, unflatten_fn: Callable, is_linearized: bool = False
):
    # Wraps `apply_fn` in a calling convention used for both
    # linearised predictions or regular predictions.
    def linearized_apply_fn(param_mean, p, x_eval):
        def apply_at_param(p):
            return apply_fn(unflatten_fn(p), x_eval)

        outputs = apply_at_param(param_mean)
        _, lin_outputs = jax.jvp(apply_at_param, (param_mean,), (p - param_mean,))
        return outputs + lin_outputs

    if is_linearized:
        _apply_fn = linearized_apply_fn
    else:
        _apply_fn = lambda _, p, x: apply_fn(unflatten_fn(p), x)

    def sample_outputs(
        x_eval,
        param_mean,
        image_samples,
        kernel_samples,
        log_scale_kernel,
        log_scale_image,
    ):
        posterior_samples = (
            param_mean
            + jnp.exp(log_scale_kernel) * kernel_samples
            + jnp.exp(log_scale_image) * image_samples
        )
        return jax.vmap(_apply_fn, in_axes=(None, 0, None))(
            param_mean, posterior_samples, x_eval
        )

    return sample_outputs


def normal_spherical_sample(key, num_samples, num_dims, end=1.0):
    """This function samples and normalizes two standard Gaussian
    points, then samples along the great circle defined by them on the
    hypersphere.
    """
    # NOTE: end=1.0 means "loop" all the way
    endpoints = jax.random.normal(key, shape=(2, num_dims))
    end_a = endpoints[0]
    end_b = endpoints[1]
    # This ensures that end_b is orthogonal to end_a, meaning we can
    # control interpolation a bit better (and enforce that end=1.0
    # means a full loop), see below
    end_a, end_b = _gram_schmidt(end_a, end_b)
    # Expected norm is proportional to sqrt(num_dims)
    expected_norm = jnp.sqrt(num_dims)
    end_a /= jnp.linalg.norm(end_a)
    end_b /= jnp.linalg.norm(end_b)
    omega = jnp.arccos(jnp.inner(end_a, end_b))
    end_a *= expected_norm
    end_b *= expected_norm
    t = jnp.linspace(0.0, end, num_samples, endpoint=True)[:, None]
    samples = (
        # Since end_a and end_b are orthogonal, they are "a quarter"
        # of the way apart vs. a full loop. Hence, we multiply t by 4
        # when parameterising the interpolation to ensure that t=1
        # results in a full loop.
        jnp.sin((1.0 - 4 * t) * omega) * end_a[None, ...]
        + jnp.sin(4 * t * omega) * end_b[None, ...]
    ) / jnp.sin(omega)
    return samples


def normal_noisy_spherical_sample(key, num_samples, num_dims, end=1.0):
    """Similar to `normal_spherical_sample`, but this function makes
    the great circle noisy along the surface of the sphere, with a (cubic)
    spline passing along the sampled points.
    """
    key, key_s, key_n = jax.random.split(key, num=3)
    num_clean_samples = max(num_samples // 5, 2)
    clean_samples = normal_spherical_sample(
        key_s, num_samples=num_clean_samples, num_dims=num_dims, end=end
    )
    # Noise should be orthogonal to the direction the clean samples
    # are, meaning we don't make samples in that direction
    # closer/further away in their natural trajectory along the loop.
    # Thus, our noise vectors only wiggle the clean samples "left and
    # right", if following that trajectory of samples in a 3d sense.
    _, noise = jax.vmap(_gram_schmidt)(
        clean_samples[1:] - clean_samples[0:-1],
        jax.random.normal(key_n, shape=clean_samples.shape)[:-1],
    )
    clean_samples = clean_samples[:-1] + 0.2 * noise
    # Adds first point as last for the periodic spline
    y_train = jnp.concatenate((clean_samples, clean_samples[0:1]), axis=0)
    y_train = y_train / jnp.linalg.norm(y_train, axis=-1, keepdims=True)
    y_train = _from_hypersphere(y_train)
    t = jnp.linspace(0, 1, num_clean_samples)
    spline = sp.interpolate.CubicSpline(t, y_train, bc_type="periodic")
    t = jnp.linspace(0, 1, num_samples)
    expected_norm = jnp.sqrt(num_dims)
    samples = _to_hypersphere(spline(t)) * expected_norm
    return samples


def _gram_schmidt(v1, v2):
    u1 = v1
    u2 = v2 - _project_onto(u1, v2)
    return u1, u2


def _project_onto(u, v):
    """Projects `v` onto `u`."""
    return jnp.inner(v, u) / jnp.inner(u, u) * u


def _to_hypersphere(x):
    d_norm = 1 + jnp.sum(x * x, axis=-1, keepdims=True)
    z = jnp.concatenate((2 * x / d_norm, 1 - 2 / d_norm), axis=-1)
    return z


def _from_hypersphere(z):
    x = z[:, :-1] / (1.0 - z[:, -1]).reshape((-1, 1))
    return x


def make_alternating_projections(loader, projection: Callable):
    """Returns a function that calls the `projection` function
    (returned from `projection_kernel_ggn` or
    `projection_kernel_param_to_loss`) successively over mini-batches
    from `loader`, starting from random samples and ending up with an
    approximation of the projection on the full data [1].

    [1] https://arxiv.org/abs/2410.16901

    """

    def project_batch(i, batch_carry):
        param_nn, carry_kernel_samples = batch_carry
        batch_x, batch_y = loader.dyn_batch(i)
        est_UUt_kernel = projection(param_nn, batch_x, batch_y)
        batch_kernel_samples, _ = jax.vmap(est_UUt_kernel)(carry_kernel_samples)
        return param_nn, batch_kernel_samples

    def projection_iter(_, iter_carry):
        return jax.lax.fori_loop(0, len(loader), project_batch, iter_carry)

    def alt_projections(param_nn, iso_samples, num_iter):
        _, batch_kernel_samples = jax.lax.fori_loop(
            0, num_iter, projection_iter, (param_nn, iso_samples)
        )
        return batch_kernel_samples

    return alt_projections


def make_alternating_projections_from_iterator(loader, projection: Callable):
    """Same as `make_alternating_projections`, but assumes loader is a
    regular iterable.

    """

    # This will be VERY slow without jax.jit()
    @jax.jit
    def projection_iter(param_nn, samples, batch_x, batch_y):
        est_UUt_kernel = projection(param_nn, batch_x, batch_y)
        samples, _ = jax.vmap(est_UUt_kernel)(samples)
        return samples

    def alt_projections(param_nn, iso_samples, num_iter):
        kernel_samples = iso_samples
        for _ in range(num_iter):
            for batch in loader:
                kernel_samples = projection_iter(
                    param_nn, kernel_samples, batch["image"], batch["label"]
                )
        return kernel_samples

    return alt_projections


def make_state_alternating_projections_from_iterator(loader, projection: Callable):
    """Same as `make_alternating_projections`, but assumes loader is a
    regular iterable and uses a Flax-like model state.

    """

    # This will be VERY slow without jax.jit()
    @jax.jit
    def projection_iter(state, samples, batch_x, batch_y):
        est_UUt_kernel, *_ = projection(state.params, state, batch_x, batch_y)
        samples, _ = jax.vmap(est_UUt_kernel)(samples)
        return samples

    def alt_projections(state, iso_samples, num_iter):
        kernel_samples = iso_samples
        for _ in range(num_iter):
            for batch in tqdm(loader, desc="Projecting"):
                kernel_samples = projection_iter(
                    state, kernel_samples, batch["image"], batch["label"]
                )
        return kernel_samples

    return alt_projections
