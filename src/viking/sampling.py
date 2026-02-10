import functools as ft
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import scipy as sp


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


# NOTE: Will be VERY slow without @jax.jit
@ft.partial(jax.jit, static_argnames=("projection_fn",))
def _projection_iter(projection_fn, param_vec, samples, batch_x, batch_y=None):
    project_onto_kernel = projection_fn(param_vec, batch_x, batch_y)
    return jax.vmap(project_onto_kernel)(samples)


def alternating_projections(
    posterior,
    iso_samples: jax.Array,
    loader: Iterable,
):
    """Projects `iso_samples` onto the kernel of the GGN of the
    parameters within `posterior` successively over mini-batches taken from
    `loader`, ending up with an approximation of the projection on the
    full data [1].

    Args:
      posterior: A :class:`KernelImagePosterior` object
      iso_samples: The samples from a standard Gaussian to be projected
      loader: Iterable that yields a mini-batch of data at every iteration

    .. [1] https://arxiv.org/abs/2410.16901
    """
    param_vec, _ = posterior.flatten_fn(posterior.params)
    kernel_samples = iso_samples
    for x, y in loader:
        kernel_samples, _ = _projection_iter(
            posterior._project, param_vec, kernel_samples, x, y
        )
    return kernel_samples
