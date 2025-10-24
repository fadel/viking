# TODO(samuel): Ensure we can use model posterior in training and
#               evaluation modes (e.g., for dropout, batchnorm)
import dataclasses
from typing import Any, Callable, NamedTuple, Union

import jax
import jax.numpy as jnp

from viking import linalg, sampling, vi


class PosteriorSamples(NamedTuple):
    mean: jax.Array
    iso: jax.Array
    kernel: jax.Array
    image: jax.Array
    solve_info: linalg.SolveInfo


class ELBOInfo(NamedTuple):
    """
    This tuple holds useful auxiliary information computed within the
    ELBO loss; see :py:func:`as_elbo_loss`.
    """

    expectation: float
    """The value of the expectation part of the ELBO."""

    kl: float
    """The value of the Kullback-Leibler divergence part of the ELBO."""

    outputs: jax.Array
    """The outputs (predictions) computed by every posterior sample."""

    projection_rank: float
    """The estimated rank of the kernel (null) space."""

    samples: jax.Array
    """The posterior samples."""


_JAX_STATIC_METADATA = {"static": True}


def _decl_static():
    return dataclasses.field(metadata=_JAX_STATIC_METADATA)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class KernelImagePosterior:
    params: Any
    log_precision: jax.Array
    log_scale_image: jax.Array
    beta: float = _decl_static()
    flatten_fn: Callable = _decl_static()
    unflatten_fn: Callable = _decl_static()
    loss_fn: Callable = _decl_static()
    _projection: Callable = _decl_static()
    _sample_outputs: Callable = _decl_static()


def make_posterior(
    apply_fn: Callable,
    params: Any,
    flatten_fn: Callable,
    log_precision: Union[float, jax.Array] = 0.0,
    log_scale_image: Union[float, jax.Array] = -2.0,
    projection_fn: Callable = linalg.projection_kernel_lsmr(),
    beta: float = 1.0,
    loss_fn: Callable = None,
    is_linearized: bool = True,
    custom_vjp: bool = True,
) -> KernelImagePosterior:
    """
    Performs the necessary intialization for a
    :py:class:`KernelImagePosterior` object.

    If ``loss_fn`` is provided, it uses the loss (computed using the
    function outputs) Jacobian instead of the function Jacobian,
    projecting into that kernel (null) space instead. This scales
    better for models with large output dimensions at the cost of
    shifting the requirement of maintaining model outputs at training
    data to maintaining the loss values at those data points. For more
    details regarding the loss Jacobian, refer to [1]_ (Sec. 4.1).

    Args:
      apply_fn: A differentiable (with respect to parameters) function
                that takes a pytree of parameters as its first
                argument and the inputs as the second argument,
                returning the function outputs.
      params: A pytree of the initial set of model parameters
      flatten_fn: A function that takes a pytree of model parameters
                  and returns two values: one is the parameters
                  flattened into a single-axis ``jax.Array`` and a
                  function that takes such an array and returns the
                  original pytree.
      log_precision: The initial (prior) log-precision (:math:`\\log
                     \\alpha`); follows the relationship :math:`\\log
                     \\sigma_{\\mathrm{ker}} = -\\frac12 \\log \\alpha`.
      log_scale_image: The initial value of :math:`\\log
                       \\sigma_{\\mathrm{im}}`.
      projection_fn: Provide here the return value of either
                     :py:func:`viking.linalg.projection_kernel_normaleq`
                     or
                     :py:func:`viking.linalg.projection_kernel_lsmr`.
      beta: The value by which the KL divergence term is rescaled with
      loss_fn: If provided, this posterior will use the loss Jacobian
               instead of function Jacobian when projecting parameter
               vectors onto the kernel (null) space of the GGN matrix
               (see above).
      is_linearized: Whether model linearization (in the weight space)
                     is done before computing predictions with the
                     posterior samples.
      custom_vjp: Whether to use custom implementations of backward
                  passes (in the gradient descent sense) with better
                  numerical properties; only change this if you are
                  aware of the consequences.

    Returns:
      A ready-to-use :py:class:`KernelImagePosterior` object.

    .. [1] https://arxiv.org/abs/2410.16901
    """
    log_precision = jnp.asarray(log_precision)
    log_scale_image = jnp.asarray(log_scale_image)
    _, unflatten_fn = flatten_fn(params)
    projection_fn = projection_fn(
        apply_fn,
        unflatten_fn,
        loss_fn=loss_fn,
        custom_vjp=custom_vjp,
    )
    sample_outputs_fn = sampling.make_output_sampler(
        apply_fn, unflatten_fn, is_linearized=is_linearized
    )
    return KernelImagePosterior(
        params=params,
        log_precision=log_precision,
        log_scale_image=log_scale_image,
        beta=beta,
        flatten_fn=flatten_fn,
        unflatten_fn=unflatten_fn,
        loss_fn=loss_fn,
        _projection=projection_fn,
        _sample_outputs=sample_outputs_fn,
    )


def sample(
    posterior: KernelImagePosterior,
    num_samples: int,
    x: jax.Array,
    y: jax.Array = None,
    *,
    key,
) -> PosteriorSamples:
    param_vec, _ = posterior.flatten_fn(posterior.params)
    D = len(param_vec)
    kernel_projection = posterior._projection(param_vec, x, y)
    iso_samples = jax.random.normal(key, (num_samples, D))
    kernel_samples, info = jax.vmap(kernel_projection)(iso_samples)
    image_samples = iso_samples - kernel_samples
    return PosteriorSamples(
        mean=param_vec,
        iso=iso_samples,
        kernel=kernel_samples,
        image=image_samples,
        solve_info=info,
    )


def predict_from_samples(
    posterior: KernelImagePosterior, samples: PosteriorSamples, x_eval: jax.Array
):
    log_scale_kernel = -0.5 * posterior.log_precision
    return posterior._sample_outputs(
        x_eval,
        param_mean=samples.mean,
        image_samples=samples.image,
        kernel_samples=samples.kernel,
        log_scale_kernel=log_scale_kernel,
        log_scale_image=posterior.log_scale_image,
    )


def elbo_kldiv(param_nn: jax.Array, log_precision, log_scale_image, *, R, D):
    """The Kullback-Leibler divergence for the VIKING ELBO."""
    alpha_tr_sigma = (
        R + jnp.exp(log_precision) * (D - R) * jnp.exp(log_scale_image) ** 2
    )
    logdet_sigma = -R * log_precision + 2 * (D - R) * log_scale_image
    kl = 0.5 * (
        alpha_tr_sigma
        - D
        + jnp.exp(log_precision) * jnp.sum(jnp.square(param_nn))
        - D * log_precision
        - logdet_sigma
    )
    return kl


def as_elbo_loss(loss_fn: Callable):
    """This wraps a target loss function into the VIKING ELBO loss.

    The returned function has signature:

    .. code:: python

      elbo_loss(
          posterior: KernelImagePosterior,
          inputs: jax.Array,
          targets: jax.Array,
          *,
          key,
          num_mc_samples=1,
      )

    Where ``posterior`` is an instance of
    :py:class:`KernelImagePosterior`; ``inputs`` is a batch of input
    data; ``targets`` is a batch of output targets; ``key`` is a key
    generated from :py:mod:`jax.random` functions; and
    ``num_mc_samples`` controls how many posterior samples are drawn
    to estimate the expectation part of the ELBO loss.

    Args:
      loss_fn: The original loss function the model would be trained
               with, with expected signature ``loss_fn(outputs,
               targets)``.

    Returns:
      A function with signature as described above.
    """

    def elbo_loss(
        posterior: KernelImagePosterior,
        inputs: jax.Array,
        targets: jax.Array,
        *,
        key,
        num_mc_samples=1,
    ):
        def logpdf(outputs):
            return jax.vmap(loss_fn)(outputs, targets)

        # The main "optimisation" done here is to sample from the
        # posterior only once and get both (i) an estimate of R, that
        # is, the rank of the kernel projection matrix (i.e., its
        # trace, since it is a projection), correspondingly, the
        # dimensionality of the null space, and (ii) computation of
        # the expectation term using those posterior samples.
        pos_samples = sample(posterior, num_mc_samples, inputs, targets, key=key)
        D = pos_samples.mean.shape[0]

        # Trace estimate:
        #
        # Let P denote the projection matrix onto the kernel space and
        # v ~ N(0, I). Then,
        #     tr(P) ≅ v.T @ P @ v = dot(pos_samples.iso, pos_samples.kernel),
        # where pos_samples.kernel = P @ v and v = pos_samples.iso.
        # The version below is a batched (over num_mc_samples) version
        # of this idea.
        iso_dot_kernel = jax.vmap(jnp.dot)(pos_samples.iso, pos_samples.kernel)
        R = jnp.mean(iso_dot_kernel, axis=-1)
        R = jnp.clip(R, max=D - 1)
        R = jax.lax.stop_gradient(R)

        # Reuse samples from trace estimation step to compute the
        # (negative) log-likelihood. The following call will add an
        # extra axis at the beginning of `outputs` with the number of
        # posterior samples as its size, such that:
        #   outputs.shape == (num_mc_samples, batch_size, ...)
        outputs = jax.vmap(predict_from_samples, in_axes=(None, None, 0), out_axes=1)(
            posterior, pos_samples, inputs
        )
        losses = jnp.sum(jax.vmap(logpdf)(outputs), axis=1)
        expectation = jnp.mean(losses, axis=0)
        kl = vi.elbo_kldiv(
            pos_samples.mean,
            log_precision=posterior.log_precision,
            log_scale_image=posterior.log_scale_image,
            R=R,
            D=D,
        )
        loss_value = expectation + posterior.beta * kl
        return loss_value, ELBOInfo(
            expectation=expectation,
            kl=posterior.beta * kl,
            outputs=outputs,
            projection_rank=R,
            samples=pos_samples,
        )

    return elbo_loss
