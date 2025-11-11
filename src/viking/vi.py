# TODO(samuel): Ensure we can use model posterior in training and
#               evaluation modes (e.g., for dropout, batchnorm)
import dataclasses
import operator
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, Union

import jax
import jax.numpy as jnp
import optax

from viking import linalg, sampling


class PosteriorSamples(NamedTuple):
    mean: jax.Array
    iso: jax.Array
    kernel: jax.Array
    image: jax.Array
    solve_info: linalg.SolveInfo


class ELBOInfo(NamedTuple):
    """
    This tuple holds useful auxiliary information computed within the
    ELBO loss; see :func:`as_elbo_loss`.
    """

    expectation: float
    """The value of the expectation part of the ELBO."""

    kl: float
    """The value of the Kullback-Leibler divergence part of the ELBO."""

    outputs: jax.Array
    """The outputs (predictions) computed by every posterior sample."""

    projection_rank: float
    """The estimated rank of the kernel (null) space."""

    samples: PosteriorSamples
    """The posterior samples."""


class ELBOState(NamedTuple):
    """
    This tuple carries necessary information to optimize the VIKING
    ELBO.
    """

    iso_samples: jax.Array
    """The original isotropic Gaussian samples that are to be
    projected onto the kernel of the model (or loss) GGN matrix."""

    kernel_samples: jax.Array
    """These are the projected iso_samples, to be updated for every
    gradient step update (when there is a new minibatch of data
    available)."""

    opt_state: optax.OptState
    """An :mod:`optax` state object, as usual for gradient step
    updates with an optimiser."""


def _make_wrapped_apply_fn(
    apply_fn: Callable, unflatten_fn: Callable, is_linearized: bool = False
):
    """
    Wraps the ``apply_fn`` argument into a calling convention used for
    both linearised predictions or regular predictions. The returned
    function is for internal use.

    Returns:
      A function ``f`` with signature

      .. code:: python
        f(theta_hat, theta, x_eval)

      where ``theta_hat`` is the linearisation point, ``theta`` is
      where the linearisation should be evaluated at and ``x_eval`` is
      a single input instance.
    """

    if not is_linearized:
        return lambda _, p, x: apply_fn(unflatten_fn(p), x)

    # We linearise around `theta_hat` and approximate on `theta`
    def linearized_apply_fn(theta_hat, theta, x_eval):
        def apply_at_param(theta):
            return apply_fn(unflatten_fn(theta), x_eval)

        outputs = apply_at_param(theta_hat)
        _, lin_outputs = jax.jvp(apply_at_param, (theta_hat,), (theta - theta_hat,))
        return outputs + lin_outputs

    return linearized_apply_fn


_JAX_STATIC_METADATA = {"static": True}


def _decl_static():
    return dataclasses.field(metadata=_JAX_STATIC_METADATA)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class KernelImagePosterior:
    """
    This object holds information about the VIKING approximate
    posterior.
    """

    params: Any
    """A pytree of the model parameters; the posterior mean."""

    log_precision: jax.Array
    """The prior precision (scalar), which always equals $-\\frac12 \\log
    \\sigma_{\\mathrm{ker}}$."""

    log_scale_image: jax.Array
    """The scale (standard deviation, scalar) of the image component of the
    approximate posterior."""

    gamma: float = _decl_static()
    """Controls the amount of injected noise while updating the kernel
    samples during ELBO optimisation.

    See also :class:`ELBOState`."""

    beta: float = _decl_static()
    """
    Scaling factor applied to the Kullback-Leibler divergence
    component of the ELBO.

    .. tip::

      Divide the intended value by the number of minibatches in your
      problem (if using minibatches) to prevent the KL term from
      dominating the ELBO.
    """

    flatten_fn: Callable = _decl_static()
    """Function used to flatten the parameter pytree into a vector."""

    unflatten_fn: Callable = _decl_static()
    """Function used to transform a parameter vector into a pytree
    usable for a model object."""

    loss_fn: Callable = _decl_static()
    """The (optional) loss function of the problem of interest. If not
    ``None``, the GGN matrix used will be based on the loss-Jacobian."""

    _project: Callable = _decl_static()
    _wrapped_apply: Callable = _decl_static()

    @property
    def num_params(self) -> int:
        """Convenience property that returns the number of parameters
        in :attr:`params`."""
        sizes = jax.tree.map(lambda p: p.size if hasattr(p, "size") else 0, self.params)
        return jax.tree.reduce(operator.add, sizes)


def make_posterior(
    apply_fn: Callable,
    params: Any,
    flatten_fn: Callable,
    log_precision: Union[float, jax.Array] = 0.0,
    log_scale_image: Union[float, jax.Array] = -2.0,
    projection_fn: Callable = linalg.projection_kernel_lsmr(),
    gamma: float = 1.0,
    beta: float = 1.0,
    loss_fn: Callable = None,
    is_linearized: bool = True,
    custom_vjp: bool = True,
) -> KernelImagePosterior:
    """
    Performs the necessary intialization for a
    :class:`KernelImagePosterior` object.

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
                     :func:`viking.linalg.projection_kernel_normaleq`
                     or
                     :func:`viking.linalg.projection_kernel_lsmr`.
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
      A ready-to-use :class:`KernelImagePosterior` object.

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
    wrapped_apply_fn = _make_wrapped_apply_fn(
        apply_fn, unflatten_fn, is_linearized=is_linearized
    )
    return KernelImagePosterior(
        params=params,
        log_precision=log_precision,
        log_scale_image=log_scale_image,
        gamma=gamma,
        beta=beta,
        flatten_fn=flatten_fn,
        unflatten_fn=unflatten_fn,
        loss_fn=loss_fn,
        _project=projection_fn,
        _wrapped_apply=wrapped_apply_fn,
    )


def make_elbo_state(
    posterior: KernelImagePosterior,
    opt_state: optax.OptState,
    num_mc_samples: int = 1,
    loader: Iterable = None,
    *,
    key,
):
    """
    Prepares and returns a new instance of :class:`ELBOState`.

    Args:
      posterior: A :class:`KernelImagePosterior` object
      opt_state: A :class:`optax.OptState` object
      num_mc_samples: Number of posterior samples to work with; notice that this means this many copies of the model will be held in memory
      loader: An :class:`Iterable` object that will be used to perform initial alternating projections on, if provided
      key: A :class:`jax.random.PRNGKey` object
    """
    params_vec, _ = posterior.flatten_fn(posterior.params)
    iso_samples = jax.random.normal(key, (num_mc_samples, len(params_vec)))
    if loader is None:
        # Samples in kernel space are initially set as those just
        # sampled from the isotropic Gaussian because we don't have
        # access to data. This should be updated externally using
        # `sampling.alternating_projections` before use. The ELBOState
        # object should not be used at least until after one call to
        # `sample_with_batch`.
        kernel_samples = iso_samples
    else:
        kernel_samples = sampling.alternating_projections(
            posterior, iso_samples, loader
        )
    return ELBOState(
        iso_samples=iso_samples,
        kernel_samples=kernel_samples,
        opt_state=opt_state,
    )


def _make_samples_from_parts(
    param_vec: jax.Array,
    iso_samples: jax.Array,
    kernel_samples: jax.Array,
    info: linalg.SolveInfo = None,
) -> PosteriorSamples:
    image_samples = iso_samples - kernel_samples
    return PosteriorSamples(
        mean=param_vec,
        iso=iso_samples,
        kernel=kernel_samples,
        image=image_samples,
        solve_info=info,
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
    iso_samples = jax.random.normal(key, (num_samples, posterior.num_params))
    kernel_projection = posterior._project(param_vec, x, y)
    kernel_samples, info = jax.vmap(kernel_projection)(iso_samples)
    return _make_samples_from_parts(param_vec, iso_samples, kernel_samples, info)


def sample_with_loader(
    posterior: KernelImagePosterior,
    num_samples: int,
    loader: Iterable,
    *,
    key,
) -> PosteriorSamples:
    iso_samples = jax.random.normal(key, (num_samples, posterior.num_params))
    kernel_samples = sampling.alternating_projections(posterior, iso_samples, loader)
    param_vec, _ = posterior.flatten_fn(posterior.params)
    return _make_samples_from_parts(param_vec, iso_samples, kernel_samples)


def sample_with_batch(
    posterior: KernelImagePosterior,
    state: ELBOState,
    batch_x: jax.Array,
    batch_y: jax.Array = None,
    *,
    key,
) -> PosteriorSamples:
    # Eq. (18)
    eta_samples = jax.random.normal(key, state.kernel_samples.shape)
    noisy_samples = (
        jnp.sqrt(posterior.gamma) * state.kernel_samples
        + jnp.sqrt(1.0 - posterior.gamma) * eta_samples
    )
    param_vec, _ = posterior.flatten_fn(posterior.params)
    project_onto_kernel = posterior._project(param_vec, batch_x, batch_y)
    kernel_samples, info = jax.vmap(project_onto_kernel)(noisy_samples)
    return _make_samples_from_parts(param_vec, state.iso_samples, kernel_samples, info)


def predict(
    posterior: KernelImagePosterior, samples: PosteriorSamples, x_eval: jax.Array
):
    """
    Performs a forward pass (computes model outputs) on the model
    whose approximate posterior is represented by ``posterior``, with
    some ``samples`` from it. This takes care of model linearisation
    as well, depending on how ``posterior`` was initialised (see
    :func:`make_posterior`).

    Args:
      posterior: A :class:`KernelImagePosterior` object
      samples: A :class:`PosteriorSamples` object with the posterior
               samples the evaluation should be performed on
      x_eval: The (single) input to the underlying model

    Returns:
      The outputs of the model(s) parameterised by ``samples``, with
      shape ``(num_samples, output_sizes...)``
    """
    # First, build parameter vectors that can be unflattened into a
    # pytree of model parameters
    log_scale_kernel = -0.5 * posterior.log_precision
    param_vecs = (
        samples.mean
        + jnp.exp(log_scale_kernel) * samples.kernel
        + jnp.exp(posterior.log_scale_image) * samples.image
    )
    return jax.vmap(posterior._wrapped_apply, in_axes=(None, 0, None))(
        samples.mean, param_vecs, x_eval
    )


def predict_on_batch(
    posterior: KernelImagePosterior, samples: PosteriorSamples, x_eval_batch: jax.Array
):
    """
    Identical to :func:`predict`, but ``x_eval_batch`` is assumed to
    be batched.

    Returns:
      The outputs of the model(s) parameterised by ``samples``, with
      shape ``(num_samples, batch_size, output_sizes...)``
    """
    # The following call will add an extra axis with the "batch size"
    # of `x_eval`, as specified by `out_axes`
    return jax.vmap(predict, in_axes=(None, None, 0), out_axes=1)(
        posterior, samples, x_eval_batch
    )


def elbo_kldiv(param_vec: jax.Array, log_precision, log_scale_image, *, R, D):
    """The Kullback-Leibler divergence term for the VIKING ELBO."""
    alpha_tr_sigma = (
        R + jnp.exp(log_precision) * (D - R) * jnp.exp(log_scale_image) ** 2
    )
    logdet_sigma = -R * log_precision + 2 * (D - R) * log_scale_image
    kl = 0.5 * (
        alpha_tr_sigma
        - D
        + jnp.exp(log_precision) * jnp.sum(jnp.square(param_vec))
        - D * log_precision
        - logdet_sigma
    )
    return kl


def as_elbo_loss(loss_fn: Callable, is_batched: bool = True):
    """This wraps a target loss function into the (possibly batched) VIKING ELBO loss.

    The returned *batched* function has signature:


    .. code:: python

      elbo_loss(
          posterior: KernelImagePosterior,
          state: ELBOState,
          inputs: jax.Array,
          targets: jax.Array,
          *,
          key,
      )

    The returned *full-batch* function has signature:

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
    :class:`KernelImagePosterior`; ``state`` is an instance of
    :class:`ELBOState`; ``inputs`` is a batch of input data;
    ``targets`` is a batch of output targets; ``key`` is a key
    generated from :mod:`jax.random` functions; and ``num_mc_samples``
    controls how many posterior samples are drawn to estimate the
    expectation part of the ELBO loss.

    Args:
      loss_fn: The original loss function the model would be trained
               with, with expected signature ``loss_fn(outputs,
               targets)``.

    Returns:
      A function with signature as described above.

    """

    def batched_elbo_loss(
        posterior: KernelImagePosterior,
        state: ELBOState,
        inputs: jax.Array,
        targets: jax.Array,
        *,
        key,
    ):
        pos_samples = sample_with_batch(posterior, state, inputs, targets, key=key)
        loss_value, info = _elbo_loss_with_samples(
            posterior, pos_samples, inputs, targets
        )
        state = state._replace(kernel_samples=pos_samples.kernel)
        return loss_value, (state, info)

    def elbo_loss(
        posterior: KernelImagePosterior,
        inputs: jax.Array,
        targets: jax.Array,
        *,
        key,
        num_mc_samples=1,
    ):
        pos_samples = sample(posterior, num_mc_samples, inputs, targets, key=key)
        return _elbo_loss_with_samples(posterior, pos_samples, inputs, targets)

    def _elbo_loss_with_samples(
        posterior: KernelImagePosterior,
        pos_samples: PosteriorSamples,
        inputs: jax.Array,
        targets: jax.Array,
    ):
        def logpdf(outputs):
            return jax.vmap(loss_fn)(outputs, targets)

        # The main "optimisation" done here is to sample from the
        # posterior only once and get both (i) an estimate of R, that
        # is, the rank of the kernel projection matrix (i.e., its
        # trace, since it is a projection), correspondingly, the
        # dimensionality of the null space, and (ii) computation of
        # the expectation term using those posterior samples.
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
        # (negative) log-likelihood.
        outputs = predict_on_batch(posterior, pos_samples, inputs)
        losses = jnp.sum(jax.vmap(logpdf)(outputs), axis=1)
        expectation = jnp.mean(losses, axis=0)
        kl = elbo_kldiv(
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

    if is_batched:
        return batched_elbo_loss
    return elbo_loss
