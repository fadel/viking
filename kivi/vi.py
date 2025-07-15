# TODO(samuel): Ensure we can use model posterior in training and
#               evaluation modes (e.g., for dropout, batchnorm)

import collections
import functools as ft
import optax
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from kivi import linalg, sampling, vi


PosteriorSamples = collections.namedtuple(
    "PosteriorSamples", ("mean", "iso", "kernel", "image", "solve_info")
)


ELBOInfo = collections.namedtuple(
    "ELBOInfo", ("expectation", "kl", "preds", "projection_rank", "samples")
)


def elbo_kldiv(param_nn: jax.Array, log_precision, log_scale_image, *, R, D):
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


def make_elbo(loss_fn):
    def elbo_loss(posterior, x, y, *, key, num_mc_samples=1):
        def logpdf(outputs):
            return loss_fn(outputs, y)

        # The main "optimisation" done here is to sample from the
        # posterior only once and get both (i) an estimate of R, that
        # is, the rank of the kernel projection matrix (i.e., its
        # trace, since it is a projection), correspondingly, the
        # dimensionality of the null space, and (ii) computation of
        # the expectation term using those posterior samples.
        pos_samples = posterior.sample(num_mc_samples, x, y, key=key)
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
        # (negative) log-likelihood
        preds = posterior.predict_from_samples(pos_samples, x)
        losses = jnp.sum(jax.vmap(logpdf)(preds), axis=-1)
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
            preds=preds,
            projection_rank=R,
            samples=pos_samples,
        )

    return elbo_loss


def make_train_step(elbo_loss_fn, optimizer, num_mc_samples=1):
    value_and_grad_fn = eqx.filter_value_and_grad(elbo_loss_fn, has_aux=True)

    def train_step(posterior, opt_state, inputs, labels, *, key):
        (loss_value, info), loss_grad = value_and_grad_fn(
            posterior, x=inputs, y=labels, key=key, num_mc_samples=num_mc_samples
        )
        updates, opt_state = optimizer.update(loss_grad, opt_state, posterior)
        posterior = eqx.apply_updates(posterior, updates)
        return loss_value, info, posterior, opt_state

    return train_step


def eqx_flatten(model):
    params, static = eqx.partition(model, eqx.is_array)
    vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    return vec, lambda v: eqx.combine(unflatten_fn(v), static)


class KernelImagePosterior(eqx.Module):
    model: Any
    log_precision: jax.Array
    log_scale_image: jax.Array
    beta: float
    flatten_fn: Callable
    unflatten_fn: Callable
    _projection: Callable
    _sample_outputs: Callable

    def __init__(
        self,
        model: Any,
        log_precision: float = 0.0,
        log_scale_image: float = -2.0,
        solve_normaleq: Callable = linalg.solve_normaleq_cg_fixed_step_reortho(
            maxiter=10
        ),
        beta: float = 1.0,
        flatten_fn: Callable = eqx_flatten,
        loss_fn: Callable = None,
        is_linearized: bool = True,
        use_custom_vjp: bool = True,
    ):
        self.model = model
        self.log_precision = jnp.asarray(log_precision)
        self.log_scale_image = jnp.asarray(log_scale_image)
        self.beta = beta
        self.flatten_fn = flatten_fn
        _, unflatten_fn = flatten_fn(self.model)
        self.unflatten_fn = unflatten_fn
        apply_fn = lambda p, x: p(x)  # FIXME: this is a hack (?)
        self._projection = linalg.projection_kernel(
            apply_fn,
            self.unflatten_fn,
            solve_normaleq,
            loss_fn=loss_fn,
            use_custom_vjp=use_custom_vjp,
        )
        self._sample_outputs = sampling.make_output_sampler(
            apply_fn, self.unflatten_fn, is_linearized=is_linearized
        )

    def wrap_loss(self, loss_fn: Callable, _has_aux=None):
        # if self.loss_fn is not None:
        #     TODO: Decide what to do with self._projection if loss_fn
        #     is now different from self.loss_fn
        return make_elbo(loss_fn)

    def sample(self, num_samples, x, y=None, *, key):
        param_vec, _ = self.flatten_fn(self.model)
        D = len(param_vec)
        kernel_projection = self._projection(param_vec, x, y)
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

    def predict_from_samples(self, samples: PosteriorSamples, x_eval):
        log_scale_kernel = -0.5 * self.log_precision
        return self._sample_outputs(
            x_eval,
            param_mean=samples.mean,
            log_scale_kernel=log_scale_kernel,
            log_scale_image=self.log_scale_image,
            image_samples=samples.image,
            kernel_samples=samples.kernel,
        )
