"""
Bayesian Optimisation
=====================

This example explores one task which was not covered in the `VIKING paper`_: Bayesian optimisation.

.. _VIKING paper: https://arxiv.org/abs/2510.23684
"""

# %%
# As usual, start with the imports.
import argparse
import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import optax

from viking import vi


# %%
# Argument parsing
# ----------------
#
# For neatness, just in case we need to play with different settings
# in an automated way. It also helps us keep in mind that
# configuration variables belong in ``args``.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--num-sel", type=int, default=5, help="Number of points selected to start"
)
parser.add_argument(
    "--log-precision",
    type=float,
    default=0.0,
    help="Controls (initial) prior precision and σ_ker",
)
parser.add_argument(
    "--log-scale-image", type=float, default=-2.0, help="Controls (initial) σ_im"
)
parser.add_argument(
    "--num-hidden", type=int, default=8, help="Number of hidden units in the MLP"
)
parser.add_argument(
    "--depth",
    type=int,
    default=2,
    help="Number layers in the MLP (excluding input layer)",
)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--num-epochs", type=int, default=2_000)
parser.add_argument("--num-epochs-bo", type=int, default=100)
parser.add_argument("--bo-steps", type=int, default=3)
parser.add_argument(
    "--num-mc-samples",
    type=int,
    default=20,
    help="Number of posterior samples drawn (in training) to estimate the expectation term of ELBO",
)
parser.add_argument(
    "--num-eval-mc-samples",
    type=int,
    default=100,
    help="Number of posterior samples drawn for evaluation",
)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument(
    "--is-linearized", action=argparse.BooleanOptionalAction, default=True
)

args = parser.parse_args()
key = jax.random.PRNGKey(seed=args.seed)


# %%
# Data
# ----
#
# Data in this example is quite straightforward, as we work with a
# simple function we want to sample from.
def fun(x):
    """The 'true' function we want to work with, but can only sample from."""
    # fun = lambda x: -0.1*(1.5*x)**2 + 0.1*x
    return jnp.exp(-0.5 * x**2) * jnp.cos(2 * jnp.pi * x)


def make_wave(key, num: int):
    x = jnp.linspace(-jnp.pi, jnp.pi, num)
    y = fun(x)
    # z = jax.random.normal(key, shape=y.shape)
    return x, y  # + 0.1 * z


def select_points(x, y, num: int, key):
    """Selects a random subset of data points."""
    idx = jax.random.choice(key, x.shape[0], shape=(num,), replace=False)
    return x[idx, ...], y[idx]


key, subkey = jax.random.split(key)
x_full, y_full = make_wave(subkey, 1000)
key, subkey = jax.random.split(key)
x_sel, y_sel = select_points(x_full, y_full, args.num_sel, subkey)


# %%
# Model and optimiser setup
# -------------------------
#
# For the model, things are also kept simple. A multi-layer perceptron
# (MLP) with a few parameters is enough for our goals.
def eqx_flatten(model):
    params, static = eqx.partition(model, eqx.is_array)
    vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    return vec, lambda v: eqx.combine(unflatten_fn(v), static)


key, subkey = jax.random.split(key)
model = eqx.nn.MLP(
    in_size="scalar",
    out_size="scalar",
    width_size=args.num_hidden,
    depth=args.depth,
    activation=jnp.tanh,
    key=subkey,
)
posterior = vi.make_posterior(
    lambda p, x: p(x),
    model,
    flatten_fn=eqx_flatten,
    log_precision=args.log_precision,
    log_scale_image=args.log_scale_image,
    beta=args.beta,
    is_linearized=args.is_linearized,
)


optimizer = optax.nadam(args.lr)
posterior_params, posterior_static = eqx.partition(posterior, eqx.is_array)
opt_state = optimizer.init(posterior_params)
print(f"Number of parameters: {posterior.num_params:,d}")


# %%
# Training loop setup
# -------------------
#
# Optimisation proceeds as with :doc:`sinusoid`, where we optimise the
# VIKING ELBO by wrapping the intended loss function with the
# :func:`viking.vi.as_elbo_loss` function. Here we additionally
# prepare a ``train`` function as the training loop needs to be called
# a few times during the Bayesian optimisation loop.
#
# One detail to notice is the use of ``@eqx.filter_jit`` to annotate
# the ``train_elbo_step`` function, instead of the usual
# :func:`jax.jit`. This is simply due to our model construction.


# The negative log-likelihood function
def loss_single(pred, u):
    return -jax.scipy.stats.norm.logpdf(u, loc=pred, scale=1e-2)


loss_fn = vi.as_elbo_loss(loss_single, is_batched=False)
value_and_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
optimizer = optax.nadam(args.lr)


@eqx.filter_jit
def train_elbo_step(posterior, opt_state, inputs, targets, key):
    out, loss_grad = value_and_grad_fn(
        posterior,
        inputs=inputs,
        targets=targets,
        key=key,
        num_mc_samples=args.num_mc_samples,
    )
    updates, opt_state = optimizer.update(loss_grad, opt_state, posterior)
    posterior = eqx.apply_updates(posterior, updates)
    return out, posterior, opt_state


def train(posterior, opt_state, x_sel, y_sel, num_epochs, key):
    for step_elbo in range(1, num_epochs + 1):
        key, subkey = jax.random.split(key)
        (loss_value, info), posterior, opt_state = train_elbo_step(
            posterior,
            opt_state,
            x_sel,
            y_sel,
            key=subkey,
        )
        if step_elbo % 10 == 0 or step_elbo == num_epochs:
            print(
                "[{:04d}] E[]={:.3e}, kl={:.3e}, σ_ker={:.2e}, σ_im={:.2e}, R={:.1f}".format(
                    step_elbo,
                    info.expectation,
                    info.kl,
                    jnp.exp(-0.5 * posterior.log_precision),
                    jnp.exp(posterior.log_scale_image),
                    info.projection_rank,
                )
            )
    return posterior, opt_state, key


# %%
# Fit on initial selection of points
# ----------------------------------
#
# This is a first attempt at getting a well-fitted model, which we
# will use for uncertainty quantification and, consequently, the
# criteria for sampling more from our `black-box` function ``fun``.
posterior, opt_state, key = train(
    posterior, opt_state, x_sel, y_sel, args.num_epochs, key
)
key, subkey = jax.random.split(key)
pos_samples = vi.sample(posterior, args.num_eval_mc_samples, x_sel, y=y_sel, key=subkey)


# %%
# UCB function and plotting BO state
# ----------------------------------
#
# The UCB function below keeps things simple and has a small hack with
# the ``sum`` and ``negate`` arguments so we can easily visualise it
# and use it for optimisation.
#
# Here we also define the function that will be used to visualise the
# state of the current (Bayesian) optimisation of ``fun``. We use that
# to also plot the state before actually performing BO.
def ucb(x_pred, posterior_params, pos_samples, kappa=1.0, negate=False, sum=False):
    """Upper confidence bound function."""
    posterior = eqx.combine(posterior_params, posterior_static)
    samples = jax.vmap(vi.predict, in_axes=(None, None, 0), out_axes=-1)(
        posterior, pos_samples, x_pred
    )
    y_pred = jnp.mean(samples, axis=0)
    y_std = jnp.std(samples, axis=0)
    acq = y_pred + kappa * y_std
    acq = acq.sum(axis=-1) if sum else acq
    if negate:
        return -acq
    return acq


x_pred = jnp.linspace(-jnp.pi, jnp.pi, 1000)


def plot_bo_state(
    posterior,
    pos_samples,
):
    fig, ax = plt.subplots(figsize=(15, 5), layout="constrained")
    for s in "top", "right":
        ax.spines[s].set_visible(False)
    ax.plot(x_full, y_full, "-", color="black", label="True function", alpha=0.5)
    ax.scatter(x_sel, y_sel, color="black", label="Selected points")
    for i in range(args.num_sel, x_sel.shape[0]):
        bo_step = 1 + i - args.num_sel
        ax.text(x_sel[i], y_sel[i], str(bo_step), size=20)

    samples = jax.vmap(vi.predict, in_axes=(None, None, 0), out_axes=-1)(
        posterior, pos_samples, x_pred
    )
    lines = ax.plot(
        x_pred,
        samples.T,
        color="C0",
        alpha=0.1,
        zorder=-100000,
    )
    ax.plot([], [], color="C0", label="BNN prediction")
    ucb_line = ax.plot(
        x_pred, ucb(x_pred, posterior, pos_samples), color="C1", label="UCB"
    )
    plt.legend(frameon=False)
    plt.show()


plot_bo_state(posterior, pos_samples)


# %%
# BO setup
# --------
#
# The acquisition function ``acq_fn`` will be used to select the next
# point to be sampled from ``fun``.
acq_fn = ft.partial(ucb, kappa=1.0, sum=True, negate=True)
acq_fn = eqx.filter_jit(acq_fn)


def _clip_safe(x, lower, upper):
    return jnp.clip(jnp.asarray(x), lower, upper)


def projection_box(x, hyperparams):
    lower, upper = hyperparams
    return jax.tree_util.tree_map(_clip_safe, x, lower, upper)


def run_solver(
    x0,
    projection_hyperparams,
    *args,
):
    opt_result = jsp.optimize.minimize(acq_fn, x0, args=args, method="BFGS")
    x = projection_box(opt_result.x, projection_hyperparams)
    return opt_result._replace(x=x, fun=acq_fn(x, *args))


# %%
# Main BO loop
# ------------
#
# This is where Bayesian optimisation actually happens. We follow
# textbook BO steps and keep plotting the current state.

# sphinx_gallery_multi_image = "single"
for bo_step in range(args.bo_steps):
    posterior, opt_state, key = train(
        posterior, opt_state, x_sel, y_sel, args.num_epochs_bo, key
    )
    key, subkey = jax.random.split(key)
    pos_samples = vi.sample(
        posterior, args.num_eval_mc_samples, x_sel, y=y_sel, key=key
    )

    x_news, y_news = [], []
    for k in range(len(x_sel)):
        key, subkey = jax.random.split(key)
        opt_result = run_solver(
            x_sel[k] + 0.05 * jax.random.normal(subkey, shape=(1,)),
            [-jnp.pi, jnp.pi],
            eqx.filter(posterior, eqx.is_array),
            pos_samples,
        )
        x_news.append(opt_result.x)
        y_news.append(opt_result.fun.item())
    x_news = jnp.array(x_news).flatten()
    y_news = jnp.array(y_news).flatten()
    idx = jnp.argmin(y_news)
    x_sel = jnp.append(x_sel, x_news[idx])
    y_sel = jnp.append(y_sel, fun(x_news[idx]))
    print("Current x_new = {}".format(x_sel[-1]))
    plot_bo_state(posterior, pos_samples)
