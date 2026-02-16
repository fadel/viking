"""
Sinusoidal regression
=====================

In this example, we build a simple MLP with a couple of functions and a :class:`typing.NamedTuple` to hold parameters. Then, we fit it to a small set of points sampled from a sinusoidal curve, with a gap between both ends of the data.
"""

# %%
# We start with the necessary imports.
import argparse
from typing import NamedTuple

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from viking import vi


# %%
# Model definition
# ----------------
#
# Our model parameters are grouped in the following class, while
# :func:`make_mlp` produces an initialisation and application
# function, respectively. This follows the style of the :mod:`stax`
# library from JAX to keep things as minimal as possible.
#
# .. note::
#
#   We assume the model application function takes a single data
#   point.
class ModelParams(NamedTuple):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array


def make_mlp(num_hidden):
    def init_fn(num_inputs=1, *, key):
        k_w1, k_b1, k_w2, k_b2, k_w3, k_b3 = jax.random.split(key, num=6)
        return ModelParams(
            w1=jax.random.normal(k_w1, shape=(num_hidden, num_inputs)),
            b1=jax.random.normal(k_b1, shape=(num_hidden,)),
            w2=jax.random.normal(k_w2, shape=(num_hidden, num_hidden)),
            b2=jax.random.normal(k_b2, shape=(num_hidden,)),
            w3=jax.random.normal(k_w3, shape=(num_hidden,)),
            b3=jax.random.normal(k_b3, shape=()),
        )

    def apply_fn(params: ModelParams, x_single):
        x = jnp.broadcast_to(x_single, (1,))
        x = params.w1 @ x + params.b1
        x = jnp.tanh(x)
        x = params.w2 @ x + params.b2
        x = jnp.tanh(x)
        x = params.w3 @ x + params.b3
        return x

    return init_fn, apply_fn


# %%
# Argument parsing
# ----------------
#
# For neatness, just in case we need to play with different settings
# in an automated way. It also helps us keep in mind that
# configuration variables belong in ``args``.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-data-points", type=int, default=10)
parser.add_argument("--num-gap-points", type=int, default=10)
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
    "--log-scale-noise",
    type=float,
    default=1e-2,
    help="Controls observation noise (for the likelihood)",
)
parser.add_argument(
    "--num-hidden", type=int, default=8, help="Number of hidden units in the MLP"
)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--num-epochs", type=int, default=3_000)
parser.add_argument(
    "--num-mc-samples",
    type=int,
    default=20,
    help="Number of posterior samples drawn (in training) to estimate the expectation term of ELBO",
)
parser.add_argument(
    "--num-plot-samples",
    type=int,
    default=100,
    help="Number of posterior samples drawn for plotting",
)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument(
    "--is-linearized", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()


# %%
# Data
# ----
#
# This will set up ``x`` and ``y`` to hold our input data and regression
# targets, respectively.
def make_wave(key, num: int):
    std = jnp.linspace(1e-3, 1e0, num)
    x = jnp.linspace(0.35, 0.65, num)
    y = 5 * jnp.sin(10 * x)

    z = jax.random.normal(key, shape=y.shape)
    return x, y + std * z


key = jax.random.PRNGKey(seed=args.seed)
key, subkey = jax.random.split(key)
x, y = make_wave(subkey, num=args.num_data_points + args.num_gap_points)
# The following trick allows us to create a "gap" in our sampled points.
first_half = args.num_data_points // 2
second_half = args.num_data_points - first_half
x = jnp.concatenate((x[:first_half], x[-second_half:]))
y = jnp.concatenate((y[:first_half], y[-second_half:]))

# %%
# Model setup
# -----------
#
# This is where we start using the library. After initialising the
# model application function and its parameters, we pass the necessary
# information to the :func:`viking.vi.make_posterior` function, which
# takes care of preparing a :class:`viking.vi.KernelImagePosterior`
# object to hold the necessary state of our approximate posterior (and
# model parameters).
#
# This function can optionally take a loss function as argument, which
# makes our approximate posterior use the loss-Jacobian instead of
# model Jacobian for VIKING. We leave that out for this model, which
# is more straightforward.
key, subkey = jax.random.split(key)
model_init_fn, model_apply_fn = make_mlp(num_hidden=args.num_hidden)
model_params = model_init_fn(num_inputs=1, key=subkey)
posterior = vi.make_posterior(
    model_apply_fn,
    model_params,
    log_precision=args.log_precision,
    log_scale_image=args.log_scale_image,
    flatten_fn=jax.flatten_util.ravel_pytree,
    beta=args.beta,
    is_linearized=args.is_linearized,
)

D = posterior.num_params
print(f"Number of parameters: {D}")


# %%
# Optimisation
# ------------
#
# Now, we need to set up our loss function to be optimised. This makes
# use of the convenient :func:`viking.vi.as_elbo_loss` function, which
# transforms a regular loss function into one that can be wrapped into
# the VIKING ELBO, allowing us to optimise for that instead, without
# writing it down manually every time.
#
# .. note::
#
#   Wrapping your loss function this way will require a few extra
#   arguments, so be sure to check out the documentation of
#   :func:`viking.vi.as_elbo_loss`.
log_scale_noise = jnp.asarray(args.log_scale_noise)


def loss_single(pred, u):
    return -jax.scipy.stats.norm.logpdf(u, loc=pred, scale=log_scale_noise)


optimizer = optax.adam(args.lr)
elbo_loss_fn = vi.as_elbo_loss(loss_single, is_batched=False)
value_and_grad_fn = jax.value_and_grad(elbo_loss_fn, has_aux=True)


@jax.jit
def train_step(posterior, opt_state, inputs, targets, *, key):
    (loss_value, info), loss_grad = value_and_grad_fn(
        posterior,
        inputs=inputs,
        targets=targets,
        key=key,
        num_mc_samples=args.num_mc_samples,
    )
    updates, opt_state = optimizer.update(loss_grad, opt_state, posterior)
    posterior = optax.apply_updates(posterior, updates)
    return posterior, (info, opt_state)


# %%
# The ``make_description`` function prepares a useful logging message
# for our optimisation loop, using the information returned by our
# ELBO optimisation procedure.
def make_description(posterior: vi.KernelImagePosterior, info: vi.ELBOInfo):
    dot = jnp.mean(jax.vmap(jnp.dot)(info.samples.kernel, info.samples.image))
    items = [
        f"E[]={info.expectation:.3e}",
        f"kl={info.kl:.3e}",
        f"σ_ker={jnp.exp(-0.5 * posterior.log_precision):.2e}",
        f"σ_im={jnp.exp(posterior.log_scale_image):.2e}",
        f"dot={dot:.2e}",
        f"R={info.projection_rank:.1f}",
    ]
    return ", ".join(items)


opt_state = optimizer.init(posterior)
for step_elbo in range(1, args.num_epochs + 1):
    key, subkey = jax.random.split(key)
    posterior, (info, opt_state) = train_step(
        posterior,
        opt_state,
        x,
        y,
        key=subkey,
    )
    if step_elbo % 1000 == 0 or step_elbo == args.num_epochs:
        print(f"[{step_elbo:04d}] {make_description(posterior, info)}")


# %%
# Plotting the results
# --------------------
#
# Finally, we visualise the (predictive) samples from our approximate posterior.
x_eval = jnp.linspace(0, 1, 500)
y_eval = jax.vmap(model_apply_fn, in_axes=(None, 0))(posterior.params, x_eval)
key, key_sample = jax.random.split(key)
pos_samples = vi.sample(posterior, args.num_plot_samples, x, y, key=key_sample)
fig, ax = plt.subplots(
    nrows=2,
    sharex=True,
    height_ratios=(1.8, 1.2),
)
for a in ax[0], ax[1]:
    for s in "top", "right", "bottom":
        a.spines[s].set_visible(False)
ax[0].set_xticks([])
ax[1].grid(which="major", axis="y", alpha=0.75, ls=":")

# Plot samples from the variational approximation
y_preds = vi.predict_on_batch(posterior, pos_samples, x_eval)
for y_pred in y_preds:
    ax[0].plot(x_eval, y_pred, alpha=0.1, color="C0", linewidth=1)

ax[0].plot([], [], color="C0", label="Posterior samples")
ax[0].plot(x_eval, y_eval, color="C3", label="Posterior mean")
ax[0].set_xlim((x_eval[0], x_eval[-1]))
ax[0].plot(x, y, marker="P", linestyle="None", color="black")

marginal_std = jnp.std(y_preds, axis=0)
ax[1].plot(x_eval, marginal_std, color="C0", linewidth=1)
ax[1].set_ylabel("Standard deviation")
ax[1].set_ylim(0, jnp.max(marginal_std))
fig.legend(frameon=False)
plt.show()
