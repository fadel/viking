"""
Classification on two moons
===========================

This example shows how to build a classifier using Equinox and how to
use VIKING for parameter inference and approximate posterior sampling
in this setting.
"""

# %%
# First, import the necessary libraries.
import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
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
parser.add_argument("--num-data-points", type=int, default=60)
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
    "--num-hidden", type=int, default=16, help="Number of hidden units in the MLP"
)
parser.add_argument(
    "--depth",
    type=int,
    default=3,
    help="Number layers in the MLP (excluding input layer)",
)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--num-epochs", type=int, default=500)
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
parser.add_argument("--beta", type=float, default=1e-2)
parser.add_argument(
    "--is-linearized", action=argparse.BooleanOptionalAction, default=True
)
args = parser.parse_args()


# %%
# Data
# ----
# This example uses the ``make_moons`` (as implemented in `scikit-learn`_, translated trivially to JAX) procedure to produce its own data to be trained on.
#
# .. _scikit-learn: https://scikit-learn.org/
def make_moons(num_samples=100, noise=0.05, normalize=True, *, key):
    # NOTE: This function is adapted from scikit-learn.
    # License: BSD 3-Clause
    # Copyright (c) 2007-2024 The scikit-learn developers.
    num_samples_out = num_samples // 2
    num_samples_in = num_samples - num_samples_out
    outer_circ_x = jnp.cos(jnp.linspace(0, jnp.pi, num_samples_out))
    outer_circ_y = jnp.sin(jnp.linspace(0, jnp.pi, num_samples_out))
    inner_circ_x = 1 - jnp.cos(jnp.linspace(0, jnp.pi, num_samples_in))
    inner_circ_y = 1 - jnp.sin(jnp.linspace(0, jnp.pi, num_samples_in)) - 0.5

    X = jnp.vstack(
        [jnp.append(outer_circ_x, inner_circ_x), jnp.append(outer_circ_y, inner_circ_y)]
    ).T
    y = jnp.hstack(
        [
            jnp.zeros(num_samples_out, dtype=jnp.int32),
            jnp.ones(num_samples_in, dtype=jnp.int32),
        ]
    )

    if noise is not None:
        X += jax.random.normal(key, X.shape) * noise

    if normalize:
        offsets = 0.5
        X_min = jnp.amin(X, axis=0, keepdims=True) - offsets
        X_max = jnp.amax(X, axis=0, keepdims=True) + offsets
        X = (X - X_min) / (X_max - X_min)

    return X, y


key = jax.random.PRNGKey(seed=args.seed)
key, key_data = jax.random.split(key)
x, y = make_moons(num_samples=args.num_data_points, key=key_data)


# %%
# Model
# -----
#
# We use a simple MLP (multi-layer perceptron) in this problem, defined below. Two differences from :doc:`sinusoid` should be noted:
#
# 1. A custom flattening function ``eqx_flatten``, since the model is built using `equinox`_;
# 2. and the use of the ``loss_fn`` argument of :func:`viking.vi.make_posterior`.
#
# .. _equinox: https://docs.kidger.site/equinox/
def eqx_flatten(model):
    params, static = eqx.partition(model, eqx.is_array)
    vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    return vec, lambda v: eqx.combine(unflatten_fn(v), static)


key, key_model = jax.random.split(key)
model = eqx.nn.MLP(
    in_size=2,
    out_size="scalar",
    width_size=args.num_hidden,
    depth=args.depth,
    activation=jax.nn.gelu,
    key=key_model,
)
posterior = vi.make_posterior(
    lambda p, x: p(x),
    model,
    flatten_fn=eqx_flatten,
    log_precision=args.log_precision,
    log_scale_image=args.log_scale_image,
    beta=args.beta,
    loss_fn=optax.losses.sigmoid_binary_cross_entropy,
    is_linearized=args.is_linearized,
)
print(f"Number of parameters: {posterior.num_params:,d}")


# %%
# Optimisation
# ------------
#
# Optimisation proceeds as with :doc:`sinusoid`, where we optimise the VIKING ELBO by wrapping the intended loss function with the :func:`viking.vi.as_elbo_loss` function.
#
# One detail to notice is the use of ``@eqx.filter_jit`` to annotate
# the ``train_elbo_step`` function, instead of the usual
# :func:`jax.jit`. This is simply due to our model construction.
loss_fn = vi.as_elbo_loss(posterior.loss_fn, is_batched=False)
value_and_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
optimizer = optax.nadam(args.lr)


@eqx.filter_jit
def train_elbo_step(posterior, opt_state, inputs, targets, key):
    (loss_value, info), loss_grad = value_and_grad_fn(
        posterior,
        inputs=inputs,
        targets=targets,
        key=key,
        num_mc_samples=args.num_mc_samples,
    )
    updates, opt_state = optimizer.update(loss_grad, opt_state, posterior)
    posterior = eqx.apply_updates(posterior, updates)
    return posterior, (info, opt_state)


def make_description(posterior, info):
    items = [
        f"E[]={info.expectation:.3e}",
        f"kl={info.kl:.3e}",
        f"σ_ker={jnp.exp(-0.5 * posterior.log_precision):.2e}",
        f"σ_im={jnp.exp(posterior.log_scale_image):.2e}",
        f"R={info.projection_rank:.1f}",
    ]
    return ", ".join(items)


posterior_params, posterior_static = eqx.partition(posterior, eqx.is_array)
opt_state = optimizer.init(posterior_params)
for step_elbo in range(1, args.num_epochs + 1):
    key, subkey = jax.random.split(key)
    posterior, (info, opt_state) = train_elbo_step(
        posterior,
        opt_state,
        x,
        y,
        key=subkey,
    )
    if step_elbo % 100 == 0 or step_elbo == args.num_epochs:
        print(f"[{step_elbo:03d}] {make_description(posterior, info)}")


# %%
# Plotting the results
# --------------------
#
# For plotting, we need to display quite a few things, so we start by
# preparing a function that can handle the repetitive tasks. In order
# to better understand the VIKING approximate posterior, we ought to
# see what the individual contributions of the kernel and image
# subspaces.
def plot_posterior_samples(title, ax1, ax2, x, posterior, pos_samples):
    # We evaluate the (approximate) predictive posterior over a grid
    # in input space
    x_eval_1d = jnp.linspace(0, 1, 100)
    mesh_x, mesh_y = jnp.meshgrid(x_eval_1d, x_eval_1d)
    x_eval = jnp.stack((mesh_x, mesh_y)).reshape((2, -1)).T
    y_eval = jax.vmap(posterior.params)(x_eval)
    y_eval_cls = jax.nn.sigmoid(y_eval)
    logits_samples = vi.predict_on_batch(posterior, pos_samples, x_eval)
    logits_samples_std = jnp.std(logits_samples, axis=0)
    logits_samples_mean = jnp.mean(logits_samples, axis=0)
    logits_mean_field = logits_samples_mean / jnp.sqrt(
        1 + jnp.pi / 8 * jnp.square(logits_samples_std)
    )
    # Uncertainties computed as in:
    # https://www.jmlr.org/papers/v24/22-0479.html
    logits_unc = 1.0 - 2.0 * jnp.abs(jax.nn.sigmoid(logits_mean_field) - 0.5)

    ax = ax1
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    mesh_shape = mesh_x.shape
    m = ax.pcolormesh(
        mesh_x,
        mesh_y,
        logits_unc.reshape(mesh_shape),
        alpha=0.75,
        vmin=0.0,
        vmax=1.0,
        cmap=plt.cm.magma,
    )
    m.set_rasterized(True)
    plt.colorbar(m)

    text_color = matplotlib.rcParams["text.color"]
    background_color = matplotlib.rcParams["axes.facecolor"]
    num_pos_samples = pos_samples.kernel.shape[0]
    ax = ax2
    ax.set_xticks([])
    ax.set_yticks([])
    ax.contour(
        mesh_x,
        mesh_y,
        y_eval_cls.reshape(mesh_shape),
        levels=[0.5],
        colors=[text_color],
    )
    for i in range(num_pos_samples):
        pred_model_i = jax.nn.sigmoid(logits_samples[i])
        ax.contour(
            mesh_x,
            mesh_y,
            pred_model_i.reshape(mesh_shape),
            levels=[0.5],
            colors=["C0"],
            alpha=min(15 / num_pos_samples, 1.0),
        )

    # Overlay data points on both plots
    for ax in ax1, ax2:
        ax.scatter(
            x[:, 0],
            x[:, 1],
            color=text_color,
            edgecolor=background_color,
            alpha=0.8,
        )


# %%
# Finally, perform the actual plotting step. Our strategy here is to
# call the same function three times: with the proper posterior
# samples and then either with the kernel component missing or the
# image component missing.
key, key_sample = jax.random.split(key)
pos_samples = vi.sample(posterior, args.num_plot_samples, x, y, key=key_sample)

nrows = 2
ncols = 3
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(4 * ncols, 3 * nrows),
    layout="constrained",
)
fig.suptitle(
    r"$\sigma_{{ker}}={:.3f}$ and $\sigma_{{im}}={:.3f}$".format(
        jnp.exp(-0.5 * posterior.log_precision), jnp.exp(posterior.log_scale_image)
    )
)
axes[0, 0].set_ylabel("Norm. uncertainties")
axes[1, 0].set_ylabel("Decision boundaries")
plot_posterior_samples(
    "Complete posterior\n({})".format(
        r"$\sigma_{{ker}} \theta_{{ker}} + \sigma_{{im}} \theta_{{im}}$"
    ),
    axes[0, 0],
    axes[1, 0],
    x,
    posterior,
    pos_samples,
)
plot_posterior_samples(
    "Kernel component ({})".format(r"$\sigma_{{ker}} \theta_{{ker}}$"),
    axes[0, 1],
    axes[1, 1],
    x,
    posterior,
    pos_samples._replace(image=jnp.zeros_like(pos_samples.image)),
)
plot_posterior_samples(
    "Image component ({})".format(r"$\sigma_{{im}} \theta_{{im}}$"),
    axes[0, 2],
    axes[1, 2],
    x,
    posterior,
    pos_samples._replace(kernel=jnp.zeros_like(pos_samples.kernel)),
)
plt.show()
