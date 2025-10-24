import argparse
from typing import NamedTuple

import jax.flatten_util
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import optax
import tqdm

from viking import vi


def main(args):
    # %%
    # Create data
    key = jax.random.PRNGKey(seed=args.seed)
    key, subkey = jax.random.split(key)
    x, y = make_wave(subkey, num=args.num_data_points + 10)
    x = jnp.concatenate((x[:5], x[-5:]))
    y = jnp.concatenate((y[:5], y[-5:]))

    # %%
    # Model setup
    key, subkey = jax.random.split(key)
    model_init_fn, model_apply_fn = make_mlp(num_hidden=args.num_hidden)
    model_params = model_init_fn(num_inputs=1, key=subkey)
    log_scale_noise = jnp.asarray(1e-2)
    posterior = vi.make_posterior(
        model_apply_fn,
        model_params,
        log_precision=args.log_precision,
        log_scale_image=args.log_scale_image,
        flatten_fn=jax.flatten_util.ravel_pytree,
        beta=args.beta,
        is_linearized=args.is_linearized,
    )

    D = len(posterior.flatten_fn(posterior.params)[0])
    print(f"Number of parameters: {D}")

    # %%
    # Optimizer setup
    def loss_single(pred, u):
        return -jax.scipy.stats.norm.logpdf(u, loc=pred, scale=log_scale_noise)

    optimizer = optax.adam(args.lr)
    loss_fn = vi.as_elbo_loss(loss_single)
    train_step = make_train_step(
        jax.value_and_grad(loss_fn, has_aux=True),
        optimizer=optimizer,
        num_mc_samples=args.num_mc_samples,
    )
    train_step = jax.jit(train_step)

    # %%
    # Optimize ELBO
    opt_state = optimizer.init(posterior)
    with tqdm.trange(args.epochs_elbo, disable=not args.plot) as progressbar:
        for step_elbo in progressbar:
            key, subkey = jax.random.split(key)
            try:
                (loss_value, info), posterior, opt_state = train_step(
                    posterior,
                    opt_state,
                    x,
                    y,
                    key=subkey,
                )
                progressbar.set_description(make_description(posterior, info))
                if step_elbo % 1000 == 0:
                    progressbar.write(
                        f"[{step_elbo:05d}] {make_description(posterior, info)}"
                    )
            except KeyboardInterrupt:
                break
    print(make_description(posterior, info))

    # %%
    # We plot on this range, starting with the mean prediction
    x_eval = jnp.linspace(0, 1, 500)
    y_eval = jax.vmap(model_apply_fn, in_axes=(None, 0))(posterior.params, x_eval)

    def plot(pos_samples):
        if not args.plot:
            matplotlib.use("Agg")
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
        y_preds = jax.vmap(
            vi.predict_from_samples, in_axes=(None, None, 0), out_axes=-1
        )(posterior, pos_samples, x_eval)
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
        if args.plot:
            plt.savefig("sinusoid.svg", bbox_inches="tight")

    key, key_sample = jax.random.split(key)
    pos_samples = vi.sample(posterior, args.num_plot_samples, x, y, key=key_sample)
    plot(pos_samples)


def make_wave(key, num: int):
    std = jnp.linspace(1e-3, 1e0, num)
    x = jnp.linspace(0.35, 0.65, num)
    y = 5 * jnp.sin(10 * x)

    z = jax.random.normal(key, shape=y.shape)
    return x, y + std * z


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


def make_train_step(value_and_grad_fn, optimizer, num_mc_samples=1):
    def train_step(posterior, opt_state, inputs, targets, *, key):
        out, loss_grad = value_and_grad_fn(
            posterior,
            inputs=inputs,
            targets=targets,
            key=key,
            num_mc_samples=num_mc_samples,
        )
        updates, opt_state = optimizer.update(loss_grad, opt_state, posterior)
        posterior = optax.apply_updates(posterior, updates)
        return out, posterior, opt_state

    return train_step


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-data-points", type=int, default=10)
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
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs-elbo", type=int, default=3_000)
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
