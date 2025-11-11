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
import tqdm

from viking import vi


def main(args):
    key = jax.random.PRNGKey(seed=args.seed)

    # The negative log-likelihood function
    def loss_single(pred, u):
        return -jax.scipy.stats.norm.logpdf(u, loc=pred, scale=1e-2)

    # %%
    # Data preparation
    key, subkey = jax.random.split(key)
    x_full, y_full = make_wave(subkey, 1000)
    key, subkey = jax.random.split(key)
    x_sel, y_sel = select_points(x_full, y_full, args.num_sel, subkey)

    # %%
    # Model and optimiser setup
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

    train_elbo_step = make_train_step(
        loss_single,
        optimizer=optimizer,
        num_mc_samples=args.num_mc_samples,
    )
    train_elbo_step = eqx.filter_jit(train_elbo_step)
    posterior_params, posterior_static = eqx.partition(posterior, eqx.is_array)
    opt_state = optimizer.init(posterior_params)

    def train(posterior, opt_state, x_sel, y_sel, num_epochs, key):
        progressbar_elbo = tqdm.trange(num_epochs, disable=not args.plot)
        for step_elbo in progressbar_elbo:
            key, subkey = jax.random.split(key)
            try:
                (loss_value, info), posterior, opt_state = train_elbo_step(
                    posterior,
                    opt_state,
                    x_sel,
                    y_sel,
                    key=subkey,
                )
                progressbar_elbo.set_description(
                    "E[]={:.3e}, kl={:.3e}, σ_ker={:.2e}, σ_im={:.2e}, R={:.1f}".format(
                        info.expectation,
                        info.kl,
                        jnp.exp(-0.5 * posterior.log_precision),
                        jnp.exp(posterior.log_scale_image),
                        info.projection_rank,
                    )
                )
                if step_elbo % 1000 == 0:
                    progressbar_elbo.write(f"[{step_elbo:04d}] loss={loss_value:.4f}")
            except KeyboardInterrupt:
                break
        return posterior, opt_state, key

    # Upper confidence bound function
    def ucb(x_pred, posterior_params, pos_samples, kappa=1.0, negate=False, sum=False):
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

    # %%
    # Fit on initial selection of points
    posterior, opt_state, key = train(
        posterior, opt_state, x_sel, y_sel, args.num_epochs, key
    )
    key, subkey = jax.random.split(key)
    pos_samples = vi.sample(
        posterior, args.num_eval_mc_samples, x_sel, y=y_sel, key=subkey
    )

    # %%
    # Plot first fit and UCB
    x_pred = jnp.linspace(-jnp.pi, jnp.pi, 1000)

    if not args.plot:
        matplotlib.use("Agg")
    plt.ion()
    fig = plt.figure(figsize=(15, 5))
    plt.plot(x_full, y_full, "-", label="True function", alpha=0.5)
    plt.scatter(x_sel, y_sel, color="red", label="Selected points")
    samples = jax.vmap(vi.predict, in_axes=(None, None, 0), out_axes=-1)(
        posterior, pos_samples, x_pred
    )
    lines = plt.plot(
        x_pred,
        samples.T,
        label="BNN prediction",
        color="green",
        alpha=0.1,
        zorder=-100000,
    )
    ucb_line = plt.plot(x_pred, ucb(x_pred, posterior, pos_samples), color="orange")

    # %%
    # BO setup
    acq_fn = ft.partial(ucb, kappa=1.0, sum=True, negate=True)
    acq_fn = eqx.filter_jit(acq_fn)

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
    for bo_step in range(args.bo_steps):
        posterior, opt_state, key = train(
            posterior, opt_state, x_sel, y_sel, args.num_epochs_bo, key
        )
        key, subkey = jax.random.split(key)
        pos_samples = vi.sample(
            posterior, args.num_eval_mc_samples, x_sel, y=y_sel, key=key
        )

        # Evaluate BNN on grid for plotting
        samples = jax.vmap(vi.predict, in_axes=(None, None, 0), out_axes=-1)(
            posterior, pos_samples, x_pred
        )
        for i in range(len(lines)):
            lines[i].set_ydata(samples[i])

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

        plt.scatter(x_sel[-1], y_sel[-1], color="green", alpha=0.5)
        plt.text(x_sel[-1], y_sel[-1], str(bo_step), size=20)

        ucb_line[0].set_ydata(ucb(x_pred, posterior, pos_samples))

        if args.plot:
            plt.axis([-jnp.pi, jnp.pi, -5.0, 5.0])
            fig.canvas.draw()
            fig.canvas.flush_events()


# *True* function
def fun(x):
    # fun = lambda x: -0.1*(1.5*x)**2 + 0.1*x
    return jnp.exp(-0.5 * x**2) * jnp.cos(2 * jnp.pi * x)


def eqx_flatten(model):
    params, static = eqx.partition(model, eqx.is_array)
    vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    return vec, lambda v: eqx.combine(unflatten_fn(v), static)


def make_wave(key, num: int):
    x = jnp.linspace(-jnp.pi, jnp.pi, num)
    y = fun(x)
    # z = jax.random.normal(key, shape=y.shape)
    return x, y  # + 0.1 * z


# select a random subset of points
def select_points(x, y, num: int, key):
    idx = jax.random.choice(key, x.shape[0], shape=(num,), replace=False)
    return x[idx, ...], y[idx]


def make_train_step(loss_fn, optimizer, num_mc_samples=1):
    elbo_loss_fn = vi.as_elbo_loss(loss_fn, is_batched=False)
    value_and_grad_fn = eqx.filter_value_and_grad(elbo_loss_fn, has_aux=True)

    def train_step(posterior, opt_state, inputs, targets, *, key):
        out, loss_grad = value_and_grad_fn(
            posterior,
            inputs=inputs,
            targets=targets,
            key=key,
            num_mc_samples=num_mc_samples,
        )
        updates, opt_state = optimizer.update(loss_grad, opt_state, posterior)
        posterior = eqx.apply_updates(posterior, updates)
        return out, posterior, opt_state

    return train_step


def _clip_safe(x, lower, upper):
    return jnp.clip(jnp.asarray(x), lower, upper)


def projection_box(x, hyperparams):
    lower, upper = hyperparams
    return jax.tree_util.tree_map(_clip_safe, x, lower, upper)


def parse_args(argv=None):
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
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
