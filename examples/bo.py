from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import optax
import tqdm

from kivi import linalg, vi

# %%
# Settings/args
lr = 1e-2
log_scale_noise = jnp.log(1e-2)
num_hidden = 8
beta = 1.0  # 0.25
epochs_elbo = 4_000
epochs_elbo_bo = 100
num_mc_samples = 20
num_eval_mc_samples = 100
num_sel = 5
is_linearized = True
bo_steps = 25

# Random seeding
seed = 42
key = jax.random.PRNGKey(seed=seed)


# %%
# *True* function
fun = lambda x: jnp.exp(-0.5 * x**2) * jnp.cos(2 * jnp.pi * x)
# fun = lambda x: -0.1*(1.5*x)**2 + 0.1*x


def make_wave(key, num: int):
    x = jnp.linspace(-jnp.pi, jnp.pi, num)
    y = fun(x)
    # z = jax.random.normal(key, shape=y.shape)
    return x, y  # + 0.1 * z


# select a random subset of points
def select_points(x, y, num: int, key):
    idx = jax.random.choice(key, x.shape[0], shape=(num,), replace=False)
    return x[idx, ...], y[idx]


# The negative log-likelihood function
def loss_single(pred, u):
    return -jax.scipy.stats.norm.logpdf(u, loc=pred, scale=jnp.exp(log_scale_noise))


# %%
# Data preparation
key, subkey = jax.random.split(key)
x_full, y_full = make_wave(subkey, 1000)
key, subkey = jax.random.split(key)
x_sel, y_sel = select_points(x_full, y_full, num_sel, subkey)


# %%
# Model and optimiser setup
key, subkey = jax.random.split(key)
model = eqx.nn.MLP(
    in_size="scalar",
    out_size="scalar",
    width_size=num_hidden,
    depth=2,
    activation=jnp.tanh,
    key=subkey,
)
posterior = vi.KernelImagePosterior(
    model,
    log_precision=0.0,
    log_scale_image=-2.0,
    # A different solver:
    # solve_normaleq=linalg.solve_normaleq_materialize(
    #     linalg.dense_solver_fn_cholesky(eps=1e-6)
    # ),
    beta=beta,
    is_linearized=is_linearized,
)
loss_fn = posterior.wrap_loss(loss_single)
optimizer = optax.nadam(lr)
train_elbo_step = vi.make_train_step(
    loss_fn, optimizer=optimizer, num_mc_samples=num_mc_samples
)
train_elbo_step = eqx.filter_jit(train_elbo_step)
posterior_params, posterior_static = eqx.partition(posterior, eqx.is_array)
opt_state = optimizer.init(posterior_params)


def train(posterior, opt_state, x_sel, y_sel, num_epochs, key):
    progressbar_elbo = tqdm.trange(num_epochs)
    for step_elbo in progressbar_elbo:
        key, subkey = jax.random.split(key)
        try:
            (loss_value, info, posterior, opt_state) = train_elbo_step(
                posterior,
                opt_state,
                x_sel,
                y_sel,
                key=subkey,
            )
            progressbar_elbo.set_description(
                "loss={:.4e}, σ_ker={:.2e}, σ_im={:.2e}, R={:.1f}".format(
                    loss_value,
                    jnp.exp(-0.5 * posterior.log_precision),
                    jnp.exp(posterior.log_scale_image),
                    info.projection_rank,
                )
            )
            if step_elbo % 1000 == 0:
                progressbar_elbo.write(str(loss_value))
        except KeyboardInterrupt:
            break
    return posterior, opt_state, key


# Upper confidence bound
def ucb(x_pred, posterior_params, pos_samples, kappa=1.0, negate=False, sum=False):
    posterior = eqx.combine(posterior_params, posterior_static)
    samples = posterior.predict_from_samples(pos_samples, x_pred)
    y_pred = jnp.mean(samples, axis=0)
    y_std = jnp.std(samples, axis=0)
    acq = y_pred + kappa * y_std
    acq = acq.sum(axis=-1) if sum else acq
    if negate:
        return -acq
    return acq


# %%
# Initial training
posterior, opt_state, key = train(posterior, opt_state, x_sel, y_sel, epochs_elbo, key)
key, subkey = jax.random.split(key)
pos_samples = posterior.sample(num_eval_mc_samples, x_sel, y=y_sel, key=subkey)


# %%
# Plot first fit and UCB
x_pred = jnp.linspace(-jnp.pi, jnp.pi, 1000)
plt.ion()
fig = plt.figure(figsize=(15, 5))
plt.plot(x_full, y_full, "-", label="True function", alpha=0.5)
plt.scatter(x_sel, y_sel, color="red", label="Selected points")
samples = posterior.predict_from_samples(pos_samples, x_pred)
lines = plt.plot(
    x_pred, samples.T, label="BNN prediction", color="green", alpha=0.1, zorder=-100000
)
ucb_line = plt.plot(x_pred, ucb(x_pred, posterior, pos_samples), color="orange")


# %%
# BO setup
acq_grad = jax.value_and_grad(partial(ucb, kappa=1.0, sum=True, negate=True))
acq_grad = eqx.filter_jit(acq_grad)
acq_solver = jaxopt.ScipyBoundedMinimize(
    fun=acq_grad, value_and_grad=True, method="L-BFGS-B"
)


# %%
# Main BO loop
for bo_step in range(bo_steps):
    posterior, opt_state, key = train(
        posterior, opt_state, x_sel, y_sel, epochs_elbo_bo, key
    )
    key, subkey = jax.random.split(key)
    pos_samples = posterior.sample(num_eval_mc_samples, x_sel, y=y_sel, key=key)

    # Evaluate BNN on grid for plotting
    samples = posterior.predict_from_samples(pos_samples, x_pred)
    y_pred = jnp.mean(samples, axis=0)
    y_std = jnp.std(samples, axis=0)
    for i in range(len(lines)):
        lines[i].set_ydata(samples[i])

    x_news, y_news = [], []
    for k in range(len(x_sel)):
        # acq_solver = jaxopt.ScipyMinimize(fun=acq_grad, method="L-BFGS-B", value_and_grad=True)
        key, subkey = jax.random.split(key)
        x_new, state = acq_solver.run(
            x_sel[k] + 0.05 * jax.random.normal(subkey, shape=(1,)),
            [-jnp.pi, jnp.pi],
            posterior_params=eqx.filter(posterior, eqx.is_array),
            pos_samples=pos_samples,
        )
        # x_new, state = acq_solver.run(x_sel[k] + 0.05*jax.random.normal(subkey, shape=[1]))
        x_news.append(x_new)
        y_news.append(state.fun_val.item())
    x_news = jnp.array(x_news).flatten()
    y_news = jnp.array(y_news).flatten()
    idx = jnp.argmin(y_news)
    x_sel = jnp.append(x_sel, x_news[idx])
    y_sel = jnp.append(y_sel, fun(x_news[idx]))
    print("Current x_new = {}".format(x_sel[-1]))

    plt.scatter(x_sel[-1], y_sel[-1], color="green", alpha=0.5)
    plt.text(x_sel[-1], y_sel[-1], str(bo_step), size=20)

    ucb_line[0].set_ydata(ucb(x_pred, posterior, pos_samples))

    plt.axis([-jnp.pi, jnp.pi, -5.0, 5.0])
    fig.canvas.draw()
    fig.canvas.flush_events()
plt.show()
