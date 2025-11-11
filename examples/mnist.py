import argparse
import itertools as it

import cv2
import equinox as eqx
import grain.python as grain
import grain.transforms
import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
import tqdm

from viking import vi


class Crop(grain.transforms.Map):
    def __init__(self, crop_by=0):
        crop_start = crop_by
        crop_end = -crop_by
        if crop_by == 0:
            crop_end = None
        self.crop_slice = slice(crop_start, crop_end)

    def map(self, inputs):
        inputs["image"] = inputs["image"][self.crop_slice, self.crop_slice, ...]
        return inputs


class Resize(grain.transforms.Map):
    def __init__(self, to_width, to_height):
        self.to_width = to_width
        self.to_height = to_height

    def map(self, inputs):
        inputs["image"] = cv2.resize(
            inputs["image"],
            dsize=(self.to_width, self.to_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return inputs


class Flatten(grain.transforms.Map):
    def map(self, inputs):
        inputs["image"] = jnp.ravel(inputs["image"]) / 255.0
        return inputs


def main(args):
    key = jax.random.PRNGKey(seed=args.seed)

    # %%
    # Data setup
    key, key_data = jax.random.split(key)
    train_data_source = tfds.data_source("mnist", split="train")
    transforms = (Crop(args.crop_by), Resize(args.to_width, args.to_height), Flatten())
    train_dataset = (
        grain.MapDataset.source(train_data_source).apply(transforms).seed(args.seed)
    )
    test_data_source = tfds.data_source("mnist", split="test")
    test_dataset = grain.MapDataset.source(test_data_source).apply(transforms)
    num_batches = len(train_data_source) // args.batch_size

    # %%
    # Model setup
    key, key_model = jax.random.split(key)
    model = eqx.nn.MLP(
        in_size=args.to_width * args.to_height,
        out_size=10,
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
        # NOTE: Using the number of batches to rescale `beta` prevents
        # the KL term from dominating the optimisation, as it is
        # scaled for the model, while the ELBO only sees a batch at a
        # time. Relatively speaking, this is equivalent to scaling up
        # the expectation term to "full data set" scale.
        beta=args.beta / num_batches,
        # Including this argument here will use the loss-Jacobian
        # (default) VIKING
        loss_fn=optax.losses.softmax_cross_entropy_with_integer_labels,
        is_linearized=args.is_linearized,
    )
    print(f"Using beta={posterior.beta:.2e} (originally {args.beta:.2e})")
    print("Model parameters: {:,d}".format(posterior.num_params))

    # %%
    # Optimiser setup
    optimizer = optax.radam(args.lr)
    elbo_train_step = make_train_step(
        posterior.loss_fn,
        optimizer=optimizer,
    )
    elbo_train_step = eqx.filter_jit(elbo_train_step)
    posterior_params, posterior_static = eqx.partition(posterior, eqx.is_array)
    opt_state = optimizer.init(posterior_params)

    def elbo_epoch(posterior, dataset, opt_state, key):
        key, key_state = jax.random.split(key)
        # This initialises the state used to carry partial kernel
        # projections and optimiser state.
        elbo_state = vi.make_elbo_state(
            posterior,
            opt_state,
            num_mc_samples=args.num_mc_samples,
            loader=data_iter(dataset, args.batch_size),
            key=key_state,
        )
        dataset = dataset.shuffle()
        step_keys = jax.random.split(key, num_batches)
        with tqdm.tqdm(
            zip(step_keys, data_iter(dataset, args.batch_size)), total=num_batches
        ) as epoch_progress:
            for step_key, (x, y) in epoch_progress:
                # Each training step updates our approximate posterior
                # and the ELBO state.
                posterior, (elbo_state, info) = elbo_train_step(
                    posterior, elbo_state, x, y, key=step_key
                )
                description = make_description(posterior, info)
                epoch_progress.set_description(f"{description}")
        return posterior, elbo_state.opt_state, info

    # %%
    # Optimise ELBO
    dataset = train_dataset
    for epoch in range(1, args.num_epochs + 1):
        key, key_train, key_eval = jax.random.split(key, 3)
        dataset = dataset.shuffle()
        posterior, opt_state, info = elbo_epoch(
            posterior, dataset, opt_state, key=key_train
        )

        # After an epoch is done, we evaluate with fresh samples on
        # training data.
        #
        # Note that alternating projections will differ slightly from
        # training (what we project on is different at each step), so
        # here we can be sure everything is working as it should:
        # robust to changes in the samples, different number of
        # samples, etc.
        dataset = dataset.shuffle()
        pos_samples = vi.sample_with_loader(
            posterior,
            args.num_eval_samples,
            data_iter(dataset, args.batch_size, drop_remainder=False),
            key=key_eval,
        )
        train_acc = posterior_acc(
            posterior,
            pos_samples,
            data_iter(train_dataset, args.eval_batch_size, drop_remainder=False),
        )
        test_acc = posterior_acc(
            posterior,
            pos_samples,
            data_iter(test_dataset, args.eval_batch_size, drop_remainder=False),
        )
        print(f"[{epoch:04d}] train_acc={train_acc:.2f}, test_acc={test_acc:.2f}")


def eqx_flatten(model):
    params, static = eqx.partition(model, eqx.is_array)
    vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    return vec, lambda v: eqx.combine(unflatten_fn(v), static)


def data_iter(dataset, batch_size, drop_remainder=True):
    for b in dataset.batch(batch_size, drop_remainder=drop_remainder):
        yield b["image"], b["label"]


def make_train_step(loss_fn, optimizer):
    # The main function we differentiate when optimising:
    # - Updates kernel samples using Eq. (18) from the VIKING paper
    # - Computes both terms of the VIKING ELBO using that
    elbo_loss_fn = vi.as_elbo_loss(loss_fn, is_batched=True)
    value_and_grad_fn = eqx.filter_value_and_grad(elbo_loss_fn, has_aux=True)

    def train_step(posterior, elbo_state, inputs, targets, *, key):
        (loss_value, (elbo_state, info)), loss_grad = value_and_grad_fn(
            posterior,
            elbo_state,
            inputs=inputs,
            targets=targets,
            key=key,
        )
        updates, opt_state = optimizer.update(
            loss_grad, elbo_state.opt_state, posterior
        )
        elbo_state = elbo_state._replace(opt_state=opt_state)
        posterior = eqx.apply_updates(posterior, updates)
        return posterior, (elbo_state, info)

    return train_step


def running_average(iterable):
    """Given an iterable object which yields numbers, this will
    compute the mean of all values using running averages. Useful for
    lazy iterators, such as the one from :func:`map`.
    """
    mean = 0
    for i, v in enumerate(iterable):
        mean += (v - mean) / (i + 1)
    return mean


@eqx.filter_jit
def batch_acc(posterior, pos_samples, x, y):
    outputs = vi.predict_on_batch(posterior, pos_samples, x)
    # Average predictions over posterior samples
    y_probs = jnp.mean(jax.nn.softmax(outputs, axis=-1), axis=0)
    acc = jnp.mean(jnp.argmax(y_probs, axis=-1) == y)
    return acc


def posterior_acc(posterior, pos_samples, loader):
    # Wrap batch_acc() to be starmap()'d
    def _batch_acc(x, y):
        return batch_acc(posterior, pos_samples, x, y)

    return running_average(it.starmap(_batch_acc, loader))


def make_description(posterior, info):
    items = [
        f"E[]={info.expectation:.3e}",
        f"kl={info.kl:.3e}",
        f"σ_ker={jnp.exp(-0.5 * posterior.log_precision):.2e}",
        f"σ_im={jnp.exp(posterior.log_scale_image):.2e}",
        f"R={info.projection_rank:.1f}",
    ]
    return ", ".join(items)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-by", type=int, default=2)
    parser.add_argument("--to-width", type=int, default=8)
    parser.add_argument("--to-height", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
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
        default=2,
        help="Number layers in the MLP (excluding input layer)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument(
        "--num-mc-samples",
        type=int,
        default=1,
        help="Number of posterior samples drawn (in training) to estimate the expectation term of ELBO",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=5,
        help="Number of posterior samples drawn for evaluation",
    )
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--is-linearized", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
