Alternating Projections
=======================

*THIS IS A DRAFT*

This tutorial aims to show users of the library how to make of its API to enable scalable posterior sampling using the KIVI posterior when model sizes and the number of data points make direct projection onto the kernel (null) space prohibitive.

We start with the same setup as usual:

.. code:: python

  from kivi.vi import KernelImagePosterior

  model = ...
  posterior = KernelImagePosterior(model)
  loss_fn = ...

  # TODO: rest

Now we need to prepare our training code to "warmup" the posterior samples by first doing alternating projections:

.. code:: python

  key, key_ap = jax.random.split(key)
  pos_samples = posterior.alternating_projections(loader, num_samples, key=key_ap)
