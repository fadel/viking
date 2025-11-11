import os

import pytest
import pytest_cases

from examples import (
    bo,
    mnist,
    sinusoid,
    two_moons,
)


@pytest_cases.parametrize("module", (bo, sinusoid, two_moons))
def test_small_example(module):
    module.main(module.parse_args(["--no-plot"]))


@pytest.mark.skipif(
    "RUNALL" not in os.environ, reason="Time-consuming and needs downloaded data"
)
def test_mnist_example():
    mnist.main(
        mnist.parse_args(["--num-epochs=2", "--num-eval-samples=1", "--num-hidden=2"])
    )
