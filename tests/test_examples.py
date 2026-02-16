import os

import pytest
import pytest_cases
import matplotlib
import warnings

# NOTE: Examples not listed here should be tested when generating
# documentation for them
from examples import mnist


@pytest.mark.skipif(
    "RUNALL" not in os.environ, reason="Time-consuming and needs downloaded data"
)
def test_mnist_example():
    mnist.main(
        mnist.parse_args(["--num-epochs=2", "--num-eval-samples=1", "--num-hidden=2"])
    )
