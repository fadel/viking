import pytest_cases

from examples import (
    bo,
    sinusoid,
    two_moons,
)


def example_bo():
    return bo


def example_sinusoid():
    return sinusoid


def example_two_moons():
    return two_moons


@pytest_cases.parametrize_with_cases("module", cases=".", prefix="example_")
def test_example(module):
    module.main(module.parse_args(["--no-plot"]))
