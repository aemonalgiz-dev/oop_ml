"""Smoke test for the example scripts.

The examples are documentation, and documentation rots. Running each one here
means a rename or a changed signature breaks the suite rather than being
discovered by whoever next opens ``examples/``.

This asserts only that each script runs to completion -- the numbers it prints
are the library's behaviour, which the rest of the suite already pins down.
"""

import pytest

from examples import (
    classification_metrics,
    gradient_descent,
    logistic_regression,
    model_selection,
    multiclass_classification,
    multiple_regression,
    polynomial_curves,
    regularization,
    simple_regression,
    standardization,
)

EXAMPLE_MODULES = [
    simple_regression,
    multiple_regression,
    gradient_descent,
    regularization,
    polynomial_curves,
    standardization,
    model_selection,
    logistic_regression,
    classification_metrics,
    multiclass_classification,
]


@pytest.mark.parametrize(
    "example_module",
    EXAMPLE_MODULES,
    ids=[module.__name__.rpartition(".")[2] for module in EXAMPLE_MODULES],
)
def test_example_runs_to_completion(example_module):
    example_module.main()
