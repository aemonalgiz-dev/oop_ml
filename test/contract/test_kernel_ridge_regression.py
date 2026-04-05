"""The contract every backend's KernelRidgeRegression keeps.

The model takes this library's kernel objects, and each of the four has to be
accepted and fit. What it learns is one weight per training row rather than
one per feature, so "addressable by name" here means the dual weights, the
row count and the target mean, which are the learned state both backends
expose.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError
from oop_ml.core.kernel.functions import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)

from .harness import provided

#: A smooth curve over one feature, sampled densely enough for a radial basis
#: kernel to recover it on the training rows.
_POSITIONS = np.linspace(0.0, 3.0, 12)
_CURVE = np.sin(_POSITIONS)
CURVE_FEATURES = [Feature("position", _POSITIONS)]
CURVE_TARGET = Feature("height", _CURVE)

#: y = 3 * area + 2 * baths + 1, with a little noise, for the linear kernel.
_AREAS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_BATHS = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
_PRICES = (
    3.0 * _AREAS
    + 2.0 * _BATHS
    + 1.0
    + np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01])
)
PLANE_FEATURES = [Feature("area", _AREAS), Feature("baths", _BATHS)]
PLANE_TARGET = Feature("price", _PRICES)

EVERY_KERNEL: list[Kernel] = [
    LinearKernel(),
    PolynomialKernel(degree=2),
    RadialBasisKernel(gamma=1.0),
    SigmoidKernel(gamma=0.1, constant=0.0),
]


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    KernelRidgeRegression = provided(backend, "KernelRidgeRegression")
    model = KernelRidgeRegression(kernel=RadialBasisKernel(gamma=2.0), penalty=0.5)

    assert model.penalty == pytest.approx(0.5)
    assert model.kernel == RadialBasisKernel(gamma=2.0)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    KernelRidgeRegression = provided(backend, "KernelRidgeRegression")
    model = KernelRidgeRegression(penalty=0.001)

    assert model.fit(PLANE_FEATURES, PLANE_TARGET) is model


def test_with_a_linear_kernel_it_predicts_the_plane(backend: ModuleType) -> None:
    KernelRidgeRegression = provided(backend, "KernelRidgeRegression")
    model = KernelRidgeRegression(kernel=LinearKernel(), penalty=0.001).fit(
        PLANE_FEATURES, PLANE_TARGET
    )

    predictions = model.predict(PLANE_FEATURES)

    assert len(predictions) == len(_PRICES)
    assert np.allclose(np.asarray(predictions), _PRICES, atol=0.2)


def test_with_a_radial_basis_kernel_it_recovers_the_curve(backend: ModuleType) -> None:
    KernelRidgeRegression = provided(backend, "KernelRidgeRegression")
    model = KernelRidgeRegression(kernel=RadialBasisKernel(gamma=1.0), penalty=1e-3)
    model.fit(CURVE_FEATURES, CURVE_TARGET)

    predictions = model.predict(CURVE_FEATURES)

    assert np.allclose(np.asarray(predictions), _CURVE, atol=0.05)


@pytest.mark.parametrize(
    "kernel", EVERY_KERNEL, ids=[type(one).__name__ for one in EVERY_KERNEL]
)
def test_every_kernel_of_the_library_is_accepted(
    backend: ModuleType, kernel: Kernel
) -> None:
    KernelRidgeRegression = provided(backend, "KernelRidgeRegression")
    model = KernelRidgeRegression(kernel=kernel, penalty=0.1).fit(
        CURVE_FEATURES, CURVE_TARGET
    )

    predictions = model.predict(CURVE_FEATURES)

    assert len(predictions) == len(_CURVE)
    assert np.all(np.isfinite(np.asarray(predictions)))


def test_it_learns_one_weight_per_training_row(backend: ModuleType) -> None:
    KernelRidgeRegression = provided(backend, "KernelRidgeRegression")
    model = KernelRidgeRegression(penalty=0.001).fit(PLANE_FEATURES, PLANE_TARGET)

    assert model.n_training_rows == len(_PRICES)
    assert model.dual_weights.shape == (len(_PRICES),)
    assert model.target_mean == pytest.approx(float(np.mean(_PRICES)))
    assert model.training_rows.feature_names == ("area", "baths")


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    KernelRidgeRegression = provided(backend, "KernelRidgeRegression")
    model = KernelRidgeRegression(kernel=RadialBasisKernel(gamma=1.0), penalty=1e-3)
    model.fit(CURVE_FEATURES, CURVE_TARGET)

    assert model.score(CURVE_FEATURES, CURVE_TARGET) > 0.99


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    KernelRidgeRegression = provided(backend, "KernelRidgeRegression")

    with pytest.raises(NotFittedError):
        KernelRidgeRegression().predict(PLANE_FEATURES)
