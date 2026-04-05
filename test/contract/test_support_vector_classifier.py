"""The contract every backend's SupportVectorClassifier keeps.

Two square clusters with a wide gap between them, so any maximum-margin
boundary separates the training rows exactly and a query deep inside either
cluster is never in doubt. The coordinates are kept small on purpose: the
numpy backend's fixed-step ascent is stable only below ``2 / lambda_max`` of
the dual's matrix, and that matrix grows with the squared coordinates.

The model takes this library's kernel objects, and each of the four has to
be accepted. What it learns is which rows the boundary rests on, so
"addressable" here means the support vectors, the multipliers and the signed
labels, which both backends expose.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.data.probabilities import Probabilities
from oop_ml.core.exceptions import NonBinaryLabelsError, NotFittedError
from oop_ml.core.kernel.functions import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)

from .harness import provided

_ACROSS = np.array([0.0, 1.0, 0.0, 1.0, 3.0, 4.0, 3.0, 4.0])
_UP = np.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0])
_CLASSES = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
FEATURES = [Feature("across", _ACROSS), Feature("up", _UP)]
TARGET = Feature("cluster", _CLASSES)

QUERIES = [Feature("across", [0.5, 3.5]), Feature("up", [0.5, 3.5])]
EXPECTED = np.array([0.0, 1.0])

#: Small enough that the margin gives up on most of the rows, so several
#: multipliers sit at the cap and none can sit above it. At the default
#: capacity of 1.0 the largest multiplier is well under 0.7 in both backends,
#: so a backend that never handed the capacity to its solver is caught here.
SMALL_CAPACITY = 0.05

#: Ten million times the default, and far looser than any movement this
#: fixture has left to make. The engine's ``tol`` and the numpy ascent's are
#: different stopping rules, so what is pinned is the direction rather than a
#: number. Measured, the default reaches a largest multiplier of 0.6675 on
#: the numpy backend and 0.2500 on scikit, where this tolerance reaches
#: 0.0010 and 0.0000.
LOOSE_TOLERANCE = 10.0

EVERY_KERNEL: list[Kernel] = [
    LinearKernel(),
    PolynomialKernel(degree=2),
    RadialBasisKernel(gamma=1.0),
    SigmoidKernel(gamma=0.1, constant=0.0),
]


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier(
        kernel=RadialBasisKernel(gamma=0.5),
        capacity=2.0,
        max_epochs=50,
        tolerance=1e-4,
    )

    assert model.kernel == RadialBasisKernel(gamma=0.5)
    assert model.capacity == pytest.approx(2.0)
    assert model.max_epochs == 50
    assert model.tolerance == pytest.approx(1e-4)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier()

    assert model.fit(FEATURES, TARGET) is model


def test_it_separates_the_clusters(backend: ModuleType) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier().fit(FEATURES, TARGET)

    assert np.array_equal(np.asarray(model.predict(FEATURES)), _CLASSES)
    assert np.array_equal(np.asarray(model.predict(QUERIES)), EXPECTED)


def test_its_decision_values_carry_the_sign_of_its_predictions(
    backend: ModuleType,
) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier().fit(FEATURES, TARGET)

    values = model.decision_values(FEATURES)

    assert values.shape == (len(_CLASSES),)
    assert np.array_equal(values >= 0.0, _CLASSES == 1.0)


def test_its_scores_are_bounded_and_side_with_the_boundary(backend: ModuleType) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier().fit(FEATURES, TARGET)

    scores = model.predict_probability(FEATURES)

    assert isinstance(scores, Probabilities)
    values = np.asarray(scores)
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.array_equal(values >= 0.5, _CLASSES == 1.0)


def test_it_rests_the_boundary_on_a_few_training_rows(backend: ModuleType) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier().fit(FEATURES, TARGET)

    support_vectors = model.support_vectors

    assert support_vectors.n_training_rows == len(_CLASSES)
    assert 2 <= support_vectors.n_vectors <= len(_CLASSES)
    assert set(support_vectors.positions()) <= set(range(len(_CLASSES)))
    assert all(vector.multiplier > 0.0 for vector in support_vectors)
    assert {vector.label for vector in support_vectors} == {-1.0, 1.0}
    assert model.multipliers.shape == (len(_CLASSES),)
    assert np.array_equal(model.signed_labels, np.where(_CLASSES == 1.0, 1.0, -1.0))
    assert model.epochs_run >= 1


@pytest.mark.parametrize(
    "kernel", EVERY_KERNEL, ids=[type(one).__name__ for one in EVERY_KERNEL]
)
def test_every_kernel_of_the_library_is_accepted(
    backend: ModuleType, kernel: Kernel
) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier(kernel=kernel).fit(FEATURES, TARGET)

    predictions = np.asarray(model.predict(FEATURES))

    assert predictions.shape == _CLASSES.shape
    assert set(np.unique(predictions)) <= {0.0, 1.0}


def test_it_refuses_a_target_that_is_not_zero_or_one(backend: ModuleType) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")

    with pytest.raises(NonBinaryLabelsError):
        SupportVectorClassifier().fit(
            FEATURES, Feature("cluster", [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        )


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier().fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(1.0)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")

    with pytest.raises(NotFittedError):
        SupportVectorClassifier().predict(FEATURES)


def test_the_capacity_caps_every_multiplier(backend: ModuleType) -> None:
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    model = SupportVectorClassifier(capacity=SMALL_CAPACITY).fit(FEATURES, TARGET)

    multipliers = model.multipliers

    assert np.all(multipliers <= SMALL_CAPACITY * (1.0 + 1e-9))
    assert multipliers.max() == pytest.approx(SMALL_CAPACITY)
    assert model.support_vectors.n_at_the_cap(SMALL_CAPACITY) >= 1


def test_a_loose_tolerance_stops_the_margin_short(backend: ModuleType) -> None:
    """``tolerance`` has to reach the solver, and it is not a cosmetic field.

    Without this, a backend that never passed it on answers the same margin
    at every setting and stays green. A factor of ten is the claim, because
    the two backends stop on different quantities and only the direction is
    shared. The measured gap is far wider, a factor of 667 on the numpy
    backend and total on scikit, where the loose fit leaves every multiplier
    at zero.
    """
    SupportVectorClassifier = provided(backend, "SupportVectorClassifier")
    tight = SupportVectorClassifier().fit(FEATURES, TARGET)
    loose = SupportVectorClassifier(tolerance=LOOSE_TOLERANCE).fit(FEATURES, TARGET)

    assert loose.multipliers.max() < tight.multipliers.max() / 10.0
