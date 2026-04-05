"""The contract every backend's KNearestNeighboursClassifier keeps.

Three well-separated pairs of rows, one class per pair, so a query inside a
pair has two nearest rows of one class under any reasonable metric and the
vote is unanimous. The expected labels and shares are written straight from
the pairs.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.data.probabilities import ProbabilityMatrix
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import (
    NonBinaryLabelsError,
    NotFittedError,
    SingleClassError,
)

from .harness import provided

_LEFT = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
_RIGHT = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
_CLASSES = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
FEATURES = [Feature("left", _LEFT), Feature("right", _RIGHT)]
TARGET = Feature("group", _CLASSES)

QUERIES = [Feature("left", [0.4, 10.6, 20.3]), Feature("right", [0.4, 10.6, 20.3])]
EXPECTED = np.array([0.0, 1.0, 2.0])
EXPECTED_SHARES = np.eye(3)

#: Class 0 on the diagonal at (3, 3) and (-3, -3); class 1 on the axis at
#: (0, 4.5) and (0, -4.5). From the origin the diagonal row is 4.24 away by
#: Euclidean, 6 by Manhattan and 3 by Chebyshev, where the axis row is 4.5
#: under all three, so the metric alone decides which class is nearest.
#: Hamming and Canberra side with the axis row too, since it agrees with
#: the origin in one coordinate. Cosine is left out, because the origin has
#: no direction and every row ties.
_SKEWED_LEFT = np.array([3.0, 0.0, -3.0, 0.0])
_SKEWED_RIGHT = np.array([3.0, 4.5, -3.0, -4.5])
SKEWED_FEATURES = [Feature("left", _SKEWED_LEFT), Feature("right", _SKEWED_RIGHT)]
SKEWED_TARGET = Feature("group", [0.0, 1.0, 0.0, 1.0])
ORIGIN = [Feature("left", [0.0]), Feature("right", [0.0])]
NEAREST_CLASS_BY_METRIC = [
    (DistanceMetric.EUCLIDEAN, 0.0),
    (DistanceMetric.MANHATTAN, 1.0),
    (DistanceMetric.CHEBYSHEV, 0.0),
    (DistanceMetric.HAMMING, 1.0),
    (DistanceMetric.CANBERRA, 1.0),
]


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")
    model = KNearestNeighboursClassifier(
        n_neighbours=2, metric=DistanceMetric.MANHATTAN
    )

    assert model.n_neighbours == 2
    assert model.metric == DistanceMetric.MANHATTAN


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")
    model = KNearestNeighboursClassifier(n_neighbours=2)

    assert model.fit(FEATURES, TARGET) is model
    assert model.n_classes == 3


def test_it_predicts_the_class_of_the_nearest_pair(backend: ModuleType) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")
    model = KNearestNeighboursClassifier(n_neighbours=2).fit(FEATURES, TARGET)

    predictions = model.predict(QUERIES)

    assert len(predictions) == len(EXPECTED)
    assert np.array_equal(np.asarray(predictions), EXPECTED)


def test_its_probability_rows_are_the_pair_s_unanimous_vote(
    backend: ModuleType,
) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")
    model = KNearestNeighboursClassifier(n_neighbours=2).fit(FEATURES, TARGET)

    probabilities = model.predict_probabilities(QUERIES)

    assert isinstance(probabilities, ProbabilityMatrix)
    assert np.allclose(np.asarray(probabilities), EXPECTED_SHARES)


@pytest.mark.parametrize(
    "metric", list(DistanceMetric), ids=[one.value for one in DistanceMetric]
)
def test_every_metric_of_the_library_is_accepted(
    backend: ModuleType, metric: DistanceMetric
) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")
    model = KNearestNeighboursClassifier(n_neighbours=2, metric=metric).fit(
        FEATURES, TARGET
    )

    predictions = np.asarray(model.predict(QUERIES))

    assert predictions.shape == EXPECTED.shape
    assert set(np.unique(predictions)) <= {0.0, 1.0, 2.0}


def test_it_remembers_every_training_row(backend: ModuleType) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")
    model = KNearestNeighboursClassifier(n_neighbours=2).fit(FEATURES, TARGET)

    assert model.n_remembered == len(_CLASSES)


def test_it_refuses_a_target_that_is_not_class_positions(backend: ModuleType) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")

    with pytest.raises(NonBinaryLabelsError):
        KNearestNeighboursClassifier(n_neighbours=2).fit(
            FEATURES, Feature("group", [0.0, 0.0, 1.5, 1.5, 2.0, 2.0])
        )
    with pytest.raises(SingleClassError):
        KNearestNeighboursClassifier(n_neighbours=2).fit(
            FEATURES, Feature("group", [0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
        )


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")
    model = KNearestNeighboursClassifier(n_neighbours=2).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(1.0)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")

    with pytest.raises(NotFittedError):
        KNearestNeighboursClassifier(n_neighbours=2).predict(FEATURES)


@pytest.mark.parametrize(
    ("metric", "nearest_class"),
    NEAREST_CLASS_BY_METRIC,
    ids=[metric.value for metric, _ in NEAREST_CLASS_BY_METRIC],
)
def test_the_metric_decides_which_row_is_nearest(
    backend: ModuleType, metric: DistanceMetric, nearest_class: float
) -> None:
    KNearestNeighboursClassifier = provided(backend, "KNearestNeighboursClassifier")
    model = KNearestNeighboursClassifier(n_neighbours=1, metric=metric).fit(
        SKEWED_FEATURES, SKEWED_TARGET
    )

    assert np.array_equal(np.asarray(model.predict(ORIGIN)), [nearest_class])
