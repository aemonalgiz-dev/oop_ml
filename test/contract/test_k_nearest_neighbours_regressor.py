"""The contract every backend's KNearestNeighboursRegressor keeps.

The fixture puts the rows in three well-separated pairs, so every query has
two nearest rows that no reasonable metric could confuse with the others and
the expected mean is written straight from the pairs.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

#: Three pairs of rows, far apart, each pair's targets straddling a round
#: number so the mean of the pair is known.
_LEFT = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
_RIGHT = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
_TARGETS = np.array([0.0, 2.0, 10.0, 12.0, 20.0, 22.0])
FEATURES = [Feature("left", _LEFT), Feature("right", _RIGHT)]
TARGET = Feature("value", _TARGETS)

#: One query inside each pair, and the mean of that pair's targets.
QUERIES = [Feature("left", [0.4, 10.6, 20.3]), Feature("right", [0.4, 10.6, 20.3])]
EXPECTED = np.array([1.0, 11.0, 21.0])

#: The skewed fixture the classifier spec uses, with numbers for targets.
#: Rows on the diagonal at (3, 3) and (-3, -3) carry 10, rows on the axis at
#: (0, 4.5) and (0, -4.5) carry 20. From the origin a diagonal row is 4.24
#: away by Euclidean, 6 by Manhattan and 3 by Chebyshev, where an axis row is
#: 4.5 under all three, so the metric alone decides which target the nearest
#: row carries. Hamming and Canberra side with the axis row, which agrees
#: with the origin in one coordinate. Cosine is left out, since the origin
#: has no direction and every row ties.
_SKEWED_LEFT = np.array([3.0, 0.0, -3.0, 0.0])
_SKEWED_RIGHT = np.array([3.0, 4.5, -3.0, -4.5])
SKEWED_FEATURES = [Feature("left", _SKEWED_LEFT), Feature("right", _SKEWED_RIGHT)]
SKEWED_TARGET = Feature("value", [10.0, 20.0, 10.0, 20.0])
ORIGIN = [Feature("left", [0.0]), Feature("right", [0.0])]
NEAREST_VALUE_BY_METRIC = [
    (DistanceMetric.EUCLIDEAN, 10.0),
    (DistanceMetric.MANHATTAN, 20.0),
    (DistanceMetric.CHEBYSHEV, 10.0),
    (DistanceMetric.HAMMING, 20.0),
    (DistanceMetric.CANBERRA, 20.0),
]


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    KNearestNeighboursRegressor = provided(backend, "KNearestNeighboursRegressor")
    model = KNearestNeighboursRegressor(n_neighbours=2, metric=DistanceMetric.MANHATTAN)

    assert model.n_neighbours == 2
    assert model.metric == DistanceMetric.MANHATTAN


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    KNearestNeighboursRegressor = provided(backend, "KNearestNeighboursRegressor")
    model = KNearestNeighboursRegressor(n_neighbours=2)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_the_mean_of_the_nearest_pair(backend: ModuleType) -> None:
    KNearestNeighboursRegressor = provided(backend, "KNearestNeighboursRegressor")
    model = KNearestNeighboursRegressor(n_neighbours=2).fit(FEATURES, TARGET)

    predictions = model.predict(QUERIES)

    assert len(predictions) == len(EXPECTED)
    assert np.allclose(np.asarray(predictions), EXPECTED)


@pytest.mark.parametrize(
    "metric", list(DistanceMetric), ids=[one.value for one in DistanceMetric]
)
def test_every_metric_of_the_library_is_accepted(
    backend: ModuleType, metric: DistanceMetric
) -> None:
    KNearestNeighboursRegressor = provided(backend, "KNearestNeighboursRegressor")
    model = KNearestNeighboursRegressor(n_neighbours=2, metric=metric).fit(
        FEATURES, TARGET
    )

    predictions = model.predict(QUERIES)

    assert len(predictions) == len(EXPECTED)
    assert np.all(np.isfinite(np.asarray(predictions)))


@pytest.mark.parametrize(
    ("metric", "nearest_value"),
    NEAREST_VALUE_BY_METRIC,
    ids=[one.value for one, _ in NEAREST_VALUE_BY_METRIC],
)
def test_the_metric_decides_which_row_is_nearest(
    backend: ModuleType, metric: DistanceMetric, nearest_value: float
) -> None:
    """Accepting a metric is not honouring one.

    The test above shows only that every metric is taken and answers
    something finite, which a wrapper handing every one of them to its
    engine as Euclidean also does. This is the companion the classifier
    spec already carries, on the same skewed fixture, and it is separate
    because the two wrappers translate the metric at separate call sites.
    """
    KNearestNeighboursRegressor = provided(backend, "KNearestNeighboursRegressor")
    model = KNearestNeighboursRegressor(n_neighbours=1, metric=metric).fit(
        SKEWED_FEATURES, SKEWED_TARGET
    )

    assert np.asarray(model.predict(ORIGIN))[0] == pytest.approx(nearest_value)


def test_it_remembers_every_training_row(backend: ModuleType) -> None:
    KNearestNeighboursRegressor = provided(backend, "KNearestNeighboursRegressor")
    model = KNearestNeighboursRegressor(n_neighbours=2).fit(FEATURES, TARGET)

    assert model.n_remembered == len(_TARGETS)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    KNearestNeighboursRegressor = provided(backend, "KNearestNeighboursRegressor")
    model = KNearestNeighboursRegressor(n_neighbours=2).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) > 0.9


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    KNearestNeighboursRegressor = provided(backend, "KNearestNeighboursRegressor")

    with pytest.raises(NotFittedError):
        KNearestNeighboursRegressor(n_neighbours=2).predict(FEATURES)
