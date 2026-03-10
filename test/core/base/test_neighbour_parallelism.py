"""The threaded neighbour search must change the timing and nothing else.

Every other neighbour test runs on fixtures far below
``PARALLEL_PAIR_THRESHOLD``, so they all take the serial path and none of them
would notice if the threaded one were broken. These force it.

The threshold is lowered rather than the data being made large. A test that
needed half a million pairs to reach the code under test would take seconds and
still only cover one block layout; dropping the threshold covers the same code
on inputs small enough to reason about, and lets the awkward shapes be chosen
deliberately.

Those shapes are the point. Queries are handed out in blocks of
``ceil(n_queries / workers)``, so a query count that divides evenly is the one
case where an off-by-one in the write-back cannot show up. The counts below are
mostly chosen to divide badly.
"""

import numpy as np
import pytest

from oop_ml.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.core.base import neighbour_model
from oop_ml.core.data.feature import Feature
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)


@pytest.fixture
def always_parallel(monkeypatch):
    """Force the threaded path on inputs of any size."""
    monkeypatch.setattr(neighbour_model, "PARALLEL_PAIR_THRESHOLD", 0)


@pytest.fixture
def never_parallel(monkeypatch):
    """Force the serial path, whatever the size."""
    monkeypatch.setattr(neighbour_model, "PARALLEL_PAIR_THRESHOLD", 10**18)


def build(n_remembered: int, n_features: int = 3, seed: int = 0):
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(n_remembered, n_features))
    features = [Feature(f"x{index}", matrix[:, index]) for index in range(n_features)]
    target = Feature("y", generator.normal(size=n_remembered))

    return features, target


def queries(n_queries: int, n_features: int = 3, seed: int = 1):
    matrix = np.random.default_rng(seed).normal(size=(n_queries, n_features))

    return [Feature(f"x{index}", matrix[:, index]) for index in range(n_features)]


class TestItAgreesWithTheSerialPath:
    @pytest.mark.parametrize("n_queries", [1, 2, 3, 7, 8, 9, 16, 17, 33, 64])
    def test_the_same_neighbours_come_back(self, n_queries, monkeypatch):
        # Query counts that divide badly across workers, because an even split
        # is the one arrangement where a write-back off-by-one cannot surface.
        features, target = build(200)
        asked = queries(n_queries)

        model = KNearestNeighboursRegressor(n_neighbours=4)
        model.fit(features, target)

        monkeypatch.setattr(neighbour_model, "PARALLEL_PAIR_THRESHOLD", 10**18)
        serial = model.predict(asked)

        monkeypatch.setattr(neighbour_model, "PARALLEL_PAIR_THRESHOLD", 0)
        parallel = model.predict(asked)

        assert np.array_equal(serial, parallel)

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_every_metric_agrees(self, metric, monkeypatch):
        features, target = build(120)
        asked = queries(23)

        model = KNearestNeighboursRegressor(n_neighbours=3, metric=metric)
        model.fit(features, target)

        monkeypatch.setattr(neighbour_model, "PARALLEL_PAIR_THRESHOLD", 10**18)
        serial = model.predict(asked)

        monkeypatch.setattr(neighbour_model, "PARALLEL_PAIR_THRESHOLD", 0)
        parallel = model.predict(asked)

        assert np.array_equal(serial, parallel)

    def test_the_answers_land_in_the_right_rows(self, always_parallel):
        # The failure a blocked write-back invites: right answers, wrong order.
        # Queried one row at a time, each answer must match the batch.
        features, target = build(150)
        asked = queries(19)

        model = KNearestNeighboursRegressor(n_neighbours=3)
        model.fit(features, target)

        batch = model.predict(asked)
        one_at_a_time = [
            model.predict(
                [Feature(column.name, [column.values[row]]) for column in asked]
            )[0]
            for row in range(19)
        ]

        assert batch == pytest.approx(one_at_a_time)


class TestTheThresholdIsRespected:
    def test_a_small_query_set_still_answers(self, never_parallel):
        features, target = build(60)

        model = KNearestNeighboursRegressor(n_neighbours=2)
        model.fit(features, target)

        assert model.predict(queries(5)).shape == (5,)

    def test_a_single_query_is_never_split(self, always_parallel):
        # n_queries < 2 short-circuits, because one block on one thread is a
        # pool started for nothing.
        features, target = build(60)

        model = KNearestNeighboursRegressor(n_neighbours=2)
        model.fit(features, target)

        assert model.predict(queries(1)).shape == (1,)

    def test_the_shipped_threshold_is_where_the_measurement_put_it(self):
        # Pinned so that lowering it "to be safe" is a deliberate act. Below
        # roughly this many pairs the pool costs more than it saves, by a
        # factor of nine at the small end.
        assert neighbour_model.PARALLEL_PAIR_THRESHOLD == 500_000
        assert neighbour_model.MAX_PARALLEL_WORKERS == 8


class TestTheClassifierToo:
    def test_votes_agree_across_both_paths(self, monkeypatch):
        generator = np.random.default_rng(5)
        matrix = generator.normal(size=(180, 3))
        features = [Feature(f"x{index}", matrix[:, index]) for index in range(3)]
        target = Feature("y", generator.integers(0, 3, size=180).astype(float))

        model = KNearestNeighboursClassifier(n_neighbours=5)
        model.fit(features, target)
        asked = queries(29)

        monkeypatch.setattr(neighbour_model, "PARALLEL_PAIR_THRESHOLD", 10**18)
        serial = model.predict(asked)
        serial_shares = model.predict_probabilities(asked)

        monkeypatch.setattr(neighbour_model, "PARALLEL_PAIR_THRESHOLD", 0)
        parallel = model.predict(asked)
        parallel_shares = model.predict_probabilities(asked)

        assert np.array_equal(serial, parallel)
        assert np.array_equal(serial_shares, parallel_shares)
