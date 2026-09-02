"""The recorded ranking must be the ranking the prediction used.

Same requirement as everywhere else in this pairing: whatever
``_neighbour_indices`` chooses, ``neighbour_search(...).result`` chooses too --
across every metric, every ``k``, and both models. If the two could diverge,
anything reading the second would be describing a prediction that never
happened.

The rest covers what the record has to hold to be worth keeping. Every distance
rather than only the surviving ones, because the interesting facts are about
the rows that lost: how much further the sixth was than the fifth, and how
little separates nearest from farthest once there are many features.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.neighbours.search import NeighbourSearch
from oop_ml.core.observation import Observation
from oop_ml.numpy.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.numpy.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)
from test.fixtures import NEIGHBOUR_GRID


def regressor(**overrides) -> KNearestNeighboursRegressor:
    model = KNearestNeighboursRegressor(**overrides)
    model.fit(NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.quantity_feature)

    return model


def classifier(**overrides) -> KNearestNeighboursClassifier:
    model = KNearestNeighboursClassifier(**overrides)
    model.fit(NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.class_feature)

    return model


def query(*points) -> list[Feature]:
    return [
        Feature("first", [point[0] for point in points]),
        Feature("second", [point[1] for point in points]),
    ]


ASKED = query((0.4, 0.4), (1.0, 1.0), (2.0, 0.0), (-3.0, 5.0))


class TestTheTwoRoutesAgree:
    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_across_every_metric(self, metric):
        model = regressor(n_neighbours=3, metric=metric)

        assert np.array_equal(
            model.neighbour_search(ASKED).result,
            model._neighbour_indices(model._matched_rows(ASKED)),
        )

    @pytest.mark.parametrize("n_neighbours", [1, 2, 3, 5, 9])
    def test_across_every_k(self, n_neighbours):
        model = regressor(n_neighbours=n_neighbours)

        assert np.array_equal(
            model.neighbour_search(ASKED).result,
            model._neighbour_indices(model._matched_rows(ASKED)),
        )

    def test_on_the_classifier(self):
        model = classifier(n_neighbours=5)

        assert np.array_equal(
            model.neighbour_search(ASKED).result,
            model._neighbour_indices(model._matched_rows(ASKED)),
        )

    def test_the_recorded_targets_are_what_combine_receives(self):
        # The record is only a record if the numbers it shows are the numbers
        # that produced the answer.
        model = regressor(n_neighbours=3)

        assert model.neighbour_search(ASKED).chosen_targets == pytest.approx(
            model._neighbour_targets(ASKED)
        )

    def test_the_prediction_follows_from_the_record(self):
        model = regressor(n_neighbours=3)
        search = model.neighbour_search(ASKED)

        by_hand = [query.chosen_targets.mean() for query in search]

        assert model.predict(ASKED) == pytest.approx(by_hand)


class TestWhatTheRecordHolds:
    def test_one_entry_per_query_in_order(self):
        search = regressor(n_neighbours=3).neighbour_search(ASKED)

        assert len(search) == 4
        for position, one in enumerate(search):
            assert one.row == pytest.approx(
                [column.values[position] for column in ASKED]
            )

    def test_every_remembered_row_gets_a_distance(self):
        # Not only the k that survived. The losers are most of the lesson.
        search = regressor(n_neighbours=3).neighbour_search(ASKED)

        for one in search:
            assert one.distances.shape == (NEIGHBOUR_GRID.n_samples,)

    def test_the_chosen_are_the_nearest_and_in_order(self):
        search = regressor(n_neighbours=4).neighbour_search(ASKED)

        for one in search:
            chosen = one.chosen_distances
            assert list(chosen) == sorted(chosen)
            assert chosen.max() <= np.sort(one.distances)[3]

    def test_the_row_that_just_missed_is_available(self):
        # Whether k was a real choice or an arbitrary one.
        search = regressor(n_neighbours=3).neighbour_search(ASKED)

        for one in search:
            missed = one.first_rejected_distance
            assert missed is not None
            assert missed >= one.chosen_distances.max()

    def test_nothing_missed_when_every_row_is_chosen(self):
        search = regressor(n_neighbours=NEIGHBOUR_GRID.n_samples).neighbour_search(
            ASKED
        )

        for one in search:
            assert one.first_rejected_distance is None

    def test_nearest_and_farthest_bracket_the_distances(self):
        search = regressor(n_neighbours=3).neighbour_search(ASKED)

        for one in search:
            assert one.nearest_distance == pytest.approx(one.distances.min())
            assert one.farthest_distance == pytest.approx(one.distances.max())
            assert one.nearest_distance <= one.farthest_distance


class TestItIsAnObservation:
    def test_it_satisfies_the_protocol(self):
        search = regressor(n_neighbours=3).neighbour_search(ASKED)

        assert isinstance(search, Observation)
        assert isinstance(search, NeighbourSearch)
        assert len(search) == 4

    def test_result_has_the_shape_predict_relies_on(self):
        search = regressor(n_neighbours=3).neighbour_search(ASKED)

        assert search.result.shape == (4, 3)
        assert search.chosen_targets.shape == (4, 3)
