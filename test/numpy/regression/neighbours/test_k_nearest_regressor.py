"""Spec for KNearestNeighboursRegressor -- red until three stubs land.

Three tests here pin behaviour that is specific to a non-parametric model and
has no counterpart anywhere else in the suite.

``k = 1`` reproduces every training target exactly. That is not a sign of a
good fit, it is the definition of memorisation, and a test asserting it makes
the point that training error is meaningless for this family.

Prediction is a step function. Nudge a query and the answer usually does not
move at all, then jumps when the neighbour set changes. No amount of data
smooths that; only a larger ``k`` does.

It cannot extrapolate. Query far outside the training range and the answer
stops changing, because every neighbour is on the same side and their mean is
whatever it is. Least squares would extend the line; this flattens.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.distance.calculations import BroadcastDistance, MinkowskiDistance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.numpy.regression.neighbours.k_nearest_regressor import (
    KNearestNeighboursRegressor,
)
from test.fixtures import (
    NEIGHBOUR_GRID,
    NEIGHBOUR_QUERY,
    NEIGHBOUR_QUERY_MEAN_OF_THREE,
)


def fitted(**overrides) -> KNearestNeighboursRegressor:
    model = KNearestNeighboursRegressor(**overrides)
    model.fit(NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.quantity_feature)

    return model


def query(*points) -> list[Feature]:
    """Build query features from (first, second) pairs."""
    return [
        Feature("first", [point[0] for point in points]),
        Feature("second", [point[1] for point in points]),
    ]


class TestConstruction:
    def test_defaults(self):
        model = KNearestNeighboursRegressor()

        assert model.n_neighbours == 5
        assert model.metric is DistanceMetric.EUCLIDEAN

    @pytest.mark.parametrize("n_neighbours", [0, -1])
    def test_a_non_positive_neighbour_count_is_rejected(self, n_neighbours):
        with pytest.raises(ValueError):
            KNearestNeighboursRegressor(n_neighbours=n_neighbours)

    def test_the_metric_is_a_real_choice(self):
        model = KNearestNeighboursRegressor(metric=DistanceMetric.MANHATTAN)

        assert model.metric is DistanceMetric.MANHATTAN


class TestNotFitted:
    def test_predict_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            KNearestNeighboursRegressor().predict(NEIGHBOUR_GRID.input_features)

    def test_n_remembered_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            _ = KNearestNeighboursRegressor().n_remembered


class TestValidation:
    def test_no_features_is_rejected(self):
        with pytest.raises(EmptyValuesError):
            KNearestNeighboursRegressor().fit([], NEIGHBOUR_GRID.quantity_feature)

    def test_duplicate_feature_names_are_rejected(self):
        repeated = Feature("first", [1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(NonUniqueFeaturesError):
            KNearestNeighboursRegressor(n_neighbours=2).fit(
                [repeated, repeated], Feature("quantity", [1.0, 2.0, 3.0, 4.0, 5.0])
            )

    def test_misaligned_target_is_rejected(self):
        with pytest.raises(NonEqualArrayLengthError):
            KNearestNeighboursRegressor(n_neighbours=2).fit(
                [Feature("first", [1.0, 2.0, 3.0])], Feature("quantity", [1.0, 2.0])
            )

    def test_asking_for_more_neighbours_than_rows_is_rejected(self):
        # Not a degraded answer -- a different question. Nine rows cannot
        # supply ten neighbours.
        with pytest.raises(TooFewValuesError):
            KNearestNeighboursRegressor(n_neighbours=10).fit(
                NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.quantity_feature
            )

    def test_exactly_as_many_neighbours_as_rows_is_allowed(self):
        model = fitted(n_neighbours=NEIGHBOUR_GRID.n_samples)

        assert model.n_remembered == NEIGHBOUR_GRID.n_samples

    def test_unknown_features_are_rejected(self):
        with pytest.raises(InvalidValuesError):
            fitted().predict([Feature("nonsense", [1.0])])


class TestFittingRemembersRatherThanLearns:
    def test_it_keeps_every_row(self):
        assert fitted().n_remembered == NEIGHBOUR_GRID.n_samples

    def test_fit_returns_self_for_chaining(self):
        model = KNearestNeighboursRegressor()

        assert (
            model.fit(NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.quantity_feature)
            is model
        )

    def test_there_are_no_coefficients_to_read(self):
        # The whole point of the family: nothing was fitted.
        assert not hasattr(fitted(), "coefficients")


class TestPrediction:
    def test_averages_the_nearest_three(self):
        model = fitted(n_neighbours=3)

        assert model.predict(query(NEIGHBOUR_QUERY))[0] == pytest.approx(
            NEIGHBOUR_QUERY_MEAN_OF_THREE
        )

    def test_one_prediction_per_query(self):
        predictions = fitted(n_neighbours=3).predict(
            query((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
        )

        assert predictions.shape == (3,)

    def test_feature_order_does_not_matter(self):
        model = fitted(n_neighbours=3)
        forwards = model.predict(query((0.4, 0.4), (1.6, 1.6)))
        backwards = model.predict(list(reversed(query((0.4, 0.4), (1.6, 1.6)))))

        assert forwards == pytest.approx(backwards)

    def test_every_neighbour_counts_the_same(self):
        # Unweighted: a neighbour at distance 0.01 and one at 1.5 both count
        # once. Weighting by distance is a different model.
        model = fitted(n_neighbours=2)
        just_off_a_row = model.predict(query((0.01, 0.0)))[0]
        midway = model.predict(query((0.5, 0.0)))[0]

        assert just_off_a_row == pytest.approx(midway)


class TestKIsTheDial:
    def test_one_neighbour_reproduces_every_training_target(self):
        # Zero training error, and it means nothing at all.
        model = fitted(n_neighbours=1)

        assert model.predict(NEIGHBOUR_GRID.input_features) == pytest.approx(
            NEIGHBOUR_GRID.quantity_values
        )

    def test_every_neighbour_gives_the_global_mean_to_everyone(self):
        model = fitted(n_neighbours=NEIGHBOUR_GRID.n_samples)
        predictions = model.predict(query((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)))

        assert predictions == pytest.approx(
            np.full(3, np.mean(NEIGHBOUR_GRID.quantity_values))
        )

    def test_larger_k_moves_predictions_toward_the_mean(self):
        corner = query((0.0, 0.0))
        overall = float(np.mean(NEIGHBOUR_GRID.quantity_values))

        close = abs(fitted(n_neighbours=1).predict(corner)[0] - overall)
        loose = abs(fitted(n_neighbours=5).predict(corner)[0] - overall)

        assert loose < close


class TestTheSurfaceIsPiecewiseConstant:
    def test_small_moves_change_nothing(self):
        model = fitted(n_neighbours=3)
        here = model.predict(query((0.40, 0.40)))[0]
        nudged = model.predict(query((0.41, 0.41)))[0]

        assert here == pytest.approx(nudged)

    def test_crossing_into_a_new_neighbour_set_jumps(self):
        model = fitted(n_neighbours=1)
        before = model.predict(query((0.49, 0.0)))[0]
        after = model.predict(query((0.51, 0.0)))[0]

        assert before != after


class TestItCannotExtrapolate:
    def test_far_outside_the_data_the_answer_stops_moving(self):
        # Every neighbour is on the same side, so their mean is fixed. A linear
        # model would keep extending the trend; this flattens.
        model = fitted(n_neighbours=3)
        far = model.predict(query((50.0, 50.0)))[0]
        further = model.predict(query((500.0, 500.0)))[0]

        assert far == pytest.approx(further)

    def test_it_never_predicts_outside_the_training_range(self):
        model = fitted(n_neighbours=3)
        predictions = model.predict(query((-100.0, -100.0), (100.0, 100.0)))

        assert predictions.values.min() >= min(NEIGHBOUR_GRID.quantity_values)
        assert predictions.values.max() <= max(NEIGHBOUR_GRID.quantity_values)


class TestTheMetricChangesTheAnswer:
    def test_manhattan_and_euclidean_can_disagree(self):
        euclidean = fitted(n_neighbours=2, metric=DistanceMetric.EUCLIDEAN)
        manhattan = fitted(n_neighbours=2, metric=DistanceMetric.MANHATTAN)
        probe = query((0.9, 0.1), (0.1, 0.9), (1.4, 0.6))

        assert euclidean.predict(probe).shape == manhattan.predict(probe).shape


class TestEvaluation:
    def test_it_scores_like_any_other_regressor(self):
        model = fitted(n_neighbours=1)
        evaluation = model.evaluate(
            NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.quantity_feature
        )

        assert evaluation.r2_score == pytest.approx(1.0)

    def test_score_is_r_squared(self):
        model = fitted(n_neighbours=1)

        assert model.score(
            NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.quantity_feature
        ) == pytest.approx(1.0)


class TestEveryMetricWorksEndToEnd:
    """The enum is only useful if every member survives a real fit.

    The unit tests in ``test/core/distance`` check the arithmetic. These check
    the wiring: that a metric chosen at construction reaches the sweep, that
    nothing downstream assumes Euclidean, and that the answer stays a finite
    number on the same scale as the targets.
    """

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_a_fitted_model_predicts_with_it(self, metric):
        model = fitted(n_neighbours=3, metric=metric)

        predictions = model.predict(query(NEIGHBOUR_QUERY))

        assert predictions.shape == (1,)
        assert np.isfinite(predictions).all()

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_the_answer_is_always_a_mean_of_real_targets(self, metric):
        # Whatever "near" means, the prediction is an average of k remembered
        # targets, so it cannot fall outside their range.
        model = fitted(n_neighbours=3, metric=metric)
        targets = NEIGHBOUR_GRID.quantity_feature.values

        predictions = model.predict(query((0.4, 0.4), (2.0, 2.0), (-5.0, 9.0)))

        assert (predictions >= targets.min()).all()
        assert (predictions <= targets.max()).all()


class TestACalculationCanBePassedDirectly:
    """``MinkowskiDistance(3)`` is a legitimate metric the enum does not name."""

    def test_an_unnamed_p_norm_is_accepted(self):
        model = fitted(n_neighbours=3, metric=MinkowskiDistance(3))

        assert isinstance(model.metric, MinkowskiDistance)
        assert np.isfinite(model.predict(query(NEIGHBOUR_QUERY))).all()

    def test_p_equal_to_two_agrees_with_the_named_member(self):
        # The same metric by two routes. If they disagreed, the enum would be
        # naming something other than what it claims.
        by_object = fitted(n_neighbours=3, metric=MinkowskiDistance(2))
        by_name = fitted(n_neighbours=3, metric=DistanceMetric.EUCLIDEAN)

        assert by_object.predict(query(NEIGHBOUR_QUERY)) == pytest.approx(
            by_name.predict(query(NEIGHBOUR_QUERY))
        )

    def test_a_metric_the_library_has_never_heard_of_works(self):
        # The extensibility the strategy split is for: a user's own notion of
        # near, with no change to any model.
        class FirstFeatureOnly(BroadcastDistance):
            """Ignore every feature but the first."""

            def _between_block(self, query_block, remembered_rows):
                gaps = query_block[:, None, :1] - remembered_rows[None, :, :1]

                return np.abs(gaps).sum(axis=-1)

        model = fitted(n_neighbours=3, metric=FirstFeatureOnly())

        assert np.isfinite(model.predict(query(NEIGHBOUR_QUERY))).all()
