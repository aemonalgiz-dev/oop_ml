"""Spec for KNearestNeighboursClassifier -- red until three stubs land.

Two specifications carry more than the rest.

Ties. With ``k`` even and two classes a tie is guaranteed, and the rule has to
be stated rather than inherited from whatever ``argmax`` happens to do. The
fixture reaches one on purpose: from (1, 1) with five neighbours the classes
split 2-2-1, and the test pins that the answer is the lower index and that it
is the same answer every time.

Probability width. ``predict_probabilities`` must return one column per class
the *fit* saw, not per class the neighbours happened to include. Otherwise the
matrix changes shape depending on which rows were queried, and column ``k``
stops meaning class ``k`` -- which is the same failure the multi-class
evaluation object guards against from the other side.
"""

import numpy as np
import pytest

from oop_ml.classification.neighbours.k_nearest_classifier import (
    KNearestNeighboursClassifier,
)
from oop_ml.core.data.feature import Feature
from oop_ml.core.distance.calculations import MinkowskiDistance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    NotFittedError,
    SingleClassError,
    TooFewValuesError,
)
from test.fixtures import (
    NEIGHBOUR_GRID,
    NEIGHBOUR_TIE_QUERY,
    NEIGHBOUR_TIE_SHARES,
)


def fitted(**overrides) -> KNearestNeighboursClassifier:
    model = KNearestNeighboursClassifier(**overrides)
    model.fit(NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.class_feature)

    return model


def query(*points) -> list[Feature]:
    return [
        Feature("first", [point[0] for point in points]),
        Feature("second", [point[1] for point in points]),
    ]


class TestConstruction:
    def test_defaults(self):
        model = KNearestNeighboursClassifier()

        assert model.n_neighbours == 5
        assert model.metric is DistanceMetric.EUCLIDEAN

    @pytest.mark.parametrize("n_neighbours", [0, -3])
    def test_a_non_positive_neighbour_count_is_rejected(self, n_neighbours):
        with pytest.raises(ValueError):
            KNearestNeighboursClassifier(n_neighbours=n_neighbours)


class TestNotFitted:
    @pytest.mark.parametrize("attribute", ["n_classes", "n_remembered"])
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(KNearestNeighboursClassifier(), attribute)

    def test_predict_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            KNearestNeighboursClassifier().predict(NEIGHBOUR_GRID.input_features)


class TestValidation:
    def test_no_features_is_rejected(self):
        with pytest.raises(EmptyValuesError):
            KNearestNeighboursClassifier().fit([], NEIGHBOUR_GRID.class_feature)

    def test_misaligned_target_is_rejected(self):
        with pytest.raises(NonEqualArrayLengthError):
            KNearestNeighboursClassifier(n_neighbours=2).fit(
                [Feature("first", [1.0, 2.0, 3.0])], Feature("outcome", [0, 1])
            )

    def test_too_few_rows_for_the_neighbour_count_is_rejected(self):
        with pytest.raises(TooFewValuesError):
            KNearestNeighboursClassifier(n_neighbours=20).fit(
                NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.class_feature
            )

    @pytest.mark.parametrize("classes", [[0, 1, -1, 2, 1], [0.0, 1.5, 2.0, 1.0, 0.0]])
    def test_non_class_targets_are_rejected(self, classes):
        with pytest.raises(NonBinaryLabelsError):
            KNearestNeighboursClassifier(n_neighbours=2).fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0, 5.0])],
                Feature("outcome", classes),
            )

    def test_a_single_class_is_rejected(self):
        with pytest.raises(SingleClassError):
            KNearestNeighboursClassifier(n_neighbours=2).fit(
                [Feature("first", [1.0, 2.0, 3.0])], Feature("outcome", [1, 1, 1])
            )

    def test_a_gap_in_the_classes_is_rejected(self):
        with pytest.raises(SingleClassError):
            KNearestNeighboursClassifier(n_neighbours=2).fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])],
                Feature("outcome", [0, 0, 2, 2]),
            )

    def test_unknown_features_are_rejected(self):
        with pytest.raises(InvalidValuesError):
            fitted().predict([Feature("nonsense", [1.0])])


class TestFitting:
    def test_it_counts_the_classes(self):
        assert fitted().n_classes == 3

    def test_it_keeps_every_row(self):
        assert fitted().n_remembered == NEIGHBOUR_GRID.n_samples

    def test_fit_returns_self_for_chaining(self):
        model = KNearestNeighboursClassifier()

        assert (
            model.fit(NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.class_feature)
            is model
        )


class TestPrediction:
    def test_one_neighbour_reproduces_every_training_class(self):
        model = fitted(n_neighbours=1)

        assert model.predict(NEIGHBOUR_GRID.input_features) == pytest.approx(
            [float(value) for value in NEIGHBOUR_GRID.class_values]
        )

    def test_predictions_are_floats_on_the_class_scale(self):
        labels = fitted(n_neighbours=3).predict(query((0.4, 0.4), (2.0, 2.0)))

        assert labels.dtype == np.float64
        assert set(np.unique(labels)) <= {0.0, 1.0, 2.0}

    def test_one_prediction_per_query(self):
        assert fitted(n_neighbours=3).predict(
            query((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
        ).shape == (3,)

    def test_predict_agrees_with_the_argmax_of_the_shares(self):
        model = fitted(n_neighbours=3)
        probe = query((0.4, 0.4), (1.5, 1.5), (2.0, 0.0))

        assert model.predict(probe) == pytest.approx(
            np.argmax(model.predict_probabilities(probe), axis=1).astype(float)
        )

    def test_feature_order_does_not_matter(self):
        model = fitted(n_neighbours=3)
        probe = query((0.4, 0.4), (1.6, 1.6))

        assert model.predict(probe) == pytest.approx(
            model.predict(list(reversed(probe)))
        )


class TestTies:
    def test_the_fixture_actually_reaches_a_tie(self):
        shares = fitted(n_neighbours=5).predict_probabilities(
            query(NEIGHBOUR_TIE_QUERY)
        )[0]

        assert shares == pytest.approx(NEIGHBOUR_TIE_SHARES)

    def test_a_tie_goes_to_the_lower_class_index(self):
        predicted = fitted(n_neighbours=5).predict(query(NEIGHBOUR_TIE_QUERY))[0]

        assert predicted == pytest.approx(0.0)

    def test_the_tie_breaks_the_same_way_every_time(self):
        # Deterministic is the property that matters. A model answering
        # differently on identical input is worse than one answering oddly.
        answers = {
            float(fitted(n_neighbours=5).predict(query(NEIGHBOUR_TIE_QUERY))[0])
            for _ in range(5)
        }

        assert len(answers) == 1


class TestProbabilities:
    def test_one_column_per_fitted_class(self):
        shares = fitted(n_neighbours=3).predict_probabilities(query((0.0, 0.0)))

        assert shares.shape == (1, 3)

    def test_the_width_does_not_depend_on_which_rows_were_queried(self):
        # A corner query draws only class 0 neighbours; the matrix must still
        # be three wide or column k stops meaning class k.
        model = fitted(n_neighbours=3)
        corner = model.predict_probabilities(query((0.0, 0.0)))

        assert corner.shape == (1, 3)
        assert corner[0].sum() == pytest.approx(1.0)

    def test_every_row_sums_to_one(self):
        shares = fitted(n_neighbours=3).predict_probabilities(
            query((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.5, 1.5))
        )

        assert shares.sum(axis=1) == pytest.approx(np.ones(4))

    def test_the_resolution_is_one_over_k(self):
        # Not calibrated confidence: with five neighbours, 0.53 is unreachable.
        shares = fitted(n_neighbours=5).predict_probabilities(
            query((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.5, 1.5))
        )
        multiples = shares * 5.0

        assert multiples == pytest.approx(np.round(multiples))


class TestMultiClassCameFree:
    def test_two_classes_need_no_special_handling(self):
        # No reference class, no wrapper, no softmax -- the same vote.
        binary_target = Feature(
            "outcome", [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        model = KNearestNeighboursClassifier(n_neighbours=3)
        model.fit(NEIGHBOUR_GRID.input_features, binary_target)

        assert model.n_classes == 2
        assert model.predict_probabilities(query((0.0, 0.0))).shape == (1, 2)

    def test_three_classes_use_the_identical_path(self):
        model = fitted(n_neighbours=3)

        assert model.n_classes == 3
        assert model.predict_probabilities(query((0.0, 0.0))).shape == (1, 3)


class TestEvaluation:
    def test_it_scores_like_any_other_multi_class_model(self):
        model = fitted(n_neighbours=1)
        evaluation = model.evaluate(
            NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.class_feature
        )

        assert evaluation.n_classes == 3
        assert evaluation.accuracy == pytest.approx(1.0)

    def test_score_is_the_accuracy(self):
        model = fitted(n_neighbours=1)

        assert model.score(
            NEIGHBOUR_GRID.input_features, NEIGHBOUR_GRID.class_feature
        ) == pytest.approx(1.0)


class TestEveryMetricWorksEndToEnd:
    """The enum is only useful if every member survives a real fit."""

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_a_fitted_model_predicts_a_known_class(self, metric):
        model = fitted(n_neighbours=3, metric=metric)

        predictions = model.predict(query((0.4, 0.4), (2.0, 2.0)))

        assert set(predictions).issubset(set(range(model.n_classes)))

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_probabilities_still_sum_to_one(self, metric):
        # A vote is a vote regardless of how "near" was defined, so nothing
        # about the metric should be able to unbalance the shares.
        model = fitted(n_neighbours=3, metric=metric)

        shares = model.predict_probabilities(query((0.4, 0.4), (2.0, 2.0)))

        assert shares.sum(axis=1) == pytest.approx(np.ones(2))
        assert (shares >= 0.0).all()


class TestClassShares:
    """The tally, which is a scatter-add flattened into one bincount."""

    def test_every_class_gets_a_column_even_with_no_votes(self):
        # The reason minlength is not optional: query a corner of the grid
        # where one class has no representatives nearby and the matrix must
        # still be as wide as the fit was.
        model = fitted(n_neighbours=1)

        shares = model.predict_probabilities(query((0.0, 0.0)))

        assert shares.shape == (1, model.n_classes)
        assert (shares == 0.0).sum() == model.n_classes - 1

    def test_repeated_votes_accumulate_rather_than_overwrite(self):
        # The bug a plain indexed += would have: several neighbours voting the
        # same way, and only the last one counting.
        model = fitted(n_neighbours=3)

        shares = model.predict_probabilities(query((0.0, 0.0)))

        assert shares.max() == pytest.approx(1.0)

    def test_the_tally_matches_a_plain_python_count(self):
        # An oracle written the obvious way, against the flattened bincount.
        model = fitted(n_neighbours=5)
        points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, -3.0)]

        shares = model.predict_probabilities(query(*points))
        neighbour_classes = model._neighbour_targets(query(*points))

        for row, votes in enumerate(neighbour_classes):
            for class_index in range(model.n_classes):
                counted = sum(1 for vote in votes if int(vote) == class_index)
                assert shares[row, class_index] == pytest.approx(counted / 5)


class TestACalculationCanBePassedDirectly:
    def test_an_unnamed_p_norm_is_accepted(self):
        model = fitted(n_neighbours=3, metric=MinkowskiDistance(3))

        assert isinstance(model.metric, MinkowskiDistance)
        assert set(model.predict(query((0.4, 0.4)))).issubset(
            set(range(model.n_classes))
        )
