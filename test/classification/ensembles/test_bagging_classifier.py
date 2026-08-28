"""Spec for BaggingClassifier -- red until the ensemble stubs land.

Most of this mirrors the regression side. Three tests do not, and they are the
ones worth reading.

``test_averages_probabilities_rather_than_counting_votes`` constructs the
disagreement directly: members whose averaged probability and whose majority
vote point at different classes. Nothing about the shape of the output
distinguishes the two rules, so a vote-counting implementation would pass every
other test in this file.

``test_probabilities_are_graded_where_members_disagree`` pins what the averaging
buys. A single unpruned tree reports 1.0 in every leaf, because every leaf is
pure; an average over disagreeing members reports something in between, which is
the first honest probability this library's trees produce.

``test_probability_width_is_the_classes_the_fit_saw`` guards the trap the tree
classifier documents. A member fitted on a resample can miss a rare class
entirely, and its matrix would then be one column narrower than its siblings'
and refuse to stack.
"""

import numpy as np
import pytest

from oop_ml.classification.ensembles.bagging_classifier import BaggingClassifier
from oop_ml.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.core.data.feature import Feature
from oop_ml.core.ensemble.member_predictions import MemberPredictions
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    NotFittedError,
    SingleClassError,
)
from test.fixtures import DOMINATED_SIGNAL, ENSEMBLE_MEMBERS, THREE_CLASSES


@pytest.fixture
def fitted() -> BaggingClassifier:
    return BaggingClassifier(n_members=ENSEMBLE_MEMBERS, random_seed=0).fit(
        DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.class_feature
    )


class TestFit:
    """What a fit produces."""

    def test_holds_the_requested_number_of_members(
        self, fitted: BaggingClassifier
    ) -> None:
        assert len(fitted.members) == ENSEMBLE_MEMBERS

    def test_every_member_is_its_own_object(self, fitted: BaggingClassifier) -> None:
        assert len({id(member) for member in fitted.members}) == ENSEMBLE_MEMBERS

    def test_records_the_class_count(self, fitted: BaggingClassifier) -> None:
        assert fitted.n_classes == 2

    def test_the_same_seed_fits_the_same_ensemble(self) -> None:
        predictions = [
            BaggingClassifier(n_members=5, random_seed=13)
            .fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.class_feature)
            .predict(DOMINATED_SIGNAL.held_out_features)
            for _ in range(2)
        ]

        assert np.array_equal(predictions[0], predictions[1])


class TestProbabilities:
    """The matrix, and why it is averaged rather than voted."""

    def test_returns_one_row_per_query_and_one_column_per_class(
        self, fitted: BaggingClassifier
    ) -> None:
        probabilities = fitted.predict_probabilities(DOMINATED_SIGNAL.held_out_features)

        assert probabilities.shape == (DOMINATED_SIGNAL.n_samples, 2)

    def test_rows_sum_to_one(self, fitted: BaggingClassifier) -> None:
        probabilities = fitted.predict_probabilities(DOMINATED_SIGNAL.held_out_features)

        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_is_the_mean_of_the_members_matrices(
        self, fitted: BaggingClassifier
    ) -> None:
        members = np.array(
            [
                member.predict_probabilities(DOMINATED_SIGNAL.held_out_features)
                for member in fitted.members
                if isinstance(member, DecisionTreeClassifier)
            ]
        )

        assert np.allclose(
            fitted.predict_probabilities(DOMINATED_SIGNAL.held_out_features),
            members.mean(axis=0),
        )

    def test_probabilities_are_graded_where_members_disagree(
        self, fitted: BaggingClassifier
    ) -> None:
        probabilities = fitted.predict_probabilities(DOMINATED_SIGNAL.held_out_features)
        strictly_between = (probabilities > 0.0) & (probabilities < 1.0)

        assert strictly_between.any()

    def test_probability_width_is_the_classes_the_fit_saw(self) -> None:
        ensemble = BaggingClassifier(n_members=10, random_seed=1).fit(
            THREE_CLASSES.input_features, THREE_CLASSES.target_feature
        )
        probabilities = ensemble.predict_probabilities(THREE_CLASSES.input_features)

        assert ensemble.n_classes == 3
        assert probabilities.shape[1] == 3
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestPredict:
    """The class chosen."""

    def test_returns_one_class_per_row(self, fitted: BaggingClassifier) -> None:
        predictions = fitted.predict(DOMINATED_SIGNAL.held_out_features)

        assert predictions.shape == (DOMINATED_SIGNAL.n_samples,)

    def test_only_ever_names_a_class_the_fit_saw(
        self, fitted: BaggingClassifier
    ) -> None:
        predictions = fitted.predict(DOMINATED_SIGNAL.held_out_features)

        assert set(np.unique(predictions).tolist()) <= {0.0, 1.0}

    def test_agrees_with_the_most_probable_column(
        self, fitted: BaggingClassifier
    ) -> None:
        probabilities = fitted.predict_probabilities(DOMINATED_SIGNAL.held_out_features)

        assert np.array_equal(
            fitted.predict(DOMINATED_SIGNAL.held_out_features),
            np.argmax(probabilities, axis=1).astype(np.float64),
        )

    def test_averages_probabilities_rather_than_counting_votes(
        self, fitted: BaggingClassifier
    ) -> None:
        """Six barely-committed members against four nearly certain ones.

        The vote says class 1, six to four. The average says class 0, 0.70 to
        0.30, because the four were sure and the six were not. Only the second
        answer uses how confident each member was, and confidence is the whole
        reason this class reads probabilities from its members.
        """
        member_predictions = np.array([[[0.49, 0.51]]] * 6 + [[[0.99, 0.01]]] * 4)

        assert fitted._combine(MemberPredictions(member_predictions)) == pytest.approx(
            [0.0]
        )

    def test_ties_go_to_the_lowest_class(self, fitted: BaggingClassifier) -> None:
        member_predictions = np.array([[[0.5, 0.5]], [[0.5, 0.5]]])

        assert fitted._combine(MemberPredictions(member_predictions)) == pytest.approx(
            [0.0]
        )

    def test_ignores_the_order_features_arrive_in(
        self, fitted: BaggingClassifier
    ) -> None:
        features = DOMINATED_SIGNAL.held_out_features

        assert np.array_equal(
            fitted.predict(features), fitted.predict(list(reversed(features)))
        )

    def test_beats_one_unpruned_tree_on_rows_it_never_saw(
        self, fitted: BaggingClassifier
    ) -> None:
        lone = DecisionTreeClassifier().fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.class_feature
        )

        assert fitted.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_classes
        ) > lone.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_classes
        )


class TestUnfitted:
    """Nothing is readable before a fit."""

    @pytest.mark.parametrize("attribute", ["members", "n_classes"])
    def test_reading_a_learned_attribute_raises(self, attribute: str) -> None:
        with pytest.raises(NotFittedError):
            getattr(BaggingClassifier(), attribute)

    def test_predicting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            BaggingClassifier().predict(DOMINATED_SIGNAL.input_features)

    def test_predicting_probabilities_raises(self) -> None:
        with pytest.raises(NotFittedError):
            BaggingClassifier().predict_probabilities(DOMINATED_SIGNAL.input_features)


class TestInvalidInput:
    """The guards, including the ones classification tightens."""

    def test_rejects_no_features(self) -> None:
        with pytest.raises(EmptyValuesError):
            BaggingClassifier(n_members=2).fit([], DOMINATED_SIGNAL.class_feature)

    def test_rejects_a_feature_of_the_wrong_length(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            BaggingClassifier(n_members=2).fit(
                [Feature("first", [1.0, 2.0, 3.0])],
                DOMINATED_SIGNAL.class_feature,
            )

    def test_rejects_a_fractional_class(self) -> None:
        with pytest.raises(NonBinaryLabelsError):
            BaggingClassifier(n_members=2).fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])],
                Feature("classes", [0.0, 1.0, 0.5, 1.0]),
            )

    def test_rejects_a_single_class(self) -> None:
        with pytest.raises(SingleClassError):
            BaggingClassifier(n_members=2).fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])],
                Feature("classes", [1.0, 1.0, 1.0, 1.0]),
            )

    def test_rejects_a_missing_feature_at_predict(
        self, fitted: BaggingClassifier
    ) -> None:
        with pytest.raises(InvalidValuesError):
            fitted.predict(DOMINATED_SIGNAL.held_out_features[:2])
