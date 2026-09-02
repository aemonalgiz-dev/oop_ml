"""Spec for GradientBoostingRegressor -- red until the boosting stubs land.

Boosting is exactly reconstructible, unlike bagging, because nothing about it
is random. That makes most of this spec arithmetic rather than statistics: the
starting constant is the target's mean, one round at a learning rate of 1.0 adds
exactly one tree's prediction to it, and the ensemble's answer is the constant
plus the shrunken sum of every round. Each of those is checked against a value
computed independently rather than against whatever the model produced.

The two tests that are not arithmetic pin the two claims that actually motivate
the family. Training error must fall with rounds, because each round is fitted
to what is left. And a small learning rate with many rounds must beat a large
one with few -- the trade that makes boosting worth the sequential cost.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NotFittedError,
)
from oop_ml.numpy.regression.ensembles.gradient_boosting_regressor import (
    GradientBoostingRegressor,
)
from oop_ml.numpy.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from test.fixtures import DOMINATED_SIGNAL

ROUNDS = 40


@pytest.fixture
def fitted() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_rounds=ROUNDS, learning_rate=0.1).fit(
        DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
    )


class TestFit:
    """What a fit produces."""

    def test_holds_one_member_per_round(
        self, fitted: GradientBoostingRegressor
    ) -> None:
        assert len(fitted.members) == ROUNDS

    def test_starts_from_the_targets_mean(
        self, fitted: GradientBoostingRegressor
    ) -> None:
        """The constant that leaves the first member only the structure to fit.

        Without it, a depth-3 tree would have to explain the target's level as
        well as its shape, and would spend its whole budget on the level.
        """
        expected = float(np.mean(DOMINATED_SIGNAL.target_feature.column.values))

        assert fitted.initial_prediction == pytest.approx(expected)

    def test_every_member_is_its_own_object(
        self, fitted: GradientBoostingRegressor
    ) -> None:
        assert len({id(member) for member in fitted.members}) == ROUNDS

    def test_a_fit_is_reproducible(self) -> None:
        """Nothing here is random, so this is exactness, not a seeded match."""
        predictions = [
            GradientBoostingRegressor(n_rounds=10)
            .fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature)
            .predict(DOMINATED_SIGNAL.held_out_features)
            for _ in range(2)
        ]

        assert np.array_equal(predictions[0], predictions[1])


class TestRounds:
    """One round, checked entirely by hand."""

    def test_one_round_at_full_rate_is_the_mean_plus_one_tree(self) -> None:
        ensemble = GradientBoostingRegressor(n_rounds=1, learning_rate=1.0).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        targets = DOMINATED_SIGNAL.target_feature.column.values
        residuals = targets - targets.mean()
        lone = DecisionTreeRegressor(max_depth=3).fit(
            DOMINATED_SIGNAL.input_features, Feature("residual", residuals)
        )

        assert np.allclose(
            ensemble.predict(DOMINATED_SIGNAL.held_out_features),
            targets.mean() + lone.predict(DOMINATED_SIGNAL.held_out_features).values,
        )

    def test_the_rate_scales_what_each_round_contributes(self) -> None:
        """Half the rate must move exactly half as far from the constant."""
        full = GradientBoostingRegressor(n_rounds=1, learning_rate=1.0).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )
        half = GradientBoostingRegressor(n_rounds=1, learning_rate=0.5).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        features = DOMINATED_SIGNAL.held_out_features
        start = full.initial_prediction

        assert np.allclose(
            half.predict(features).values - start,
            (full.predict(features).values - start) / 2.0,
        )

    def test_the_answer_is_the_constant_plus_every_shrunken_round(
        self, fitted: GradientBoostingRegressor
    ) -> None:
        features = DOMINATED_SIGNAL.held_out_features
        expected = np.full(DOMINATED_SIGNAL.n_samples, fitted.initial_prediction)
        for member in fitted.members:
            assert isinstance(member, DecisionTreeRegressor)
            expected = expected + fitted.learning_rate * member.predict(features).values

        assert np.allclose(fitted.predict(features), expected)


class TestResiduals:
    """What a round is fitted on."""

    def test_squared_error_leaves_the_plain_difference(
        self, fitted: GradientBoostingRegressor
    ) -> None:
        targets = np.array([3.0, -1.0, 0.5, 10.0])
        predictions = np.array([1.0, -1.0, 2.5, 4.0])

        assert np.allclose(
            fitted._residuals(targets, predictions),
            [2.0, 0.0, -2.0, 6.0],
        )

    def test_a_perfect_prediction_leaves_nothing(
        self, fitted: GradientBoostingRegressor
    ) -> None:
        targets = np.array([3.0, -1.0, 0.5])

        assert np.allclose(fitted._residuals(targets, targets), 0.0)


class TestConvergence:
    """The claims that make the family worth its sequential cost."""

    def test_more_rounds_fit_the_training_data_better(self) -> None:
        scores = [
            GradientBoostingRegressor(n_rounds=count, learning_rate=0.1)
            .fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature)
            .score(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature)
            for count in (1, 10, 50)
        ]

        assert scores[0] < scores[1] < scores[2]

    def test_a_small_rate_with_many_rounds_beats_a_large_one_with_few(
        self,
    ) -> None:
        """Equal total travel, unequal results -- the whole case for shrinkage.

        Both cover a nominal distance of 5.0. The patient one is better on rows
        it never saw, because committing less per step leaves less room to
        commit to noise.
        """
        patient = GradientBoostingRegressor(n_rounds=100, learning_rate=0.05)
        hasty = GradientBoostingRegressor(n_rounds=5, learning_rate=1.0)
        for model in (patient, hasty):
            model.fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature)

        assert patient.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        ) > hasty.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        )

    def test_beats_one_unpruned_tree_on_rows_it_never_saw(
        self, fitted: GradientBoostingRegressor
    ) -> None:
        lone = DecisionTreeRegressor().fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        assert fitted.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        ) > lone.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        )


class TestUnfitted:
    """Nothing is readable before a fit."""

    @pytest.mark.parametrize("attribute", ["members", "initial_prediction"])
    def test_reading_a_learned_attribute_raises(self, attribute: str) -> None:
        with pytest.raises(NotFittedError):
            getattr(GradientBoostingRegressor(), attribute)

    def test_predicting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GradientBoostingRegressor().predict(DOMINATED_SIGNAL.input_features)


class TestInvalidInput:
    """The guards the base class promises."""

    def test_rejects_no_features(self) -> None:
        with pytest.raises(EmptyValuesError):
            GradientBoostingRegressor(n_rounds=2).fit(
                [], DOMINATED_SIGNAL.target_feature
            )

    def test_rejects_a_feature_of_the_wrong_length(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            GradientBoostingRegressor(n_rounds=2).fit(
                [Feature("first", [1.0, 2.0, 3.0])],
                DOMINATED_SIGNAL.target_feature,
            )

    def test_rejects_a_missing_feature_at_predict(
        self, fitted: GradientBoostingRegressor
    ) -> None:
        with pytest.raises(InvalidValuesError):
            fitted.predict(DOMINATED_SIGNAL.held_out_features[:2])

    @pytest.mark.parametrize("learning_rate", [0.0, -0.1, 1.5])
    def test_rejects_a_meaningless_learning_rate(self, learning_rate: float) -> None:
        with pytest.raises(ValueError):
            GradientBoostingRegressor(learning_rate=learning_rate)
