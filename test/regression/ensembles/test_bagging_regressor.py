"""Spec for BaggingRegressor -- red until the ensemble stubs land.

Two of these tests carry the argument and the rest are contract.

``test_beats_one_unpruned_tree_on_rows_it_never_saw`` is the whole reason the
class exists, and it has to be measured on held-out rows because an unpruned
tree scores a perfect 1.0 on its training data. Any test written against the
training rows would be satisfied by memorisation, which is precisely the failure
bagging exists to fix.

``test_one_member_is_a_lone_tree_on_its_own_resample`` pins the contract chain
end to end without appealing to statistics: seed the ensemble, draw the same
sample independently, fit one tree on exactly those rows, and the predictions
must agree to the bit. It nails down the generator, the draw order, the
resampling of both features and target, and the fitting of a copy -- five things
that a statistical test would let slide.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.ensemble.bootstrap import BootstrapSample
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NotFittedError,
)
from oop_ml.regression.ensembles.bagging_regressor import BaggingRegressor
from oop_ml.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from test.fixtures import DOMINATED_SIGNAL, ENSEMBLE_MEMBERS


@pytest.fixture
def fitted() -> BaggingRegressor:
    return BaggingRegressor(n_members=ENSEMBLE_MEMBERS, random_seed=0).fit(
        DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
    )


class TestFit:
    """What a fit produces."""

    def test_holds_the_requested_number_of_members(
        self, fitted: BaggingRegressor
    ) -> None:
        assert len(fitted.members) == ENSEMBLE_MEMBERS

    def test_every_member_is_its_own_object(self, fitted: BaggingRegressor) -> None:
        """A copy per member, not the prototype fitted repeatedly.

        Fitting one shared object twenty times leaves twenty references to
        whichever member went last, and the average of a model with itself is
        that model.
        """
        assert len({id(member) for member in fitted.members}) == ENSEMBLE_MEMBERS

    def test_members_differ_from_each_other(self, fitted: BaggingRegressor) -> None:
        """If they agreed everywhere the average would buy nothing."""
        predictions = np.array(
            [
                member.predict(DOMINATED_SIGNAL.held_out_features)
                for member in fitted.members
                if isinstance(member, DecisionTreeRegressor)
            ]
        )

        assert predictions.std(axis=0).mean() > 0.0

    def test_one_member_is_a_lone_tree_on_its_own_resample(self) -> None:
        ensemble = BaggingRegressor(n_members=1, random_seed=7).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        drawn = BootstrapSample.draw(
            DOMINATED_SIGNAL.n_samples, np.random.default_rng(7)
        ).drawn
        target = DOMINATED_SIGNAL.target_feature
        lone = DecisionTreeRegressor().fit(
            [
                Feature(feature.name, feature.column.values[drawn])
                for feature in DOMINATED_SIGNAL.input_features
            ],
            Feature(target.name, target.column.values[drawn]),
        )

        assert np.allclose(
            ensemble.predict(DOMINATED_SIGNAL.held_out_features),
            lone.predict(DOMINATED_SIGNAL.held_out_features),
        )

    def test_the_same_seed_fits_the_same_ensemble(self) -> None:
        predictions = [
            BaggingRegressor(n_members=5, random_seed=11)
            .fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature)
            .predict(DOMINATED_SIGNAL.held_out_features)
            for _ in range(2)
        ]

        assert np.array_equal(predictions[0], predictions[1])

    def test_a_different_seed_fits_a_different_ensemble(self) -> None:
        first = BaggingRegressor(n_members=5, random_seed=11).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )
        second = BaggingRegressor(n_members=5, random_seed=12).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        assert not np.array_equal(
            first.predict(DOMINATED_SIGNAL.held_out_features),
            second.predict(DOMINATED_SIGNAL.held_out_features),
        )


class TestPredict:
    """What the ensemble answers."""

    def test_returns_one_value_per_row(self, fitted: BaggingRegressor) -> None:
        predictions = fitted.predict(DOMINATED_SIGNAL.held_out_features)

        assert predictions.shape == (DOMINATED_SIGNAL.n_samples,)

    def test_is_the_mean_of_what_the_members_said(
        self, fitted: BaggingRegressor
    ) -> None:
        members = np.array(
            [
                member.predict(DOMINATED_SIGNAL.held_out_features)
                for member in fitted.members
                if isinstance(member, DecisionTreeRegressor)
            ]
        )

        assert np.allclose(
            fitted.predict(DOMINATED_SIGNAL.held_out_features),
            members.mean(axis=0),
        )

    def test_ignores_the_order_features_arrive_in(
        self, fitted: BaggingRegressor
    ) -> None:
        features = DOMINATED_SIGNAL.held_out_features

        assert np.allclose(
            fitted.predict(features), fitted.predict(list(reversed(features)))
        )

    def test_beats_one_unpruned_tree_on_rows_it_never_saw(
        self, fitted: BaggingRegressor
    ) -> None:
        lone = DecisionTreeRegressor().fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        assert fitted.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        ) > lone.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        )

    def test_more_members_never_makes_it_worse(self) -> None:
        """The property that separates averaging from every other knob here.

        Extra members stop helping once the correlation floor is reached; they
        are not supposed to start hurting. Checked as a floor rather than a
        strict improvement, because past the floor the difference is noise.
        """
        scores = [
            BaggingRegressor(n_members=count, random_seed=3)
            .fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature)
            .score(
                DOMINATED_SIGNAL.held_out_features,
                DOMINATED_SIGNAL.held_out_target,
            )
            for count in (1, 5, 25)
        ]

        assert scores[1] > scores[0]
        assert scores[2] > scores[1] - 0.02


class TestUnfitted:
    """Nothing is readable before a fit."""

    @pytest.mark.parametrize("attribute", ["members", "feature_importances"])
    def test_reading_a_learned_attribute_raises(self, attribute: str) -> None:
        with pytest.raises(NotFittedError):
            getattr(BaggingRegressor(), attribute)

    def test_predicting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            BaggingRegressor().predict(DOMINATED_SIGNAL.input_features)


class TestInvalidInput:
    """The guards the base class promises."""

    def test_rejects_no_features(self) -> None:
        with pytest.raises(EmptyValuesError):
            BaggingRegressor(n_members=2).fit([], DOMINATED_SIGNAL.target_feature)

    def test_rejects_a_feature_of_the_wrong_length(self) -> None:
        with pytest.raises(NonEqualArrayLengthError):
            BaggingRegressor(n_members=2).fit(
                [Feature("first", [1.0, 2.0, 3.0])],
                DOMINATED_SIGNAL.target_feature,
            )

    def test_rejects_an_unknown_feature_at_predict(
        self, fitted: BaggingRegressor
    ) -> None:
        with pytest.raises(InvalidValuesError):
            fitted.predict(
                [*DOMINATED_SIGNAL.held_out_features, Feature("extra", [0.0] * 200)]
            )

    def test_rejects_a_missing_feature_at_predict(
        self, fitted: BaggingRegressor
    ) -> None:
        with pytest.raises(InvalidValuesError):
            fitted.predict(DOMINATED_SIGNAL.held_out_features[:2])
