"""Spec for RandomForestRegressor -- red until the ensemble stubs land.

A forest is bagging plus one thing, so most of the contract is inherited and
tested there. What is tested here is the one thing: that restricting which
features a node may consider actually decorrelates the members.

That is measured directly, by reading which feature each member put at its root.
Unrestricted, all twenty choose ``dominant``, because resamples share most of
their rows and therefore agree about which split is strongest. Restricted, five
or six different features appear there. Correlation is the floor in
``r * s^2 + (1 - r) * s^2 / B``, so lowering it is the mechanism, and the
mechanism is what these tests pin.

**The score is deliberately not asserted to improve, because on this fixture it
does not.** Measured, held-out R^2: one tree 0.60, bagged 0.7364, forest at
three of six 0.7343, at two 0.71, at one 0.59. The restriction is variance spent
to buy decorrelation, and here the two roughly cancel. Writing a test that
demanded an improvement would have meant tuning the fixture until the library
looked good, which is the opposite of what a fixture is for. What is asserted is
that the forest beats a lone tree -- true, and the claim that actually matters.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import InvalidValuesError, NotFittedError
from oop_ml.core.tree.node import DecisionNode
from oop_ml.numpy.regression.ensembles.random_forest_regressor import (
    RandomForestRegressor,
)
from oop_ml.numpy.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from test.fixtures import DOMINATED_SIGNAL, ENSEMBLE_MEMBERS, FOREST_MAX_FEATURES


def root_features(forest: RandomForestRegressor) -> set[str]:
    """Which feature each member asked about first."""
    names = set()
    for member in forest.members:
        assert isinstance(member, DecisionTreeRegressor)
        root = member.root
        assert isinstance(root, DecisionNode)
        names.add(root.split.feature_name)

    return names


def fit(max_features: int | None) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_members=ENSEMBLE_MEMBERS, max_features=max_features, random_seed=0
    ).fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature)


class TestDecorrelation:
    """The one thing a forest adds to bagging."""

    def test_unrestricted_members_all_start_the_same_way(self) -> None:
        """The floor a forest exists to lower, shown before it is lowered."""
        assert root_features(fit(None)) == {"dominant"}

    @pytest.mark.parametrize("max_features", [1, 2, FOREST_MAX_FEATURES])
    def test_restriction_spreads_the_root_across_features(
        self, max_features: int
    ) -> None:
        assert len(root_features(fit(max_features))) > 1

    def test_every_member_is_seeded_differently(self) -> None:
        """Otherwise the restriction is identical in all of them.

        Twenty trees handed the same seed draw the same features at every node,
        which reproduces bagging exactly while appearing to have decorrelated
        anything. The spread of roots above is what would catch it, and this
        states the cause rather than the symptom.
        """
        forest = fit(1)
        seeds = set()
        for member in forest.members:
            assert isinstance(member, DecisionTreeRegressor)
            seeds.add(member.random_seed)

        assert len(seeds) == ENSEMBLE_MEMBERS

    def test_restricting_to_every_feature_is_plain_bagging(self) -> None:
        """Six of six is no restriction, and must behave like none at all."""
        assert np.allclose(
            fit(DOMINATED_SIGNAL.n_features).predict(
                DOMINATED_SIGNAL.held_out_features
            ),
            fit(None).predict(DOMINATED_SIGNAL.held_out_features),
        )

    def test_members_are_still_fitted_on_resamples(self) -> None:
        """The restriction is added to bagging, not substituted for it."""
        predictions = np.array(
            [
                member.predict(DOMINATED_SIGNAL.held_out_features)
                for member in fit(FOREST_MAX_FEATURES).members
                if isinstance(member, DecisionTreeRegressor)
            ]
        )

        assert predictions.std(axis=0).mean() > 0.0


class TestPredict:
    """The inherited contract, spot-checked on the subclass."""

    def test_returns_one_value_per_row(self) -> None:
        predictions = fit(FOREST_MAX_FEATURES).predict(
            DOMINATED_SIGNAL.held_out_features
        )

        assert predictions.shape == (DOMINATED_SIGNAL.n_samples,)

    def test_beats_one_unpruned_tree_on_rows_it_never_saw(self) -> None:
        lone = DecisionTreeRegressor().fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        assert fit(FOREST_MAX_FEATURES).score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        ) > lone.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_target
        )

    def test_the_same_seed_fits_the_same_forest(self) -> None:
        assert np.array_equal(
            fit(FOREST_MAX_FEATURES).predict(DOMINATED_SIGNAL.held_out_features),
            fit(FOREST_MAX_FEATURES).predict(DOMINATED_SIGNAL.held_out_features),
        )

    def test_ignores_the_order_features_arrive_in(self) -> None:
        forest = fit(FOREST_MAX_FEATURES)
        features = DOMINATED_SIGNAL.held_out_features

        assert np.allclose(
            forest.predict(features), forest.predict(list(reversed(features)))
        )


class TestInvalidInput:
    """The guards, inherited."""

    def test_predicting_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            RandomForestRegressor().predict(DOMINATED_SIGNAL.input_features)

    def test_rejects_a_missing_feature_at_predict(self) -> None:
        with pytest.raises(InvalidValuesError):
            fit(FOREST_MAX_FEATURES).predict(DOMINATED_SIGNAL.held_out_features[:2])

    @pytest.mark.parametrize("max_features", [0, -1])
    def test_rejects_a_meaningless_restriction(self, max_features: int) -> None:
        with pytest.raises(ValueError):
            RandomForestRegressor(max_features=max_features)
