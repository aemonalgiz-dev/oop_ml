"""Spec for RandomForestClassifier -- red until the ensemble stubs land.

The decorrelation tests mirror the regression forest's. What is here and not
there is parity, where a lone greedy tree does not merely do worse -- it fails
completely, and an ensemble is what rescues it.

Parity gives the right questions a gain of essentially zero: split on either
real feature and both sides come back half and half, which is the impurity the
node already had. Measured on this fixture, 0.0037 for a real feature against
0.0084 for a pure noise column, so the search roots on noise and a depth-3 tree
lands at 0.537 -- barely above the 0.5 a coin gets.

**What rescues it is the ensemble, and the restriction only sharpens that.**
Measured across two seeds: lone tree 0.52-0.59, unrestricted forest 0.95-0.97,
restricted to one feature 0.99. Bootstrap resampling alone is already enough to
make different members root on different spurious splits, so an honest test
asserts that the ensemble recovers the signal and does not credit the
restriction with all of it. An earlier draft of this file claimed the
restriction was the mechanism; measuring the unrestricted case is what showed
that to be most of the way wrong.
"""

import numpy as np
import pytest

from oop_ml.classification.ensembles.random_forest_classifier import (
    RandomForestClassifier,
)
from oop_ml.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.core.exceptions import InvalidValuesError, NotFittedError
from oop_ml.core.tree.node import DecisionNode
from test.fixtures import (
    DOMINATED_SIGNAL,
    ENSEMBLE_MEMBERS,
    EXCLUSIVE_OR,
    FOREST_MAX_FEATURES,
    PARITY_ENSEMBLE_FLOOR,
    PARITY_LONE_TREE_CEILING,
    PARITY_MAX_DEPTH,
)


def root_features(forest: RandomForestClassifier) -> set[str]:
    """Which feature each member asked about first."""
    names = set()
    for member in forest.members:
        assert isinstance(member, DecisionTreeClassifier)
        root = member.root
        assert isinstance(root, DecisionNode)
        names.add(root.split.feature_name)

    return names


def fit(max_features: int | None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_members=ENSEMBLE_MEMBERS, max_features=max_features, random_seed=0
    ).fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.class_feature)


class TestDecorrelation:
    """The one thing a forest adds to bagging."""

    def test_unrestricted_members_all_start_the_same_way(self) -> None:
        assert root_features(fit(None)) == {"dominant"}

    @pytest.mark.parametrize("max_features", [1, 2, FOREST_MAX_FEATURES])
    def test_restriction_spreads_the_root_across_features(
        self, max_features: int
    ) -> None:
        assert len(root_features(fit(max_features))) > 1

    def test_every_member_is_seeded_differently(self) -> None:
        """Identical seeds would reproduce bagging while looking decorrelated."""
        seeds = set()
        for member in fit(1).members:
            assert isinstance(member, DecisionTreeClassifier)
            seeds.add(member.random_seed)

        assert len(seeds) == ENSEMBLE_MEMBERS

    def test_restricting_to_every_feature_is_plain_bagging(self) -> None:
        assert np.array_equal(
            fit(DOMINATED_SIGNAL.n_features).predict(
                DOMINATED_SIGNAL.held_out_features
            ),
            fit(None).predict(DOMINATED_SIGNAL.held_out_features),
        )


class TestParity:
    """Where a lone greedy tree fails outright and an ensemble does not."""

    def test_a_noise_column_outscores_the_real_features(self) -> None:
        """The premise: greed is not unlucky here, it is correctly blind.

        No single question about a parity target reduces impurity, so a
        continuous column that happens to line up with a handful of rows wins
        the root on nothing but sampling noise.
        """
        tree = DecisionTreeClassifier(max_depth=1).fit(
            EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature
        )
        real_only = DecisionTreeClassifier(max_depth=1).fit(
            EXCLUSIVE_OR.real_features, EXCLUSIVE_OR.class_feature
        )

        assert isinstance(tree.root, DecisionNode)
        assert isinstance(real_only.root, DecisionNode)
        assert tree.root.split.feature_name == "distractor"
        assert tree.root.split.gain > real_only.root.split.gain

    def test_a_lone_tree_scores_near_chance(self) -> None:
        """Depth-capped, so the failure is greed rather than a lack of room.

        Left unstopped the tree memorises all three hundred rows perfectly,
        which would hide exactly the thing being demonstrated.
        """
        tree = DecisionTreeClassifier(max_depth=PARITY_MAX_DEPTH).fit(
            EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature
        )

        assert (
            tree.score(EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature)
            < PARITY_LONE_TREE_CEILING
        )

    @pytest.mark.parametrize("max_features", [None, 1, 2])
    def test_an_ensemble_recovers_the_signal(self, max_features: int | None) -> None:
        forest = RandomForestClassifier(
            n_members=ENSEMBLE_MEMBERS,
            max_features=max_features,
            max_depth=PARITY_MAX_DEPTH,
            random_seed=0,
        ).fit(EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature)

        assert (
            forest.score(EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature)
            > PARITY_ENSEMBLE_FLOOR
        )


class TestProbabilities:
    """The inherited matrix, spot-checked on the subclass."""

    def test_returns_one_row_per_query_and_one_column_per_class(self) -> None:
        probabilities = fit(FOREST_MAX_FEATURES).predict_probabilities(
            DOMINATED_SIGNAL.held_out_features
        )

        assert probabilities.shape == (DOMINATED_SIGNAL.n_samples, 2)

    def test_rows_sum_to_one(self) -> None:
        probabilities = fit(FOREST_MAX_FEATURES).predict_probabilities(
            DOMINATED_SIGNAL.held_out_features
        )

        assert np.allclose(probabilities.values.sum(axis=1), 1.0)


class TestPredict:
    """The inherited contract, spot-checked."""

    def test_returns_one_class_per_row(self) -> None:
        predictions = fit(FOREST_MAX_FEATURES).predict(
            DOMINATED_SIGNAL.held_out_features
        )

        assert predictions.shape == (DOMINATED_SIGNAL.n_samples,)

    def test_beats_one_unpruned_tree_on_rows_it_never_saw(self) -> None:
        lone = DecisionTreeClassifier().fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.class_feature
        )

        assert fit(FOREST_MAX_FEATURES).score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_classes
        ) > lone.score(
            DOMINATED_SIGNAL.held_out_features, DOMINATED_SIGNAL.held_out_classes
        )

    def test_ignores_the_order_features_arrive_in(self) -> None:
        forest = fit(FOREST_MAX_FEATURES)
        features = DOMINATED_SIGNAL.held_out_features

        assert np.array_equal(
            forest.predict(features), forest.predict(list(reversed(features)))
        )


class TestInvalidInput:
    """The guards, inherited."""

    def test_predicting_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            RandomForestClassifier().predict(DOMINATED_SIGNAL.input_features)

    def test_rejects_a_missing_feature_at_predict(self) -> None:
        with pytest.raises(InvalidValuesError):
            fit(FOREST_MAX_FEATURES).predict(DOMINATED_SIGNAL.held_out_features[:2])

    @pytest.mark.parametrize("max_features", [0, -1])
    def test_rejects_a_meaningless_restriction(self, max_features: int) -> None:
        with pytest.raises(ValueError):
            RandomForestClassifier(max_features=max_features)
