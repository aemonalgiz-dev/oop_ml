"""What sample weights do to a whole tree, rather than to one impurity.

The measures have their own spec; this is the end-to-end version of the same
claim, and the one that matters to a caller. A tree fitted on weighted rows has
to be the tree fitted on the rows written out that many times -- not close to
it, the same tree, because every quantity the growth consults is a sum and a
weight is exactly a multiplicity.

Both tasks, because the regressor and the classifier read the weights in
different places: one in a mean and a variance, the other in class counts and
shares.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import InvalidValuesError, NonEqualArrayLengthError
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.numpy.regression.trees.decision_tree_regressor import (
    DecisionTreeRegressor,
)

_GENERATOR = np.random.default_rng(3)
ROWS = _GENERATOR.normal(size=(20, 2))
TARGETS = 1.0 + 2.0 * ROWS[:, 0] - ROWS[:, 1] + _GENERATOR.normal(scale=0.3, size=20)
CLASSES = (float(np.median(TARGETS)) < TARGETS).astype(float)

#: Whole numbers, so that "counts twice" and "written twice" are comparable at
#: all. Three rows carry more than their share and the rest carry one.
REPEATS = np.array([3, 1, 2, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1])

FEATURES = [Feature("first", ROWS[:, 0]), Feature("second", ROWS[:, 1])]
PROBE = [
    Feature("first", np.linspace(-2.0, 2.0, 25)),
    Feature("second", np.linspace(2.0, -2.0, 25)),
]

_COPIED = np.repeat(ROWS, REPEATS, axis=0)
COPIED_FEATURES = [
    Feature("first", _COPIED[:, 0]),
    Feature("second", _COPIED[:, 1]),
]


class TestWeighingIsCopying:
    def test_the_regressor_grows_the_same_tree(self):
        weighted = DecisionTreeRegressor(max_depth=3).fit(
            FEATURES, Feature("y", TARGETS), REPEATS
        )
        copied = DecisionTreeRegressor(max_depth=3).fit(
            COPIED_FEATURES, Feature("y", np.repeat(TARGETS, REPEATS))
        )

        assert np.array_equal(
            np.asarray(weighted.predict(PROBE)), np.asarray(copied.predict(PROBE))
        )
        assert weighted.depth == copied.depth
        assert weighted.n_leaves == copied.n_leaves

    def test_the_classifier_grows_the_same_tree_and_the_same_shares(self):
        weighted = DecisionTreeClassifier(max_depth=3).fit(
            FEATURES, Feature("y", CLASSES), REPEATS
        )
        copied = DecisionTreeClassifier(max_depth=3).fit(
            COPIED_FEATURES, Feature("y", np.repeat(CLASSES, REPEATS))
        )

        assert np.array_equal(
            np.asarray(weighted.predict(PROBE)), np.asarray(copied.predict(PROBE))
        )
        assert np.allclose(
            np.asarray(weighted.predict_probabilities(PROBE)),
            np.asarray(copied.predict_probabilities(PROBE)),
        )

    def test_the_fixture_actually_makes_a_difference(self):
        """A guard on the guard. If the weights changed nothing, the two tests
        above would hold for an implementation that ignored them."""
        weighted = DecisionTreeRegressor(max_depth=3).fit(
            FEATURES, Feature("y", TARGETS), REPEATS
        )
        plain = DecisionTreeRegressor(max_depth=3).fit(FEATURES, Feature("y", TARGETS))

        assert not np.array_equal(
            np.asarray(weighted.predict(PROBE)), np.asarray(plain.predict(PROBE))
        )


class TestUniformWeightsAreNoWeights:
    @pytest.mark.parametrize("model_type", [DecisionTreeRegressor])
    def test_stating_them_matches_leaving_them_out(self, model_type):
        stated = model_type(max_depth=3).fit(
            FEATURES, Feature("y", TARGETS), np.ones(len(TARGETS))
        )
        omitted = model_type(max_depth=3).fit(FEATURES, Feature("y", TARGETS))

        assert np.array_equal(
            np.asarray(stated.predict(PROBE)), np.asarray(omitted.predict(PROBE))
        )

    def test_scaling_every_weight_alike_changes_almost_nothing(self):
        """Weights are shares of a total, so multiplying them all by seven
        leaves every ratio where it was -- mathematically.

        Not to the last bit, and the tolerance here is the honest reading
        rather than a loosened one: the sums are then formed at a different
        magnitude and round differently, which moves a leaf's mean by 4.4e-16.
        The tree itself, meaning every split and every routing, is identical.
        """
        ones = DecisionTreeRegressor(max_depth=3).fit(
            FEATURES, Feature("y", TARGETS), np.ones(len(TARGETS))
        )
        sevens = DecisionTreeRegressor(max_depth=3).fit(
            FEATURES, Feature("y", TARGETS), np.full(len(TARGETS), 7.0)
        )

        assert ones.describe() == sevens.describe()
        assert np.allclose(
            np.asarray(ones.predict(PROBE)),
            np.asarray(sevens.predict(PROBE)),
            atol=1e-12,
        )


class TestWhatTheStructureStillCounts:
    def test_n_samples_on_a_node_stays_a_row_count(self):
        """Weights change what the tree learns, not how big it says its nodes
        are. A node of three rows counting double holds three rows."""
        weighted = DecisionTreeRegressor(max_depth=1).fit(
            FEATURES, Feature("y", TARGETS), np.full(len(TARGETS), 5.0)
        )

        assert weighted.root.n_samples == len(TARGETS)


class TestWhatItRefuses:
    def test_one_weight_per_row(self):
        with pytest.raises(NonEqualArrayLengthError):
            DecisionTreeRegressor().fit(FEATURES, Feature("y", TARGETS), [1.0, 2.0])

    def test_a_negative_weight(self):
        weights = np.ones(len(TARGETS))
        weights[0] = -1.0

        with pytest.raises(InvalidValuesError):
            DecisionTreeRegressor().fit(FEATURES, Feature("y", TARGETS), weights)

    def test_weights_that_total_nothing(self):
        with pytest.raises(InvalidValuesError):
            DecisionTreeRegressor().fit(
                FEATURES, Feature("y", TARGETS), np.zeros(len(TARGETS))
            )
