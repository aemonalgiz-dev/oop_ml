"""Spec for DecisionTreeRegressor -- red until the growth stubs land.

The step fixture is chosen so that the right answer is not a matter of degree.
One split at 3.5 removes the entire parent variance, both leaves are constant,
and the predictions are exactly 10 and 50. An implementation that is nearly
right on this data is wrong.

Three groups here have no counterpart elsewhere in the suite.

The stopping rules get a test each, and each one is written so that the *same*
data grows a different tree with the rule on. Testing that a limit exists is
easy and worthless; testing that it bites is the point, because every one of
them is the only thing standing between the recursion and one leaf per row.

Extrapolation is pinned deliberately. Query far past the training range and the
answer must stop changing, because every remaining neighbour is on one side and
there is no slope anywhere in the model to continue. That is not a defect being
tolerated, it is the defining behaviour of a piecewise-constant fit.

The candidate-threshold tests cover the search space rather than the search. A
constant column must offer no thresholds at all, and ``n`` distinct values must
offer exactly ``n - 1`` -- if that set is wrong, every gain computed over it is
answering the wrong question.
"""

import numpy as np
import pytest

from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.row_block import rows_of
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
)
from oop_ml.core.tree.criterion import RegressionCriterion
from oop_ml.core.tree.node import DecisionNode, LeafNode
from oop_ml.core.validation import ValueRole
from oop_ml.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from test.fixtures import (
    STEP_FUNCTION,
    STEP_LEAF_MEANS,
    STEP_ROOT_GAIN,
    STEP_SPLIT,
)


def fitted(**overrides) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(**overrides)
    model.fit(STEP_FUNCTION.input_features, STEP_FUNCTION.target_feature)

    return model


def query(*positions) -> list[Feature]:
    return [Feature("position", list(positions))]


class TestConstruction:
    def test_defaults(self):
        model = DecisionTreeRegressor()

        assert model.criterion is RegressionCriterion.SQUARED_ERROR
        assert model.max_depth is None
        assert model.min_samples_split == 2
        assert model.min_samples_leaf == 1
        assert model.min_impurity_decrease == 0.0

    @pytest.mark.parametrize("max_depth", [0, -1])
    def test_a_non_positive_depth_is_rejected(self, max_depth):
        with pytest.raises(ValueError):
            DecisionTreeRegressor(max_depth=max_depth)

    @pytest.mark.parametrize("min_samples_split", [0, 1])
    def test_splitting_fewer_than_two_rows_is_rejected(self, min_samples_split):
        # One row cannot be split into two non-empty children at all.
        with pytest.raises(ValueError):
            DecisionTreeRegressor(min_samples_split=min_samples_split)

    def test_a_negative_impurity_decrease_is_rejected(self):
        with pytest.raises(ValueError):
            DecisionTreeRegressor(min_impurity_decrease=-0.1)


class TestNotFitted:
    @pytest.mark.parametrize(
        "call",
        [
            lambda model: model.predict(STEP_FUNCTION.input_features),
            lambda model: model.root,
            lambda model: model.depth,
            lambda model: model.n_leaves,
            lambda model: model.describe(),
            lambda model: model.split_search(
                rows_of(np.array([[1.0], [2.0], [3.0]]), ("position",)),
                Column.of([1.0, 2.0, 3.0], ValueRole.TARGET_VALUES),
            ),
        ],
    )
    def test_it_raises_before_fit(self, call):
        with pytest.raises(NotFittedError):
            call(DecisionTreeRegressor())


class TestTheStepItCanFitExactly:
    def test_it_reproduces_the_target(self):
        model = fitted()

        assert model.predict(STEP_FUNCTION.input_features) == pytest.approx(
            STEP_FUNCTION.target_feature.values
        )

    def test_one_split_is_enough(self):
        model = fitted()

        assert model.depth == 1
        assert model.n_leaves == 2

    def test_the_split_is_the_one_worked_out_by_hand(self):
        model = fitted()
        root = model.root

        assert isinstance(root, DecisionNode)
        assert root.split.feature_name == STEP_SPLIT[0]
        assert root.split.threshold == pytest.approx(STEP_SPLIT[1])
        assert root.split.gain == pytest.approx(STEP_ROOT_GAIN)

    def test_the_leaves_predict_the_two_means(self):
        root = fitted().root

        assert isinstance(root, DecisionNode)
        assert isinstance(root.left, LeafNode)
        assert isinstance(root.right, LeafNode)
        assert root.left.prediction == pytest.approx(STEP_LEAF_MEANS[0])
        assert root.right.prediction == pytest.approx(STEP_LEAF_MEANS[1])

    def test_both_leaves_are_pure(self):
        root = fitted().root

        assert isinstance(root, DecisionNode)
        assert root.left.n_samples == 4
        assert root.right.n_samples == 4


class TestStoppingRules:
    def test_max_depth_caps_the_questions(self):
        wide = [
            Feature("position", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        ]
        target = Feature("quantity", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        capped = DecisionTreeRegressor(max_depth=1)
        capped.fit(wide, target)

        assert capped.depth == 1
        assert capped.n_leaves == 2

    def test_without_a_cap_a_distinct_target_is_memorised(self):
        # The failure the rules exist to prevent, stated plainly: every row in
        # its own leaf, training error zero, nothing learned.
        wide = [Feature("position", [float(index) for index in range(8)])]
        target = Feature("quantity", [float(index) for index in range(8)])

        grown = DecisionTreeRegressor()
        grown.fit(wide, target)

        assert grown.n_leaves == 8
        assert grown.predict(wide) == pytest.approx(target.values)

    def test_min_samples_split_refuses_a_small_node(self):
        model = DecisionTreeRegressor(min_samples_split=9)
        model.fit(STEP_FUNCTION.input_features, STEP_FUNCTION.target_feature)

        assert model.n_leaves == 1
        assert model.depth == 0

    def test_min_samples_leaf_rejects_a_lopsided_split(self):
        # The step's only useful split is four against four, so demanding five
        # in every leaf leaves nothing admissible at all.
        model = DecisionTreeRegressor(min_samples_leaf=5)
        model.fit(STEP_FUNCTION.input_features, STEP_FUNCTION.target_feature)

        assert model.n_leaves == 1

    def test_min_impurity_decrease_refuses_a_cheap_split(self):
        model = DecisionTreeRegressor(min_impurity_decrease=1000.0)
        model.fit(STEP_FUNCTION.input_features, STEP_FUNCTION.target_feature)

        assert model.n_leaves == 1

    def test_a_split_worth_exactly_the_threshold_is_taken(self):
        model = DecisionTreeRegressor(min_impurity_decrease=STEP_ROOT_GAIN)
        model.fit(STEP_FUNCTION.input_features, STEP_FUNCTION.target_feature)

        assert model.n_leaves == 2

    def test_a_constant_target_is_one_leaf(self):
        flat = Feature("quantity", [5.0] * 8)

        model = DecisionTreeRegressor()
        model.fit(STEP_FUNCTION.input_features, flat)

        assert model.n_leaves == 1
        assert model.predict(query(0.0, 99.0)) == pytest.approx([5.0, 5.0])

    def test_a_constant_predictor_offers_no_split(self):
        model = DecisionTreeRegressor()
        model.fit([Feature("position", [2.0] * 8)], STEP_FUNCTION.target_feature)

        assert model.n_leaves == 1


class TestThePredictionSurface:
    def test_it_is_piecewise_constant(self):
        model = fitted()

        assert model.predict(query(0.0, 1.0, 2.0, 3.4)) == pytest.approx(
            [10.0, 10.0, 10.0, 10.0]
        )

    def test_it_jumps_at_the_threshold(self):
        model = fitted()

        below, above = model.predict(query(3.49, 3.51))

        assert below == pytest.approx(10.0)
        assert above == pytest.approx(50.0)

    def test_it_cannot_extrapolate(self):
        # Past the edge every answer is the outermost leaf's mean, forever.
        model = fitted()

        assert model.predict(query(7.0, 50.0, 1_000_000.0)) == pytest.approx(
            [50.0, 50.0, 50.0]
        )


class TestCandidateThresholds:
    def test_n_distinct_values_offer_n_minus_one_thresholds(self):
        model = DecisionTreeRegressor()

        thresholds = model._candidate_thresholds(np.array([1.0, 2.0, 3.0, 4.0]))

        assert thresholds == pytest.approx([1.5, 2.5, 3.5])

    def test_repeats_do_not_add_thresholds(self):
        model = DecisionTreeRegressor()

        thresholds = model._candidate_thresholds(np.array([1.0, 1.0, 1.0, 4.0]))

        assert thresholds == pytest.approx([2.5])

    def test_a_constant_column_offers_none(self):
        model = DecisionTreeRegressor()

        assert model._candidate_thresholds(np.full(6, 3.0)).size == 0

    def test_a_single_row_offers_none(self):
        model = DecisionTreeRegressor()

        assert model._candidate_thresholds(np.array([3.0])).size == 0

    def test_they_are_midpoints_and_therefore_between_the_rows(self):
        # A threshold sitting on top of a training value would place the
        # boundary at an observation rather than between two of them.
        model = DecisionTreeRegressor()
        column = np.array([0.0, 10.0])

        assert model._candidate_thresholds(column) == pytest.approx([5.0])


class TestFeatureMatching:
    def test_column_order_does_not_matter(self):
        first = Feature("first", [0.0, 1.0, 2.0, 3.0])
        second = Feature("second", [3.0, 2.0, 1.0, 0.0])
        target = Feature("quantity", [1.0, 1.0, 9.0, 9.0])

        model = DecisionTreeRegressor()
        model.fit([first, second], target)

        assert model.predict([second, first]) == pytest.approx(
            model.predict([first, second])
        )

    def test_a_missing_feature_is_refused(self):
        model = fitted()

        with pytest.raises(InvalidValuesError):
            model.predict([Feature("elsewhere", [1.0])])

    def test_an_extra_feature_is_refused(self):
        model = fitted()

        with pytest.raises(InvalidValuesError):
            model.predict([Feature("position", [1.0]), Feature("spare", [1.0])])


class TestInvalidInput:
    def test_no_features_is_refused(self):
        with pytest.raises(EmptyValuesError):
            DecisionTreeRegressor().fit([], STEP_FUNCTION.target_feature)

    def test_duplicate_names_are_refused(self):
        column = Feature("position", [1.0, 2.0])

        with pytest.raises(NonUniqueFeaturesError):
            DecisionTreeRegressor().fit(
                [column, column], Feature("quantity", [1.0, 2.0])
            )

    def test_a_target_of_the_wrong_length_is_refused(self):
        with pytest.raises(NonEqualArrayLengthError):
            DecisionTreeRegressor().fit(
                STEP_FUNCTION.input_features, Feature("quantity", [1.0, 2.0])
            )


class TestScoring:
    def test_it_scores_the_step_perfectly(self):
        model = fitted()

        assert model.score(
            STEP_FUNCTION.input_features, STEP_FUNCTION.target_feature
        ) == pytest.approx(1.0)


class TestDescription:
    def test_it_renders_the_question_and_both_leaves(self):
        text = fitted().describe()

        assert "position < 3.5 ?" in text
        assert text.count("predict") == 2
