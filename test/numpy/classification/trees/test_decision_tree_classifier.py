"""Spec for DecisionTreeClassifier -- red until the growth stubs land.

The exam fixture's tree was worked out by exhaustive search over every
candidate threshold under both criteria, so these are not "whatever the code
did" assertions. Root split, second split, leaf sizes, leaf shares and the
single row it gets wrong are all known in advance.

Four things here are worth reading rather than skimming.

The misclassified row is load-bearing. A fixture the tree fits perfectly cannot
distinguish a correct implementation from one that grows until every leaf is
pure, and those are very different models. Row 5 slept well, studied little,
passed anyway, and lands among three fails -- the tree must get it wrong.

The interaction test pins the reason to reach for a tree at all. ``studied`` is
tested only inside the region where ``slept`` cleared its threshold, which is a
conditional relationship no linear model expresses without being handed a
product column.

Both criteria are asserted to choose the same splits. They usually do, and
pinning it means a future disagreement on this data shows up as the bug it
would be rather than as an interesting finding.

Probabilities are checked for width as well as content. Every fitted class needs
a column even where no row in the reached leaf belongs to it, or the matrix
quietly changes shape depending on which query was asked.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    NotFittedError,
    SingleClassError,
)
from oop_ml.core.tree.criterion import ClassificationCriterion
from oop_ml.core.tree.node import ClassificationLeaf, DecisionNode
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from test.fixtures import (
    EXAM_LEAF_GINI,
    EXAM_LEAF_PREDICTIONS,
    EXAM_LEAF_SHARES,
    EXAM_LEAF_SIZES,
    EXAM_MIN_SAMPLES_SPLIT,
    EXAM_MISCLASSIFIED_ROW,
    EXAM_OUTCOMES,
    EXAM_ROOT_GINI,
    EXAM_ROOT_GINI_GAIN,
    EXAM_ROOT_SPLIT,
    EXAM_SECOND_SPLIT,
    EXAM_TREE_ACCURACY,
    EXAM_TREE_DEPTH,
    EXAM_TREE_LEAF_COUNT,
    EXAM_UNSTOPPED_ACCURACY,
    EXAM_UNSTOPPED_LEAF_COUNT,
)


def fitted(**overrides) -> DecisionTreeClassifier:
    """The canonical tree: stopped, so the impure leaf survives to be checked.

    Without ``min_samples_split`` the recursion carries on until every leaf is
    pure, which reproduces all fifteen rows and would make most of the
    assertions below vacuous. ``TestTheDefaultMemorises`` covers that case
    deliberately instead.
    """
    model = DecisionTreeClassifier(
        **{"min_samples_split": EXAM_MIN_SAMPLES_SPLIT, **overrides}
    )
    model.fit(EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature)

    return model


def query(*points) -> list[Feature]:
    """Build query features from (studied, slept) pairs."""
    return [
        Feature("studied", [point[0] for point in points]),
        Feature("slept", [point[1] for point in points]),
    ]


class TestConstruction:
    def test_defaults(self):
        model = DecisionTreeClassifier()

        assert model.criterion is ClassificationCriterion.GINI
        assert model.max_depth is None

    def test_the_criterion_is_a_real_choice(self):
        model = DecisionTreeClassifier(criterion=ClassificationCriterion.ENTROPY)

        assert model.criterion is ClassificationCriterion.ENTROPY


class TestNotFitted:
    @pytest.mark.parametrize(
        "call",
        [
            lambda model: model.predict(EXAM_OUTCOMES.input_features),
            lambda model: model.predict_probabilities(EXAM_OUTCOMES.input_features),
            lambda model: model.n_classes,
            lambda model: model.root,
            lambda model: model.describe(),
        ],
    )
    def test_it_raises_before_fit(self, call):
        with pytest.raises(NotFittedError):
            call(DecisionTreeClassifier())


class TestTheTreeItGrows:
    def test_the_root_split_is_the_one_found_by_exhaustive_search(self):
        root = fitted().root

        assert isinstance(root, DecisionNode)
        assert root.split.feature_name == EXAM_ROOT_SPLIT[0]
        assert root.split.threshold == pytest.approx(EXAM_ROOT_SPLIT[1])

    def test_the_root_impurity_and_gain_are_the_hand_computed_ones(self):
        root = fitted().root

        assert isinstance(root, DecisionNode)
        assert root.impurity == pytest.approx(EXAM_ROOT_GINI)
        assert root.split.gain == pytest.approx(EXAM_ROOT_GINI_GAIN)

    def test_the_second_split_sits_under_the_first(self):
        root = fitted().root

        assert isinstance(root, DecisionNode)
        assert isinstance(root.right, DecisionNode)
        assert root.right.split.feature_name == EXAM_SECOND_SPLIT[0]
        assert root.right.split.threshold == pytest.approx(EXAM_SECOND_SPLIT[1])

    def test_it_is_two_deep_with_three_leaves(self):
        model = fitted()

        assert model.depth == EXAM_TREE_DEPTH
        assert model.n_leaves == EXAM_TREE_LEAF_COUNT

    def test_the_first_branch_is_already_a_leaf(self):
        # Everyone who slept under 6.25 hours failed, so there is nothing left
        # to separate and the recursion stops immediately on that side.
        root = fitted().root

        assert isinstance(root, DecisionNode)
        assert isinstance(root.left, ClassificationLeaf)
        assert root.left.n_samples == EXAM_LEAF_SIZES[0]
        assert root.left.impurity == pytest.approx(0.0)


class TestTheDefaultMemorises:
    """Left unstopped, the tree carves out a box per row and learns nothing."""

    def test_every_leaf_ends_up_pure(self):
        model = DecisionTreeClassifier()
        model.fit(EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature)

        assert model.n_leaves == EXAM_UNSTOPPED_LEAF_COUNT
        assert model.n_leaves > EXAM_TREE_LEAF_COUNT

    def test_training_accuracy_reaches_one_and_means_nothing(self):
        # The same tell k=1 gave a neighbour model: a perfect training score
        # produced by construction rather than by having learned anything.
        model = DecisionTreeClassifier()
        model.fit(EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature)

        assert model.score(
            EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature
        ) == pytest.approx(EXAM_UNSTOPPED_ACCURACY)

    def test_it_reaches_the_row_the_stopped_tree_cannot(self):
        # Not a better model -- the same row, memorised rather than predicted.
        model = DecisionTreeClassifier()
        model.fit(EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature)

        predictions = model.predict(EXAM_OUTCOMES.input_features)

        assert predictions[EXAM_MISCLASSIFIED_ROW] == pytest.approx(1.0)


class TestTheInteractionItRepresents:
    def test_studying_is_only_tested_where_sleeping_cleared_the_bar(self):
        # The reason to reach for a tree: a conditional relationship, expressed
        # by nesting one question inside the region another one carved out.
        root = fitted().root

        assert isinstance(root, DecisionNode)
        assert root.split.feature_name == "slept"
        assert isinstance(root.right, DecisionNode)
        assert root.right.split.feature_name == "studied"

    def test_studying_hard_on_no_sleep_still_predicts_failure(self):
        model = fitted()

        assert model.predict(query((9.0, 4.0))) == pytest.approx([0.0])

    def test_studying_hard_after_sleeping_predicts_a_pass(self):
        model = fitted()

        assert model.predict(query((9.0, 8.0))) == pytest.approx([1.0])


class TestTheRowItCannotReach:
    def test_training_accuracy_is_not_perfect(self):
        model = fitted()

        assert model.score(
            EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature
        ) == pytest.approx(EXAM_TREE_ACCURACY)

    def test_it_is_the_row_worked_out_by_hand(self):
        model = fitted()
        predictions = model.predict(EXAM_OUTCOMES.input_features)
        actual = np.array(EXAM_OUTCOMES.class_values, dtype=float)

        wrong = np.flatnonzero(predictions != actual)

        assert wrong.tolist() == [EXAM_MISCLASSIFIED_ROW]

    def test_the_leaf_it_lands_in_is_impure_and_stays_impure(self):
        root = fitted().root

        assert isinstance(root, DecisionNode)
        assert isinstance(root.right, DecisionNode)
        assert isinstance(root.right.left, ClassificationLeaf)
        assert root.right.left.impurity == pytest.approx(EXAM_LEAF_GINI[1])
        assert root.right.left.n_samples == EXAM_LEAF_SIZES[1]


class TestLeafPredictions:
    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            ((3.0, 5.0), EXAM_LEAF_PREDICTIONS[0]),
            ((2.0, 7.0), EXAM_LEAF_PREDICTIONS[1]),
            ((6.0, 7.0), EXAM_LEAF_PREDICTIONS[2]),
        ],
    )
    def test_each_region_predicts_its_majority(self, point, expected):
        assert fitted().predict(query(point)) == pytest.approx([expected])

    def test_a_prediction_is_always_a_class_the_fit_saw(self):
        model = fitted()
        generator = np.random.default_rng(0)
        points = [
            (float(value), float(other))
            for value, other in generator.uniform(0.0, 12.0, size=(40, 2))
        ]

        predictions = model.predict(query(*points))

        assert set(predictions).issubset({0.0, 1.0})


class TestProbabilities:
    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            ((3.0, 5.0), EXAM_LEAF_SHARES[0]),
            ((2.0, 7.0), EXAM_LEAF_SHARES[1]),
            ((6.0, 7.0), EXAM_LEAF_SHARES[2]),
        ],
    )
    def test_each_leaf_reports_its_own_composition(self, point, expected):
        assert fitted().predict_probabilities(query(point))[0] == pytest.approx(
            expected
        )

    def test_every_class_gets_a_column_even_with_no_rows(self):
        # A pure leaf still reports a full-width row, or the matrix would
        # change shape depending on which query was asked.
        shares = fitted().predict_probabilities(query((3.0, 5.0), (6.0, 7.0)))

        assert shares.shape == (2, 2)

    def test_rows_sum_to_one(self):
        shares = fitted().predict_probabilities(
            query((3.0, 5.0), (2.0, 7.0), (6.0, 7.0))
        )

        assert shares.values.sum(axis=1) == pytest.approx(np.ones(3))

    def test_a_pure_leaf_claims_total_certainty(self):
        # Worth pinning because it is the honest weakness: five rows are enough
        # for this leaf to report 1.0, and an unstopped tree makes every leaf
        # pure, so an unstopped tree is certain everywhere.
        shares = fitted().predict_probabilities(query((6.0, 7.0)))

        assert shares[0] == pytest.approx([0.0, 1.0])


class TestBothCriteriaAgreeHere:
    @pytest.mark.parametrize("criterion", list(ClassificationCriterion))
    def test_each_grows_the_same_shape(self, criterion):
        model = fitted(criterion=criterion)

        assert model.depth == EXAM_TREE_DEPTH
        assert model.n_leaves == EXAM_TREE_LEAF_COUNT

    @pytest.mark.parametrize("criterion", list(ClassificationCriterion))
    def test_each_chooses_the_same_root_split(self, criterion):
        root = fitted(criterion=criterion).root

        assert isinstance(root, DecisionNode)
        assert root.split.feature_name == EXAM_ROOT_SPLIT[0]
        assert root.split.threshold == pytest.approx(EXAM_ROOT_SPLIT[1])

    def test_the_gains_differ_even_though_the_choice_does_not(self):
        # Same split, different units. Gains are not comparable across
        # criteria and nothing should ever compare them.
        by_gini = fitted(criterion=ClassificationCriterion.GINI).root
        by_entropy = fitted(criterion=ClassificationCriterion.ENTROPY).root

        assert isinstance(by_gini, DecisionNode)
        assert isinstance(by_entropy, DecisionNode)
        assert by_gini.split.gain != pytest.approx(by_entropy.split.gain)


class TestStoppingRules:
    def test_max_depth_one_keeps_only_the_root_question(self):
        model = fitted(max_depth=1)

        assert model.depth == 1
        assert model.n_leaves == 2

    def test_max_depth_cannot_be_zero(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(max_depth=0)

    def test_min_samples_leaf_can_forbid_the_second_split(self):
        # The second split leaves four rows on one side, so demanding five
        # rules it out while leaving the root split admissible.
        model = fitted(min_samples_leaf=5)

        assert model.n_leaves == 2

    def test_min_impurity_decrease_can_stop_it_dead(self):
        model = fitted(min_impurity_decrease=1.0)

        assert model.n_leaves == 1

    def test_a_single_leaf_predicts_the_global_majority(self):
        model = fitted(min_impurity_decrease=1.0)

        assert model.predict(query((0.0, 0.0), (9.0, 9.0))) == pytest.approx([0.0, 0.0])


class TestTargetValidation:
    def test_a_fractional_label_is_refused(self):
        with pytest.raises(NonBinaryLabelsError):
            DecisionTreeClassifier().fit(
                EXAM_OUTCOMES.input_features,
                Feature("passed", [0.5] * EXAM_OUTCOMES.n_samples),
            )

    def test_a_single_class_is_refused(self):
        with pytest.raises(SingleClassError):
            DecisionTreeClassifier().fit(
                EXAM_OUTCOMES.input_features,
                Feature("passed", [0.0] * EXAM_OUTCOMES.n_samples),
            )

    def test_n_classes_is_what_the_fit_saw(self):
        assert fitted().n_classes == 2


class TestInvalidInput:
    def test_no_features_is_refused(self):
        with pytest.raises(EmptyValuesError):
            DecisionTreeClassifier().fit([], EXAM_OUTCOMES.class_feature)

    def test_a_target_of_the_wrong_length_is_refused(self):
        with pytest.raises(NonEqualArrayLengthError):
            DecisionTreeClassifier().fit(
                EXAM_OUTCOMES.input_features, Feature("passed", [0.0, 1.0])
            )

    def test_unknown_features_at_predict_are_refused(self):
        with pytest.raises(InvalidValuesError):
            fitted().predict([Feature("elsewhere", [1.0])])


class TestDescription:
    def test_it_names_both_features_it_split_on(self):
        text = fitted().describe()

        assert "slept < 6.25 ?" in text
        assert "studied < 4.5 ?" in text

    def test_there_is_one_line_per_node(self):
        model = fitted()

        assert len(model.describe().splitlines()) == 5
