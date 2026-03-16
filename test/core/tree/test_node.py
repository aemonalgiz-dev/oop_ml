"""Spec for the nodes a fitted tree is made of.

The tests that matter here are about routing, and they are cheap to write and
easy to skip. Routing is where a plausible misreading lives -- that prediction
compares the leaves and picks the best one -- and the shape of the code is what
rules it out: a row's path is forced by its own values, so a leaf is an address
rather than a choice.

Hence the two leaves carrying the same prediction. If anything anywhere were
ranking leaves, that arrangement would need a tie-break rule; because nothing
is, the row simply arrives where its answers send it.

The boundary test is the other one worth keeping. A threshold equal to some
row's value must send that row exactly one way, and ``<`` rather than ``<=`` is
the convention the whole library has to agree on, because a candidate threshold
is a midpoint that could later coincide with an unseen query's value.
"""

import numpy as np
import pytest

from oop_ml.core.tree.node import (
    ClassificationLeaf,
    DecisionNode,
    LeafNode,
    TreeNode,
)
from oop_ml.core.tree.split import Split


def leaf(prediction: float, n_samples: int = 3, impurity: float = 0.0) -> LeafNode:
    return LeafNode(prediction, n_samples, impurity)


def stump(threshold: float = 5.0, feature_index: int = 0) -> DecisionNode:
    """One question, two leaves, predicting 10 left and 20 right."""
    return DecisionNode(
        Split(feature_index, "first", threshold, 0.25),
        leaf(10.0),
        leaf(20.0),
        n_samples=6,
        impurity=0.5,
    )


class TestLeaf:
    def test_a_leaf_routes_to_itself(self):
        only = leaf(4.0)

        assert only.leaf_for(np.array([1.0, 2.0])) is only

    def test_a_leaf_is_depth_zero_and_one_leaf(self):
        only = leaf(4.0)

        assert only.depth == 0
        assert only.n_leaves == 1

    def test_it_remembers_what_it_was_built_from(self):
        only = LeafNode(2.0, n_samples=7, impurity=0.375)

        assert only.prediction == 2.0
        assert only.n_samples == 7
        assert only.impurity == pytest.approx(0.375)

    def test_a_leaf_need_not_be_pure(self):
        # A leaf is a node that stopped, not a node that succeeded.
        impure = LeafNode(0.0, n_samples=4, impurity=0.375)

        assert impure.impurity > 0.0


class TestRouting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, 10.0), (4.9, 10.0), (5.0, 20.0), (5.1, 20.0), (99.0, 20.0)],
    )
    def test_below_the_threshold_goes_left(self, value, expected):
        node = stump(threshold=5.0)

        assert node.leaf_for(np.array([value, 0.0])).prediction == expected

    def test_the_threshold_itself_goes_right(self):
        # Strictly less than, so a row sitting exactly on the boundary is sent
        # one way rather than both. The convention has to be stated somewhere,
        # and every candidate threshold is a midpoint that an unseen query can
        # land on exactly.
        node = stump(threshold=5.0)

        assert node.leaf_for(np.array([5.0, 0.0])).prediction == 20.0

    def test_it_reads_only_the_feature_its_split_names(self):
        node = stump(threshold=5.0, feature_index=1)

        assert node.leaf_for(np.array([100.0, 1.0])).prediction == 10.0
        assert node.leaf_for(np.array([0.0, 9.0])).prediction == 20.0

    def test_every_row_reaches_exactly_one_leaf(self):
        node = stump(threshold=5.0)
        reached = {
            id(node.leaf_for(np.array([value, 0.0])))
            for value in np.linspace(0.0, 10.0, 40)
        }

        assert reached == {id(node.left), id(node.right)}

    def test_two_leaves_may_carry_the_same_prediction(self):
        # Nothing ranks leaves, so agreeing costs nothing and needs no rule.
        node = DecisionNode(Split(0, "first", 5.0, 0.1), leaf(7.0), leaf(7.0), 6, 0.5)

        assert node.leaf_for(np.array([1.0])).prediction == 7.0
        assert node.leaf_for(np.array([9.0])).prediction == 7.0
        assert node.leaf_for(np.array([1.0])) is not node.leaf_for(np.array([9.0]))


class TestShape:
    def test_a_stump_is_depth_one_with_two_leaves(self):
        node = stump()

        assert node.depth == 1
        assert node.n_leaves == 2

    def test_depth_is_the_longest_path_not_the_shortest(self):
        deep = DecisionNode(Split(0, "first", 2.0, 0.1), leaf(1.0), stump(), 9, 0.5)

        assert deep.depth == 2
        assert deep.n_leaves == 3

    def test_a_lopsided_tree_counts_every_leaf(self):
        deeper = DecisionNode(
            Split(0, "first", 1.0, 0.1),
            leaf(1.0),
            DecisionNode(Split(0, "first", 2.0, 0.1), leaf(2.0), stump(), 9, 0.4),
            12,
            0.5,
        )

        assert deeper.depth == 3
        assert deeper.n_leaves == 4


class TestDescription:
    def test_a_leaf_describes_its_prediction(self):
        lines = leaf(3.5).description_lines()

        assert len(lines) == 1
        assert "predict 3.5" in lines[0]

    def test_a_stump_describes_the_question_then_both_answers(self):
        lines = stump(threshold=6.25).description_lines()

        assert len(lines) == 3
        assert "first < 6.25 ?" in lines[0]
        assert "predict 10" in lines[1]
        assert "predict 20" in lines[2]

    def test_children_are_indented_under_their_question(self):
        lines = stump().description_lines()

        assert not lines[0].startswith(" ")
        assert lines[1].startswith("  ")

    def test_one_line_per_node(self):
        deeper = DecisionNode(Split(0, "first", 1.0, 0.1), leaf(1.0), stump(), 9, 0.5)

        assert len(deeper.description_lines()) == 5


class TestClassificationLeaf:
    def test_it_carries_shares_alongside_the_prediction(self):
        shares = np.array([0.75, 0.25])
        node = ClassificationLeaf(0.0, shares, n_samples=4, impurity=0.375)

        assert node.prediction == 0.0
        assert node.class_shares == pytest.approx(shares)
        assert node.n_samples == 4

    def test_it_is_still_a_leaf(self):
        node = ClassificationLeaf(1.0, np.array([0.0, 1.0]), 5, 0.0)

        assert isinstance(node, LeafNode)
        assert node.leaf_for(np.array([1.0])) is node
        assert node.n_leaves == 1


class TestTheHierarchy:
    @pytest.mark.parametrize("node", [leaf(1.0), stump()])
    def test_both_kinds_are_tree_nodes(self, node):
        assert isinstance(node, TreeNode)

    def test_the_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            TreeNode()  # pyright: ignore[reportAbstractUsage]
