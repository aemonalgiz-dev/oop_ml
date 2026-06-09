"""Spec for HuffmanTree and HuffmanCode -- the tree hierarchical softmax walks.

Pinned against the counts the module docstring works by hand, ``5, 2, 1, 1``.
The two ones merge first into internal node 0, that node and the two merge into
node 1, and node 1 and the five merge into node 2, the root. So the commonest
word is one choice from the root, the two rarest are three, the expected depth
is ``15 / 9`` against a balanced tree's 2, and ``sum 2 ** -depth`` is exactly 1.
"""

from __future__ import annotations

import pytest

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.embeddings.prediction.huffman import (
    HuffmanCode,
    HuffmanTree,
)

PAPER_COUNTS = [5, 2, 1, 1]


class TestFromCounts:
    def test_the_worked_counts_give_depths_one_two_three_three(self) -> None:
        tree = HuffmanTree.from_counts(PAPER_COUNTS)

        assert [code.depth for code in tree] == [1, 2, 3, 3]

    def test_the_expected_depth_is_fifteen_ninths(self) -> None:
        tree = HuffmanTree.from_counts(PAPER_COUNTS)

        assert tree.expected_depth(PAPER_COUNTS) == pytest.approx(15 / 9)

    def test_the_expected_depth_beats_a_balanced_tree_over_four_words(self) -> None:
        """A balanced tree over four leaves costs exactly two choices per word."""
        tree = HuffmanTree.from_counts(PAPER_COUNTS)

        assert tree.expected_depth(PAPER_COUNTS) < 2.0

    def test_the_codes_are_the_hand_worked_paths(self) -> None:
        tree = HuffmanTree.from_counts(PAPER_COUNTS)

        assert tree.code_of(0) == HuffmanCode(0, [2], [1])
        assert tree.code_of(1) == HuffmanCode(1, [2, 1], [0, 0])
        assert tree.code_of(2) == HuffmanCode(2, [2, 1, 0], [0, 1, 0])
        assert tree.code_of(3) == HuffmanCode(3, [2, 1, 0], [0, 1, 1])

    @pytest.mark.parametrize(
        "counts",
        [
            PAPER_COUNTS,
            [1, 1, 1, 1],
            [10, 3, 3, 2, 1, 1],
            [7, 1],
            [100, 1, 1, 1, 1, 1, 1, 1],
            [2, 3, 5, 8, 13, 21],
        ],
    )
    def test_kraft_equality_holds_for_a_full_binary_tree(
        self, counts: list[int]
    ) -> None:
        tree = HuffmanTree.from_counts(counts)

        assert sum(2.0**-code.depth for code in tree) == pytest.approx(1.0)

    @pytest.mark.parametrize("counts", [PAPER_COUNTS, [1, 1, 1, 1], [7, 1], [4]])
    def test_there_is_one_internal_node_fewer_than_words(
        self, counts: list[int]
    ) -> None:
        tree = HuffmanTree.from_counts(counts)

        assert tree.n_internal_nodes == len(counts) - 1
        assert tree.n_words == len(counts)

    @pytest.mark.parametrize("counts", [PAPER_COUNTS, [10, 3, 3, 2, 1, 1]])
    def test_no_code_is_a_prefix_of_another(self, counts: list[int]) -> None:
        """What makes the leaves distinguishable by their bits alone."""
        bit_strings = [code.bits for code in HuffmanTree.from_counts(counts)]

        for first in bit_strings:
            for second in bit_strings:
                if first is not second:
                    assert second[: len(first)] != first

    @pytest.mark.parametrize("counts", [PAPER_COUNTS, [10, 3, 3, 2, 1, 1], [1, 9, 4]])
    def test_a_commoner_word_is_never_deeper_than_a_rarer_one(
        self, counts: list[int]
    ) -> None:
        tree = HuffmanTree.from_counts(counts)

        for first_id, first_count in enumerate(counts):
            for second_id, second_count in enumerate(counts):
                if first_count > second_count:
                    assert tree.code_of(first_id).depth <= tree.code_of(second_id).depth

    def test_ties_break_by_creation_order(self) -> None:
        """``1, 1, 2``: the ones merge into node 0, count 2. At the tie between
        that node and word 2, the leaf was created first, so it goes first and
        takes bit 0 at the root."""
        tree = HuffmanTree.from_counts([1, 1, 2])

        assert tree.code_of(2) == HuffmanCode(2, [1], [0])
        assert tree.code_of(0) == HuffmanCode(0, [1, 0], [1, 0])
        assert tree.code_of(1) == HuffmanCode(1, [1, 0], [1, 1])

    def test_two_equal_leaves_take_bits_in_id_order(self) -> None:
        tree = HuffmanTree.from_counts([3, 3])

        assert tree.code_of(0).bits == (0,)
        assert tree.code_of(1).bits == (1,)

    def test_the_tree_is_a_function_of_the_counts_alone(self) -> None:
        assert HuffmanTree.from_counts(PAPER_COUNTS) == HuffmanTree.from_counts(
            PAPER_COUNTS
        )
        assert hash(HuffmanTree.from_counts(PAPER_COUNTS)) == hash(
            HuffmanTree.from_counts(PAPER_COUNTS)
        )

    def test_every_path_starts_at_the_root(self) -> None:
        """The root is the last internal node created, so its number is ``n - 2``."""
        tree = HuffmanTree.from_counts([10, 3, 3, 2, 1, 1])

        assert {code.nodes[0] for code in tree} == {tree.n_internal_nodes - 1}

    def test_a_single_word_has_an_empty_code_and_no_internal_nodes(self) -> None:
        tree = HuffmanTree.from_counts([7])

        assert tree.n_internal_nodes == 0
        assert tree.code_of(0) == HuffmanCode(0, [], [])
        assert tree.code_of(0).depth == 0
        assert tree.expected_depth([7]) == 0.0


class TestIteration:
    def test_iterates_the_codes_in_id_order(self) -> None:
        tree = HuffmanTree.from_counts(PAPER_COUNTS)

        assert [code.word_id for code in tree] == [0, 1, 2, 3]
        assert len(tree) == 4

    def test_repr_names_the_sizes(self) -> None:
        assert repr(HuffmanTree.from_counts(PAPER_COUNTS)) == (
            "HuffmanTree(n_words=4, n_internal_nodes=3)"
        )
        assert repr(HuffmanCode(2, [2, 1, 0], [0, 1, 0])) == (
            "HuffmanCode(word_id=2, bits=010)"
        )

    def test_codes_compare_by_value(self) -> None:
        assert HuffmanCode(1, [2, 1], [0, 0]) == HuffmanCode(1, (2, 1), (0, 0))
        assert HuffmanCode(1, [2, 1], [0, 0]) != HuffmanCode(1, [2, 1], [0, 1])
        assert HuffmanCode(1, [2, 1], [0, 0]) != "10"


class TestRefusals:
    def test_no_counts_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            HuffmanTree.from_counts([])

    @pytest.mark.parametrize("counts", [[0, 1], [3, -1], [0]])
    def test_a_count_below_one_is_refused(self, counts: list[int]) -> None:
        with pytest.raises(InvalidValuesError):
            HuffmanTree.from_counts(counts)

    def test_a_code_with_more_bits_than_nodes_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HuffmanCode(0, [1], [0, 1])

    def test_a_bit_other_than_zero_or_one_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HuffmanCode(0, [1], [2])

    def test_a_tree_with_no_codes_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            HuffmanTree([], 0)

    def test_a_code_out_of_position_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HuffmanTree([HuffmanCode(1, [], [])], 0)

    def test_a_node_outside_the_internal_nodes_is_refused(self) -> None:
        with pytest.raises(InvalidValuesError):
            HuffmanTree([HuffmanCode(0, [3], [0]), HuffmanCode(1, [0], [1])], 1)

    @pytest.mark.parametrize("word_id", [4, -1])
    def test_an_id_outside_the_tree_is_refused(self, word_id: int) -> None:
        with pytest.raises(InvalidValuesError):
            HuffmanTree.from_counts(PAPER_COUNTS).code_of(word_id)

    def test_expected_depth_needs_one_count_per_word(self) -> None:
        with pytest.raises(InvalidValuesError):
            HuffmanTree.from_counts(PAPER_COUNTS).expected_depth([5, 2, 1])
