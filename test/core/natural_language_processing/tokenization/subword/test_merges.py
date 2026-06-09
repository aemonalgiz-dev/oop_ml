"""Spec for Merge / Merges -- one learned join, and the ranked list of them."""

import pytest

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.subword.merges import (
    Merge,
    Merges,
)


def make_merges() -> Merges:
    return Merges([Merge("e", "s", 9), Merge("es", "t</w>", 9), Merge("l", "o", 7)])


class TestMerge:
    def test_carries_both_symbols_and_the_score(self):
        merge = Merge("es", "t</w>", 9)

        assert merge.left == "es"
        assert merge.right == "t</w>"
        assert merge.score == pytest.approx(9.0)

    def test_merged_is_the_concatenation(self):
        assert Merge("es", "t</w>", 9).merged == "est</w>"

    def test_the_score_is_a_python_float(self):
        assert isinstance(Merge("a", "b", 3).score, float)

    @pytest.mark.parametrize(("left", "right"), [("", "b"), ("a", "")])
    def test_an_empty_symbol_raises(self, left, right):
        with pytest.raises(EmptyValuesError):
            Merge(left, right, 1)

    def test_equal_when_symbols_and_score_match(self):
        assert Merge("a", "b", 3) == Merge("a", "b", 3)
        assert hash(Merge("a", "b", 3)) == hash(Merge("a", "b", 3))
        assert Merge("a", "b", 3) != Merge("a", "b", 4)
        assert Merge("a", "b", 3) != ("a", "b")


class TestMerges:
    def test_rank_is_learned_order(self):
        merges = make_merges()

        assert merges.rank_of("e", "s") == 0
        assert merges.rank_of("l", "o") == 2

    def test_a_pair_never_merged_has_no_rank(self):
        assert make_merges().rank_of("o", "w") is None

    def test_iterates_merge_objects_in_order(self):
        assert [merge.merged for merge in make_merges()] == ["es", "est</w>", "lo"]

    def test_counts_and_indexes(self):
        merges = make_merges()

        assert len(merges) == 3
        assert merges.n_merges == 3
        assert merges[1] == Merge("es", "t</w>", 9)

    def test_may_be_empty(self):
        assert len(Merges([])) == 0

    def test_a_pair_merged_twice_raises(self):
        with pytest.raises(InvalidValuesError):
            Merges([Merge("a", "b", 2), Merge("a", "b", 1)])

    def test_equal_when_the_sequences_match(self):
        assert make_merges() == make_merges()
        assert hash(make_merges()) == hash(make_merges())

    def test_unequal_when_the_order_differs(self):
        assert Merges([Merge("a", "b", 1), Merge("c", "d", 1)]) != Merges(
            [Merge("c", "d", 1), Merge("a", "b", 1)]
        )

    def test_compares_unequal_to_a_bare_list(self):
        assert make_merges() != []
