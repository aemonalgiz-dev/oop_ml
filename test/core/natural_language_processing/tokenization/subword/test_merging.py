"""Spec for the shared merge arithmetic -- the one loop six tokenizers reuse."""

import random

import pytest

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.subword.merges import (
    Merge,
    Merges,
)
from oop_ml.core.natural_language_processing.tokenization.subword.merging import (
    PairScoring,
    SpelledWord,
    alphabet_of,
    apply_merges,
    best_pair,
    learn_merges,
    with_pair_merged,
)
from test.core.natural_language_processing.fixtures import (
    SENNRICH_ALPHABET,
    SENNRICH_FIRST_TEN_MERGES,
    SENNRICH_WORD_COUNTS,
)


def spelled_sennrich() -> list[SpelledWord]:
    return [
        SpelledWord((*word[:-1], word[-1] + "</w>"), count)
        for word, count in SENNRICH_WORD_COUNTS.items()
    ]


class TestSpelledWord:
    def test_carries_symbols_and_count(self):
        word = SpelledWord(["l", "o", "w</w>"], 5)

        assert word.symbols == ("l", "o", "w</w>")
        assert word.count == 5
        assert list(word) == ["l", "o", "w</w>"]
        assert len(word) == 3

    def test_with_pair_merged_keeps_the_count(self):
        merged = SpelledWord(["l", "o", "w</w>"], 5).with_pair_merged(
            Merge("l", "o", 7)
        )

        assert merged == SpelledWord(["lo", "w</w>"], 5)

    @pytest.mark.parametrize("symbols", [[], ["a", ""]])
    def test_needs_non_empty_symbols(self, symbols):
        with pytest.raises(EmptyValuesError):
            SpelledWord(symbols, 1)

    def test_a_count_below_one_raises(self):
        with pytest.raises(InvalidValuesError):
            SpelledWord(["a"], 0)


class TestWithPairMerged:
    def test_joins_every_occurrence_left_to_right(self):
        assert with_pair_merged(["a", "b", "a", "b", "c"], Merge("a", "b", 1)) == (
            "ab",
            "ab",
            "c",
        )

    def test_does_not_re_scan_the_new_symbol(self):
        """``a a a`` under ``a + a`` is ``aa a``, not ``aaa``."""
        assert with_pair_merged(["a", "a", "a"], Merge("a", "a", 1)) == ("aa", "a")

    def test_leaves_a_sequence_without_the_pair_alone(self):
        assert with_pair_merged(["a", "c"], Merge("a", "b", 1)) == ("a", "c")


class TestAlphabet:
    def test_is_every_distinct_symbol_in_codepoint_order(self):
        assert list(alphabet_of(spelled_sennrich())) == SENNRICH_ALPHABET


class TestBestPair:
    def test_frequency_picks_the_most_frequent_pair(self):
        assert best_pair(spelled_sennrich(), PairScoring.FREQUENCY) == Merge(
            "e", "s", 9
        )

    def test_frequency_ties_go_to_the_lexicographically_smaller_pair(self):
        """``e s`` and ``s t</w>`` both score 9."""
        assert best_pair(spelled_sennrich(), PairScoring.FREQUENCY) == Merge(
            "e", "s", 9
        )

    def test_likelihood_picks_the_pair_its_parts_least_explain(self):
        """``i d`` scores 3 / (3 * 3); ``e s`` scores 9 / (17 * 9)."""
        assert best_pair(spelled_sennrich(), PairScoring.LIKELIHOOD) == Merge(
            "i", "d", 3
        )

    def test_the_score_carried_is_the_count_under_either_rule(self):
        frequency = best_pair(spelled_sennrich(), PairScoring.FREQUENCY)
        likelihood = best_pair(spelled_sennrich(), PairScoring.LIKELIHOOD)

        assert frequency is not None and frequency.score == pytest.approx(9.0)
        assert likelihood is not None and likelihood.score == pytest.approx(3.0)

    def test_nothing_adjacent_gives_none(self):
        assert best_pair([SpelledWord(["a</w>"], 3)], PairScoring.FREQUENCY) is None


class TestLearnMerges:
    def test_learns_sennrichs_merges_in_order(self):
        merges = learn_merges(spelled_sennrich(), n_merges=10, minimum_pair_frequency=2)

        assert [(m.left, m.right, int(m.score)) for m in merges] == (
            SENNRICH_FIRST_TEN_MERGES
        )

    def test_stops_at_the_minimum_frequency(self):
        merges = learn_merges(
            spelled_sennrich(), n_merges=100, minimum_pair_frequency=3
        )

        assert len(merges) == 10

    def test_stops_when_nothing_is_left(self):
        merges = learn_merges(
            spelled_sennrich(), n_merges=100, minimum_pair_frequency=1
        )

        assert len(merges) == 13

    def test_zero_merges_is_allowed(self):
        assert learn_merges(spelled_sennrich(), 0, 1) == Merges([])


class TestApplyMerges:
    def test_applies_the_earliest_rank_first(self):
        merges = learn_merges(spelled_sennrich(), n_merges=10, minimum_pair_frequency=2)

        assert apply_merges(["l", "o", "w", "e", "s", "t</w>"], merges) == (
            "lo",
            "w",
            "est</w>",
        )

    def test_one_occurrence_at_a_time_matches_all_at_once_without_dropout(self):
        merges = Merges([Merge("a", "b", 3)])

        assert apply_merges(["a", "b", "a", "b"], merges) == ("ab", "ab")

    def test_no_applicable_merge_leaves_the_spelling_alone(self):
        assert apply_merges(["x", "y"], Merges([Merge("a", "b", 1)])) == ("x", "y")

    def test_dropout_needs_a_generator(self):
        with pytest.raises(InvalidValuesError):
            apply_merges(["a", "b"], Merges([Merge("a", "b", 1)]), dropout=0.5)

    def test_full_dropout_skips_every_merge(self):
        """0.999 rather than 1.0, which the callers' fields refuse."""
        merges = Merges([Merge("a", "b", 1)])
        generator = random.Random(0)

        skipped = [
            apply_merges(["a", "b"], merges, 0.999, generator) for _ in range(50)
        ]

        assert skipped.count(("a", "b")) >= 49

    def test_zero_dropout_ignores_the_generator(self):
        merges = Merges([Merge("a", "b", 1)])

        assert apply_merges(["a", "b"], merges, 0.0, random.Random(0)) == ("ab",)
