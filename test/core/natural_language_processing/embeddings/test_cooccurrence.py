"""Spec for CooccurrenceMatrix -- which words appear near which, counted once."""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    InvalidValuesError,
    ShapeMismatchError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary

# the cat sat on the mat, in id order of first appearance.
WORDS = ["the", "cat", "sat", "on", "mat"]
SENTENCE = (0, 1, 2, 3, 0, 4)


def make_matrix(weighting: ContextWeighting, window: int = 2) -> CooccurrenceMatrix:
    return CooccurrenceMatrix.from_id_sequences(
        Vocabulary(WORDS), [SENTENCE], window=window, weighting=weighting
    )


class TestTheWorkedExample:
    def test_harmonic_row_for_the(self):
        """From the docstring: cat 1, sat 1 (1/2 + 1/2), on 1, mat 1."""
        matrix = make_matrix(ContextWeighting.HARMONIC)

        assert matrix.count_between("the", "cat") == 1.0
        assert matrix.count_between("the", "sat") == 1.0
        assert matrix.count_between("the", "on") == 1.0
        assert matrix.count_between("the", "mat") == 1.0
        assert matrix.count_between("the", "the") == 0.0

    def test_uniform_row_for_the(self):
        """sat is within two of both occurrences of the, so it counts twice."""
        matrix = make_matrix(ContextWeighting.UNIFORM)

        assert matrix.count_between("the", "sat") == 2.0
        assert matrix.count_between("the", "cat") == 1.0

    def test_is_symmetric_by_construction(self):
        counts = make_matrix(ContextWeighting.HARMONIC).counts

        assert np.array_equal(counts, counts.T)

    def test_the_total_counts_every_ordered_pair_from_both_ends(self):
        """Uniform, window 2, six positions: pairs at distance 1 are 5, at
        distance 2 are 4, each counted twice: 2 * 9 = 18."""
        assert make_matrix(ContextWeighting.UNIFORM).total == 18.0

    def test_a_sentence_boundary_ends_the_window(self):
        matrix = CooccurrenceMatrix.from_id_sequences(
            Vocabulary(WORDS), [(0, 1), (2, 3)], window=5
        )

        assert matrix.count_between("cat", "sat") == 0.0
        assert matrix.count_between("the", "cat") == 1.0

    def test_a_window_of_one_counts_only_neighbours(self):
        matrix = make_matrix(ContextWeighting.UNIFORM, window=1)

        assert matrix.count_between("the", "sat") == 0.0
        assert matrix.count_between("the", "cat") == 1.0


class TestMarginals:
    def test_word_totals_are_row_sums(self):
        matrix = make_matrix(ContextWeighting.UNIFORM)

        assert np.array_equal(matrix.word_totals, matrix.counts.sum(axis=1))
        assert np.array_equal(matrix.context_totals, matrix.counts.sum(axis=0))

    def test_marginals_are_frozen(self):
        matrix = make_matrix(ContextWeighting.UNIFORM)

        with pytest.raises(ValueError):
            matrix.word_totals[0] = 1.0


class TestConstruction:
    def test_carries_window_and_weighting(self):
        matrix = make_matrix(ContextWeighting.HARMONIC, window=3)

        assert matrix.window == 3
        assert matrix.weighting is ContextWeighting.HARMONIC
        assert matrix.n_words == 5

    def test_the_counts_are_frozen(self):
        with pytest.raises(ValueError):
            make_matrix(ContextWeighting.UNIFORM).counts[0, 0] = 1.0

    def test_np_array_is_a_real_copy(self):
        matrix = make_matrix(ContextWeighting.UNIFORM)
        converted = np.array(matrix)
        converted[0, 1] = 99.0

        assert matrix.count_between("the", "cat") == 1.0

    def test_a_window_below_one_is_refused(self):
        with pytest.raises(InvalidValuesError):
            make_matrix(ContextWeighting.UNIFORM, window=0)

    def test_an_id_outside_the_vocabulary_is_refused(self):
        with pytest.raises(InvalidValuesError):
            CooccurrenceMatrix.from_id_sequences(Vocabulary(WORDS), [(0, 9)], window=1)

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(InvalidValuesError):
            CooccurrenceMatrix(
                Vocabulary(WORDS), np.zeros((5, 4)), 1, ContextWeighting.UNIFORM
            )

    def test_a_matrix_of_the_wrong_size_is_refused(self):
        with pytest.raises(ShapeMismatchError):
            CooccurrenceMatrix(
                Vocabulary(WORDS), np.zeros((4, 4)), 1, ContextWeighting.UNIFORM
            )

    def test_a_negative_count_is_refused(self):
        with pytest.raises(InvalidValuesError):
            CooccurrenceMatrix(
                Vocabulary(WORDS), -np.ones((5, 5)), 1, ContextWeighting.UNIFORM
            )

    def test_an_unknown_word_raises_on_lookup(self):
        with pytest.raises(UnknownTokenError):
            make_matrix(ContextWeighting.UNIFORM).count_between("the", "kettle")

    def test_equal_when_words_counts_and_rule_match(self):
        assert make_matrix(ContextWeighting.UNIFORM) == make_matrix(
            ContextWeighting.UNIFORM
        )
        assert make_matrix(ContextWeighting.UNIFORM) != make_matrix(
            ContextWeighting.HARMONIC
        )
        assert make_matrix(ContextWeighting.UNIFORM) != "a matrix"
