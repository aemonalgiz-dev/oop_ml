"""Spec for WordVector / WordEmbeddings / SimilarWords -- a table you can ask."""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    UndefinedMetricError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    SimilarWord,
    SimilarWords,
    WordEmbeddings,
    WordVector,
    cosine_similarity,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary

# Four words in two dimensions, chosen so every cosine can be read off a
# sketch: king and queen point the same way, man is at right angles to them,
# and woman is at right angles to man on the far side.
WORDS = ["king", "queen", "man", "woman", "zero"]
TABLE = np.array(
    [
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [0.0, 3.0],
        [0.0, 0.0],
    ]
)


def make_embeddings() -> WordEmbeddings:
    return WordEmbeddings(Vocabulary(WORDS), TABLE)


class TestCosineSimilarity:
    def test_parallel_vectors_score_one(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([3.0, 0.0])) == 1.0

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == 0.0

    def test_opposite_vectors_score_minus_one(self):
        assert cosine_similarity(
            np.array([1.0, 1.0]), np.array([-2.0, -2.0])
        ) == pytest.approx(-1.0)

    def test_the_forty_five_degree_case_by_hand(self):
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 1.0])) == (
            pytest.approx(1 / np.sqrt(2))
        )

    def test_length_is_ignored(self):
        first = cosine_similarity(np.array([1.0, 2.0]), np.array([3.0, 1.0]))
        second = cosine_similarity(np.array([10.0, 20.0]), np.array([0.3, 0.1]))

        assert first == pytest.approx(second)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ShapeMismatchError):
            cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_a_zero_vector_has_no_direction(self):
        with pytest.raises(UndefinedMetricError):
            cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


class TestWordVector:
    def test_carries_word_and_frozen_values(self):
        vector = WordVector("king", np.array([1.0, 0.0]))

        assert vector.word == "king"
        assert vector.dimension == 2
        assert np.array_equal(vector.values, [1.0, 0.0])
        with pytest.raises(ValueError):
            vector.values[0] = 5.0

    def test_copies_what_it_is_given(self):
        source = np.array([1.0, 0.0])
        vector = WordVector("king", source)
        source[0] = 9.0

        assert vector.values[0] == 1.0

    def test_np_array_is_a_real_copy(self):
        vector = WordVector("king", np.array([1.0, 0.0]))
        converted = np.array(vector)
        converted[0] = 7.0

        assert vector.values[0] == 1.0

    def test_cosine_to_another(self):
        king = WordVector("king", np.array([1.0, 0.0]))
        man = WordVector("man", np.array([0.0, 1.0]))

        assert king.cosine_similarity_to(man) == 0.0

    def test_empty_word_raises(self):
        with pytest.raises(EmptyValuesError):
            WordVector("", np.array([1.0]))

    @pytest.mark.parametrize("values", [[[1.0, 2.0]], [1.0, float("nan")], "abc"])
    def test_a_non_finite_or_non_vector_raises(self, values):
        with pytest.raises(InvalidValuesError):
            WordVector("king", np.asarray(values) if values != "abc" else values)  # type: ignore[arg-type]

    def test_equal_when_word_and_values_match(self):
        assert WordVector("a", np.array([1.0])) == WordVector("a", np.array([1.0]))
        assert WordVector("a", np.array([1.0])) != WordVector("b", np.array([1.0]))
        assert WordVector("a", np.array([1.0])) != "a"


class TestSimilarWords:
    def test_reads_words_and_similarities_off(self):
        similar = SimilarWords([SimilarWord("queen", 1.0), SimilarWord("man", 0.0)])

        assert similar.words == ("queen", "man")
        assert similar.similarities == (1.0, 0.0)
        assert len(similar) == 2
        assert similar[0] == SimilarWord("queen", 1.0)

    def test_must_be_in_descending_order(self):
        with pytest.raises(InvalidValuesError):
            SimilarWords([SimilarWord("man", 0.0), SimilarWord("queen", 1.0)])

    def test_a_repeated_word_raises(self):
        with pytest.raises(InvalidValuesError):
            SimilarWords([SimilarWord("man", 0.5), SimilarWord("man", 0.5)])

    def test_may_be_empty(self):
        assert SimilarWords([]).words == ()

    def test_a_similarity_outside_the_cosine_range_raises(self):
        with pytest.raises(InvalidValuesError):
            SimilarWord("man", 1.5)

    def test_compares_unequal_to_a_bare_tuple(self):
        assert SimilarWords([SimilarWord("a", 1.0)]) != ("a",)


class TestWordEmbeddingsConstruction:
    def test_pairs_the_vocabulary_with_the_table(self):
        embeddings = make_embeddings()

        assert embeddings.n_words == 5
        assert embeddings.dimension == 2
        assert len(embeddings) == 5
        assert embeddings.vocabulary == Vocabulary(WORDS)

    def test_the_table_is_frozen_and_copied(self):
        source = TABLE.copy()
        embeddings = WordEmbeddings(Vocabulary(WORDS), source)
        source[0, 0] = 9.0

        assert embeddings.table[0, 0] == 1.0
        with pytest.raises(ValueError):
            embeddings.table[0, 0] = 5.0

    def test_np_array_is_a_real_copy(self):
        embeddings = make_embeddings()
        converted = np.array(embeddings)
        converted[0, 0] = 7.0

        assert embeddings.table[0, 0] == 1.0

    def test_a_row_count_that_disagrees_with_the_vocabulary_raises(self):
        with pytest.raises(ShapeMismatchError):
            WordEmbeddings(Vocabulary(WORDS[:3]), TABLE)

    @pytest.mark.parametrize(
        "table",
        [np.array([1.0, 2.0]), np.zeros((5, 0)), np.full((5, 2), np.nan)],
    )
    def test_a_table_that_is_not_a_finite_matrix_raises(self, table):
        with pytest.raises(InvalidValuesError):
            WordEmbeddings(Vocabulary(WORDS), table)

    def test_iterates_word_vectors_in_id_order(self):
        assert [vector.word for vector in make_embeddings()] == WORDS

    def test_membership_is_by_word(self):
        assert "king" in make_embeddings()
        assert "kettle" not in make_embeddings()

    def test_equal_when_vocabulary_and_table_match(self):
        assert make_embeddings() == make_embeddings()
        assert make_embeddings() != WordEmbeddings(Vocabulary(WORDS), TABLE * 2)
        assert make_embeddings() != "a table"


class TestLookups:
    def test_vector_of_a_word(self):
        assert make_embeddings().vector_of("queen") == WordVector(
            "queen", np.array([2.0, 0.0])
        )

    def test_an_unknown_word_raises(self):
        with pytest.raises(UnknownTokenError):
            make_embeddings().vector_of("kettle")

    def test_similarity_between_two_words(self):
        embeddings = make_embeddings()

        assert embeddings.similarity("king", "queen") == 1.0
        assert embeddings.similarity("king", "man") == 0.0

    def test_similarity_to_a_zero_vector_is_undefined(self):
        with pytest.raises(UndefinedMetricError):
            make_embeddings().similarity("king", "zero")


class TestMostSimilar:
    def test_excludes_the_word_itself_and_orders_by_cosine(self):
        similar = make_embeddings().most_similar("king", n_results=3)

        assert similar.words == ("queen", "man", "woman")
        assert similar.similarities[0] == 1.0

    def test_skips_words_with_a_zero_vector(self):
        assert "zero" not in make_embeddings().most_similar("king", n_results=10).words

    def test_asks_for_at_most_what_exists(self):
        assert len(make_embeddings().most_similar("king", n_results=10)) == 3

    def test_n_results_below_one_raises(self):
        with pytest.raises(InvalidValuesError):
            make_embeddings().most_similar("king", n_results=0)

    def test_a_zero_query_word_is_undefined(self):
        with pytest.raises(UndefinedMetricError):
            make_embeddings().most_similar("zero")


class TestAnalogy:
    def test_king_minus_man_plus_woman(self):
        """Unit vectors: king (1, 0) - man (0, 1) + woman (0, 1) = (1, 0),
        which points at queen once king itself is excluded."""
        answer = make_embeddings().analogy(["king", "woman"], ["man"], n_results=1)

        assert answer.words == ("queen",)

    def test_the_question_words_are_excluded(self):
        answer = make_embeddings().analogy(["king"], n_results=10)

        assert "king" not in answer.words
        assert answer.words[0] == "queen"

    def test_needs_a_positive_word(self):
        with pytest.raises(EmptyValuesError):
            make_embeddings().analogy([])

    def test_a_combination_that_cancels_is_undefined(self):
        with pytest.raises(UndefinedMetricError):
            make_embeddings().analogy(["king"], ["queen"])


class TestSimilarToVector:
    def test_a_query_of_the_wrong_width_raises(self):
        with pytest.raises(ShapeMismatchError):
            make_embeddings().similar_to_vector(np.array([1.0, 0.0, 0.0]))

    def test_the_zero_query_is_undefined(self):
        with pytest.raises(UndefinedMetricError):
            make_embeddings().similar_to_vector(np.array([0.0, 0.0]))

    def test_excluding_removes_named_words(self):
        similar = make_embeddings().similar_to_vector(
            np.array([1.0, 0.0]), n_results=5, excluding=["king", "queen"]
        )

        assert similar.words == ("man", "woman")
