"""Spec for TermDocumentMatrix -- the counts every count-based embedding starts from.

Pinned against the module docstring's three documents, whose inverse document
frequencies are worked by hand there.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    InvalidValuesError,
    ShapeMismatchError,
    UnknownTokenError,
)
from oop_ml.core.natural_language_processing.embeddings.counts.term_document import (
    TermDocumentMatrix,
    TermWeighting,
    inverse_document_frequencies,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import TokenisedCorpus
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.word_level.whitespace import (
    WhitespacePreTokenizer,
)
from test.core.natural_language_processing.embeddings.counts.fixtures import (
    THREE_DOCUMENTS,
)

# log(4 / 3) + 1 and log(4 / 2) + 1, the docstring's two figures.
IDF_IN_TWO_OF_THREE = 1.2876820724517809
IDF_IN_ONE_OF_THREE = 1.6931471805599454


def tokenised(corpus: list[str]) -> TokenisedCorpus:
    return TokenisedCorpus.from_texts(corpus, WhitespacePreTokenizer())


def three_documents(
    weighting: TermWeighting = TermWeighting.COUNT, minimum_count: int = 1
) -> TermDocumentMatrix:
    corpus = tokenised(THREE_DOCUMENTS)
    return TermDocumentMatrix.from_tokenised(
        corpus, corpus.vocabulary(minimum_count), weighting
    )


class TestTheDocstringExample:
    def test_the_vocabulary_is_commonest_first_then_alphabetical(self):
        assert list(three_documents().vocabulary) == ["cat", "sat", "the", "a", "dog"]

    def test_counts_are_one_per_use(self):
        expected = np.array(
            [
                [1.0, 0.0, 1.0],  # cat
                [1.0, 1.0, 0.0],  # sat
                [1.0, 1.0, 0.0],  # the
                [0.0, 0.0, 1.0],  # a
                [0.0, 1.0, 0.0],  # dog
            ]
        )

        assert np.array_equal(three_documents().values, expected)

    def test_document_frequencies_count_documents_not_uses(self):
        assert np.array_equal(
            three_documents().document_frequencies, [2.0, 2.0, 2.0, 1.0, 1.0]
        )

    def test_a_word_in_two_of_three_documents_has_idf_1_2877(self):
        matrix = three_documents()

        assert matrix.inverse_document_frequencies[matrix.vocabulary.id_of("the")] == (
            pytest.approx(IDF_IN_TWO_OF_THREE, abs=1e-12)
        )
        assert round(IDF_IN_TWO_OF_THREE, 4) == 1.2877

    def test_a_word_in_one_document_has_idf_1_6931(self):
        matrix = three_documents()

        assert matrix.inverse_document_frequencies[matrix.vocabulary.id_of("dog")] == (
            pytest.approx(IDF_IN_ONE_OF_THREE, abs=1e-12)
        )
        assert round(IDF_IN_ONE_OF_THREE, 4) == 1.6931

    def test_the_smoothed_formula_is_log_of_one_plus_each_over_the_other_plus_one(self):
        frequencies = np.array([2.0, 1.0, 0.0])

        assert np.allclose(
            inverse_document_frequencies(frequencies, 3),
            np.log((1.0 + 3) / (1.0 + frequencies)) + 1.0,
        )

    def test_tf_idf_multiplies_each_count_by_its_rows_idf(self):
        counts = three_documents(TermWeighting.COUNT)
        weighted = three_documents(
            TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY
        )

        assert np.allclose(
            weighted.values,
            counts.values * counts.inverse_document_frequencies[:, None],
        )
        assert weighted.weight_of("dog", 1) == pytest.approx(IDF_IN_ONE_OF_THREE)
        assert weighted.weight_of("the", 0) == pytest.approx(IDF_IN_TWO_OF_THREE)

    def test_a_word_in_every_document_gets_weight_exactly_one(self):
        corpus = tokenised(["the cat", "the dog"])
        matrix = TermDocumentMatrix.from_tokenised(
            corpus,
            corpus.vocabulary(),
            TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY,
        )

        assert matrix.weight_of("the", 0) == 1.0
        assert matrix.weight_of("the", 1) == 1.0

    def test_a_repeated_use_counts_twice_but_is_one_document(self):
        corpus = tokenised(["cat cat sat"])
        matrix = TermDocumentMatrix.from_tokenised(corpus, corpus.vocabulary())

        assert matrix.weight_of("cat", 0) == 2.0
        assert matrix.document_frequencies[matrix.vocabulary.id_of("cat")] == 1.0


class TestShapes:
    def test_the_matrix_is_terms_by_documents(self):
        matrix = three_documents()

        assert matrix.values.shape == (5, 3)
        assert matrix.n_terms == 5
        assert matrix.n_documents == 3

    def test_tf_idf_keeps_the_shape(self):
        matrix = three_documents(
            TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY
        )

        assert matrix.values.shape == (5, 3)
        assert (
            matrix.weighting is TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY
        )

    def test_a_rare_word_is_not_a_row_and_a_document_of_only_rare_words_is_zeros(self):
        corpus = tokenised([*THREE_DOCUMENTS, "zebra"])
        matrix = TermDocumentMatrix.from_tokenised(corpus, corpus.vocabulary(2))

        assert list(matrix.vocabulary) == ["cat", "sat", "the"]
        assert matrix.values.shape == (3, 4)
        assert np.array_equal(matrix.values[:, 3], [0.0, 0.0, 0.0])

    def test_the_weighting_is_recorded(self):
        assert three_documents().weighting is TermWeighting.COUNT

    def test_the_enum_values_are_the_familiar_names(self):
        assert TermWeighting.COUNT.value == "count"
        assert TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY.value == "tf_idf"


class TestRefusals:
    def test_a_one_dimensional_matrix_is_refused(self):
        with pytest.raises(InvalidValuesError):
            TermDocumentMatrix(
                Vocabulary(["a", "b"]), np.zeros(2), TermWeighting.COUNT, np.zeros(2)
            )

    def test_rows_must_match_the_vocabulary(self):
        with pytest.raises(ShapeMismatchError):
            TermDocumentMatrix(
                Vocabulary(["a", "b"]),
                np.zeros((3, 2)),
                TermWeighting.COUNT,
                np.zeros(3),
            )

    def test_negative_weights_are_refused(self):
        with pytest.raises(InvalidValuesError):
            TermDocumentMatrix(
                Vocabulary(["a"]), np.array([[-1.0]]), TermWeighting.COUNT, np.zeros(1)
            )

    def test_non_finite_weights_are_refused(self):
        with pytest.raises(InvalidValuesError):
            TermDocumentMatrix(
                Vocabulary(["a"]),
                np.array([[np.nan]]),
                TermWeighting.COUNT,
                np.zeros(1),
            )

    def test_non_numeric_values_are_refused(self):
        with pytest.raises(InvalidValuesError):
            TermDocumentMatrix(
                Vocabulary(["a"]),
                np.array([["text"]]),  # type: ignore[arg-type]
                TermWeighting.COUNT,
                np.zeros(1),
            )

    def test_document_frequencies_must_be_one_per_term(self):
        with pytest.raises(InvalidValuesError):
            TermDocumentMatrix(
                Vocabulary(["a", "b"]),
                np.zeros((2, 1)),
                TermWeighting.COUNT,
                np.zeros(3),
            )

    def test_a_negative_document_frequency_is_refused(self):
        with pytest.raises(InvalidValuesError):
            TermDocumentMatrix(
                Vocabulary(["a"]),
                np.zeros((1, 1)),
                TermWeighting.COUNT,
                np.array([-1.0]),
            )

    def test_idf_needs_at_least_one_document(self):
        with pytest.raises(InvalidValuesError):
            inverse_document_frequencies(np.array([1.0]), 0)

    def test_idf_refuses_a_negative_frequency(self):
        with pytest.raises(InvalidValuesError):
            inverse_document_frequencies(np.array([-1.0]), 3)

    def test_the_weight_of_an_unknown_word_is_refused(self):
        with pytest.raises(UnknownTokenError):
            three_documents().weight_of("zebra", 0)

    @pytest.mark.parametrize("document", [3, -1])
    def test_the_weight_of_an_out_of_range_document_is_refused(self, document: int):
        with pytest.raises(InvalidValuesError):
            three_documents().weight_of("cat", document)


class TestEncapsulation:
    def test_values_are_frozen(self):
        matrix = three_documents()

        assert not matrix.values.flags.writeable
        with pytest.raises(ValueError):
            matrix.values[0, 0] = 5.0

    def test_document_frequencies_are_frozen(self):
        assert not three_documents().document_frequencies.flags.writeable

    def test_the_constructor_copies_its_input(self):
        source = np.ones((1, 2))
        matrix = TermDocumentMatrix(
            Vocabulary(["a"]), source, TermWeighting.COUNT, np.array([2.0])
        )
        source[0, 0] = 7.0

        assert matrix.values[0, 0] == 1.0

    def test_np_array_returns_a_private_copy(self):
        matrix = three_documents()

        assert not np.shares_memory(np.array(matrix), matrix.values)

    def test_np_asarray_may_share_the_frozen_buffer(self):
        matrix = three_documents()

        assert np.shares_memory(np.asarray(matrix), matrix.values)

    def test_a_dtype_change_under_copy_false_is_refused(self):
        matrix = three_documents()

        with pytest.raises(ValueError, match="copy"):
            matrix.__array__(dtype=np.float32, copy=False)

    def test_equality_is_a_verdict_over_terms_weighting_and_numbers(self):
        assert three_documents() == three_documents()
        assert three_documents() != three_documents(
            TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY
        )
        assert three_documents() != three_documents(minimum_count=2)
        assert (three_documents() == 3) is False
        assert hash(three_documents()) == hash(three_documents())

    def test_repr_names_the_shape_and_weighting(self):
        assert repr(three_documents()) == (
            "TermDocumentMatrix(n_terms=5, n_documents=3, weighting='count')"
        )
