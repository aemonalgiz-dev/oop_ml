"""Spec for BagOfWords -- counts per vocabulary word, weighted by the fitted idf.

Pinned against three documents whose counts and smoothed inverse document
frequencies are worked by hand in the module docstring and in ``fixtures.py``.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    TooFewValuesError,
    UndefinedMetricError,
)
from oop_ml.core.natural_language_processing.embeddings.counts.term_document import (
    TermWeighting,
)
from oop_ml.core.natural_language_processing.embeddings.documents.bag_of_words import (
    BagOfWords,
    unit_rows,
)
from test.core.natural_language_processing.embeddings.documents.fixtures import (
    IDF_IN_ONE_OF_THREE,
    IDF_IN_TWO_OF_THREE,
    THREE_DOCUMENT_COUNTS,
    THREE_DOCUMENT_VOCABULARY,
    THREE_DOCUMENTS,
)

TF_IDF = TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY

# One weight per vocabulary word, in vocabulary order: cat, sat, the, a, dog.
THREE_DOCUMENT_IDF = np.array(
    [
        IDF_IN_TWO_OF_THREE,
        IDF_IN_TWO_OF_THREE,
        IDF_IN_TWO_OF_THREE,
        IDF_IN_ONE_OF_THREE,
        IDF_IN_ONE_OF_THREE,
    ]
)


def fit_counts() -> BagOfWords:
    return BagOfWords().fit(THREE_DOCUMENTS)


def fit_tf_idf() -> BagOfWords:
    return BagOfWords(weighting=TF_IDF).fit(THREE_DOCUMENTS)


class TestFit:
    def test_the_vocabulary_is_commonest_first_with_ties_alphabetical(self):
        assert list(fit_counts().vocabulary) == THREE_DOCUMENT_VOCABULARY

    def test_the_worked_inverse_document_frequencies(self):
        bag = fit_counts()
        frequencies = bag.inverse_document_frequencies

        assert frequencies[bag.vocabulary.id_of("the")] == pytest.approx(
            1.2877, abs=1e-4
        )
        assert frequencies[bag.vocabulary.id_of("dog")] == pytest.approx(
            1.6931, abs=1e-4
        )
        assert np.allclose(frequencies, THREE_DOCUMENT_IDF, rtol=0.0, atol=1e-15)

    def test_the_training_matrix_is_the_counts(self):
        assert np.array_equal(
            fit_counts().term_document_matrix.values, THREE_DOCUMENT_COUNTS
        )

    def test_the_training_matrix_is_weighted_under_tf_idf(self):
        expected = THREE_DOCUMENT_COUNTS * THREE_DOCUMENT_IDF[:, None]

        assert fit_tf_idf().term_document_matrix.weighting is TF_IDF
        assert np.allclose(
            fit_tf_idf().term_document_matrix.values, expected, rtol=0.0, atol=1e-15
        )

    def test_the_idf_is_learned_under_both_weightings(self):
        assert np.array_equal(
            fit_counts().inverse_document_frequencies,
            fit_tf_idf().inverse_document_frequencies,
        )

    def test_the_idf_is_frozen(self):
        assert fit_counts().inverse_document_frequencies.flags.writeable is False

    def test_minimum_count_drops_the_rare_words(self):
        bag = BagOfWords(minimum_count=2).fit(THREE_DOCUMENTS)

        assert list(bag.vocabulary) == ["cat", "sat", "the"]
        assert bag.transform(["a dog"]).dimension == 3

    def test_no_word_reaching_the_minimum_is_refused(self):
        with pytest.raises(TooFewValuesError):
            BagOfWords(minimum_count=3).fit(THREE_DOCUMENTS)

    def test_fit_returns_self(self):
        bag = BagOfWords()

        assert bag.fit(THREE_DOCUMENTS) is bag

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            BagOfWords().fit("the cat sat")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            BagOfWords().fit(["", "  "])

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            BagOfWords().fit([])

    def test_refitting_replaces_the_vocabulary(self):
        bag = fit_counts().fit(["one two", "two three"])

        assert list(bag.vocabulary) == ["two", "one", "three"]


class TestTransform:
    @pytest.mark.parametrize("weighting", list(TermWeighting))
    def test_the_training_texts_reproduce_the_training_matrix_transposed(
        self, weighting
    ):
        """Two routes to one table: the matrix's own counting, and transform's."""
        bag = BagOfWords(weighting=weighting).fit(THREE_DOCUMENTS)

        assert np.array_equal(
            bag.transform(THREE_DOCUMENTS).vectors, bag.term_document_matrix.values.T
        )

    def test_the_dimension_is_the_vocabulary_size(self):
        assert fit_counts().transform(["anything"]).dimension == 5

    def test_a_new_text_is_counted(self):
        assert np.array_equal(
            fit_counts().transform(["the dog sat"]).vector_of(0), [0, 1, 1, 0, 1]
        )

    def test_a_repeated_word_counts_each_time(self):
        assert np.array_equal(
            fit_counts().transform(["cat cat cat"]).vector_of(0), [3, 0, 0, 0, 0]
        )

    def test_a_new_text_is_weighted_by_the_fitted_idf(self):
        expected = [0.0, 1.2877, 1.2877, 0.0, 1.6931]

        assert np.allclose(
            fit_tf_idf().transform(["the dog sat"]).vector_of(0), expected, atol=1e-4
        )

    def test_an_unseen_word_contributes_nothing(self):
        assert np.allclose(
            fit_tf_idf().transform(["the zebra sat"]).vector_of(0),
            [0.0, IDF_IN_TWO_OF_THREE, IDF_IN_TWO_OF_THREE, 0.0, 0.0],
        )

    def test_the_idf_is_the_training_corpus_and_not_the_batch(self):
        """Recomputed on this batch every word would be in every document and
        weigh exactly one; the fitted weights are 1.2877 and 1.6931."""
        vectors = fit_tf_idf().transform(["the dog sat", "the dog sat"])

        assert np.allclose(vectors.vector_of(0), vectors.vector_of(1))
        assert not np.allclose(vectors.vector_of(0), [0.0, 1.0, 1.0, 0.0, 1.0])
        assert vectors.vector_of(0)[4] == pytest.approx(IDF_IN_ONE_OF_THREE)

    def test_normalised_rows_have_unit_length(self):
        bag = BagOfWords(weighting=TF_IDF, normalise=True).fit(THREE_DOCUMENTS)
        vectors = bag.transform(["the dog sat", "a cat", "cat"])

        assert np.allclose(np.linalg.norm(vectors.vectors, axis=1), 1.0)

    def test_the_worked_normalised_vector(self):
        """(0, 1.2877, 1.2877, 0, 1.6931) has norm 2.4866."""
        bag = BagOfWords(weighting=TF_IDF, normalise=True).fit(THREE_DOCUMENTS)

        assert np.allclose(
            bag.transform(["the dog sat"]).vector_of(0),
            [0.0, 0.5179, 0.5179, 0.0, 0.6809],
            atol=1e-4,
        )

    def test_normalisation_keeps_the_direction(self):
        raw = fit_tf_idf().transform(["the dog sat"]).vector_of(0)
        normalised = (
            BagOfWords(weighting=TF_IDF, normalise=True)
            .fit(THREE_DOCUMENTS)
            .transform(["the dog sat"])
            .vector_of(0)
        )

        assert np.allclose(normalised, raw / np.linalg.norm(raw))

    def test_a_document_of_unknown_words_is_the_zero_vector(self):
        assert np.array_equal(
            fit_counts().transform(["zebra quokka"]).vector_of(0), np.zeros(5)
        )

    def test_a_zero_document_stays_zero_when_normalised(self):
        bag = BagOfWords(normalise=True).fit(THREE_DOCUMENTS)
        vectors = bag.transform(["zebra quokka", "the cat"])

        assert np.array_equal(vectors.vector_of(0), np.zeros(5))
        assert np.linalg.norm(vectors.vector_of(1)) == pytest.approx(1.0)

    def test_a_zero_document_has_no_similarity(self):
        vectors = fit_counts().transform(["zebra quokka", "the cat"])

        with pytest.raises(UndefinedMetricError):
            vectors.similarity(0, 1)

    def test_shared_words_make_documents_similar(self):
        vectors = fit_counts().transform(["the cat sat", "the cat sat down", "a dog"])

        assert vectors.similarity(0, 1) > vectors.similarity(0, 2)
        assert vectors.similarity(0, 2) == pytest.approx(0.0)

    def test_fit_transform_matches_fit_then_transform(self):
        assert BagOfWords(weighting=TF_IDF).fit_transform(
            THREE_DOCUMENTS
        ) == fit_tf_idf().transform(THREE_DOCUMENTS)

    def test_before_fit_raises_not_fitted(self):
        bag = BagOfWords()

        with pytest.raises(NotFittedError):
            bag.transform(THREE_DOCUMENTS)
        with pytest.raises(NotFittedError):
            _ = bag.vocabulary
        with pytest.raises(NotFittedError):
            _ = bag.inverse_document_frequencies
        with pytest.raises(NotFittedError):
            _ = bag.term_document_matrix

    def test_a_single_string_is_refused(self):
        with pytest.raises(InvalidValuesError):
            fit_counts().transform("the dog sat")  # type: ignore[arg-type]

    def test_a_blank_batch_is_refused(self):
        with pytest.raises(EmptyValuesError):
            fit_counts().transform(["", " "])

    def test_a_blank_text_beside_a_real_one_is_the_zero_vector(self):
        vectors = fit_counts().transform(["", "the cat"])

        assert np.array_equal(vectors.vector_of(0), np.zeros(5))
        assert vectors.n_documents == 2


class TestUnitRows:
    def test_every_non_zero_row_gets_unit_norm(self):
        rows = unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))

        assert np.allclose(rows, [[0.6, 0.8], [0.0, 1.0]])

    def test_a_zero_row_stays_zero(self):
        rows = unit_rows(np.array([[0.0, 0.0], [1.0, 1.0]]))

        assert np.array_equal(rows[0], [0.0, 0.0])
        assert np.isfinite(rows).all()


class TestConstruction:
    @pytest.mark.parametrize(
        "keywords",
        [
            {"minimum_count": 0},
            {"weighting": "binary"},
            {"normalize": True},
            {"normalise": "sometimes"},
            {"vocabulary_size": 10},
        ],
    )
    def test_out_of_range_or_unknown_keywords_are_refused(self, keywords):
        with pytest.raises(ValidationError):
            BagOfWords(**keywords)

    def test_the_weighting_accepts_its_string_value(self):
        assert BagOfWords(weighting="tf_idf").weighting is TF_IDF  # type: ignore[arg-type]

    def test_the_defaults(self):
        bag = BagOfWords()

        assert bag.weighting is TermWeighting.COUNT
        assert bag.normalise is False
        assert bag.minimum_count == 1
