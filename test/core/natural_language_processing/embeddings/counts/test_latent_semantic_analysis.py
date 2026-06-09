"""Spec for LatentSemanticAnalysis, its decomposition and its document vectors.

Pinned against the four-document example worked in the module docstring, whose
singular values are sqrt(7), sqrt(3), 1, 1 exactly, and against the designed
two-topic corpus in ``fixtures.py``.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.natural_language_processing.embeddings.counts.latent_semantic_analysis import (
    DocumentVectors,
    LatentSemanticAnalysis,
    TruncatedSingularValueDecomposition,
)
from oop_ml.core.natural_language_processing.embeddings.counts.term_document import (
    TermWeighting,
)
from test.core.natural_language_processing.embeddings.counts.fixtures import (
    COOKING_WORDS,
    N_COOKING_DOCUMENTS,
    SAILING_WORDS,
    TINY_LSA_CORPUS,
    TWO_TOPIC_CORPUS,
)

SQRT_SEVEN = 7.0**0.5
SQRT_THREE = 3.0**0.5
# 23 terms by 24 documents, so the most components the matrix has.
TWO_TOPIC_FULL_RANK = 23
BOTH_WEIGHTINGS = list(TermWeighting)


def tiny(dimension: int = 2) -> LatentSemanticAnalysis:
    return LatentSemanticAnalysis(
        dimension=dimension, weighting=TermWeighting.COUNT
    ).fit(TINY_LSA_CORPUS)


def two_topics(
    dimension: int = 2,
    weighting: TermWeighting = TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY,
) -> LatentSemanticAnalysis:
    return LatentSemanticAnalysis(dimension=dimension, weighting=weighting).fit(
        TWO_TOPIC_CORPUS
    )


def mean_similarity(
    model: LatentSemanticAnalysis, first: list[str], second: list[str]
) -> float:
    pairs = [
        model.similarity(one, other)
        for one in first
        for other in second
        if one != other
    ]
    return float(np.mean(pairs))


class TestTheFourDocumentExample:
    def test_the_vocabulary_is_commonest_first_then_alphabetical(self):
        assert list(tiny().vocabulary) == [
            "the",
            "boat",
            "cat",
            "ran",
            "sailed",
            "sank",
            "sat",
        ]

    def test_the_singular_values_are_sqrt_seven_sqrt_three_one_one(self):
        assert tiny(2).singular_values == pytest.approx([SQRT_SEVEN, SQRT_THREE])
        assert tiny(4).singular_values == pytest.approx(
            [SQRT_SEVEN, SQRT_THREE, 1.0, 1.0]
        )

    def test_the_first_component_is_the_shared_direction(self):
        model = tiny()
        first = model.embeddings.table[:, 0]

        assert first[model.vocabulary.id_of("the")] == pytest.approx(2.0)
        assert first[model.vocabulary.id_of("boat")] == pytest.approx(1.0)
        assert first[model.vocabulary.id_of("cat")] == pytest.approx(1.0)
        for word in ("ran", "sailed", "sank", "sat"):
            assert first[model.vocabulary.id_of(word)] == pytest.approx(0.5)
        assert model.document_vectors.vectors[:, 0] == pytest.approx(
            [SQRT_SEVEN / 2] * 4
        )

    def test_the_second_component_is_the_topic_contrast_up_to_its_sign(self):
        """``cat`` and ``boat`` tie for the largest magnitude, so the sign rule
        cannot be pinned; everything else about the component can."""
        model = tiny()
        second = model.embeddings.table[:, 1]
        cat_sign = np.sign(second[model.vocabulary.id_of("cat")])

        assert second[model.vocabulary.id_of("the")] == pytest.approx(0.0, abs=1e-12)
        assert second[model.vocabulary.id_of("cat")] == pytest.approx(cat_sign * 1.0)
        assert second[model.vocabulary.id_of("boat")] == pytest.approx(-cat_sign * 1.0)
        for word in ("ran", "sat"):
            assert second[model.vocabulary.id_of(word)] == pytest.approx(cat_sign * 0.5)
        for word in ("sailed", "sank"):
            assert second[model.vocabulary.id_of(word)] == pytest.approx(
                -cat_sign * 0.5
            )
        assert model.document_vectors.vectors[:, 1] == pytest.approx(
            cat_sign * np.array([1.0, 1.0, -1.0, -1.0]) * SQRT_THREE / 2
        )

    def test_the_variance_shares_are_seven_and_three_twelfths(self):
        assert tiny().variance_shares == pytest.approx([7 / 12, 3 / 12])

    def test_shares_sum_to_one_only_at_full_rank(self):
        assert sum(tiny(2).variance_shares) == pytest.approx(10 / 12)
        assert sum(tiny(4).variance_shares) == pytest.approx(1.0)

    def test_a_new_document_folds_in_as_the_sum_of_its_words_left_vectors(self):
        """``the cat purred``: ``purred`` is unknown and ignored, so the fold-in
        is ``U_k^T (e_the + e_cat) = (3 / sqrt(7), +-1 / sqrt(3))``."""
        model = tiny()
        cat_sign = np.sign(model.embeddings.table[model.vocabulary.id_of("cat"), 1])
        folded = model.transform(["the cat purred"])

        assert folded.n_documents == 1
        assert folded.vector_of(0) == pytest.approx(
            [3 / SQRT_SEVEN, cat_sign / SQRT_THREE]
        )

    def test_a_document_of_only_unknown_words_folds_to_the_zero_vector(self):
        assert np.array_equal(
            tiny().transform(["zebra giraffe"]).vector_of(0), [0.0, 0.0]
        )

    def test_the_fold_in_uses_the_fitted_idf_not_one_from_the_new_texts(self):
        """One training text alone lands exactly where it did among four; had
        the idf been recomputed from that one text every word would weigh 1."""
        model = LatentSemanticAnalysis(
            dimension=2,
            weighting=TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY,
        ).fit(TINY_LSA_CORPUS)

        assert model.transform([TINY_LSA_CORPUS[0]]).vector_of(0) == pytest.approx(
            model.document_vectors.vector_of(0), abs=1e-12
        )

    def test_reconstruction_at_full_rank_recovers_the_count_matrix(self):
        model = tiny(4)

        assert np.allclose(
            model.decomposition.reconstruction(),
            model.term_document_matrix.values,
            atol=1e-12,
        )


class TestTwoTopics:
    @pytest.mark.parametrize("weighting", BOTH_WEIGHTINGS)
    @pytest.mark.parametrize("dimension", [2, 4])
    def test_every_document_has_one_sign_on_the_first_component(
        self, dimension: int, weighting: TermWeighting
    ):
        """Perron: the matrix is non-negative, so its leading singular vector is."""
        first = two_topics(dimension, weighting).document_vectors.vectors[:, 0]

        assert np.all(first > 0.0)

    @pytest.mark.parametrize("weighting", BOTH_WEIGHTINGS)
    @pytest.mark.parametrize("dimension", [2, 4])
    def test_the_second_component_separates_the_topics(
        self, dimension: int, weighting: TermWeighting
    ):
        second = two_topics(dimension, weighting).document_vectors.vectors[:, 1]
        cooking = np.sign(second[:N_COOKING_DOCUMENTS])
        sailing = np.sign(second[N_COOKING_DOCUMENTS:])

        assert len(set(cooking.tolist())) == 1
        assert len(set(sailing.tolist())) == 1
        assert cooking[0] == -sailing[0]

    def test_two_cooking_words_are_nearer_than_a_cooking_and_a_sailing_word(self):
        model = two_topics()

        assert model.similarity("flour", "sugar") == pytest.approx(1.0, abs=1e-3)
        assert model.similarity("flour", "anchor") == pytest.approx(-0.0028, abs=1e-3)
        assert model.similarity("flour", "sugar") > model.similarity("flour", "anchor")

    @pytest.mark.parametrize("weighting", BOTH_WEIGHTINGS)
    def test_within_topic_similarity_exceeds_across_on_average(
        self, weighting: TermWeighting
    ):
        model = two_topics(2, weighting)
        within = mean_similarity(model, COOKING_WORDS, COOKING_WORDS)
        across = mean_similarity(model, COOKING_WORDS, SAILING_WORDS)

        assert within > 0.99
        assert across < 0.05

    def test_the_nearest_words_to_flour_are_the_other_cooking_words(self):
        nearest = two_topics().most_similar("flour", n_results=9)

        assert set(nearest.words) == set(COOKING_WORDS) - {"flour"}

    def test_a_new_cooking_text_lands_on_the_cooking_side(self):
        model = two_topics()
        cooking_sign = np.sign(model.document_vectors.vectors[0, 1])
        folded = model.transform(["we bake the dough and whisk the eggs"])

        assert np.sign(folded.vector_of(0)[1]) == cooking_sign

    @pytest.mark.parametrize("weighting", BOTH_WEIGHTINGS)
    def test_folding_in_the_training_texts_reproduces_the_document_vectors(
        self, weighting: TermWeighting
    ):
        model = two_topics(4, weighting)

        assert np.allclose(
            model.transform(TWO_TOPIC_CORPUS).vectors,
            model.document_vectors.vectors,
            atol=1e-10,
        )

    def test_singular_values_descend(self):
        singular_values = two_topics(8).singular_values

        assert np.all(np.diff(singular_values) <= 0.0)

    @pytest.mark.parametrize("weighting", BOTH_WEIGHTINGS)
    def test_reconstruction_at_full_rank_recovers_the_matrix(
        self, weighting: TermWeighting
    ):
        model = two_topics(TWO_TOPIC_FULL_RANK, weighting)

        assert np.allclose(
            model.decomposition.reconstruction(),
            model.term_document_matrix.values,
            atol=1e-10,
        )

    def test_reconstruction_at_low_rank_does_not(self):
        model = two_topics(2)

        assert not np.allclose(
            model.decomposition.reconstruction(),
            model.term_document_matrix.values,
            atol=1e-3,
        )

    def test_variance_shares_are_squared_singular_values_over_the_frobenius_norm(
        self,
    ):
        """The sum of every squared singular value is the squared Frobenius
        norm of the matrix, which is an oracle that never truncates."""
        model = two_topics(4)
        frobenius_squared = float(np.sum(model.term_document_matrix.values**2))

        assert model.variance_shares == pytest.approx(
            model.singular_values**2 / frobenius_squared
        )
        assert sum(two_topics(TWO_TOPIC_FULL_RANK).variance_shares) == pytest.approx(
            1.0
        )

    def test_the_document_vectors_are_one_per_text_of_the_asked_dimension(self):
        vectors = two_topics(4).document_vectors

        assert vectors.n_documents == len(TWO_TOPIC_CORPUS)
        assert vectors.dimension == 4
        assert len(list(vectors)) == len(TWO_TOPIC_CORPUS)

    def test_the_word_vectors_are_one_per_word_of_the_asked_dimension(self):
        embeddings = two_topics(4).embeddings

        assert embeddings.n_words == TWO_TOPIC_FULL_RANK
        assert embeddings.dimension == 4


class TestRefusals:
    def test_a_dimension_above_the_number_of_terms_is_refused_at_fit(self):
        with pytest.raises(TooFewValuesError):
            LatentSemanticAnalysis(dimension=TWO_TOPIC_FULL_RANK + 1).fit(
                TWO_TOPIC_CORPUS
            )

    def test_a_dimension_above_the_number_of_documents_is_refused_at_fit(self):
        with pytest.raises(TooFewValuesError):
            LatentSemanticAnalysis(dimension=5).fit(TINY_LSA_CORPUS)

    def test_exactly_the_rank_is_accepted(self):
        assert LatentSemanticAnalysis(dimension=4).fit(TINY_LSA_CORPUS).is_fitted

    @pytest.mark.parametrize(
        "keywords",
        [
            {"dimension": 0},
            {"minimum_count": 0},
            {"n_components": 2},
            {"weighting": "idf"},
        ],
    )
    def test_bad_construction_is_refused_by_pydantic(self, keywords: dict[str, object]):
        with pytest.raises(ValidationError):
            LatentSemanticAnalysis(**keywords)  # type: ignore[arg-type]

    def test_a_single_string_corpus_is_refused(self):
        with pytest.raises(InvalidValuesError):
            LatentSemanticAnalysis(dimension=1).fit("the cat sat")  # type: ignore[arg-type]

    def test_a_blank_corpus_is_refused(self):
        with pytest.raises(EmptyValuesError):
            LatentSemanticAnalysis(dimension=1).fit(["  ", ""])

    def test_no_word_reaching_the_minimum_count_is_refused(self):
        with pytest.raises(TooFewValuesError):
            LatentSemanticAnalysis(dimension=1, minimum_count=10).fit(TINY_LSA_CORPUS)

    def test_minimum_count_drops_the_rare_words(self):
        model = LatentSemanticAnalysis(dimension=2, minimum_count=2).fit(
            TINY_LSA_CORPUS
        )

        assert list(model.vocabulary) == ["the", "boat", "cat"]

    def test_before_fit_everything_learned_raises_not_fitted(self):
        model = LatentSemanticAnalysis(dimension=2)

        for attribute in (
            "embeddings",
            "document_vectors",
            "singular_values",
            "variance_shares",
            "decomposition",
            "term_document_matrix",
            "vocabulary",
        ):
            with pytest.raises(NotFittedError):
                getattr(model, attribute)
        with pytest.raises(NotFittedError):
            model.transform(["the cat"])
        with pytest.raises(NotFittedError):
            model.similarity("the", "cat")

    def test_transform_refuses_a_single_string(self):
        with pytest.raises(InvalidValuesError):
            tiny().transform("the cat")  # type: ignore[arg-type]

    def test_transform_refuses_blank_texts(self):
        with pytest.raises(EmptyValuesError):
            tiny().transform(["   "])

    def test_fit_returns_self(self):
        model = LatentSemanticAnalysis(dimension=2)

        assert model.fit(TINY_LSA_CORPUS) is model

    def test_a_failed_fit_leaves_the_model_unfitted(self):
        model = LatentSemanticAnalysis(dimension=5)

        with pytest.raises(TooFewValuesError):
            model.fit(TINY_LSA_CORPUS)
        assert not model.is_fitted


def random_matrix(n_rows: int = 6, n_columns: int = 4) -> np.ndarray:
    return np.random.default_rng(11).normal(size=(n_rows, n_columns))


class TestTruncatedSingularValueDecomposition:
    def test_left_and_right_vectors_are_orthonormal(self):
        decomposition = TruncatedSingularValueDecomposition.of(random_matrix(), 3)

        assert np.allclose(
            decomposition.left_vectors.T @ decomposition.left_vectors, np.eye(3)
        )
        assert np.allclose(
            decomposition.right_vectors.T @ decomposition.right_vectors, np.eye(3)
        )

    def test_the_largest_entry_of_every_left_vector_is_positive(self):
        left = TruncatedSingularValueDecomposition.of(random_matrix(), 4).left_vectors

        for column in range(4):
            assert left[np.argmax(np.abs(left[:, column])), column] > 0.0

    def test_flipping_the_sign_leaves_the_reconstruction_unchanged(self):
        matrix = random_matrix()
        decomposition = TruncatedSingularValueDecomposition.of(matrix, 4)

        assert np.allclose(decomposition.reconstruction(), matrix)

    def test_the_discarded_energy_is_the_reconstruction_error(self):
        """Eckart-Young: ``||X - X_k||_F^2 = sum of the discarded s^2``."""
        matrix = random_matrix()
        decomposition = TruncatedSingularValueDecomposition.of(matrix, 2)
        discarded = decomposition.total_squared_singular_values - float(
            np.sum(decomposition.singular_values**2)
        )

        assert float(np.sum((matrix - decomposition.reconstruction()) ** 2)) == (
            pytest.approx(discarded)
        )

    def test_the_total_is_the_squared_frobenius_norm(self):
        matrix = random_matrix()

        assert TruncatedSingularValueDecomposition.of(
            matrix, 1
        ).total_squared_singular_values == pytest.approx(float(np.sum(matrix**2)))

    def test_dimension_and_shares(self):
        decomposition = TruncatedSingularValueDecomposition.of(random_matrix(), 2)

        assert decomposition.dimension == 2
        assert len(decomposition.variance_shares) == 2
        assert 0.0 < sum(decomposition.variance_shares) < 1.0

    def test_a_zero_matrix_has_zero_shares(self):
        decomposition = TruncatedSingularValueDecomposition.of(np.zeros((3, 3)), 2)

        assert decomposition.variance_shares == (0.0, 0.0)
        assert np.array_equal(decomposition.singular_values, [0.0, 0.0])

    def test_a_dimension_above_the_rank_is_refused(self):
        with pytest.raises(TooFewValuesError):
            TruncatedSingularValueDecomposition.of(random_matrix(6, 4), 5)

    def test_a_dimension_below_one_is_refused(self):
        with pytest.raises(InvalidValuesError):
            TruncatedSingularValueDecomposition.of(random_matrix(), 0)

    def test_a_non_finite_matrix_is_refused(self):
        matrix = random_matrix()
        matrix[0, 0] = np.inf

        with pytest.raises(InvalidValuesError):
            TruncatedSingularValueDecomposition.of(matrix, 1)

    def test_a_one_dimensional_matrix_is_refused(self):
        with pytest.raises(InvalidValuesError):
            TruncatedSingularValueDecomposition.of(np.ones(3), 1)

    def test_the_constructor_refuses_ascending_singular_values(self):
        with pytest.raises(InvalidValuesError):
            TruncatedSingularValueDecomposition(
                np.eye(3, 2), np.array([1.0, 2.0]), np.eye(3, 2), 5.0
            )

    def test_the_constructor_refuses_a_negative_singular_value(self):
        with pytest.raises(InvalidValuesError):
            TruncatedSingularValueDecomposition(
                np.eye(3, 2), np.array([2.0, -1.0]), np.eye(3, 2), 5.0
            )

    def test_the_constructor_refuses_a_total_below_what_was_kept(self):
        with pytest.raises(InvalidValuesError):
            TruncatedSingularValueDecomposition(
                np.eye(3, 2), np.array([2.0, 1.0]), np.eye(3, 2), 4.0
            )

    def test_the_constructor_refuses_blocks_that_disagree_on_the_rank(self):
        with pytest.raises(ShapeMismatchError):
            TruncatedSingularValueDecomposition(
                np.eye(3, 2), np.array([2.0, 1.0]), np.eye(3, 3), 5.0
            )

    def test_the_constructor_refuses_no_components(self):
        with pytest.raises(InvalidValuesError):
            TruncatedSingularValueDecomposition(
                np.zeros((3, 0)), np.zeros(0), np.zeros((3, 0)), 0.0
            )

    def test_the_constructor_refuses_a_non_finite_block(self):
        with pytest.raises(InvalidValuesError):
            TruncatedSingularValueDecomposition(
                np.full((3, 1), np.nan), np.array([1.0]), np.eye(3, 1), 1.0
            )

    def test_every_block_is_frozen(self):
        decomposition = TruncatedSingularValueDecomposition.of(random_matrix(), 2)

        assert not decomposition.left_vectors.flags.writeable
        assert not decomposition.singular_values.flags.writeable
        assert not decomposition.right_vectors.flags.writeable

    def test_equality_is_a_verdict(self):
        matrix = random_matrix()
        two = TruncatedSingularValueDecomposition.of(matrix, 2)

        assert two == TruncatedSingularValueDecomposition.of(matrix, 2)
        assert two != TruncatedSingularValueDecomposition.of(matrix, 3)
        assert (two == "not a decomposition") is False
        assert hash(two) == hash(TruncatedSingularValueDecomposition.of(matrix, 2))

    def test_repr_names_the_shape_and_rank(self):
        assert repr(TruncatedSingularValueDecomposition.of(random_matrix(), 2)) == (
            "TruncatedSingularValueDecomposition(n_rows=6, n_columns=4, dimension=2)"
        )


class TestDocumentVectors:
    def test_iterates_rows_in_document_order(self):
        vectors = DocumentVectors(np.array([[1.0, 2.0], [3.0, 4.0]]))

        assert [row.tolist() for row in vectors] == [[1.0, 2.0], [3.0, 4.0]]
        assert len(vectors) == 2

    def test_vector_of_reads_a_frozen_row(self):
        vectors = DocumentVectors(np.array([[1.0, 2.0], [3.0, 4.0]]))

        assert np.array_equal(vectors.vector_of(1), [3.0, 4.0])
        assert not vectors.vector_of(1).flags.writeable

    @pytest.mark.parametrize("position", [2, -1])
    def test_vector_of_an_out_of_range_document_is_refused(self, position: int):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.zeros((2, 2))).vector_of(position)

    def test_n_documents_and_dimension(self):
        vectors = DocumentVectors(np.zeros((5, 3)))

        assert vectors.n_documents == 5
        assert vectors.dimension == 3

    def test_a_zero_row_is_allowed(self):
        assert DocumentVectors(np.zeros((1, 3))).n_documents == 1

    def test_a_one_dimensional_array_is_refused(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.zeros(3))

    def test_no_documents_is_refused(self):
        with pytest.raises(EmptyValuesError):
            DocumentVectors(np.zeros((0, 3)))

    def test_no_dimensions_is_refused(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.zeros((3, 0)))

    def test_a_non_finite_value_is_refused(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.array([[1.0, np.nan]]))

    def test_non_numeric_values_are_refused(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.array([["text"]]))  # type: ignore[arg-type]

    def test_the_vectors_are_copied_and_frozen(self):
        source = np.ones((2, 2))
        vectors = DocumentVectors(source)
        source[0, 0] = 9.0

        assert vectors.vectors[0, 0] == 1.0
        assert not vectors.vectors.flags.writeable

    def test_the_array_protocol_copies_when_asked_and_shares_when_allowed(self):
        vectors = DocumentVectors(np.ones((2, 2)))

        assert not np.shares_memory(np.array(vectors), vectors.vectors)
        assert np.shares_memory(np.asarray(vectors), vectors.vectors)
        with pytest.raises(ValueError, match="copy"):
            vectors.__array__(dtype=np.float32, copy=False)

    def test_equality_is_a_verdict(self):
        vectors = DocumentVectors(np.ones((2, 2)))

        assert vectors == DocumentVectors(np.ones((2, 2)))
        assert vectors != DocumentVectors(np.zeros((2, 2)))
        assert (vectors == "not document vectors") is False
        assert hash(vectors) == hash(DocumentVectors(np.ones((2, 2))))

    def test_repr_names_the_shape(self):
        assert repr(DocumentVectors(np.ones((2, 3)))) == (
            "DocumentVectors(n_documents=2, dimension=3)"
        )
