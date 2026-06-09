"""Spec for the documents frame: DocumentVectors, and what every embedder shares."""

from collections.abc import Sequence
from typing import Self

import numpy as np
import pytest
from pydantic import PrivateAttr, ValidationError

from oop_ml.core.base.estimator import Transformer
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NotFittedError,
    UndefinedMetricError,
)
from oop_ml.core.natural_language_processing.embeddings.documents.bag_of_words import (
    BagOfWords,
)
from oop_ml.core.natural_language_processing.embeddings.documents.embedder import (
    DocumentEmbedder,
    DocumentVectors,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import WordEmbedder
from test.core.natural_language_processing.embeddings.documents.fixtures import (
    THREE_DOCUMENTS,
    cosine,
)

TWO_BY_TWO = np.array([[1.0, 0.0], [1.0, 1.0]])


class TestDocumentVectorsConstruction:
    def test_holds_a_frozen_copy(self):
        source = TWO_BY_TWO.copy()
        vectors = DocumentVectors(source)
        source[0, 0] = 99.0

        assert vectors.vectors[0, 0] == 1.0
        assert vectors.vectors.flags.writeable is False

    def test_counts_documents_and_dimension(self):
        vectors = DocumentVectors(np.zeros((3, 5)))

        assert vectors.n_documents == 3
        assert vectors.dimension == 5
        assert len(vectors) == 3

    def test_a_zero_row_is_a_legitimate_answer(self):
        vectors = DocumentVectors(np.array([[0.0, 0.0], [1.0, 2.0]]))

        assert np.array_equal(vectors.vector_of(0), [0.0, 0.0])

    def test_a_single_vector_is_refused_as_one_dimension(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.array([1.0, 2.0, 3.0]))

    def test_three_dimensions_are_refused(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.zeros((2, 2, 2)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_value_is_refused(self, bad):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.array([[1.0, bad]]))

    def test_no_documents_is_refused(self):
        with pytest.raises(EmptyValuesError):
            DocumentVectors(np.zeros((0, 4)))

    def test_no_dimension_is_refused(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(np.zeros((3, 0)))

    def test_a_non_numeric_table_is_refused(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors([["a", "b"]])  # type: ignore[arg-type]

    def test_repr_names_the_shape(self):
        assert repr(DocumentVectors(TWO_BY_TWO)) == (
            "DocumentVectors(n_documents=2, dimension=2)"
        )


class TestDocumentVectorsAccess:
    def test_vector_of_is_the_row(self):
        vectors = DocumentVectors(TWO_BY_TWO)

        assert np.array_equal(vectors.vector_of(1), [1.0, 1.0])

    def test_vector_of_is_read_only(self):
        row = DocumentVectors(TWO_BY_TWO).vector_of(0)

        with pytest.raises(ValueError):
            row[0] = 5.0

    @pytest.mark.parametrize("position", [-1, 2, 100])
    def test_an_out_of_range_position_is_refused(self, position):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(TWO_BY_TWO).vector_of(position)

    def test_similarity_is_the_cosine_of_the_two_rows(self):
        vectors = DocumentVectors(TWO_BY_TWO)

        assert vectors.similarity(0, 1) == pytest.approx(1.0 / np.sqrt(2.0))
        assert vectors.similarity(0, 1) == pytest.approx(
            cosine(TWO_BY_TWO[0], TWO_BY_TWO[1])
        )

    def test_similarity_of_a_document_with_itself_is_one(self):
        assert DocumentVectors(TWO_BY_TWO).similarity(1, 1) == pytest.approx(1.0)

    def test_similarity_is_symmetric(self):
        vectors = DocumentVectors(np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]))

        assert vectors.similarity(0, 1) == vectors.similarity(1, 0)

    def test_similarity_with_a_zero_row_is_undefined(self):
        vectors = DocumentVectors(np.array([[0.0, 0.0], [1.0, 1.0]]))

        with pytest.raises(UndefinedMetricError):
            vectors.similarity(0, 1)

    def test_similarity_refuses_an_out_of_range_position(self):
        with pytest.raises(InvalidValuesError):
            DocumentVectors(TWO_BY_TWO).similarity(0, 7)

    def test_iterates_the_rows_in_order(self):
        rows = list(DocumentVectors(TWO_BY_TWO))

        assert len(rows) == 2
        assert np.array_equal(rows[0], [1.0, 0.0])
        assert np.array_equal(rows[1], [1.0, 1.0])


class TestDocumentVectorsProtocols:
    def test_the_array_protocol_reads_the_table(self):
        vectors = DocumentVectors(TWO_BY_TWO)

        assert np.array_equal(np.asarray(vectors), TWO_BY_TWO)

    def test_a_requested_copy_does_not_share_memory(self):
        vectors = DocumentVectors(TWO_BY_TWO)

        assert not np.shares_memory(np.array(vectors, copy=True), vectors.vectors)

    def test_equality_is_a_verdict_on_the_whole_table(self):
        assert DocumentVectors(TWO_BY_TWO) == DocumentVectors(TWO_BY_TWO.copy())
        assert DocumentVectors(TWO_BY_TWO) != DocumentVectors(TWO_BY_TWO + 1.0)
        assert DocumentVectors(TWO_BY_TWO) != DocumentVectors(np.zeros((3, 2)))

    def test_equality_defers_on_a_different_type(self):
        assert DocumentVectors(TWO_BY_TWO).__eq__("text") is NotImplemented
        assert DocumentVectors(TWO_BY_TWO) != "text"

    def test_equal_tables_hash_alike(self):
        assert hash(DocumentVectors(TWO_BY_TWO)) == hash(
            DocumentVectors(TWO_BY_TWO.copy())
        )


class RecordingEmbedder(DocumentEmbedder):
    """The smallest concrete embedder: one dimension, always one, calls recorded."""

    _calls: list[str] = PrivateAttr(default_factory=list)

    def fit(self, corpus: Sequence[str]) -> Self:
        self._tokenised(corpus)
        self._calls.append("fit")
        self._mark_fitted()
        return self

    def transform(self, texts: Sequence[str]) -> DocumentVectors:
        self._check_fitted()
        tokenised = self._tokenised(texts)
        self._calls.append("transform")
        return DocumentVectors(np.ones((tokenised.n_sentences, 1)))

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)


class TestDocumentEmbedderFrame:
    def test_the_frame_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            DocumentEmbedder()  # type: ignore[abstract]

    def test_is_a_sibling_of_the_word_embedder_and_not_a_transformer(self):
        assert not issubclass(DocumentEmbedder, Transformer)
        assert not issubclass(DocumentEmbedder, WordEmbedder)

    def test_fit_transform_fits_then_transforms_the_same_texts(self):
        embedder = RecordingEmbedder()
        vectors = embedder.fit_transform(THREE_DOCUMENTS)

        assert embedder.calls == ("fit", "transform")
        assert vectors.n_documents == len(THREE_DOCUMENTS)

    def test_transform_before_fit_raises_not_fitted(self):
        with pytest.raises(NotFittedError):
            RecordingEmbedder().transform(THREE_DOCUMENTS)

    def test_fit_refuses_a_single_string(self):
        with pytest.raises(InvalidValuesError):
            RecordingEmbedder().fit("the cat sat")  # type: ignore[arg-type]

    def test_transform_refuses_a_single_string(self):
        embedder = RecordingEmbedder().fit(THREE_DOCUMENTS)

        with pytest.raises(InvalidValuesError):
            embedder.transform("the cat sat")  # type: ignore[arg-type]

    def test_transform_refuses_a_blank_batch(self):
        embedder = RecordingEmbedder().fit(THREE_DOCUMENTS)

        with pytest.raises(EmptyValuesError):
            embedder.transform(["", "   "])

    def test_transform_refuses_a_non_string(self):
        embedder = RecordingEmbedder().fit(THREE_DOCUMENTS)

        with pytest.raises(InvalidValuesError):
            embedder.transform(["the cat", 3])  # type: ignore[list-item]

    def test_the_pre_tokenizer_decides_where_the_words_are(self):
        embedder = RecordingEmbedder().fit(THREE_DOCUMENTS)

        assert embedder._tokenised(["a b c", "d"]).sentences == (
            ("a", "b", "c"),
            ("d",),
        )

    @pytest.mark.parametrize(
        "keywords", [{"minimum_count": 0}, {"minimum_count": -1}, {"min_count": 2}]
    )
    def test_the_frame_fields_are_validated_at_construction(self, keywords):
        with pytest.raises(ValidationError):
            RecordingEmbedder(**keywords)

    def test_a_real_embedder_keeps_the_same_contract(self):
        vectors = BagOfWords().fit_transform(THREE_DOCUMENTS)

        assert isinstance(vectors, DocumentVectors)
        assert vectors.n_documents == 3
