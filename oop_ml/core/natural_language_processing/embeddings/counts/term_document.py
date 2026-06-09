"""Which words each document uses, and how much each use should count.

The oldest embedding
--------------------
Before a word was a point, a document was a row of counts: one column per
word in the vocabulary, the number of times the document used it. Salton's
vector space model (1975) is that table, and two things are still read off it.
Read a *column* and you have a document as a vector over words, which is the
bag of words. Read a *row* and you have a word as a vector over documents,
which is where latent semantic analysis starts. Both families in this package
begin from this one matrix, so it is built once, here, with its vocabulary
attached so a row can never be mistaken for a different word.

Why raw counts are the wrong weight
-----------------------------------
A word used ten times in a document is more about that document than a word
used once, so the count matters; but ``the`` is used ten times in every
document and says nothing about any of them. Term frequency times inverse
document frequency (Sparck Jones, 1972) keeps the first fact and removes the
second: each count is multiplied by a weight that falls with the number of
documents the word appears in. The form here is the smoothed one that
scikit-learn uses, ``idf = log((1 + n_documents) / (1 + document_frequency)) +
1``, which behaves as though one extra document contained every word, so a word
in every document gets weight one rather than zero and a word in no document
cannot divide by zero.

Worked, on three documents
--------------------------
``the cat sat``, ``the dog sat``, ``a cat`` -- ``the`` is in two of three
documents, ``cat`` in two, ``sat`` in two, ``dog`` in one, ``a`` in one. With
the smoothing, ``idf(the) = log(4 / 3) + 1 = 1.2877`` and
``idf(dog) = log(4 / 2) + 1 = 1.6931``: the word in one document counts a
third more than the word in two, and neither is zero. Under raw counts the two
would count alike.

The matrix is dense. Real collections make it sparse and enormous, and the
usual repair is a sparse matrix; here vocabularies are small enough that a
dense ``(n_terms, n_documents)`` array is the honest, readable choice.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import DTypeLike

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.natural_language_processing.embeddings.embedder import (
    TokenisedCorpus,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray, array_for_protocol


class TermWeighting(StrEnum):
    """How much one use of a word in a document counts."""

    COUNT = "count"
    """One, per use: the raw bag of words."""

    TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY = "tf_idf"
    """The count times the smoothed inverse document frequency of the word."""


def inverse_document_frequencies(
    document_frequencies: FloatArray, n_documents: int
) -> FloatArray:
    """``log((1 + n_documents) / (1 + document_frequency)) + 1`` per term.

    Raises
    ------
    InvalidValuesError
        If ``n_documents`` is below one or a frequency is negative.
    """
    if n_documents < 1:
        raise InvalidValuesError(
            f"inverse document frequency needs at least one document, got {n_documents}"
        )
    frequencies = np.asarray(document_frequencies, dtype=np.float64)
    if bool(np.any(frequencies < 0.0)):
        raise InvalidValuesError("a document frequency is a count, so non-negative")
    return np.log((1.0 + n_documents) / (1.0 + frequencies)) + 1.0


class TermDocumentMatrix:
    """One row per word, one column per document, weighted as asked.

    Parameters
    ----------
    vocabulary:
        Which word each row belongs to; ids are positions.
    values:
        ``(n_terms, n_documents)``, finite and non-negative. Copied and frozen.
    weighting:
        How the counts were weighted.
    document_frequencies:
        How many documents each term appears in, one per row, kept so a new
        document can be weighted the same way.

    Raises
    ------
    InvalidValuesError
        If the values are not a finite non-negative matrix, or the document
        frequencies are not one non-negative count per term.
    ShapeMismatchError
        If the rows do not match the vocabulary.
    """

    __slots__ = ("_document_frequencies", "_values", "_vocabulary", "_weighting")

    def __init__(
        self,
        vocabulary: Vocabulary,
        values: FloatArray,
        weighting: TermWeighting,
        document_frequencies: FloatArray,
    ) -> None:
        try:
            as_array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError(
                "a term-document matrix must be numeric"
            ) from error

        if as_array.ndim != 2:
            raise InvalidValuesError(
                f"a term-document matrix is (n_terms, n_documents), got shape "
                f"{as_array.shape}"
            )

        if as_array.shape[0] != vocabulary.n_tokens:
            raise ShapeMismatchError(
                f"the matrix has {as_array.shape[0]} rows for a vocabulary of "
                f"{vocabulary.n_tokens} terms"
            )

        if not np.all(np.isfinite(as_array)) or bool(np.any(as_array < 0.0)):
            raise InvalidValuesError("term weights must be finite and non-negative")

        frequencies = np.asarray(document_frequencies, dtype=np.float64)
        if frequencies.shape != (vocabulary.n_tokens,) or bool(
            np.any(frequencies < 0.0)
        ):
            raise InvalidValuesError(
                "document frequencies are one non-negative count per term"
            )

        frozen = as_array.copy()
        frozen.setflags(write=False)
        frozen_frequencies = frequencies.copy()
        frozen_frequencies.setflags(write=False)

        self._vocabulary = vocabulary
        self._values = frozen
        self._weighting = weighting
        self._document_frequencies = frozen_frequencies

    @classmethod
    def from_tokenised(
        cls,
        tokenised: TokenisedCorpus,
        vocabulary: Vocabulary,
        weighting: TermWeighting = TermWeighting.COUNT,
    ) -> TermDocumentMatrix:
        """Count every word of ``vocabulary`` in every sentence of the corpus.

        A sentence is a document here. Words outside the vocabulary are
        ignored, so a document of only rare words is a column of zeros rather
        than an error.
        """
        counts = np.zeros((vocabulary.n_tokens, tokenised.n_sentences))
        for column, sentence in enumerate(tokenised.sentences):
            for word in sentence:
                if word in vocabulary:
                    counts[vocabulary.id_of(word), column] += 1.0

        document_frequencies = (counts > 0.0).sum(axis=1).astype(np.float64)
        if weighting is TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY:
            values = (
                counts
                * inverse_document_frequencies(
                    document_frequencies, tokenised.n_sentences
                )[:, None]
            )
        else:
            values = counts

        return cls(vocabulary, values, weighting, document_frequencies)

    @property
    def vocabulary(self) -> Vocabulary:
        """Which word each row belongs to."""
        return self._vocabulary

    @property
    def values(self) -> FloatArray:
        """``(n_terms, n_documents)``, frozen."""
        return self._values

    @property
    def weighting(self) -> TermWeighting:
        """How the counts were weighted."""
        return self._weighting

    @property
    def document_frequencies(self) -> FloatArray:
        """How many documents each term appears in, frozen, one per row."""
        return self._document_frequencies

    @property
    def inverse_document_frequencies(self) -> FloatArray:
        """The smoothed inverse document frequency of each term."""
        return inverse_document_frequencies(
            self._document_frequencies, self.n_documents
        )

    @property
    def n_terms(self) -> int:
        """How many words have a row."""
        return int(self._values.shape[0])

    @property
    def n_documents(self) -> int:
        """How many documents have a column."""
        return int(self._values.shape[1])

    def weight_of(self, word: str, document: int) -> float:
        """The weighted count of ``word`` in document ``document``.

        Raises
        ------
        UnknownTokenError
            If the word is not in the vocabulary.
        InvalidValuesError
            If the document index is out of range.
        """
        if not 0 <= document < self.n_documents:
            raise InvalidValuesError(
                f"document {document} is outside the {self.n_documents} documents"
            )
        return float(self._values[self._vocabulary.id_of(word), document])

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __eq__(self, other: object) -> bool:
        """A verdict: the same terms, the same weighting, the same numbers."""
        if not isinstance(other, TermDocumentMatrix):
            return NotImplemented
        return (
            self._vocabulary == other._vocabulary
            and self._weighting is other._weighting
            and bool(np.array_equal(self._values, other._values))
        )

    def __hash__(self) -> int:
        return hash((self._vocabulary, self._weighting, self._values.tobytes()))

    def __repr__(self) -> str:
        return (
            f"TermDocumentMatrix(n_terms={self.n_terms}, "
            f"n_documents={self.n_documents}, weighting={self._weighting.value!r})"
        )
