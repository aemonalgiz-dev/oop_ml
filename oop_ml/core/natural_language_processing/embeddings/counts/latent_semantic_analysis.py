"""Latent semantic analysis: words and documents as points in one low-rank space.

The idea
--------
A term-document matrix says which words each document uses. Read literally it
is too sharp: two documents about the same thing that happen to use different
words share no column, and a word never used in a document contributes nothing
to it, however much its synonyms did. Deerwester, Dumais, Furnas, Landauer and
Harshman (1990) proposed replacing the matrix by its best rank-``k``
approximation. Writing ``X = U S V^T`` and keeping the ``k`` largest singular
values, ``U_k S_k V_k^T`` is the closest rank-``k`` matrix to ``X`` in the
least-squares sense (Eckart and Young, 1936), and the approximation is where
the "latent" structure lives: a word's row in ``U_k S_k`` is a point in a
``k``-dimensional space, a document's row in ``V_k S_k`` is a point in the same
space, and two words that were used in the same *kind* of document sit near
each other there even if they were never used in the same document.

Why the rows are scaled by the singular values
----------------------------------------------
``U_k`` alone has orthonormal columns, so every direction counts alike;
scaling by ``S_k`` weights each direction by how much of the matrix it
explains. The word-word inner products then reproduce those of the
approximation exactly: ``(U_k S_k)(U_k S_k)^T = U_k S_k V_k^T V_k S_k U_k^T``,
which is ``X_k X_k^T``. The same holds for documents through ``V_k S_k``. Both
tables here carry that scaling, as the paper's do.

Signs, and the rule that fixes them
-----------------------------------
A singular value decomposition determines each pair of singular vectors only
up to a joint sign: flipping column ``j`` of ``U`` together with column ``j``
of ``V`` leaves ``U S V^T`` unchanged, and LAPACK returns whichever sign its
arithmetic reached. So that a refit gives the same numbers rather than the
same numbers up to sign, :class:`TruncatedSingularValueDecomposition` makes the
entry of largest magnitude in each left singular vector positive, and flips
the matching right vector with it. That rule is arbitrary -- it is the one
scikit-learn's ``svd_flip`` applies -- and it is deterministic on exact ties
(the lowest row wins), but a tie in exact arithmetic is seldom one in float,
so nothing downstream may depend on which of two equal-magnitude entries won.

Worked, on four documents
-------------------------
``the cat sat``, ``the cat ran``, ``the boat sailed``, ``the boat sank``. The
document-document inner products ``X^T X`` are 3 on the diagonal, 2 between
the two cat documents and between the two boat documents, and 1 everywhere
else, because every document shares ``the``. Its eigenvectors are the all-ones
direction with eigenvalue ``3 + 2 + 1 + 1 = 7``, the cat-against-boat contrast
with eigenvalue ``3 + 2 - 1 - 1 = 3``, and two within-topic contrasts with
eigenvalue 1 each, so the singular values are ``sqrt(7) = 2.6458``,
``sqrt(3) = 1.7321``, 1 and 1.

At ``k = 2`` the word vectors ``U_k S_k`` are: ``the`` at ``(2, 0)``, ``cat``
at ``(1, 1)``, ``boat`` at ``(1, -1)``, ``sat`` and ``ran`` at ``(0.5, 0.5)``,
``sailed`` and ``sank`` at ``(0.5, -0.5)``, up to the sign of the second
component, which the tie between ``cat`` and ``boat`` leaves to the last bit of
the arithmetic. Every document sits at ``sqrt(7) / 2 = 1.3229`` on the first
component -- the shared direction, positive for all of them -- and at
``+-sqrt(3) / 2 = +-0.8660`` on the second, by topic. The variance shares are
``7 / 12 = 0.5833`` and ``3 / 12 = 0.25``: the denominator is the sum over
*all* four squared singular values, so a truncated fit reports what it kept
rather than claiming everything.

The first component never separates topics
------------------------------------------
That is worth stating because it is the natural expectation and it is wrong.
A term-document matrix is non-negative, so ``X X^T`` is too, and the leading
eigenvector of a non-negative matrix has entries of one sign (Perron). Every
document therefore has the same sign on the first component, which measures
how much of the corpus's common vocabulary it uses. Topics can only show from
the second component on. On the twenty-four document fixture in the specs the
second component's document coordinates are one sign for every cooking
document and the other for every sailing one.

Folding in a new document
-------------------------
A new document ``x``, a column of weighted counts over the fitted vocabulary,
is placed in the fitted space without refitting: Deerwester's fold-in
``q = S_k^-1 U_k^T x`` gives its coordinates in ``V``'s frame, and scaling
back by ``S_k`` puts it in the frame the document vectors here use, so
:meth:`LatentSemanticAnalysis.transform` returns ``U_k^T x``. Applied to the
training documents that reproduces ``V_k S_k`` (measured, to 3.1e-14 on the
specs' two-topic fixture), which is the agreement test. The weighting uses the
*fitted* inverse document frequencies, never ones recomputed from the new
texts, since a new document's weights must be comparable to the old ones.

What is dense here
------------------
The term-document matrix is dense and the decomposition is a full
``numpy.linalg.svd``, which is ``O(min(m, n) m n)``. Real collections use a
sparse matrix and a randomised or Lanczos truncated decomposition that only
ever touches the ``k`` leading triples; both are optimisations of this
computation rather than different ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import (
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.natural_language_processing.embeddings.counts.term_document import (
    TermDocumentMatrix,
    TermWeighting,
)
from oop_ml.core.natural_language_processing.embeddings.documents.embedder import (
    DocumentVectors,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import (
    TokenisedCorpus,
    WordEmbedder,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray


class TruncatedSingularValueDecomposition:
    """The ``k`` leading singular triples of a matrix, with the signs settled.

    Parameters
    ----------
    left_vectors:
        ``U_k``, ``(n_rows, k)``. Copied and frozen.
    singular_values:
        ``(k,)``, non-negative, non-increasing. Copied and frozen.
    right_vectors:
        ``V_k``, ``(n_columns, k)``. Copied and frozen.
    total_squared_singular_values:
        The sum of *every* squared singular value of the matrix, kept because
        after truncation the discarded ones are gone and the variance shares
        would otherwise sum to one whatever was kept.

    Raises
    ------
    InvalidValuesError
        If a block is not finite, the singular values are negative or not in
        descending order, or the total is below what the kept values sum to.
    ShapeMismatchError
        If the three blocks do not agree on ``k``.
    """

    __slots__ = (
        "_left_vectors",
        "_right_vectors",
        "_singular_values",
        "_total_squared_singular_values",
    )

    def __init__(
        self,
        left_vectors: FloatArray,
        singular_values: FloatArray,
        right_vectors: FloatArray,
        total_squared_singular_values: float,
    ) -> None:
        left = np.asarray(left_vectors, dtype=np.float64)
        values = np.asarray(singular_values, dtype=np.float64)
        right = np.asarray(right_vectors, dtype=np.float64)

        if left.ndim != 2 or right.ndim != 2 or values.ndim != 1:
            raise InvalidValuesError(
                "a decomposition is two matrices and a vector of singular values"
            )
        if left.shape[1] != values.shape[0] or right.shape[1] != values.shape[0]:
            raise ShapeMismatchError(
                f"the blocks disagree on the rank: left {left.shape[1]}, singular "
                f"values {values.shape[0]}, right {right.shape[1]}"
            )
        if values.shape[0] == 0:
            raise InvalidValuesError("a decomposition keeps at least one component")
        for block, name in ((left, "left"), (values, "singular"), (right, "right")):
            if not np.all(np.isfinite(block)):
                raise InvalidValuesError(f"the {name} vectors hold a non-finite value")
        if bool(np.any(values < 0.0)):
            raise InvalidValuesError("singular values are non-negative")
        if bool(np.any(np.diff(values) > 0.0)):
            raise InvalidValuesError("singular values are in descending order")

        kept_total = float((values**2).sum())
        total = float(total_squared_singular_values)
        if not np.isfinite(total) or total < kept_total * (1.0 - 1e-12):
            raise InvalidValuesError(
                f"the total squared singular value {total} is below the kept "
                f"{kept_total}"
            )

        frozen_left = left.copy()
        frozen_left.setflags(write=False)
        frozen_values = values.copy()
        frozen_values.setflags(write=False)
        frozen_right = right.copy()
        frozen_right.setflags(write=False)

        self._left_vectors = frozen_left
        self._singular_values = frozen_values
        self._right_vectors = frozen_right
        self._total_squared_singular_values = total

    @classmethod
    def of(
        cls, matrix: FloatArray, dimension: int
    ) -> TruncatedSingularValueDecomposition:
        """Decompose ``matrix`` and keep its ``dimension`` leading triples.

        The full decomposition is computed and truncated, and each left
        singular vector is flipped, together with its right vector, so that its
        entry of largest magnitude is positive; an exact tie goes to the
        lowest row.

        Raises
        ------
        InvalidValuesError
            If the matrix is not a finite two-dimensional array, or the
            dimension is below one.
        TooFewValuesError
            If ``dimension`` exceeds ``min(n_rows, n_columns)``, the most
            components the matrix has.
        """
        as_array = np.asarray(matrix, dtype=np.float64)
        if as_array.ndim != 2 or not np.all(np.isfinite(as_array)):
            raise InvalidValuesError(
                "a singular value decomposition needs a finite two-dimensional matrix"
            )
        if dimension < 1:
            raise InvalidValuesError(f"dimension must be at least 1, got {dimension}")

        largest_rank = min(as_array.shape)
        if dimension > largest_rank:
            raise TooFewValuesError(
                f"dimension={dimension} exceeds the {largest_rank} components a "
                f"{as_array.shape[0]} x {as_array.shape[1]} matrix has"
            )

        left, singular_values, right_transposed = np.linalg.svd(
            as_array, full_matrices=False
        )
        for column in range(dimension):
            largest = int(np.argmax(np.abs(left[:, column])))
            if left[largest, column] < 0.0:
                left[:, column] *= -1.0
                right_transposed[column, :] *= -1.0

        return cls(
            left[:, :dimension],
            singular_values[:dimension],
            right_transposed[:dimension, :].T,
            float((singular_values**2).sum()),
        )

    @property
    def left_vectors(self) -> FloatArray:
        """``U_k``, ``(n_rows, k)``, frozen. Orthonormal columns."""
        return self._left_vectors

    @property
    def singular_values(self) -> FloatArray:
        """The ``k`` kept singular values, descending, frozen."""
        return self._singular_values

    @property
    def right_vectors(self) -> FloatArray:
        """``V_k``, ``(n_columns, k)``, frozen. Orthonormal columns."""
        return self._right_vectors

    @property
    def dimension(self) -> int:
        """How many components were kept."""
        return int(self._singular_values.shape[0])

    @property
    def total_squared_singular_values(self) -> float:
        """The sum of every squared singular value, kept and discarded."""
        return self._total_squared_singular_values

    @property
    def variance_shares(self) -> tuple[float, ...]:
        """Each kept component's ``s^2`` over the total of every ``s^2``.

        Sums to one only at full rank; a truncated fit reports what it kept.
        A zero matrix has no variance to share, and every share is zero.
        """
        if self._total_squared_singular_values == 0.0:
            return tuple(0.0 for _ in self._singular_values)
        return tuple(
            float(value**2 / self._total_squared_singular_values)
            for value in self._singular_values
        )

    def reconstruction(self) -> FloatArray:
        """``U_k S_k V_k^T``: the closest rank-``k`` matrix to the original.

        Equal to the original, to rounding, when every component was kept.
        """
        return (self._left_vectors * self._singular_values) @ self._right_vectors.T

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TruncatedSingularValueDecomposition):
            return NotImplemented
        return (
            bool(np.array_equal(self._left_vectors, other._left_vectors))
            and bool(np.array_equal(self._singular_values, other._singular_values))
            and bool(np.array_equal(self._right_vectors, other._right_vectors))
            and self._total_squared_singular_values
            == other._total_squared_singular_values
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._left_vectors.tobytes(),
                self._singular_values.tobytes(),
                self._right_vectors.tobytes(),
                self._total_squared_singular_values,
            )
        )

    def __repr__(self) -> str:
        n_rows = self._left_vectors.shape[0]
        n_columns = self._right_vectors.shape[0]
        return (
            f"TruncatedSingularValueDecomposition(n_rows={n_rows}, "
            f"n_columns={n_columns}, dimension={self.dimension})"
        )


class LatentSemanticAnalysis(WordEmbedder):
    """Word and document vectors from a truncated term-document decomposition.

    Each text of the corpus is one document. The word vectors are
    ``U_k S_k`` and the document vectors ``V_k S_k``, both scaled by the
    singular values as the module docstring explains.

    Parameters
    ----------
    dimension:
        How many components to keep. At most ``min(n_terms, n_documents)``,
        which is known only at ``fit``.
    weighting:
        How a word's count in a document is weighted before the
        decomposition. Term frequency times inverse document frequency by
        default, so a word in every document counts least.
    """

    dimension: int = Field(default=50, ge=1)
    weighting: TermWeighting = TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY

    _embeddings: WordEmbeddings = PrivateAttr()
    _document_vectors: DocumentVectors = PrivateAttr()
    _decomposition: TruncatedSingularValueDecomposition = PrivateAttr()
    _term_document_matrix: TermDocumentMatrix = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Build the term-document matrix of ``corpus`` and decompose it.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If no word reaches ``minimum_count``, or ``dimension`` exceeds
            ``min(n_terms, n_documents)``, the most components the matrix has.
        """
        tokenised = self._tokenised(corpus)
        vocabulary = tokenised.vocabulary(self.minimum_count)
        term_document_matrix = TermDocumentMatrix.from_tokenised(
            tokenised, vocabulary, self.weighting
        )

        largest_rank = min(
            term_document_matrix.n_terms, term_document_matrix.n_documents
        )
        if self.dimension > largest_rank:
            raise TooFewValuesError(
                f"dimension={self.dimension} exceeds the {largest_rank} components "
                f"a matrix of {term_document_matrix.n_terms} terms by "
                f"{term_document_matrix.n_documents} documents has"
            )

        decomposition = TruncatedSingularValueDecomposition.of(
            term_document_matrix.values, self.dimension
        )
        word_table = decomposition.left_vectors * decomposition.singular_values
        document_table = decomposition.right_vectors * decomposition.singular_values

        self._embeddings = WordEmbeddings(vocabulary, word_table)
        self._document_vectors = DocumentVectors(document_table)
        self._decomposition = decomposition
        self._term_document_matrix = term_document_matrix
        self._mark_fitted()
        return self

    def transform(self, texts: Sequence[str]) -> DocumentVectors:
        """Place new documents in the fitted space without refitting.

        Each text is weighted over the fitted vocabulary with the fitted
        inverse document frequencies, then projected as ``U_k^T x``. Words
        outside the vocabulary are ignored, so a text of only unknown words
        gets the zero vector. On the training texts this reproduces
        :attr:`document_vectors`.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``texts`` is a single string or holds a non-string.
        EmptyValuesError
            If there are no texts, or no text holds a word.
        """
        self._check_fitted()
        tokenised = self._tokenised(texts)
        weighted = self._weighted_columns(tokenised)
        return DocumentVectors(weighted.T @ self._decomposition.left_vectors)

    @property
    def embeddings(self) -> WordEmbeddings:
        """The word vectors ``U_k S_k``, one row per word.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._embeddings

    @property
    def document_vectors(self) -> DocumentVectors:
        """The training documents' vectors ``V_k S_k``, one row per text.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._document_vectors

    @property
    def decomposition(self) -> TruncatedSingularValueDecomposition:
        """The kept singular triples themselves.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._decomposition

    @property
    def term_document_matrix(self) -> TermDocumentMatrix:
        """The weighted matrix that was decomposed.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._term_document_matrix

    @property
    def singular_values(self) -> FloatArray:
        """The ``dimension`` kept singular values, descending, frozen.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._decomposition.singular_values

    @property
    def variance_shares(self) -> tuple[float, ...]:
        """Each kept component's share of the total squared singular value.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._decomposition.variance_shares

    def _weighted_columns(self, tokenised: TokenisedCorpus) -> FloatArray:
        """``(n_terms, n_texts)`` of counts over the fitted vocabulary, weighted
        with the fitted inverse document frequencies where the fit was."""
        vocabulary: Vocabulary = self._embeddings.vocabulary
        counts = np.zeros((vocabulary.n_tokens, tokenised.n_sentences))
        for column, sentence in enumerate(tokenised.sentences):
            for word in sentence:
                if word in vocabulary:
                    counts[vocabulary.id_of(word), column] += 1.0

        if self.weighting is TermWeighting.TERM_FREQUENCY_INVERSE_DOCUMENT_FREQUENCY:
            return (
                counts
                * self._term_document_matrix.inverse_document_frequencies[:, None]
            )
        return counts
