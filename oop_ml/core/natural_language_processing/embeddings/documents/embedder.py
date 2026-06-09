"""A document as a point, and a corpus's worth of them as a table you can ask.

From words to documents
-----------------------
Every embedder in the sibling packages answers a
:class:`~oop_ml.core.natural_language_processing.embeddings.vectors.WordEmbeddings`:
one vector per word, addressed by the word. This package answers one vector per
*text*, and it is a different question with two families of answer. The bag of
words never asks what a word means and counts which words a document uses, so
two documents are near when they share vocabulary. Pooling starts from word
vectors someone else learned and combines the vectors of a document's words, so
two documents are near when their words are near, whether or not any word is
shared. Both hand back a :class:`DocumentVectors`, which is a frozen
``(n_documents, dimension)`` table that knows how many documents it holds and
can compare any two of them.

Why a sibling of ``WordEmbedder`` and not a ``Transformer``
-----------------------------------------------------------
A :class:`~oop_ml.core.base.estimator.Transformer` promises that ``transform``
answers in its input type: a standardiser takes features and answers features.
A document embedder takes texts and answers vectors, so the type changes across
the call, which is exactly what that signature says cannot happen. And a
:class:`~oop_ml.core.natural_language_processing.embeddings.embedder.WordEmbedder`
has no ``transform`` at all, because its fit produces a closed table with a row
per word, and a word outside the table has nothing to be asked about. A
document is different: a new text can always be given a vector from whatever
words it shares with the fit, so ``transform`` on unseen texts is the whole
point of the class. What the two embedders share -- a pre-tokenizer, a minimum
count and one line that tokenises through them -- is repeated here rather than
inherited, because a base holding two fields and one method would live in the
foundation module and would exist only to hold them.

The rule ``Transformer`` states holds here all the same. ``fit`` learns from
the training corpus alone -- a vocabulary, the inverse document frequencies,
the word probabilities, a principal direction -- and ``transform`` applies
those unchanged to whatever texts it is handed. A new text's counts are
weighted by the *training* document frequencies and never by frequencies
re-estimated on the texts being transformed, so a held-out set cannot leak
into its own representation. ``fit_transform`` exists for the training corpus
and for nothing else.

Why a single string is refused
------------------------------
A Python string is a sequence of one-character strings, so
``transform("the cat sat")`` would iterate as eleven texts of one character
each and answer eleven vectors, most of them zero, without raising anything.
:class:`~oop_ml.core.natural_language_processing.tokenization.corpus.Corpus`
refuses a bare string at the text boundary, and
:meth:`~oop_ml.core.natural_language_processing.embeddings.embedder.TokenisedCorpus.from_texts`
routes through it, so ``fit`` and ``transform`` meet the same refusal:
:class:`~oop_ml.core.exceptions.InvalidValuesError`, and the remedy is to
write ``transform(["the cat sat"])``.

Why similarity is cosine
------------------------
The length of a document vector mostly reflects how many words the document
had: a bag of words grows with every word counted, and a pooled vector grows
with the words that happened to have long vectors. The direction is where
what the document is about was put, so every comparison here is the angle. A
zero vector -- a document none of whose words the fit knows -- has no
direction, and is refused rather than given a similarity of zero it did not
earn.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
from numpy.typing import DTypeLike

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    UndefinedMetricError,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import (
    CorpusEmbedder,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    checked_vector,
    cosine_similarity,
)
from oop_ml.core.types import FloatArray, array_for_protocol


class DocumentVectors:
    """One vector per document, row ``i`` for the ``i``-th text handed in.

    Parameters
    ----------
    vectors:
        ``(n_documents, dimension)``, finite. Copied and frozen. A row may be
        entirely zero, which is how a document with no known word comes back;
        it is a legitimate answer, though one that cannot be compared.

    Raises
    ------
    InvalidValuesError
        If the table is not numeric, not two-dimensional, has no columns, or
        holds a non-finite value.
    EmptyValuesError
        If the table has no rows.
    """

    __slots__ = ("_norms", "_vectors")

    def __init__(self, vectors: FloatArray) -> None:
        try:
            as_array = np.asarray(vectors, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError("document vectors must be numeric") from error

        if as_array.ndim != 2:
            raise InvalidValuesError(
                f"document vectors are (n_documents, dimension), got shape "
                f"{as_array.shape}; one document is a table of one row"
            )

        if as_array.shape[0] == 0:
            raise EmptyValuesError("document vectors need at least one document")

        if as_array.shape[1] == 0:
            raise InvalidValuesError("a document vector needs at least one dimension")

        if not np.all(np.isfinite(as_array)):
            raise InvalidValuesError("document vectors hold a non-finite value")

        frozen = as_array.copy()
        frozen.setflags(write=False)
        self._vectors = frozen
        self._norms = np.linalg.norm(frozen, axis=1)

    @property
    def vectors(self) -> FloatArray:
        """``(n_documents, dimension)``, frozen."""
        return self._vectors

    @property
    def n_documents(self) -> int:
        """How many documents have a vector."""
        return int(self._vectors.shape[0])

    @property
    def dimension(self) -> int:
        """How many numbers each vector holds."""
        return int(self._vectors.shape[1])

    def vector_of(self, position: int) -> FloatArray:
        """The vector of the document at ``position``, counting from zero.

        A position is a position, so a negative one is refused rather than
        read from the end.

        Raises
        ------
        InvalidValuesError
            If ``position`` is not in ``0 .. n_documents - 1``.
        """
        return self._vectors[self._checked_position(position)]

    def similarity(self, first_position: int, second_position: int) -> float:
        """Cosine similarity between two documents' vectors.

        Raises
        ------
        InvalidValuesError
            If either position is out of range.
        UndefinedMetricError
            If either vector is zero, since it then has no direction.
        """
        return cosine_similarity(
            self.vector_of(first_position), self.vector_of(second_position)
        )

    def most_similar(self, position: int, n_results: int = 10) -> tuple[int, ...]:
        """The positions of the ``n_results`` documents nearest to one, itself
        excluded, most similar first.

        A tuple of positions rather than an object, because the positions are
        like items and the table itself answers any further question about
        them. Ties go to the lower position.

        Raises
        ------
        InvalidValuesError
            If the position is not a document's, or ``n_results`` is below one.
        UndefinedMetricError
            If the document's vector is zero.
        """
        return self.similar_to_vector(
            self.vector_of(position), n_results, excluding=(position,)
        )

    def similar_to_vector(
        self,
        values: FloatArray,
        n_results: int = 10,
        excluding: Sequence[int] = (),
    ) -> tuple[int, ...]:
        """The positions of the ``n_results`` documents nearest to a vector.

        Cosine, most similar first. Documents whose vector is zero are skipped,
        since their similarity to anything is undefined. The route
        :meth:`most_similar` takes, and the one an inferred vector is compared
        to the fitted documents by.

        Raises
        ------
        InvalidValuesError
            If ``n_results`` is below one, ``values`` is not a finite vector,
            or an excluded position is not a document's.
        ShapeMismatchError
            If ``values`` is not of this table's dimension.
        UndefinedMetricError
            If ``values`` is the zero vector.
        """
        if n_results < 1:
            raise InvalidValuesError(f"n_results must be at least 1, got {n_results}")

        query = checked_vector(values, "the query vector")
        if query.shape[0] != self.dimension:
            raise ShapeMismatchError(
                f"the query has {query.shape[0]} dimensions against a table of "
                f"{self.dimension}"
            )

        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            raise UndefinedMetricError(
                "the zero vector has no direction, so nothing is similar to it"
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            similarities = (self._vectors @ query) / (self._norms * query_norm)
        similarities = np.where(self._norms == 0.0, -np.inf, similarities)
        for excluded in excluding:
            similarities[self._checked_position(excluded)] = -np.inf

        order = np.argsort(-similarities, kind="stable")
        results: list[int] = []
        for candidate in order:
            if len(results) == n_results or not np.isfinite(similarities[candidate]):
                break
            results.append(int(candidate))
        return tuple(results)

    def _checked_position(self, position: int) -> int:
        if not 0 <= position < self.n_documents:
            raise InvalidValuesError(
                f"no document has position {position}; positions run from 0 to "
                f"{self.n_documents - 1}"
            )
        return int(position)

    def __iter__(self) -> Iterator[FloatArray]:
        """Iterate the document vectors in document order."""
        return iter(self._vectors)

    def __len__(self) -> int:
        return self.n_documents

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._vectors, dtype, copy)

    def __eq__(self, other: object) -> bool:
        """A verdict rather than an element-wise answer: the same table."""
        if not isinstance(other, DocumentVectors):
            return NotImplemented
        return bool(np.array_equal(self._vectors, other._vectors))

    def __hash__(self) -> int:
        return hash((self._vectors.shape, self._vectors.tobytes()))

    def __repr__(self) -> str:
        return (
            f"DocumentVectors(n_documents={self.n_documents}, "
            f"dimension={self.dimension})"
        )


class DocumentEmbedder(CorpusEmbedder):
    """Learns from a corpus how to give any text a vector.

    Construction configures the rule for where the words are and how rare a
    word may be; ``fit`` learns whatever the technique needs from the training
    corpus; ``transform`` gives new texts vectors from that alone.

    Parameters
    ----------
    pre_tokenizer:
        Decides where the words are in each text.
    minimum_count:
        A word seen fewer times than this across the training corpus is not
        given a place in the vocabulary. Read by the embedders that build a
        vocabulary from the corpus; one that is handed its vocabulary refuses
        any value but the default.
    """

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn from ``corpus`` and return ``self``.

        Implementations should tokenise through :meth:`_tokenised`, compute
        into locals, assign the private attributes at the end, then call
        ``self._mark_fitted()``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        """

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> DocumentVectors:
        """One vector per text, from what ``fit`` learned and nothing else.

        Must call ``_check_fitted()`` first. A text with no word the fit knows
        answers the zero vector.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``texts`` is a single string or holds a non-string.
        EmptyValuesError
            If ``texts`` is empty, blank, or yields no words at all.
        """

    def fit_transform(self, corpus: Sequence[str]) -> DocumentVectors:
        """Fit on ``corpus`` and immediately give its texts vectors.

        For the training corpus only. On held-out texts it re-learns the
        statistics from them, which is the leak the fit/transform split exists
        to prevent.
        """
        return self.fit(corpus).transform(corpus)
