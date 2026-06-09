"""Random indexing: a co-occurrence row, projected down by sparse random vectors.

The idea
--------
Give every word a fixed random *index vector*: ``dimension`` long, almost all
zeros, a few ``+1`` and as many ``-1`` scattered at random. Then read the
corpus once, and every time a word appears, add to its *context vector* the
index vectors of the words in the window around it. Kanerva, Kristoferson and
Holst (2000) proposed it and Sahlgren (2005) made the case for it; there is no
matrix to decompose and no objective to descend, just one accumulating pass.

Why it works
------------
Write the co-occurrence matrix ``C`` with a row per word and the index vectors
as the rows of a matrix ``R`` of shape ``(n_words, dimension)``. The context
vector of word ``w`` is the sum over its neighbours ``c`` of ``C[w, c] R[c]``,
which is row ``w`` of ``C R``. So random indexing is the co-occurrence matrix
multiplied by a random matrix: a random projection of each word's
co-occurrence row from ``n_words`` numbers down to ``dimension`` of them. The
Johnson-Lindenstrauss lemma says a random linear map into
``O(log n / epsilon^2)`` dimensions keeps every pairwise distance among ``n``
points within a factor ``1 +- epsilon``, and Achlioptas (2001) showed a sparse
map with entries in ``{-1, 0, +1}`` does as well as a Gaussian one. The
index vectors are that sparse map, and the geometry of the co-occurrence rows
survives the projection to the accuracy the lemma allows.

The intuition behind the lemma is that random vectors in a high-dimensional
space are nearly orthogonal. Two index vectors with four non-zeros each in
``d`` positions share a position ``16 / d`` times in expectation, each shared
position contributes ``+-1`` to a dot product whose denominator is 4, so the
expected absolute cosine is about ``4 / d``: 0.08 at ``d = 50``, 0.008 at
``d = 500``. Measured on 200 index vectors drawn with seed 0, the mean absolute
cosine is 0.0744 at 50 and 0.0077 at 500, and the pairs that are exactly
orthogonal are 71.8% and 96.9% of them. If the index vectors were exactly
orthogonal the context vectors would be the co-occurrence rows in a rotated
basis and nothing would be lost; they are nearly so, and a little is.

What is not here
----------------
No weighting of the counts. A word's context vector is a sum of raw
co-occurrences, so a word that appears beside everything -- ``the`` -- pulls
every context vector toward its own index vector, and two words that share
only their function-word company can look alike. Sahlgren removes frequent
words or weights by distance before summing; the harmonic weighting is
offered here and the removal is left to the pre-tokenizer. The specs' fixture
keeps function words to one per document for exactly this reason, and the
pointwise mutual information module is the count-based repair.

The identity, and the tie rule
------------------------------
The specs check ``C R`` against a direct sum over every position of every
sentence, because the identity above is the whole argument that this is a
projection and not a heuristic. The index vectors are drawn in vocabulary
order from one seeded generator: for each word, ``n_nonzero`` distinct
positions are chosen without replacement, the first half set to ``+1`` and the
second half to ``-1``. Nothing else is random, so a seed fixes the fit.

What is dense here
------------------
The co-occurrence matrix is ``(n_words, n_words)`` and dense, and the product
``C R`` costs ``O(n_words^2 dimension)``. The point of the method in practice
is that neither needs to exist: the reference implementations never form
``C``, and add each neighbour's sparse index vector into a context vector as
the corpus streams past, which is ``O(corpus length x window x n_nonzero)`` and
touches only the ``dimension``-wide context table. The one-pass form is the
one the specs test the matrix form against.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
from numpy.typing import DTypeLike
from pydantic import Field, PrivateAttr, model_validator

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import WordEmbedder
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    WordEmbeddings,
    WordVector,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray, array_for_protocol


class IndexVectors:
    """One fixed sparse ternary vector per word, row ``i`` for id ``i``.

    Not an embedding: these carry no meaning, only identity, and are what the
    context vectors are summed *from*.

    Parameters
    ----------
    vocabulary:
        Which word each row belongs to.
    table:
        ``(n_words, dimension)`` with entries in ``{-1, 0, +1}``, each row
        holding exactly ``n_nonzero`` non-zeros, half of each sign. Copied and
        frozen.
    n_nonzero:
        How many non-zeros each row holds. Even, at least two.

    Raises
    ------
    InvalidValuesError
        If the table is not two-dimensional, an entry is outside
        ``{-1, 0, +1}``, ``n_nonzero`` is odd or below two, or a row does not
        hold ``n_nonzero / 2`` of each sign.
    ShapeMismatchError
        If the rows do not match the vocabulary.
    """

    __slots__ = ("_n_nonzero", "_table", "_vocabulary")

    def __init__(
        self, vocabulary: Vocabulary, table: FloatArray, n_nonzero: int
    ) -> None:
        try:
            as_array = np.asarray(table, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError("index vectors must be numeric") from error

        if as_array.ndim != 2:
            raise InvalidValuesError(
                f"index vectors are (n_words, dimension), got shape {as_array.shape}"
            )
        if as_array.shape[0] != vocabulary.n_tokens:
            raise ShapeMismatchError(
                f"the table has {as_array.shape[0]} rows for a vocabulary of "
                f"{vocabulary.n_tokens} words"
            )
        if n_nonzero < 2 or n_nonzero % 2 != 0:
            raise InvalidValuesError(
                f"n_nonzero is even and at least two, got {n_nonzero}"
            )
        if not np.all(np.isin(as_array, (-1.0, 0.0, 1.0))):
            raise InvalidValuesError("index vector entries are -1, 0 or +1")

        positives = np.count_nonzero(as_array == 1.0, axis=1)
        negatives = np.count_nonzero(as_array == -1.0, axis=1)
        half = n_nonzero // 2
        if bool(np.any(positives != half)) or bool(np.any(negatives != half)):
            raise InvalidValuesError(
                f"every index vector holds {half} entries of +1 and {half} of -1"
            )

        frozen = as_array.copy()
        frozen.setflags(write=False)

        self._vocabulary = vocabulary
        self._table = frozen
        self._n_nonzero = int(n_nonzero)

    @classmethod
    def drawn(
        cls,
        vocabulary: Vocabulary,
        dimension: int,
        n_nonzero: int,
        generator: np.random.Generator,
    ) -> IndexVectors:
        """Draw one index vector per word, in vocabulary order.

        For each word ``n_nonzero`` distinct positions are drawn without
        replacement; the first half become ``+1`` and the second half ``-1``.

        Raises
        ------
        InvalidValuesError
            If ``n_nonzero`` is odd, below two, or above ``dimension``.
        """
        if n_nonzero < 2 or n_nonzero % 2 != 0:
            raise InvalidValuesError(
                f"n_nonzero is even and at least two, got {n_nonzero}"
            )
        if n_nonzero > dimension:
            raise InvalidValuesError(
                f"n_nonzero={n_nonzero} non-zeros cannot fit in dimension={dimension}"
            )

        table = np.zeros((vocabulary.n_tokens, dimension))
        half = n_nonzero // 2
        for row in range(vocabulary.n_tokens):
            positions = generator.choice(dimension, size=n_nonzero, replace=False)
            table[row, positions[:half]] = 1.0
            table[row, positions[half:]] = -1.0

        return cls(vocabulary, table, n_nonzero)

    @property
    def vocabulary(self) -> Vocabulary:
        """Which word each row belongs to."""
        return self._vocabulary

    @property
    def table(self) -> FloatArray:
        """``(n_words, dimension)``, frozen, entries in ``{-1, 0, +1}``."""
        return self._table

    @property
    def dimension(self) -> int:
        """How long each index vector is."""
        return int(self._table.shape[1])

    @property
    def n_words(self) -> int:
        """How many words have an index vector."""
        return int(self._table.shape[0])

    @property
    def n_nonzero(self) -> int:
        """How many non-zeros each index vector holds."""
        return self._n_nonzero

    def vector_of(self, word: str) -> WordVector:
        """The index vector for ``word``.

        Raises
        ------
        UnknownTokenError
            If the word is not in the vocabulary.
        """
        return WordVector(word, self._table[self._vocabulary.id_of(word)])

    def __iter__(self) -> Iterator[WordVector]:
        """Iterate the index vectors in id order."""
        for token_id, word in enumerate(self._vocabulary):
            yield WordVector(word, self._table[token_id])

    def __len__(self) -> int:
        return self.n_words

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._table, dtype, copy)

    def __eq__(self, other: object) -> bool:
        """A verdict: the same words with the same index vectors."""
        if not isinstance(other, IndexVectors):
            return NotImplemented
        return (
            self._vocabulary == other._vocabulary
            and self._n_nonzero == other._n_nonzero
            and bool(np.array_equal(self._table, other._table))
        )

    def __hash__(self) -> int:
        return hash((self._vocabulary, self._n_nonzero, self._table.tobytes()))

    def __repr__(self) -> str:
        return (
            f"IndexVectors(n_words={self.n_words}, dimension={self.dimension}, "
            f"n_nonzero={self._n_nonzero})"
        )


class RandomIndexing(WordEmbedder):
    """Context vectors summed from random index vectors, in one pass.

    Kanerva, Kristoferson and Holst (2000); Sahlgren (2005). A word's vector
    is the weighted sum, over every occurrence, of its window neighbours'
    index vectors, which is its co-occurrence row projected by the index
    vector matrix.

    Parameters
    ----------
    dimension:
        How long every vector is. At least ``n_nonzero``.
    n_nonzero:
        How many non-zeros each index vector holds, half ``+1`` and half
        ``-1``. Even, at least two.
    window:
        How many positions on each side of a word count as its context.
    weighting:
        How a neighbour's distance weights its count. Uniform by default.
    random_seed:
        Seeds the draw of the index vectors, which is the only randomness.
    """

    dimension: int = Field(default=50, ge=1)
    n_nonzero: int = Field(default=4, ge=2)
    window: int = Field(default=5, ge=1)
    weighting: ContextWeighting = ContextWeighting.UNIFORM
    random_seed: int | None = None

    _embeddings: WordEmbeddings = PrivateAttr()
    _index_vectors: IndexVectors = PrivateAttr()
    _cooccurrence: CooccurrenceMatrix = PrivateAttr()

    @model_validator(mode="after")
    def _check_the_nonzeros_are_even_and_fit(self) -> Self:
        if self.n_nonzero % 2 != 0:
            raise ValueError(
                f"n_nonzero={self.n_nonzero} is odd; an index vector holds as many "
                f"+1 entries as -1 entries"
            )
        if self.n_nonzero > self.dimension:
            raise ValueError(
                f"n_nonzero={self.n_nonzero} non-zeros cannot fit in "
                f"dimension={self.dimension}"
            )
        return self

    def fit(self, corpus: Sequence[str]) -> Self:
        """Draw the index vectors and sum them over every window in ``corpus``.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If no word reaches ``minimum_count``.
        """
        tokenised = self._tokenised(corpus)
        vocabulary = tokenised.vocabulary(self.minimum_count)
        cooccurrence = CooccurrenceMatrix.from_id_sequences(
            vocabulary,
            tokenised.id_sequences(vocabulary),
            self.window,
            self.weighting,
        )
        generator = np.random.default_rng(self.random_seed)
        index_vectors = IndexVectors.drawn(
            vocabulary, self.dimension, self.n_nonzero, generator
        )
        context_vectors = cooccurrence.counts @ index_vectors.table

        self._embeddings = WordEmbeddings(vocabulary, context_vectors)
        self._index_vectors = index_vectors
        self._cooccurrence = cooccurrence
        self._mark_fitted()
        return self

    @property
    def embeddings(self) -> WordEmbeddings:
        """The context vectors, one row per word.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._embeddings

    @property
    def index_vectors(self) -> IndexVectors:
        """The random index vectors the context vectors were summed from.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._index_vectors

    @property
    def cooccurrence(self) -> CooccurrenceMatrix:
        """The counts that weighted the sum.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._cooccurrence
