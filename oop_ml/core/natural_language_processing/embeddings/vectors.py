"""A word as a point, and a vocabulary's worth of them as a table you can ask.

What an embedding is
--------------------
A tokenizer ends in a lookup table from token to id, and an id is a name: it
says which row, and nothing about what the row is near. An embedding gives each
id a vector, so that the geometry of the vectors carries what the ids alone
cannot -- ``king`` sits nearer ``queen`` than ``kettle``, and the direction from
``man`` to ``woman`` is roughly the direction from ``king`` to ``queen``. Every
technique in this package is a different answer to how those vectors should be
learned; every one of them hands back the same object, a :class:`WordEmbeddings`,
which is a
:class:`~oop_ml.core.natural_language_processing.tokenization.vocabulary.Vocabulary`
paired with a table whose row ``i`` is the vector for token ``i``.

That pairing is the point of the class. A bare ``(n_words, dimension)`` array
has rows addressed by position, and the position means nothing without the
vocabulary that assigned it. Holding the two together means a caller cannot ask
for row 17 and get a different word than the one the fit meant, and a network's
:class:`~oop_ml.core.network.embedding.Embedding` layer can be seeded from the
table knowing that its positions are the vocabulary's ids.

Why similarity is cosine
------------------------
The length of a learned vector mostly reflects how often the word was seen
rather than what it means: a frequent word gets more updates and drifts further
from its start. Cosine similarity, the angle between two vectors, ignores that
length and reads only the direction, which is where the meaning was put. Every
comparison here is cosine, and a zero vector, which has no direction, is
refused rather than given a similarity of zero it did not earn.

Analogies by arithmetic
-----------------------
Mikolov et al. (2013) noticed that ``king - man + woman`` lands near ``queen``.
:meth:`WordEmbeddings.analogy` is that arithmetic, in the form Levy and Goldberg
(2014) call 3CosAdd: unit vectors are added and subtracted, the result is
compared to every word by cosine, and the words that were in the question are
excluded from the answer -- without that exclusion the top answer to
``king - man + woman`` is usually ``king``, since it is the largest term.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import DTypeLike

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    UndefinedMetricError,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray, array_for_protocol


def checked_vector(values: object, description: str) -> FloatArray:
    """A finite one-dimensional float array, or a refusal that says why.

    Raises
    ------
    InvalidValuesError
        If ``values`` is not numeric, not one-dimensional, or not finite.
    """
    try:
        as_array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidValuesError(f"{description} must be numeric") from error

    if as_array.ndim != 1:
        raise InvalidValuesError(
            f"{description} must be one vector, so one dimension; got shape "
            f"{as_array.shape}"
        )

    if not np.all(np.isfinite(as_array)):
        raise InvalidValuesError(f"{description} holds a non-finite value")

    return as_array


def cosine_similarity(first: FloatArray, second: FloatArray) -> float:
    """The cosine of the angle between two vectors, in ``[-1, 1]``.

    Raises
    ------
    ShapeMismatchError
        If the vectors differ in length.
    UndefinedMetricError
        If either vector has zero length, since it then has no direction.
    """
    if first.shape != second.shape:
        raise ShapeMismatchError(
            f"cosine similarity needs two vectors of one length, got "
            f"{first.shape[0]} and {second.shape[0]}"
        )

    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise UndefinedMetricError(
            "cosine similarity is undefined for a zero vector, which has no direction"
        )

    return float(np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0))


class WordVector:
    """One word's vector, bound to the word.

    Parameters
    ----------
    word:
        The word. Non-empty.
    values:
        Its vector, finite and one-dimensional. Copied and frozen.

    Raises
    ------
    EmptyValuesError
        If ``word`` is empty.
    InvalidValuesError
        If ``values`` is not a finite one-dimensional numeric vector.
    """

    __slots__ = ("_values", "_word")

    def __init__(self, word: str, values: FloatArray) -> None:
        if not isinstance(word, str) or not word:
            raise EmptyValuesError("a word vector needs a non-empty word")

        frozen = checked_vector(values, f"the vector for {word!r}").copy()
        frozen.setflags(write=False)

        self._word = word
        self._values = frozen

    @property
    def word(self) -> str:
        """The word this is the vector of."""
        return self._word

    @property
    def values(self) -> FloatArray:
        """The vector, frozen."""
        return self._values

    @property
    def dimension(self) -> int:
        """How many numbers the vector holds."""
        return int(self._values.shape[0])

    def cosine_similarity_to(self, other: WordVector) -> float:
        """The cosine similarity between this word's vector and another's.

        Raises
        ------
        ShapeMismatchError
            If the two vectors differ in dimension.
        UndefinedMetricError
            If either vector is zero.
        """
        return cosine_similarity(self._values, other._values)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WordVector):
            return NotImplemented
        return self._word == other._word and bool(
            np.array_equal(self._values, other._values)
        )

    def __hash__(self) -> int:
        return hash((self._word, self._values.tobytes()))

    def __repr__(self) -> str:
        return f"WordVector({self._word!r}, dimension={self.dimension})"


class SimilarWord:
    """One word and how similar it was to what was asked about.

    Parameters
    ----------
    word:
        The word. Non-empty.
    similarity:
        Its cosine similarity to the query, in ``[-1, 1]``.

    Raises
    ------
    EmptyValuesError
        If ``word`` is empty.
    InvalidValuesError
        If ``similarity`` is outside ``[-1, 1]`` or not finite.
    """

    __slots__ = ("_similarity", "_word")

    def __init__(self, word: str, similarity: float) -> None:
        if not isinstance(word, str) or not word:
            raise EmptyValuesError("a similar word needs a non-empty word")

        value = float(similarity)
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise InvalidValuesError(
                f"a cosine similarity lies in [-1, 1], got {similarity!r} for {word!r}"
            )

        self._word = word
        self._similarity = value

    @property
    def word(self) -> str:
        """The word."""
        return self._word

    @property
    def similarity(self) -> float:
        """Its cosine similarity to the query."""
        return self._similarity

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimilarWord):
            return NotImplemented
        return self._word == other._word and self._similarity == other._similarity

    def __hash__(self) -> int:
        return hash((self._word, self._similarity))

    def __repr__(self) -> str:
        return f"SimilarWord({self._word!r}, {self._similarity:.4f})"


class SimilarWords:
    """The nearest words to a query, most similar first.

    Parameters
    ----------
    similar_words:
        In descending order of similarity. May be empty, since a vocabulary of
        one word has no neighbours.

    Raises
    ------
    InvalidValuesError
        If the similarities are not in descending order, or a word repeats.
    """

    __slots__ = ("_similar_words",)

    def __init__(self, similar_words: Sequence[SimilarWord]) -> None:
        seen: set[str] = set()
        previous = 1.0
        for similar_word in similar_words:
            if similar_word.similarity > previous:
                raise InvalidValuesError(
                    "similar words must be in descending order of similarity"
                )
            if similar_word.word in seen:
                raise InvalidValuesError(
                    f"{similar_word.word!r} appears twice among the similar words"
                )
            seen.add(similar_word.word)
            previous = similar_word.similarity

        self._similar_words = tuple(similar_words)

    @property
    def words(self) -> tuple[str, ...]:
        """The words alone, most similar first."""
        return tuple(similar_word.word for similar_word in self._similar_words)

    @property
    def similarities(self) -> tuple[float, ...]:
        """The similarities alone, in the same order."""
        return tuple(similar_word.similarity for similar_word in self._similar_words)

    @property
    def n_words(self) -> int:
        """How many neighbours were returned."""
        return len(self._similar_words)

    def __iter__(self) -> Iterator[SimilarWord]:
        return iter(self._similar_words)

    def __len__(self) -> int:
        return len(self._similar_words)

    def __getitem__(self, position: int) -> SimilarWord:
        return self._similar_words[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimilarWords):
            return NotImplemented
        return self._similar_words == other._similar_words

    def __hash__(self) -> int:
        return hash(self._similar_words)

    def __repr__(self) -> str:
        return f"SimilarWords({list(self.words)!r})"


class WordEmbeddings:
    """A vocabulary and one vector per token, row ``i`` for id ``i``.

    Parameters
    ----------
    vocabulary:
        The tokens, in id order.
    table:
        ``(n_tokens, dimension)``, finite. Copied and frozen.

    Raises
    ------
    InvalidValuesError
        If the table is not two-dimensional, has no columns, or is not finite.
    ShapeMismatchError
        If the table has a different number of rows than the vocabulary has
        tokens.
    """

    __slots__ = ("_norms", "_table", "_vocabulary")

    def __init__(self, vocabulary: Vocabulary, table: FloatArray) -> None:
        try:
            as_array = np.asarray(table, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError("an embedding table must be numeric") from error

        if as_array.ndim != 2:
            raise InvalidValuesError(
                f"an embedding table is (n_tokens, dimension), got shape "
                f"{as_array.shape}"
            )

        if as_array.shape[1] == 0:
            raise InvalidValuesError("an embedding needs at least one dimension")

        if as_array.shape[0] != vocabulary.n_tokens:
            raise ShapeMismatchError(
                f"the table has {as_array.shape[0]} rows for a vocabulary of "
                f"{vocabulary.n_tokens} tokens"
            )

        if not np.all(np.isfinite(as_array)):
            raise InvalidValuesError("an embedding table holds a non-finite value")

        frozen = as_array.copy()
        frozen.setflags(write=False)

        self._vocabulary = vocabulary
        self._table = frozen
        self._norms = np.linalg.norm(frozen, axis=1)

    @property
    def vocabulary(self) -> Vocabulary:
        """Which token each row belongs to."""
        return self._vocabulary

    @property
    def table(self) -> FloatArray:
        """``(n_tokens, dimension)``, frozen. Row ``i`` is token ``i``'s vector.

        The arrangement an
        :class:`~oop_ml.core.network.embedding.Embedding` layer reads, so a
        network can start from what a fit learned.
        """
        return self._table

    @property
    def dimension(self) -> int:
        """How many numbers each vector holds."""
        return int(self._table.shape[1])

    @property
    def n_words(self) -> int:
        """How many tokens have a vector."""
        return int(self._table.shape[0])

    def vector_of(self, word: str) -> WordVector:
        """The vector for ``word``.

        Raises
        ------
        UnknownTokenError
            If the word is not in the vocabulary and the vocabulary has no
            unknown token to fall back on.
        """
        return WordVector(word, self._table[self._vocabulary.id_of(word)])

    def similarity(self, first_word: str, second_word: str) -> float:
        """Cosine similarity between two words' vectors.

        Raises
        ------
        UnknownTokenError
            If either word is not in the vocabulary.
        UndefinedMetricError
            If either vector is zero.
        """
        return cosine_similarity(
            self._table[self._vocabulary.id_of(first_word)],
            self._table[self._vocabulary.id_of(second_word)],
        )

    def most_similar(self, word: str, n_results: int = 10) -> SimilarWords:
        """The ``n_results`` words nearest to ``word`` by cosine, itself excluded.

        Raises
        ------
        UnknownTokenError
            If the word is not in the vocabulary.
        UndefinedMetricError
            If the word's vector is zero.
        InvalidValuesError
            If ``n_results`` is below one.
        """
        query = self._table[self._vocabulary.id_of(word)]
        return self.similar_to_vector(query, n_results, excluding=(word,))

    def analogy(
        self,
        positive: Sequence[str],
        negative: Sequence[str] = (),
        n_results: int = 10,
    ) -> SimilarWords:
        """The words nearest to ``sum(positive) - sum(negative)``, in unit vectors.

        ``analogy(["king", "woman"], ["man"])`` asks what is to ``woman`` as
        ``king`` is to ``man``. The words in the question are excluded from the
        answer.

        Raises
        ------
        EmptyValuesError
            If ``positive`` is empty.
        UnknownTokenError
            If a word is not in the vocabulary.
        UndefinedMetricError
            If a word's vector is zero, or the combination is.
        """
        if len(positive) == 0:
            raise EmptyValuesError("an analogy needs at least one positive word")

        combination = np.zeros(self.dimension)
        for word in positive:
            combination += self._unit_vector_of(word)
        for word in negative:
            combination -= self._unit_vector_of(word)

        return self.similar_to_vector(
            combination, n_results, excluding=(*positive, *negative)
        )

    def similar_to_vector(
        self,
        values: FloatArray,
        n_results: int = 10,
        excluding: Sequence[str] = (),
    ) -> SimilarWords:
        """The ``n_results`` words nearest to an arbitrary vector by cosine.

        Words with a zero vector are skipped, since their similarity to
        anything is undefined.

        Raises
        ------
        InvalidValuesError
            If ``n_results`` is below one, or ``values`` is not a finite vector.
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
            similarities = (self._table @ query) / (self._norms * query_norm)
        similarities = np.where(self._norms == 0.0, -np.inf, similarities)
        for word in excluding:
            if word in self._vocabulary:
                similarities[self._vocabulary.id_of(word)] = -np.inf

        order = np.argsort(-similarities, kind="stable")
        results: list[SimilarWord] = []
        for token_id in order:
            if len(results) == n_results or not np.isfinite(similarities[token_id]):
                break
            results.append(
                SimilarWord(
                    self._vocabulary.token_of(int(token_id)),
                    float(np.clip(similarities[token_id], -1.0, 1.0)),
                )
            )
        return SimilarWords(results)

    def _unit_vector_of(self, word: str) -> FloatArray:
        token_id = self._vocabulary.id_of(word)
        norm = self._norms[token_id]
        if norm == 0.0:
            raise UndefinedMetricError(
                f"{word!r} has a zero vector, which has no direction"
            )
        return self._table[token_id] / norm

    def __contains__(self, word: object) -> bool:
        return word in self._vocabulary

    def __iter__(self) -> Iterator[WordVector]:
        """Iterate the word vectors in id order."""
        for token_id, word in enumerate(self._vocabulary):
            yield WordVector(word, self._table[token_id])

    def __len__(self) -> int:
        return self.n_words

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._table, dtype, copy)

    def __eq__(self, other: object) -> bool:
        """A verdict rather than an element-wise answer, because two embeddings
        are equal when they name the same tokens with the same vectors."""
        if not isinstance(other, WordEmbeddings):
            return NotImplemented
        return self._vocabulary == other._vocabulary and bool(
            np.array_equal(self._table, other._table)
        )

    def __hash__(self) -> int:
        return hash((self._vocabulary, self._table.tobytes()))

    def __repr__(self) -> str:
        return f"WordEmbeddings(n_words={self.n_words}, dimension={self.dimension})"
