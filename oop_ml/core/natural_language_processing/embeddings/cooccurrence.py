"""Which words appear near which, counted once for every technique that reads it.

The distributional hypothesis, as a table
-----------------------------------------
Harris (1954) and Firth (1957): a word is characterised by the company it
keeps. Make that a number and it is a matrix with a row per word and a column
per context word, holding how often the two occurred within a few positions of
each other. Every count-based embedding in this package -- pointwise mutual
information followed by a decomposition, GloVe's weighted least squares -- is a
different way of reading that one matrix, so it is built once, here, and handed
to each of them.

Two choices decide what "near" means
------------------------------------
The **window** is how many positions on each side count as context; five is
the usual default, and words outside a sentence are never context for words
inside it. The **weighting** is how much a neighbour at distance ``d`` counts.
:attr:`ContextWeighting.UNIFORM` gives every position in the window one full
count; :attr:`ContextWeighting.HARMONIC` gives ``1 / d``, so the word right
next door counts once and the word five away counts a fifth, which is GloVe's
choice and the reason its co-occurrence counts are not whole numbers. Word2vec
reaches a similar effect by another route, shrinking its window at random so
nearer positions are inside it more often.

Worked, on one sentence
-----------------------
``the cat sat on the mat`` with a window of 2, harmonic weighting. From the
first ``the`` (position 0): ``cat`` at distance 1 counts 1, ``sat`` at distance
2 counts 1/2. From the second ``the`` (position 4): ``sat`` at distance 2
counts 1/2, ``on`` 1, ``mat`` 1. So the row for ``the`` reads ``cat`` 1,
``sat`` 1, ``on`` 1, ``mat`` 1 -- and every pair is counted from both ends, so
the matrix is symmetric by construction, and its total is
``2 * sum over ordered pairs``. Under uniform weighting the same row reads
``cat`` 1, ``sat`` 2, ``on`` 1, ``mat`` 1, because ``sat`` is within two of
both occurrences of ``the``.

The matrix is dense. Real corpora make it sparse and enormous, and the usual
repair is a hash map of pairs or a sparse matrix; here the vocabularies are
small enough that a dense ``(n, n)`` array is the honest, readable choice, and
the docstrings say so where the cost would bite.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
from numpy.typing import DTypeLike

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray, array_for_protocol


class ContextWeighting(StrEnum):
    """How much a neighbour at distance ``d`` counts."""

    UNIFORM = "uniform"
    """One full count at every position in the window."""

    HARMONIC = "harmonic"
    """``1 / d``, GloVe's rule: the word next door counts once, the word five
    away counts a fifth."""


class CooccurrenceMatrix:
    """How often each word appeared near each other word.

    Parameters
    ----------
    vocabulary:
        Which word each row and column belongs to; ids are positions.
    counts:
        ``(n_words, n_words)``, finite and non-negative. Copied and frozen.
    window:
        How many positions on each side were counted.
    weighting:
        How a neighbour's distance weighted its count.

    Raises
    ------
    InvalidValuesError
        If the counts are not a square, finite, non-negative matrix, or the
        window is below one.
    ShapeMismatchError
        If the matrix is not ``n_words`` on a side.
    """

    __slots__ = ("_counts", "_vocabulary", "_weighting", "_window")

    def __init__(
        self,
        vocabulary: Vocabulary,
        counts: FloatArray,
        window: int,
        weighting: ContextWeighting,
    ) -> None:
        try:
            as_array = np.asarray(counts, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError("co-occurrence counts must be numeric") from error

        if as_array.ndim != 2 or as_array.shape[0] != as_array.shape[1]:
            raise InvalidValuesError(
                f"co-occurrence counts are a square matrix, got shape {as_array.shape}"
            )

        if as_array.shape[0] != vocabulary.n_tokens:
            raise ShapeMismatchError(
                f"the matrix is {as_array.shape[0]} on a side for a vocabulary of "
                f"{vocabulary.n_tokens} words"
            )

        if not np.all(np.isfinite(as_array)) or bool(np.any(as_array < 0.0)):
            raise InvalidValuesError(
                "co-occurrence counts must be finite and non-negative"
            )

        if window < 1:
            raise InvalidValuesError(f"the window must be at least 1, got {window}")

        frozen = as_array.copy()
        frozen.setflags(write=False)

        self._vocabulary = vocabulary
        self._counts = frozen
        self._window = int(window)
        self._weighting = weighting

    @classmethod
    def from_id_sequences(
        cls,
        vocabulary: Vocabulary,
        id_sequences: Sequence[Sequence[int]],
        window: int,
        weighting: ContextWeighting = ContextWeighting.HARMONIC,
    ) -> CooccurrenceMatrix:
        """Count every pair within ``window`` positions, in every sentence.

        Each pair is counted from both ends, so the matrix is symmetric. A
        sentence boundary ends the window.

        Raises
        ------
        InvalidValuesError
            If the window is below one, or an id is outside the vocabulary.
        """
        if window < 1:
            raise InvalidValuesError(f"the window must be at least 1, got {window}")

        n_words = vocabulary.n_tokens
        for sentence in id_sequences:
            for word_id in sentence:
                if not 0 <= word_id < n_words:
                    raise InvalidValuesError(
                        f"id {word_id} is outside a vocabulary of {n_words} words"
                    )

        counts = np.zeros((n_words, n_words))
        for sentence in id_sequences:
            for position, word_id in enumerate(sentence):
                for distance in range(1, window + 1):
                    weight = (
                        1.0 / distance
                        if weighting is ContextWeighting.HARMONIC
                        else 1.0
                    )
                    if position + distance < len(sentence):
                        counts[word_id, sentence[position + distance]] += weight
                    if position - distance >= 0:
                        counts[word_id, sentence[position - distance]] += weight

        return cls(vocabulary, counts, window, weighting)

    @property
    def vocabulary(self) -> Vocabulary:
        """Which word each row and column belongs to."""
        return self._vocabulary

    @property
    def counts(self) -> FloatArray:
        """``(n_words, n_words)``, frozen; row is the word, column the context."""
        return self._counts

    @property
    def window(self) -> int:
        """How many positions on each side were counted."""
        return self._window

    @property
    def weighting(self) -> ContextWeighting:
        """How distance weighted each count."""
        return self._weighting

    @property
    def n_words(self) -> int:
        """How many words the matrix is over."""
        return self._vocabulary.n_tokens

    @property
    def total(self) -> float:
        """The sum of every count, which pointwise mutual information divides by."""
        return float(self._counts.sum())

    @property
    def word_totals(self) -> FloatArray:
        """Row sums: how much context each word had in all, frozen."""
        totals = self._counts.sum(axis=1)
        totals.setflags(write=False)
        return totals

    @property
    def context_totals(self) -> FloatArray:
        """Column sums: how often each word served as context, frozen."""
        totals = self._counts.sum(axis=0)
        totals.setflags(write=False)
        return totals

    def count_between(self, word: str, context: str) -> float:
        """How often ``context`` appeared within the window of ``word``.

        Raises
        ------
        UnknownTokenError
            If either word is not in the vocabulary.
        """
        return float(
            self._counts[self._vocabulary.id_of(word), self._vocabulary.id_of(context)]
        )

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._counts, dtype, copy)

    def __eq__(self, other: object) -> bool:
        """A verdict: the same words, the same counts, the same window rule."""
        if not isinstance(other, CooccurrenceMatrix):
            return NotImplemented
        return (
            self._vocabulary == other._vocabulary
            and self._window == other._window
            and self._weighting is other._weighting
            and bool(np.array_equal(self._counts, other._counts))
        )

    def __hash__(self) -> int:
        return hash(
            (self._vocabulary, self._window, self._weighting, self._counts.tobytes())
        )

    def __repr__(self) -> str:
        return (
            f"CooccurrenceMatrix(n_words={self.n_words}, window={self._window}, "
            f"weighting={self._weighting.value!r})"
        )
