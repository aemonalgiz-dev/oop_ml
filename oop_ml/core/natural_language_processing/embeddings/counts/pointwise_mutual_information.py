"""Word vectors from the decomposition of a pointwise mutual information matrix.

From counts to association
--------------------------
A co-occurrence count says how often ``w`` and ``c`` appeared together. It is
a poor measure of how much they have to do with each other, because a frequent
word appears beside everything: ``the`` co-occurs with ``sail`` far more often
than ``mast`` does, and means nothing by it. Pointwise mutual information
(Church and Hanks, 1990) divides out what frequency alone would predict,

    PMI(w, c) = log( P(w, c) / (P(w) P(c)) )
              = log( #(w, c) * total / (#(w) #(c)) ),

so it is positive when the pair appears more than chance, zero at chance and
negative below it. Levy, Goldberg and Dagan (2015) showed that a matrix of
these values, decomposed by a truncated singular value decomposition, gives
word vectors competitive with word2vec's -- and that skip-gram with negative
sampling is itself implicitly factorising a shifted version of this very
matrix (Levy and Goldberg, 2014). Three of their adjustments are fields here.

Three adjustments from the paper
--------------------------------
**Positive PMI.** A pair that never co-occurred has ``PMI = log 0 = -inf``,
and a pair that co-occurred less than chance has a negative value that says
little more than "these were not seen together often", which on a small corpus
is most pairs. Both are clipped to zero: ``PPMI = max(PMI, 0)``. A zero count
therefore gives 0, not ``-inf``, and the matrix is non-negative.

**Context distribution smoothing.** Rare contexts get a large PMI simply
because ``#(c)`` in the denominator is small. Raising every context count to
``alpha = 0.75`` and renormalising, ``P_alpha(c) = #(c)^alpha / sum #(c)^alpha``,
lifts the probability of rare contexts and lowers that of frequent ones, and
so *lowers* the PMI of a pair with a rare context and raises that of a pair
with a frequent one. It is the same 0.75 word2vec uses when drawing negatives,
and the paper found it the single most useful of its adjustments.

**A shift.** Skip-gram with ``k`` negatives factorises ``PMI - log k``, so the
matrix can be shifted by ``log k`` before clipping, which zeroes every entry
below ``log k``. It is a sparsity knob: more shift, fewer positive entries.
The default is no shift.

Worked, on three sentences
--------------------------
``the cat sat``, ``the dog sat``, ``the cat ran``, window 1, uniform
weighting. Each adjacent pair is counted from both ends, so the row for
``the`` reads ``cat`` 2, ``dog`` 1; ``cat`` reads ``the`` 2, ``sat`` 1, ``ran``
1; ``sat`` reads ``cat`` 1, ``dog`` 1; ``dog`` reads ``the`` 1, ``sat`` 1;
``ran`` reads ``cat`` 1. The word totals are 3, 4, 2, 2, 1, the context
totals the same by symmetry, and the grand total is 12. Then, unsmoothed:

    PMI(the, cat) = log(2 * 12 / (3 * 4)) = log 2   = 0.693147
    PMI(cat, ran) = log(1 * 12 / (4 * 1)) = log 3   = 1.098612
    PMI(cat, sat) = log(1 * 12 / (4 * 2)) = log 1.5 = 0.405465

and ``PMI(the, sat) = 0`` because the pair never occurred. At
``alpha = 0.75`` the smoothed context weights are ``3^0.75 + 4^0.75 +
2 * 2^0.75 + 1 = 9.47153``, so ``P_alpha(ran) = 1 / 9.47153 = 0.10558``
against ``P(ran) = 1 / 12 = 0.08333``, and ``P_alpha(cat) = 2.82843 / 9.47153 =
0.29862`` against ``0.33333``. ``PMI(cat, ran)`` falls to
``log(0.083333 / (0.333333 * 0.10558)) = 0.861995`` and ``PMI(the, cat)`` rises
to ``log(0.166667 / (0.25 * 0.29862)) = 0.803104``: the rare context lost,
the frequent one gained, which is the direction the smoothing exists for. A
shift of ``log 5 = 1.6094`` is larger than every entry, so on this corpus it
zeroes the whole matrix.

The vectors
-----------
The positive matrix ``M`` is decomposed as ``U S V^T`` and the word vectors
are ``U_k S_k^p``. The textbook choice ``p = 1`` reproduces the matrix's
inner products; the paper found ``p = 0.5`` consistently better on word
similarity, and ``p = 0`` (the bare orthonormal ``U_k``) close behind, so the
exponent is a field with the paper's default. The right vectors ``V_k`` are
the context vectors and are not exposed as embeddings, though at ``p = 1``
and full rank ``(U_k S_k) V_k^T`` reconstructs ``M`` exactly, which the specs
check. Signs follow the rule in
:mod:`~oop_ml.core.natural_language_processing.embeddings.counts.latent_semantic_analysis`.

A word whose every association is at or below chance has a zero row in ``M``
and comes out with a zero vector; its similarity to anything is then
undefined and asking for it raises. That is the honest answer for a word the
corpus said nothing about.

What is dense here
------------------
``M`` is ``(n_words, n_words)`` and mostly zero on any real corpus, and the
decomposition is a full ``numpy.linalg.svd`` at ``O(n^3)``. The usual repair
is a sparse matrix and a truncated solver that touches only the ``k`` leading
triples. Both are optimisations of this calculation, not different ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import DTypeLike
from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import (
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.counts.latent_semantic_analysis import (  # noqa: E501
    TruncatedSingularValueDecomposition,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import WordEmbedder
from oop_ml.core.natural_language_processing.embeddings.vectors import WordEmbeddings
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.types import FloatArray, array_for_protocol


class PointwiseMutualInformationMatrix:
    """Shifted positive pointwise mutual information per word and context.

    Row is the word, column the context, entry ``max(PMI - shift, 0)`` with the
    context marginal smoothed as the module docstring describes. Non-negative
    by construction; a pair never seen together is exactly zero.

    Parameters
    ----------
    vocabulary:
        Which word each row and column belongs to; ids are positions.
    values:
        ``(n_words, n_words)``, finite and non-negative. Copied and frozen.
    shift:
        What was subtracted before clipping. Non-negative.
    context_distribution_smoothing:
        The exponent the context counts were raised to. In ``(0, 1]``.

    Raises
    ------
    InvalidValuesError
        If the values are not a square, finite, non-negative matrix, the
        shift is negative, or the smoothing is outside ``(0, 1]``.
    ShapeMismatchError
        If the matrix is not ``n_words`` on a side.
    """

    __slots__ = ("_context_distribution_smoothing", "_shift", "_values", "_vocabulary")

    def __init__(
        self,
        vocabulary: Vocabulary,
        values: FloatArray,
        shift: float,
        context_distribution_smoothing: float,
    ) -> None:
        try:
            as_array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError(
                "pointwise mutual information must be numeric"
            ) from error

        if as_array.ndim != 2 or as_array.shape[0] != as_array.shape[1]:
            raise InvalidValuesError(
                f"pointwise mutual information is a square matrix, got shape "
                f"{as_array.shape}"
            )
        if as_array.shape[0] != vocabulary.n_tokens:
            raise ShapeMismatchError(
                f"the matrix is {as_array.shape[0]} on a side for a vocabulary of "
                f"{vocabulary.n_tokens} words"
            )
        if not np.all(np.isfinite(as_array)) or bool(np.any(as_array < 0.0)):
            raise InvalidValuesError(
                "positive pointwise mutual information is finite and non-negative"
            )
        if not np.isfinite(shift) or shift < 0.0:
            raise InvalidValuesError(f"the shift is non-negative, got {shift}")
        if not 0.0 < context_distribution_smoothing <= 1.0:
            raise InvalidValuesError(
                f"context distribution smoothing lies in (0, 1], got "
                f"{context_distribution_smoothing}"
            )

        frozen = as_array.copy()
        frozen.setflags(write=False)

        self._vocabulary = vocabulary
        self._values = frozen
        self._shift = float(shift)
        self._context_distribution_smoothing = float(context_distribution_smoothing)

    @classmethod
    def from_cooccurrence(
        cls,
        cooccurrence: CooccurrenceMatrix,
        context_distribution_smoothing: float = 0.75,
        shift: float = 0.0,
    ) -> PointwiseMutualInformationMatrix:
        """``max(log(P(w, c) / (P(w) P_alpha(c))) - shift, 0)`` from the counts.

        ``P(w, c)`` is the count over the grand total, ``P(w)`` the row total
        over the grand total, and ``P_alpha(c)`` the column total raised to
        the smoothing exponent and renormalised over every column. Entries
        whose count is zero are zero, so no logarithm of zero is ever taken;
        a matrix with no co-occurrences at all is all zeros.

        Raises
        ------
        InvalidValuesError
            If the shift is negative or the smoothing outside ``(0, 1]``.
        """
        if not np.isfinite(shift) or shift < 0.0:
            raise InvalidValuesError(f"the shift is non-negative, got {shift}")
        if not 0.0 < context_distribution_smoothing <= 1.0:
            raise InvalidValuesError(
                f"context distribution smoothing lies in (0, 1], got "
                f"{context_distribution_smoothing}"
            )

        counts = cooccurrence.counts
        total = cooccurrence.total
        values = np.zeros_like(counts)
        if total > 0.0:
            word_probabilities = cooccurrence.word_totals / total
            smoothed_context_totals = (
                cooccurrence.context_totals**context_distribution_smoothing
            )
            context_probabilities = (
                smoothed_context_totals / smoothed_context_totals.sum()
            )
            expected = np.outer(word_probabilities, context_probabilities)
            observed = counts > 0.0
            values[observed] = np.log((counts[observed] / total) / expected[observed])
            values = np.maximum(values - shift, 0.0)

        return cls(
            cooccurrence.vocabulary, values, shift, context_distribution_smoothing
        )

    @property
    def vocabulary(self) -> Vocabulary:
        """Which word each row and column belongs to."""
        return self._vocabulary

    @property
    def values(self) -> FloatArray:
        """``(n_words, n_words)``, frozen; row is the word, column the context."""
        return self._values

    @property
    def shift(self) -> float:
        """What was subtracted from every entry before clipping at zero."""
        return self._shift

    @property
    def context_distribution_smoothing(self) -> float:
        """The exponent the context counts were raised to."""
        return self._context_distribution_smoothing

    @property
    def n_words(self) -> int:
        """How many words the matrix is over."""
        return self._vocabulary.n_tokens

    @property
    def n_positive(self) -> int:
        """How many entries are above zero, which the shift decides."""
        return int(np.count_nonzero(self._values > 0.0))

    def value_between(self, word: str, context: str) -> float:
        """The shifted positive PMI of ``context`` given ``word``.

        Raises
        ------
        UnknownTokenError
            If either word is not in the vocabulary.
        """
        return float(
            self._values[self._vocabulary.id_of(word), self._vocabulary.id_of(context)]
        )

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __eq__(self, other: object) -> bool:
        """A verdict: the same words, the same rule, the same numbers."""
        if not isinstance(other, PointwiseMutualInformationMatrix):
            return NotImplemented
        return (
            self._vocabulary == other._vocabulary
            and self._shift == other._shift
            and self._context_distribution_smoothing
            == other._context_distribution_smoothing
            and bool(np.array_equal(self._values, other._values))
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._vocabulary,
                self._shift,
                self._context_distribution_smoothing,
                self._values.tobytes(),
            )
        )

    def __repr__(self) -> str:
        return (
            f"PointwiseMutualInformationMatrix(n_words={self.n_words}, "
            f"shift={self._shift}, "
            f"context_distribution_smoothing={self._context_distribution_smoothing})"
        )


class PointwiseMutualInformationEmbeddings(WordEmbedder):
    """Word vectors ``U_k S_k^p`` from the decomposition of a positive PMI matrix.

    Levy, Goldberg and Dagan (2015). The co-occurrence matrix is counted over
    a window, turned into shifted positive pointwise mutual information with a
    smoothed context distribution, and decomposed.

    Parameters
    ----------
    dimension:
        How many components to keep. At most the vocabulary size, which is
        known only at ``fit``.
    window:
        How many positions on each side of a word count as its context.
    weighting:
        How a neighbour's distance weights its count. Uniform by default,
        which is the paper's setting; harmonic is GloVe's.
    context_distribution_smoothing:
        The exponent ``alpha`` the context counts are raised to. One is no
        smoothing; the paper's 0.75 is the default.
    shift:
        Subtracted from every PMI before clipping at zero; ``log k`` makes the
        matrix the one skip-gram with ``k`` negatives factorises.
    singular_value_exponent:
        The power ``p`` of the singular values in the word vectors. One is the
        textbook scaling, zero the bare singular vectors, 0.5 the paper's
        default.
    """

    dimension: int = Field(default=50, ge=1)
    window: int = Field(default=5, ge=1)
    weighting: ContextWeighting = ContextWeighting.UNIFORM
    context_distribution_smoothing: float = Field(default=0.75, gt=0.0, le=1.0)
    shift: float = Field(default=0.0, ge=0.0)
    singular_value_exponent: float = Field(default=0.5, ge=0.0, le=1.0)

    _embeddings: WordEmbeddings = PrivateAttr()
    _cooccurrence: CooccurrenceMatrix = PrivateAttr()
    _pointwise_mutual_information: PointwiseMutualInformationMatrix = PrivateAttr()
    _decomposition: TruncatedSingularValueDecomposition = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Count co-occurrences in ``corpus``, take their PMI, and decompose it.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If no word reaches ``minimum_count``, or ``dimension`` exceeds the
            vocabulary size, the most components a square matrix has.
        """
        tokenised = self._tokenised(corpus)
        vocabulary = tokenised.vocabulary(self.minimum_count)
        if self.dimension > vocabulary.n_tokens:
            raise TooFewValuesError(
                f"dimension={self.dimension} exceeds the {vocabulary.n_tokens} "
                f"components a matrix over {vocabulary.n_tokens} words has"
            )

        cooccurrence = CooccurrenceMatrix.from_id_sequences(
            vocabulary,
            tokenised.id_sequences(vocabulary),
            self.window,
            self.weighting,
        )
        pointwise_mutual_information = (
            PointwiseMutualInformationMatrix.from_cooccurrence(
                cooccurrence, self.context_distribution_smoothing, self.shift
            )
        )
        decomposition = TruncatedSingularValueDecomposition.of(
            pointwise_mutual_information.values, self.dimension
        )
        word_table = decomposition.left_vectors * (
            decomposition.singular_values**self.singular_value_exponent
        )

        self._embeddings = WordEmbeddings(vocabulary, word_table)
        self._cooccurrence = cooccurrence
        self._pointwise_mutual_information = pointwise_mutual_information
        self._decomposition = decomposition
        self._mark_fitted()
        return self

    @property
    def embeddings(self) -> WordEmbeddings:
        """The word vectors ``U_k S_k^p``, one row per word.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._embeddings

    @property
    def cooccurrence(self) -> CooccurrenceMatrix:
        """The counts the PMI was computed from.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._cooccurrence

    @property
    def pointwise_mutual_information(self) -> PointwiseMutualInformationMatrix:
        """The shifted positive PMI matrix that was decomposed.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._pointwise_mutual_information

    @property
    def decomposition(self) -> TruncatedSingularValueDecomposition:
        """The kept singular triples of the PMI matrix.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._decomposition

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
