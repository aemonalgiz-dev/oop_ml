"""GloVe: word vectors fitted to the logarithm of how often words co-occur.

Why a ratio, and why a logarithm
--------------------------------
Pennington, Socher and Manning (2014) start from the co-occurrence matrix
rather than from a window sliding over the text: how often each word ``i``
appeared near each context word ``k``, ``X_ik``, counted once for the whole
corpus. The question is what in that table carries meaning. Their example is
``ice`` and ``steam``. Each is compared, through the probability of a probe
word appearing beside it, with four probes. For ``solid`` the probability is
much higher beside ``ice`` than beside ``steam``; for ``gas`` it is the other
way round; for ``water``, which belongs to both, both probabilities are high;
for ``fashion``, which belongs to neither, both are low. The raw probabilities
are not much use on their own: ``water`` scores high beside both words and
``fashion`` low beside both, and neither tells ``ice`` from ``steam``. The
*ratio* of the two probabilities does. It is large for ``solid``, small for
``gas``, and near one for both ``water`` and ``fashion``, so the ratio cancels
what the two words share and what neither has, and leaves what separates them.

A model of ratios wants the vectors to reproduce ``P_ik / P_jk`` from ``w_i``,
``w_j`` and a context vector for ``k``. Asking that the answer depend only on
the difference ``w_i - w_j``, and that differences in the vector space turn
into ratios of probabilities, ``F(a - b) = F(a) / F(b)``, leaves the
exponential as the only choice. Taking logarithms turns the product into
``w_i . w~_k = log X_ik - log X_i``. The term ``log X_i`` does not depend on
``k`` and becomes a bias ``b_i``; a second bias ``b~_k`` is added so that
nothing distinguishes the two roles. What is left is a least-squares fit:

    w_i . w~_j + b_i + b~_j  should equal  log X_ij

for every pair that ever co-occurred. Word2vec reaches a similar factorisation
by predicting one word from another, a window at a time; this model reads the
counts first and fits them directly, which is what "global" in the name means.

Why the weighting, and why it stops at ``x_max``
------------------------------------------------
Most entries of ``X`` are zero, and ``log 0`` is not a number, so the sum runs
over the nonzero entries only. Among those, a pair seen once is a poor
estimate of anything and a pair seen a thousand times is a good one, so each
term is weighted by ``f(X_ij) = (X_ij / x_max) ** alpha``, but only up to
``x_max``, above which ``f`` is exactly one. Without the cap the commonest
pairs would dominate: ``the`` beside ``of`` occurs so often that its term alone
would outweigh most of the vocabulary. The cap says that once a pair has been
seen often enough, seeing it more often makes it no more important. The paper
found ``alpha = 0.75`` to beat ``alpha = 1`` by a small margin and
``x_max = 100`` to work over corpora of many sizes, and both are the defaults.

On a corpus of a few hundred words nearly every count is far below 100, so the
cap never binds and the weights are small: at ``x_max = 100`` a pair seen once
weighs ``0.01 ** 0.75 = 0.0316``, and a pair seen fifty times weighs
``0.5 ** 0.75 = 0.5946``. Measured on the two-topic spec corpus, 24 sentences
of six to nine words, the largest count at a window of two is 16.5 and every
weight is below ``0.00026`` once ``x_max`` is raised to a million.

What a small corpus does to the logarithms
------------------------------------------
Harmonic weighting makes a count of ``1 / d`` for a neighbour at distance
``d``, and ``log`` of anything below one is negative. On the spec corpus a
window of five reaches every function word from every content word at a weight
of 0.2 to 0.5, so 52% of the nonzero counts sit below one and the mean
logarithm is -0.15; the biases, which should absorb that mean, reach only
-0.02 in 25 epochs, and the negative residual lands in the dot products. The
within-topic cosine of ``w + w~`` then comes out *below* the across-topic one
on some seeds. At a window of two the counts are 0.5 or 1 per occurrence and
mostly above one once repeated: 31% below one, mean logarithm +0.26, and the
within-topic cosine beats the across-topic one on every seed tried, +0.119
against -0.078 on the first. The paper's window of ten works because its
counts are in the thousands; on two dozen sentences the window is what decides
whether the logarithms have a sign the biases can absorb.

Why the final vector is ``w + w~``
----------------------------------
``X`` is symmetric, so the word and context tables are asked the same question
and differ only in where they started. The paper hands back their sum,
observing that two estimates of one thing added are less noisy than either, in
the way an ensemble of networks is. That is the choice here:
:attr:`GloVe.embeddings` holds ``w + w~``, and the two tables are exposed
separately for anyone who wants one of them.

How the fit proceeds
--------------------
Every parameter starts uniform in ``(-0.5, 0.5) / dimension``, drawn from one
generator seeded by ``random_seed`` in a fixed order: word vectors, context
vectors, word biases, context biases. Each epoch visits every nonzero pair
once, in an order shuffled by the same generator, and for each computes the
error ``e = w_i . w~_j + b_i + b~_j - log X_ij`` and the gradient of the
pair's term ``f(X_ij) e^2``: ``2 f(X_ij) e`` times the partner vector for
either vector, and ``2 f(X_ij) e`` for either bias. The step is AdaGrad's, as
in the paper: every parameter keeps a running sum of its squared gradients,
the sum takes the new square first, and the parameter moves by
``learning_rate * gradient / (sqrt(sum) + epsilon)``, so the first step a
parameter ever takes is the base rate in the gradient's direction and every
later one is smaller. The reference C code instead starts every sum at one and
divides before adding, which on a corpus of a few hundred words is not AdaGrad
at all: with weights near 0.03 the squared gradients never lift the sum above
one, the step stays at ``learning_rate * gradient`` throughout, and measured
on the spec corpus at the defaults ``J`` fell from 17.6 to only 10.4 in 25
epochs where the textbook rule takes it to 0.19. The nonzero pairs are
enumerated in row-major order before the first shuffle, so two fits with one
seed are identical to the last bit.

What is recorded, and why it is not the objective
-------------------------------------------------
Each epoch's record holds the mean of the pair terms as they were computed
during the epoch, before each pair's update, over the number of nonzero pairs;
that is the reference implementation's ``cost``. The parameters move between
one pair and the next, so that mean is not the objective at any one set of
parameters. :meth:`GloVe.objective_value` recomputes ``J`` from the fitted
parameters at rest. The two agree when the walk stands still: at a learning
rate of 1e-12 for one epoch on the two-topic fixture the recorded total is
22.1271538003 and the recomputed one differs from it by 9.5e-11, which is
4.3e-12 of its size. After a real fit the recomputed value is below the first
epoch's record, 0.51 against 18.93 on that fixture, which is the assertion the
spec makes.

Worked, on three words
----------------------
``["a b a"]`` with a window of 1 gives ``X_ab = X_ba = 2`` and nothing else.
With every parameter at zero the error on both pairs is ``-log 2``, so at
``maximum_count = 1``, where the weight is one, ``J = 2 (log 2)^2 = 0.9609``;
at the defaults the weight is ``0.02 ** 0.75 = 0.0532`` and ``J = 0.0511``.

The cost
--------
The co-occurrence matrix is dense, ``n_words`` on a side, and the fit is a
Python loop over the nonzero pairs, once per epoch. The reference
implementation streams the nonzero triples from disk and updates them from
parallel threads without locks; a vectorised pass over blocks of pairs is the
numpy repair. Neither is done here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.exceptions import (
    DivergenceError,
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.natural_language_processing.embeddings.cooccurrence import (
    ContextWeighting,
    CooccurrenceMatrix,
)
from oop_ml.core.natural_language_processing.embeddings.embedder import WordEmbedder
from oop_ml.core.natural_language_processing.embeddings.prediction.word2vec import (
    EpochRecord,
    TrainingHistory,
)
from oop_ml.core.natural_language_processing.embeddings.vectors import (
    WordEmbeddings,
    checked_vector,
)
from oop_ml.core.types import FloatArray

ADAGRAD_EPSILON = 1e-10
"""Added to the root of the accumulated squared gradient before dividing.

A gradient component of exactly zero over an accumulator of exactly zero is
``0 / 0`` without it. At this size it changes a step only once the accumulated
root is itself near 1e-10, which is a parameter whose gradient has been
vanishing for a long time; PyTorch's default for the same purpose.
"""


def _checked_table(values: object, description: str) -> FloatArray:
    """A finite two-dimensional float array, or a refusal that says why.

    Raises
    ------
    InvalidValuesError
        If ``values`` is not numeric, not two-dimensional, or not finite.
    """
    try:
        as_array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidValuesError(f"{description} must be numeric") from error

    if as_array.ndim != 2:
        raise InvalidValuesError(
            f"{description} is (n_words, dimension), got shape {as_array.shape}"
        )

    if not np.all(np.isfinite(as_array)):
        raise InvalidValuesError(f"{description} holds a non-finite value")

    return as_array


class GloVeParameters:
    """The four tables a GloVe fit learns, held together so they cannot drift apart.

    Parameters
    ----------
    word_vectors:
        ``(n_words, dimension)``, the vector each word has as the word of a
        pair. Copied and frozen.
    context_vectors:
        ``(n_words, dimension)``, the vector each word has as the context of a
        pair. Copied and frozen.
    biases:
        ``(n_words,)``, one bias per word in the word role. Copied and frozen.
    context_biases:
        ``(n_words,)``, one bias per word in the context role. Copied and frozen.

    Raises
    ------
    InvalidValuesError
        If a table is not numeric, not of the stated rank, or not finite.
    ShapeMismatchError
        If the four do not agree on the word count, or the two vector tables
        on the dimension.
    """

    __slots__ = ("_biases", "_context_biases", "_context_vectors", "_word_vectors")

    def __init__(
        self,
        word_vectors: FloatArray,
        context_vectors: FloatArray,
        biases: FloatArray,
        context_biases: FloatArray,
    ) -> None:
        word_table = _checked_table(word_vectors, "the word vectors")
        context_table = _checked_table(context_vectors, "the context vectors")
        bias_vector = checked_vector(biases, "the biases")
        context_bias_vector = checked_vector(context_biases, "the context biases")

        if word_table.shape != context_table.shape:
            raise ShapeMismatchError(
                f"word vectors of shape {word_table.shape} against context vectors "
                f"of shape {context_table.shape}"
            )

        n_words = word_table.shape[0]
        if bias_vector.shape != (n_words,) or context_bias_vector.shape != (n_words,):
            raise ShapeMismatchError(
                f"{n_words} words need {n_words} biases of each kind, got "
                f"{bias_vector.shape[0]} and {context_bias_vector.shape[0]}"
            )

        self._word_vectors = self._frozen(word_table)
        self._context_vectors = self._frozen(context_table)
        self._biases = self._frozen(bias_vector)
        self._context_biases = self._frozen(context_bias_vector)

    @staticmethod
    def _frozen(values: FloatArray) -> FloatArray:
        copy = values.copy()
        copy.setflags(write=False)
        return copy

    @property
    def word_vectors(self) -> FloatArray:
        """``(n_words, dimension)``, frozen: each word in the word role."""
        return self._word_vectors

    @property
    def context_vectors(self) -> FloatArray:
        """``(n_words, dimension)``, frozen: each word in the context role."""
        return self._context_vectors

    @property
    def biases(self) -> FloatArray:
        """``(n_words,)``, frozen: each word's bias in the word role."""
        return self._biases

    @property
    def context_biases(self) -> FloatArray:
        """``(n_words,)``, frozen: each word's bias in the context role."""
        return self._context_biases

    @property
    def n_words(self) -> int:
        """How many words have parameters."""
        return int(self._word_vectors.shape[0])

    @property
    def dimension(self) -> int:
        """How many numbers each vector holds."""
        return int(self._word_vectors.shape[1])

    @property
    def combined_vectors(self) -> FloatArray:
        """``word_vectors + context_vectors``, frozen: the paper's final vectors."""
        combined = self._word_vectors + self._context_vectors
        combined.setflags(write=False)
        return combined

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GloVeParameters):
            return NotImplemented
        return (
            bool(np.array_equal(self._word_vectors, other._word_vectors))
            and bool(np.array_equal(self._context_vectors, other._context_vectors))
            and bool(np.array_equal(self._biases, other._biases))
            and bool(np.array_equal(self._context_biases, other._context_biases))
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._word_vectors.tobytes(),
                self._context_vectors.tobytes(),
                self._biases.tobytes(),
                self._context_biases.tobytes(),
            )
        )

    def __repr__(self) -> str:
        return f"GloVeParameters(n_words={self.n_words}, dimension={self.dimension})"


class PairGradient:
    """What one co-occurring pair asks of its four parameters, and what it cost.

    Parameters
    ----------
    loss:
        The pair's weighted squared error ``f(X_ij) e^2``, finite and
        non-negative.
    word_gradient:
        ``d loss / d w_i``, one vector. Copied and frozen.
    context_gradient:
        ``d loss / d w~_j``, one vector of the same width. Copied and frozen.
    bias_gradient:
        ``d loss / d b_i``.
    context_bias_gradient:
        ``d loss / d b~_j``.

    Raises
    ------
    InvalidValuesError
        If the loss is negative or not finite, a gradient is not a finite
        vector, or a bias gradient is not finite.
    ShapeMismatchError
        If the two vector gradients differ in width.
    """

    __slots__ = (
        "_bias_gradient",
        "_context_bias_gradient",
        "_context_gradient",
        "_loss",
        "_word_gradient",
    )

    def __init__(
        self,
        loss: float,
        word_gradient: FloatArray,
        context_gradient: FloatArray,
        bias_gradient: float,
        context_bias_gradient: float,
    ) -> None:
        if not np.isfinite(loss) or loss < 0.0:
            raise InvalidValuesError(
                f"a weighted squared error is finite and non-negative, got {loss}"
            )

        word_values = checked_vector(word_gradient, "the word gradient")
        context_values = checked_vector(context_gradient, "the context gradient")
        if word_values.shape != context_values.shape:
            raise ShapeMismatchError(
                f"a word gradient of width {word_values.shape[0]} against a context "
                f"gradient of width {context_values.shape[0]}"
            )

        if not np.isfinite(bias_gradient) or not np.isfinite(context_bias_gradient):
            raise InvalidValuesError("a bias gradient must be finite")

        self._loss = float(loss)
        self._word_gradient = GloVeParameters._frozen(word_values)
        self._context_gradient = GloVeParameters._frozen(context_values)
        self._bias_gradient = float(bias_gradient)
        self._context_bias_gradient = float(context_bias_gradient)

    @property
    def loss(self) -> float:
        """The pair's term of the objective, ``f(X_ij) e^2``."""
        return self._loss

    @property
    def word_gradient(self) -> FloatArray:
        """``d loss / d w_i``, frozen."""
        return self._word_gradient

    @property
    def context_gradient(self) -> FloatArray:
        """``d loss / d w~_j``, frozen."""
        return self._context_gradient

    @property
    def bias_gradient(self) -> float:
        """``d loss / d b_i``."""
        return self._bias_gradient

    @property
    def context_bias_gradient(self) -> float:
        """``d loss / d b~_j``."""
        return self._context_bias_gradient

    def __repr__(self) -> str:
        return (
            f"PairGradient(loss={self._loss:.4f}, "
            f"dimension={self._word_gradient.shape[0]})"
        )


class GloVe(WordEmbedder):
    """Global vectors: weighted least squares on the log co-occurrence counts.

    Parameters
    ----------
    dimension:
        How many numbers each word's vector holds.
    window:
        How many positions on each side of a word count as its context.
    weighting:
        How a context word's distance weights its count; the paper's ``1 / d``
        is the default.
    maximum_count:
        The paper's ``x_max``: the count at and above which a pair's weight is
        exactly one.
    weighting_exponent:
        The paper's ``alpha``: the power in ``(X_ij / x_max) ** alpha`` below
        ``x_max``. At most one, where the weight is proportional to the count.
    epochs:
        How many passes over the nonzero pairs.
    learning_rate:
        AdaGrad's base rate, divided per parameter by the root of that
        parameter's accumulated squared gradient.
    random_seed:
        Seeds the initial parameters and each epoch's order of pairs, so a
        seeded fit is reproducible.
    """

    dimension: int = Field(default=50, ge=1)
    window: int = Field(default=5, ge=1)
    weighting: ContextWeighting = ContextWeighting.HARMONIC
    maximum_count: float = Field(default=100.0, gt=0.0)
    weighting_exponent: float = Field(default=0.75, gt=0.0, le=1.0)
    epochs: int = Field(default=25, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0)
    random_seed: int | None = None

    _cooccurrence: CooccurrenceMatrix = PrivateAttr()
    _parameters: GloVeParameters = PrivateAttr()
    _embeddings: WordEmbeddings = PrivateAttr()
    _history: TrainingHistory = PrivateAttr()

    def fit(self, corpus: Sequence[str]) -> Self:
        """Count the co-occurrences of ``corpus`` and fit their logarithms.

        Raises
        ------
        InvalidValuesError
            If ``corpus`` is a single string or holds a non-string.
        EmptyValuesError
            If the corpus is empty, blank, or yields no words.
        TooFewValuesError
            If fewer than two words reach ``minimum_count``, or no two words
            ever fall within a window of each other, since either way there is
            no pair to fit.
        DivergenceError
            If the parameters overflow, which a learning rate far too large
            for the counts can do.
        """
        tokenised = self._tokenised(corpus)
        vocabulary = tokenised.vocabulary(self.minimum_count)
        if vocabulary.n_tokens < 2:
            raise TooFewValuesError(
                "GloVe needs at least two words in the vocabulary: one word "
                "co-occurs with nothing, so there is no pair to fit"
            )

        cooccurrence = CooccurrenceMatrix.from_id_sequences(
            vocabulary, tokenised.id_sequences(vocabulary), self.window, self.weighting
        )
        counts = cooccurrence.counts
        rows, columns = np.nonzero(counts)
        n_pairs = int(rows.shape[0])
        if n_pairs == 0:
            raise TooFewValuesError(
                f"no two words ever appeared within {self.window} positions of each "
                f"other, so there is no pair to fit"
            )

        generator = np.random.default_rng(self.random_seed)
        n_words = vocabulary.n_tokens
        word_vectors = self._initial_values(generator, (n_words, self.dimension))
        context_vectors = self._initial_values(generator, (n_words, self.dimension))
        biases = self._initial_values(generator, (n_words,))
        context_biases = self._initial_values(generator, (n_words,))

        word_accumulators = np.zeros((n_words, self.dimension))
        context_accumulators = np.zeros((n_words, self.dimension))
        bias_accumulators = np.zeros(n_words)
        context_bias_accumulators = np.zeros(n_words)

        records: list[EpochRecord] = []
        for epoch in range(1, self.epochs + 1):
            loss_total = 0.0
            for position in generator.permutation(n_pairs):
                word_id = int(rows[position])
                context_id = int(columns[position])
                gradient = self.pair_gradient(
                    float(counts[word_id, context_id]),
                    word_vectors[word_id],
                    context_vectors[context_id],
                    float(biases[word_id]),
                    float(context_biases[context_id]),
                )
                loss_total += gradient.loss

                word_accumulators[word_id] += gradient.word_gradient**2
                context_accumulators[context_id] += gradient.context_gradient**2
                bias_accumulators[word_id] += gradient.bias_gradient**2
                context_bias_accumulators[context_id] += (
                    gradient.context_bias_gradient**2
                )

                word_vectors[word_id] -= self._adagrad_step(
                    gradient.word_gradient, word_accumulators[word_id]
                )
                context_vectors[context_id] -= self._adagrad_step(
                    gradient.context_gradient, context_accumulators[context_id]
                )
                biases[word_id] -= self._adagrad_step(
                    gradient.bias_gradient, bias_accumulators[word_id]
                )
                context_biases[context_id] -= self._adagrad_step(
                    gradient.context_bias_gradient,
                    context_bias_accumulators[context_id],
                )

            records.append(
                EpochRecord(epoch, loss_total / n_pairs, n_pairs, self.learning_rate)
            )

        parameters = GloVeParameters(
            word_vectors, context_vectors, biases, context_biases
        )

        self._cooccurrence = cooccurrence
        self._parameters = parameters
        self._embeddings = WordEmbeddings(vocabulary, parameters.combined_vectors)
        self._history = TrainingHistory(records)
        self._mark_fitted()
        return self

    def weight_of_count(self, count: float) -> float:
        """The paper's ``f``: ``(count / x_max) ** alpha`` below ``x_max``, else 1.

        Zero at a count of zero, which is why the objective is a sum over the
        nonzero pairs alone.

        Raises
        ------
        InvalidValuesError
            If the count is negative or not finite.
        """
        if not math.isfinite(count) or count < 0.0:
            raise InvalidValuesError(
                f"a co-occurrence count is finite and non-negative, got {count}"
            )
        if count >= self.maximum_count:
            return 1.0
        return float((count / self.maximum_count) ** self.weighting_exponent)

    def pair_gradient(
        self,
        count: float,
        word_vector: FloatArray,
        context_vector: FloatArray,
        bias: float,
        context_bias: float,
    ) -> PairGradient:
        """One pair's term ``f(X_ij) e^2`` and its gradient in all four parameters.

        With ``e = w_i . w~_j + b_i + b~_j - log X_ij`` the gradient is
        ``2 f(X_ij) e w~_j`` for the word vector, ``2 f(X_ij) e w_i`` for the
        context vector, and ``2 f(X_ij) e`` for either bias. The gradients are
        of the loss, so a caller subtracts a multiple of them.

        Raises
        ------
        InvalidValuesError
            If the count is not positive and finite, a vector is not a finite
            one-dimensional array, or a bias is not finite.
        ShapeMismatchError
            If the two vectors differ in width.
        DivergenceError
            If the error overflows, which finite inputs do only once a fit has
            run away.
        """
        if not math.isfinite(count) or count <= 0.0:
            raise InvalidValuesError(
                f"a pair that co-occurred has a positive, finite count, got {count}"
            )
        word_values = checked_vector(word_vector, "the word vector")
        context_values = checked_vector(context_vector, "the context vector")
        if word_values.shape != context_values.shape:
            raise ShapeMismatchError(
                f"a word vector of width {word_values.shape[0]} against a context "
                f"vector of width {context_values.shape[0]}"
            )
        if not math.isfinite(bias) or not math.isfinite(context_bias):
            raise InvalidValuesError("a bias must be finite")

        weight = self.weight_of_count(count)
        with np.errstate(over="ignore", invalid="ignore"):
            error = (
                float(word_values @ context_values)
                + bias
                + context_bias
                - math.log(count)
            )
            loss = weight * error * error
            scaled_error = 2.0 * weight * error
            word_gradient = scaled_error * context_values
            context_gradient = scaled_error * word_values

        if not (
            math.isfinite(loss)
            and bool(np.all(np.isfinite(word_gradient)))
            and bool(np.all(np.isfinite(context_gradient)))
        ):
            raise DivergenceError(
                "the parameters have overflowed; lower the learning rate"
            )

        return PairGradient(
            loss=loss,
            word_gradient=word_gradient,
            context_gradient=context_gradient,
            bias_gradient=scaled_error,
            context_bias_gradient=scaled_error,
        )

    def objective_of(
        self, cooccurrence: CooccurrenceMatrix, parameters: GloVeParameters
    ) -> float:
        """The paper's ``J`` for any counts and any parameters, at rest.

        ``sum over nonzero X_ij of f(X_ij) (w_i . w~_j + b_i + b~_j - log X_ij)^2``,
        summed pair by pair. Needs no fit, so a hand-built matrix and hand-set
        parameters can be scored, which is how the spec pins the arithmetic.

        Raises
        ------
        ShapeMismatchError
            If the parameters and the matrix disagree on the word count.
        """
        if parameters.n_words != cooccurrence.n_words:
            raise ShapeMismatchError(
                f"parameters for {parameters.n_words} words against a matrix over "
                f"{cooccurrence.n_words}"
            )

        counts = cooccurrence.counts
        rows, columns = np.nonzero(counts)
        total = 0.0
        for word_id, context_id in zip(rows, columns, strict=True):
            total += self.pair_gradient(
                float(counts[word_id, context_id]),
                parameters.word_vectors[word_id],
                parameters.context_vectors[context_id],
                float(parameters.biases[word_id]),
                float(parameters.context_biases[context_id]),
            ).loss
        return total

    def objective_value(self) -> float:
        """``J`` recomputed from the fitted parameters on the matrix they fit.

        Not the last epoch's recorded loss: that was accumulated while the
        parameters moved, and this is the objective where they stopped. The
        module docstring says how far apart the two are.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        self._check_fitted()
        return self.objective_of(self._cooccurrence, self._parameters)

    @property
    def embeddings(self) -> WordEmbeddings:
        """``w + w~`` per word, the paper's final vectors.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._embeddings

    @property
    def parameters(self) -> GloVeParameters:
        """All four learned tables together.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._parameters

    @property
    def word_vectors(self) -> FloatArray:
        """``(n_words, dimension)``, frozen: the word-role table alone.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._parameters.word_vectors

    @property
    def context_vectors(self) -> FloatArray:
        """``(n_words, dimension)``, frozen: the context-role table alone.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._parameters.context_vectors

    @property
    def biases(self) -> FloatArray:
        """``(n_words,)``, frozen: each word's bias in the word role.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._parameters.biases

    @property
    def context_biases(self) -> FloatArray:
        """``(n_words,)``, frozen: each word's bias in the context role.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._parameters.context_biases

    @property
    def history(self) -> TrainingHistory:
        """Each epoch's mean pair term, accumulated as the epoch ran.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._history

    @property
    def cooccurrence(self) -> CooccurrenceMatrix:
        """The matrix the fit was made on.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        return self._cooccurrence

    def _adagrad_step(
        self, gradient: FloatArray | float, accumulated: FloatArray | float
    ) -> FloatArray | float:
        """``learning_rate * gradient / (sqrt(accumulated) + epsilon)``.

        ``accumulated`` already holds this gradient's square, so the first step
        a parameter ever takes is the base rate in the gradient's direction and
        every later one is smaller. The epsilon keeps a zero gradient over a
        zero sum at a step of zero rather than ``0 / 0``; see
        :data:`ADAGRAD_EPSILON`.
        """
        return self.learning_rate * gradient / (np.sqrt(accumulated) + ADAGRAD_EPSILON)

    def _initial_values(
        self, generator: np.random.Generator, shape: tuple[int, ...]
    ) -> FloatArray:
        """Uniform in ``[-0.5, 0.5) / dimension``, the reference initialisation.

        Small, so no pair starts with a strong opinion, and spread, so the two
        tables start apart; with both tables random there is no symmetry to
        break the way word2vec's zero output table needs.
        """
        return (generator.random(shape) - 0.5) / self.dimension
