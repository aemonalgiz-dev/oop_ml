"""What a language model says about the next word: one number per word of the
vocabulary, and the questions asked of them.

A distribution rather than a table
----------------------------------
Asked what follows ``the cat``, a model answers for every word it knows at
once, and the answers only mean anything together: they sum to one, the
largest is the model's guess, and their spread is how unsure it is. A bare
array of probabilities loses which word each belongs to; a dictionary loses the
guarantee that they sum to one. :class:`NextWordDistribution` keeps both, the
vocabulary that names each position and the check that the numbers are a
distribution.

The check can be switched off for a method that answers scores rather than
probabilities, stupid backoff being the one it was written for, and the object
says so through :attr:`is_normalised` rather than pretending. Entropy is
refused on scores, because ``-sum p log p`` of numbers that do not sum to one
is not the entropy of anything. No smoothing method here answers scores yet,
so every distribution built by the language model today is normalised.

Sampling with a temperature
---------------------------
Generating text means drawing the next word from this distribution and
repeating. A temperature reshapes the draw: every probability is raised to
``1 / temperature`` and the result renormalised, so below one the likely words
become likelier and the text more predictable, and above one the tail comes
up and the text more surprising. At one the draw is the distribution itself.
The draw is by inversion of the cumulative sum against one uniform number, so
a seeded generator makes a seeded text.

Entropy in bits
---------------
``-sum p log2 p`` is how many bits the next word costs on average under the
model, and ``2 ** entropy`` is the effective number of choices: a
distribution spread evenly over eight words has three bits and eight
choices, and one certain of its word has zero of either. Perplexity is the
same idea measured against a held-out text rather than the model's own
opinion, which is why the two are reported in the same units.
"""

from __future__ import annotations

from collections.abc import Iterator

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

NORMALISATION_TOLERANCE = 1e-9
"""How far from one a distribution's total may be before it is refused."""


class WordProbability:
    """One word and the probability (or score) the model gave it.

    Parameters
    ----------
    word:
        The word. Non-empty.
    probability:
        Its probability, finite and non-negative.

    Raises
    ------
    EmptyValuesError
        If ``word`` is empty.
    InvalidValuesError
        If ``probability`` is negative or not finite.
    """

    __slots__ = ("_probability", "_word")

    def __init__(self, word: str, probability: float) -> None:
        if not isinstance(word, str) or not word:
            raise EmptyValuesError("a word probability needs a non-empty word")

        value = float(probability)
        if not np.isfinite(value) or value < 0.0:
            raise InvalidValuesError(
                f"a probability is finite and non-negative, got {probability!r} "
                f"for {word!r}"
            )

        self._word = word
        self._probability = value

    @property
    def word(self) -> str:
        """The word."""
        return self._word

    @property
    def probability(self) -> float:
        """What the model gave it."""
        return self._probability

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WordProbability):
            return NotImplemented
        return self._word == other._word and self._probability == other._probability

    def __hash__(self) -> int:
        return hash((self._word, self._probability))

    def __repr__(self) -> str:
        return f"WordProbability({self._word!r}, {self._probability:.4f})"


class NextWordDistribution:
    """A number per word of the vocabulary, summing to one unless it is a score.

    Parameters
    ----------
    vocabulary:
        Which word each position belongs to.
    values:
        One per token of the vocabulary, finite and non-negative. Copied and
        frozen.
    normalised:
        Whether the values are a probability distribution and must sum to
        one. False only for a scoring method such as stupid backoff, which is
        not implemented here, so the language model always passes True.

    Raises
    ------
    ShapeMismatchError
        If there is not one value per word.
    InvalidValuesError
        If a value is negative or not finite, or the values are declared
        normalised and do not sum to one.
    """

    __slots__ = ("_normalised", "_values", "_vocabulary")

    def __init__(
        self, vocabulary: Vocabulary, values: FloatArray, normalised: bool = True
    ) -> None:
        try:
            as_array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError(
                "a distribution's values must be numeric"
            ) from error

        if as_array.shape != (vocabulary.n_tokens,):
            raise ShapeMismatchError(
                f"a distribution has one value per word, got shape {as_array.shape} "
                f"for a vocabulary of {vocabulary.n_tokens}"
            )

        if not np.all(np.isfinite(as_array)) or bool(np.any(as_array < 0.0)):
            raise InvalidValuesError(
                "a distribution's values are finite and non-negative"
            )

        if normalised and abs(float(as_array.sum()) - 1.0) > NORMALISATION_TOLERANCE:
            raise InvalidValuesError(
                f"a probability distribution sums to one, got {float(as_array.sum())!r}"
            )

        frozen = as_array.copy()
        frozen.setflags(write=False)

        self._vocabulary = vocabulary
        self._values = frozen
        self._normalised = bool(normalised)

    @property
    def vocabulary(self) -> Vocabulary:
        """Which word each position belongs to."""
        return self._vocabulary

    @property
    def values(self) -> FloatArray:
        """One value per word, in vocabulary order, frozen."""
        return self._values

    @property
    def is_normalised(self) -> bool:
        """Whether the values are probabilities rather than scores."""
        return self._normalised

    @property
    def n_words(self) -> int:
        """How many words have a value."""
        return int(self._values.shape[0])

    @property
    def entropy(self) -> float:
        """``-sum p log2 p``, in bits: how uncertain the model is about the next
        word. Zero terms contribute nothing.

        Raises
        ------
        UndefinedMetricError
            If the values are scores rather than probabilities.
        """
        if not self._normalised:
            raise UndefinedMetricError(
                "entropy is defined for a probability distribution, and these "
                "values are scores that do not sum to one"
            )
        positive = self._values[self._values > 0.0]
        return float(-np.sum(positive * np.log2(positive)))

    def probability_of(self, word: str) -> float:
        """The value for ``word``, or for the unknown word if it is unknown.

        Raises
        ------
        UnknownTokenError
            If the word is unknown and the vocabulary has no unknown token.
        """
        return float(self._values[self._vocabulary.id_of(word)])

    def most_likely(self, n_results: int = 10) -> tuple[WordProbability, ...]:
        """The ``n_results`` words with the highest values, highest first.

        Ties break alphabetically, so the answer depends on the values and on
        nothing else.

        Raises
        ------
        InvalidValuesError
            If ``n_results`` is below one.
        """
        if n_results < 1:
            raise InvalidValuesError(f"n_results must be at least 1, got {n_results}")

        ranked = sorted(
            (
                WordProbability(word, float(self._values[position]))
                for position, word in enumerate(self._vocabulary)
            ),
            key=lambda entry: (-entry.probability, entry.word),
        )
        return tuple(ranked[:n_results])

    def sample(self, generator: np.random.Generator, temperature: float = 1.0) -> str:
        """Draw one word, with the values reshaped by ``temperature``.

        Raises
        ------
        InvalidValuesError
            If the temperature is not positive, or every value is zero.
        """
        if not temperature > 0.0:
            raise InvalidValuesError(f"a temperature is positive, got {temperature}")

        with np.errstate(divide="ignore"):
            log_weights = np.where(
                self._values > 0.0, np.log(self._values) / temperature, -np.inf
            )
        if not np.any(np.isfinite(log_weights)):
            raise InvalidValuesError("every value is zero, so nothing can be drawn")

        weights = np.exp(log_weights - np.max(log_weights))
        cumulative = np.cumsum(weights / weights.sum())
        position = int(np.searchsorted(cumulative, generator.random()))
        return self._vocabulary.token_of(min(position, self.n_words - 1))

    def __iter__(self) -> Iterator[WordProbability]:
        """Iterate every word with its value, in vocabulary order."""
        return (
            WordProbability(word, float(self._values[position]))
            for position, word in enumerate(self._vocabulary)
        )

    def __len__(self) -> int:
        return self.n_words

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __eq__(self, other: object) -> bool:
        """A verdict: the same words with the same values."""
        if not isinstance(other, NextWordDistribution):
            return NotImplemented
        return (
            self._vocabulary == other._vocabulary
            and self._normalised == other._normalised
            and bool(np.array_equal(self._values, other._values))
        )

    def __hash__(self) -> int:
        return hash((self._vocabulary, self._normalised, self._values.tobytes()))

    def __repr__(self) -> str:
        return (
            f"NextWordDistribution(n_words={self.n_words}, "
            f"normalised={self._normalised})"
        )
