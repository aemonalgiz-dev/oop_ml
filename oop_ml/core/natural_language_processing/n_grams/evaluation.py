"""How well a language model predicts text it did not see, as one object.

Perplexity, in plain words
--------------------------
Score a held-out text by the probability the model gives each of its words in
turn, multiply them up, and ask: over how many equally likely choices would a
model have to be guessing to do this well? That number is the perplexity. A
model that gives every word one chance in a hundred has perplexity 100; a model
certain of every word has perplexity 1. It is ``exp`` of the average negative
log probability per word, so it is the geometric mean of the inverse
probabilities, and lower is better.

Cross-entropy is the same quantity in bits per word, ``-(1 / M) sum log2 P``,
and perplexity is two to that power. Both are reported here because a reader
of the literature meets both, and neither is more fundamental.

What counts as a word
---------------------
Every predicted position, which includes the end marker of every sentence and
excludes the start markers, since those are context and not events. A
held-out text is scored with the fitted vocabulary, so a word the model never
saw is scored as the unknown word; how many were is reported, because a
perplexity over a text that was mostly ``<unk>`` is a perplexity over
``<unk>``, and a reader deserves to know.

A zero
------
A model that gives one word of the text probability zero gives the whole text
probability zero, and its perplexity is infinite. That is a real answer and
the one maximum-likelihood estimation produces on almost any held-out text; it
is why every other smoothing method exists, and it is reported as ``inf``
rather than raised, because "this model cannot score this text" is a finding
rather than an error.
"""

from __future__ import annotations

import math

from oop_ml.core.exceptions import InvalidValuesError


class LanguageModelEvaluation:
    """The probability a model gave a held-out text, and the two readings of it.

    Parameters
    ----------
    log_probability:
        The natural log of the probability of the whole text: the sum over
        every predicted word. At most zero; ``-inf`` if any word had
        probability zero.
    n_words:
        How many words were predicted, end markers included. At least one.
    n_unknown:
        How many of them were the unknown word, between zero and ``n_words``.

    Raises
    ------
    InvalidValuesError
        If the log probability is positive or ``nan``, there are no words, or
        the unknown count is out of range.
    """

    __slots__ = ("_log_probability", "_n_unknown", "_n_words")

    def __init__(self, log_probability: float, n_words: int, n_unknown: int) -> None:
        value = float(log_probability)
        if math.isnan(value) or value > 0.0:
            raise InvalidValuesError(
                f"a log probability is at most zero, got {log_probability!r}"
            )
        if n_words < 1:
            raise InvalidValuesError(
                f"an evaluation needs at least one word, got {n_words}"
            )
        if not 0 <= n_unknown <= n_words:
            raise InvalidValuesError(
                f"n_unknown lies between 0 and n_words={n_words}, got {n_unknown}"
            )

        self._log_probability = value
        self._n_words = int(n_words)
        self._n_unknown = int(n_unknown)

    @property
    def log_probability(self) -> float:
        """The natural log of the whole text's probability; ``-inf`` for a zero."""
        return self._log_probability

    @property
    def n_words(self) -> int:
        """How many words were predicted, end markers included."""
        return self._n_words

    @property
    def n_unknown(self) -> int:
        """How many of them the model scored as the unknown word."""
        return self._n_unknown

    @property
    def unknown_share(self) -> float:
        """What fraction of the predicted words were unknown."""
        return self._n_unknown / self._n_words

    @property
    def cross_entropy(self) -> float:
        """Bits per word: ``-log2 P / n_words``. ``inf`` for a zero."""
        if math.isinf(self._log_probability):
            return math.inf
        return -self._log_probability / (self._n_words * math.log(2.0))

    @property
    def perplexity(self) -> float:
        """``exp(-log P / n_words)``: the effective number of equally likely
        choices per word. ``inf`` for a zero, one for a certain model."""
        if math.isinf(self._log_probability):
            return math.inf
        return math.exp(-self._log_probability / self._n_words)

    @property
    def is_finite(self) -> bool:
        """Whether the model gave every word some probability."""
        return not math.isinf(self._log_probability)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LanguageModelEvaluation):
            return NotImplemented
        return (
            self._log_probability == other._log_probability
            and self._n_words == other._n_words
            and self._n_unknown == other._n_unknown
        )

    def __hash__(self) -> int:
        return hash((self._log_probability, self._n_words, self._n_unknown))

    def __repr__(self) -> str:
        return (
            f"LanguageModelEvaluation(perplexity={self.perplexity:.4g}, "
            f"n_words={self._n_words}, n_unknown={self._n_unknown})"
        )
