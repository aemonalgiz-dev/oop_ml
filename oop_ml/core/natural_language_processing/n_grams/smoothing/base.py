"""What a smoothing method is: a rule from counts to a probability, and the two
rules every other one is measured against.

The problem every method here solves
------------------------------------
Divide the count of ``the cat`` by the count of ``the`` and you have the
maximum-likelihood estimate of ``P(cat | the)``. It is the estimate that makes
the training corpus most probable, and it is useless for anything else, because
every n-gram the corpus did not contain gets probability zero, a sentence
containing one gets probability zero, and its perplexity is infinite. Language
is mostly n-grams nobody has written down yet: on any real corpus a third or
more of the trigrams in a new text are new. So every practical model takes
probability away from what was seen and gives it to what was not, and the
methods differ only in how much to take and where to put it. That is what
smoothing means.

The contract
------------
A smoothing method reads a fitted
:class:`~oop_ml.core.natural_language_processing.n_grams.counts.NGramCounts`
and answers ``P(word | context)`` for a context of at most ``order - 1`` words.
It is a pydantic model rather than an enum for the reason
:mod:`oop_ml.core.kernel` gives: most methods carry a parameter, a discount or
an additive constant, and an enum member cannot. :meth:`Smoothing.probability_of`
is the template, checking the context's length once; a subclass supplies
:meth:`Smoothing._probability` and, when it recurses to a shorter context,
does so through the same counts.

Every method here is normalised -- its probabilities over the vocabulary sum to
one for every context -- and the language model's spec asserts that for each,
because it is the property a smoothing method most easily loses.

:attr:`Smoothing.is_normalised` exists for the method that will lose it on
purpose. Stupid backoff answers a relative score rather than a probability, so
a perplexity over its numbers would be meaningless and the model refuses to
report one. Nothing here answers False today, so that refusal is unreached
until stupid backoff is written.

The two here
------------
:class:`MaximumLikelihood` is the estimate above, kept as the control: any
other method must agree with it on frequent n-grams and disagree on absent
ones. An unseen context has no trials at all, and the answer is zero rather
than an error, since a probability of zero is exactly the failure the method
is named for.

:class:`AdditiveSmoothing` is Laplace's rule from 1814, ``(C(hw) + k) /
(C(h) + kV)``: pretend every one of the ``V`` words followed every context
``k`` more times than it did. At ``k = 1`` on ``the cat sat / the cat
ran / a dog sat``, whose vocabulary is eight words (the six seen, the end
marker and the unknown word), ``P(cat | the)`` falls from ``2 / 2 = 1`` to
``3 / 10``, and the seven words that never followed ``the`` share ``7 / 10``
between them. That is far too much taken from the seen -- Gale and
Church (1994) measured it giving unseen bigrams a thousand times their real
share on the AP corpus -- which is why it is the baseline the others improve
on and not a method anyone ships.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.natural_language_processing.n_grams.counts import NGramCounts


class Smoothing(BaseModel, ABC):
    """A rule from n-gram counts to ``P(word | context)``.

    A pydantic model so that a method's parameter is validated at
    construction, where every other hyperparameter in this library is.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    def is_normalised(self) -> bool:
        """Whether the probabilities over the vocabulary sum to one.

        True for both methods implemented here. The property exists for
        stupid backoff, which answers a relative score rather than a
        probability; until that is written nothing returns False.
        """
        return True

    def probability_of(
        self, counts: NGramCounts, context: Sequence[str], word: str
    ) -> float:
        """``P(word | context)`` under this method, read from ``counts``.

        Parameters
        ----------
        counts:
            The fitted table.
        context:
            At most ``counts.order - 1`` words, the most recent last. Shorter
            contexts are allowed and are answered by the corresponding lower
            order, which is what a method recursing to a shorter context asks.
        word:
            The word predicted. Should be in ``counts.vocabulary``; a word
            outside it has a count of zero everywhere and is what the unknown
            word exists for.

        Raises
        ------
        InvalidValuesError
            If the context is too long for the table, or ``word`` is empty.
        """
        context_words = tuple(context)
        if len(context_words) > counts.order - 1:
            raise InvalidValuesError(
                f"an order-{counts.order} model conditions on at most "
                f"{counts.order - 1} words, got {len(context_words)}"
            )
        if not isinstance(word, str) or not word:
            raise InvalidValuesError("the predicted word is a non-empty string")
        return float(self._probability(counts, context_words, word))

    @abstractmethod
    def _probability(
        self, counts: NGramCounts, context: tuple[str, ...], word: str
    ) -> float:
        """The rule itself, on a context already known to fit the table."""


class MaximumLikelihood(Smoothing):
    """``C(context word) / C(context)``, and zero for anything unseen.

    The control every other method is measured against, and the one that
    gives a new sentence probability zero.
    """

    def _probability(
        self, counts: NGramCounts, context: tuple[str, ...], word: str
    ) -> float:
        total = counts.context_total(context)
        if total == 0:
            return 0.0
        return counts.count_of((*context, word)) / total


class AdditiveSmoothing(Smoothing):
    """``(C(context word) + k) / (C(context) + k V)``: Laplace's rule.

    Parameters
    ----------
    pretended_count:
        How many times every word is pretended to have followed every
        context, the ``k`` of the formula above. One is Laplace; a fraction
        (Lidstone's rule) takes less from the seen.
    """

    pretended_count: float = Field(default=1.0, gt=0.0)

    def _probability(
        self, counts: NGramCounts, context: tuple[str, ...], word: str
    ) -> float:
        vocabulary_size = counts.vocabulary.n_tokens
        return (counts.count_of((*context, word)) + self.pretended_count) / (
            counts.context_total(context) + self.pretended_count * vocabulary_size
        )
