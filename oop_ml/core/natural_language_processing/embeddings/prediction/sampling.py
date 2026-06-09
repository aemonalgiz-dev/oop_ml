"""Where negative samples come from: the unigram distribution, flattened.

Why negatives are needed at all
-------------------------------
A skip-gram model is asked, for a centre word, how likely each other word is
to appear beside it. Answering that exactly is a softmax over the whole
vocabulary, and its gradient touches every output vector for every training
pair -- forty thousand rows per word seen. Mikolov et al. (2013) replaced the
question with a cheaper one: given this centre word, is this particular word a
true neighbour or a word drawn at random? A handful of random words per pair,
and only their rows move. The random words are the negatives, and how they are
drawn is the whole of this module.

Why the counts are raised to the three quarters
-----------------------------------------------
Drawn in proportion to frequency, the negatives would nearly always be ``the``,
``of`` and ``and``, and the model would learn mostly to say no to those. Drawn
uniformly, every rare word would be a negative as often as a common one, which
tells the model nothing about the words it actually meets. Word2vec raises each
count to the power ``0.75`` before normalising, which flattens the distribution
part-way towards uniform. On counts of ``1000, 10, 1`` the raw shares are
``0.989, 0.010, 0.001``; after the exponent they are ``0.964, 0.030, 0.005``,
so the rarest word is drawn five times as often and the commonest a little less.
The paper reports the value as found by trial, and it has stayed.

Exact rather than tabulated
---------------------------
The reference implementation fills a table of a hundred million cells with word
ids in proportion to their share and indexes it at random, which approximates
the distribution to one part in a hundred million. Here the cumulative shares
are searched directly with a uniform draw, which is the same distribution
exactly and needs no table. A draw that lands on the word being predicted is
redrawn, since a word is not its own negative.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import DTypeLike

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.types import FloatArray, IndexArray, array_for_protocol


class UnigramSampler:
    """Draws word ids in proportion to ``count ** exponent``.

    Parameters
    ----------
    counts:
        One count per word id, each at least one.
    exponent:
        The flattening power, in ``(0, 1]``. Word2vec's value is ``0.75``; one
        draws in proportion to raw frequency.

    Raises
    ------
    EmptyValuesError
        If there are no counts.
    InvalidValuesError
        If a count is below one or the exponent is outside ``(0, 1]``.
    """

    __slots__ = ("_cumulative", "_probabilities")

    def __init__(self, counts: Sequence[int], exponent: float = 0.75) -> None:
        if len(counts) == 0:
            raise EmptyValuesError("a sampler needs at least one word to draw")

        if any(count < 1 for count in counts):
            raise InvalidValuesError(
                "every word of the vocabulary was seen at least once"
            )

        if not 0.0 < exponent <= 1.0:
            raise InvalidValuesError(
                f"the flattening exponent lies in (0, 1], got {exponent}"
            )

        weights = np.asarray(counts, dtype=np.float64) ** exponent
        probabilities = weights / weights.sum()
        probabilities.setflags(write=False)

        self._probabilities = probabilities
        self._cumulative = np.cumsum(probabilities)

    @property
    def probabilities(self) -> FloatArray:
        """The share of draws each word id receives, frozen, summing to one."""
        return self._probabilities

    @property
    def n_words(self) -> int:
        """How many word ids can be drawn."""
        return int(self._probabilities.shape[0])

    def probability_of(self, word_id: int) -> float:
        """The share of draws ``word_id`` receives."""
        return float(self._probabilities[word_id])

    def draw(
        self,
        generator: np.random.Generator,
        n_draws: int,
        excluding: int | None = None,
    ) -> IndexArray:
        """``n_draws`` word ids, independently, none equal to ``excluding``.

        Raises
        ------
        InvalidValuesError
            If ``n_draws`` is below one, or every word is excluded.
        """
        if n_draws < 1:
            raise InvalidValuesError(f"n_draws must be at least 1, got {n_draws}")

        if excluding is not None and self.n_words == 1:
            raise InvalidValuesError(
                "a vocabulary of one word has nothing to draw as a negative"
            )

        drawn = np.searchsorted(self._cumulative, generator.random(n_draws))
        drawn = np.minimum(drawn, self.n_words - 1)
        if excluding is not None:
            while True:
                collisions = drawn == excluding
                if not np.any(collisions):
                    break
                replacements = np.searchsorted(
                    self._cumulative, generator.random(int(collisions.sum()))
                )
                drawn[collisions] = np.minimum(replacements, self.n_words - 1)
        return drawn.astype(np.intp)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._probabilities, dtype, copy)

    def __repr__(self) -> str:
        return f"UnigramSampler(n_words={self.n_words})"
