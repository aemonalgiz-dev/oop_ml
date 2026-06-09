"""A run of ``n`` adjacent words, and the three markers a sentence is framed in.

What an n-gram is, and why it is the oldest language model
-----------------------------------------------------------
Take a sentence and slide a window of ``n`` words along it. Each window is an
n-gram: a unigram is one word, a bigram two, a trigram three. Count how often
each window appears across a corpus and you have a model of the language that
Shannon (1948) wrote down before there were computers to run it on: the
probability of the next word depends on the ``n - 1`` words before it and on
nothing earlier. That is a false assumption about language and a useful one
about counting, since a table of trigrams can be filled from text alone and
read in constant time.

The three markers
-----------------
A sentence has edges, and the model has to see them. ``<s>`` stands before the
first word so that the first word has a context to be predicted from, and a
trigram model needs two of them; ``</s>`` stands after the last word so that
the model can learn where sentences end, which is what lets it stop generating.
``<unk>`` is the word every word the model was not taught becomes, so that a
held-out text is never simply unscorable: the vocabulary is closed, and a
closed vocabulary answers every lookup.

The markers are reserved. A corpus that contains one of them as an ordinary
word is refused rather than read, because a text with ``</s>`` in the middle
would teach the model that sentences end where they do not.

Why ``<s>`` is never predicted
------------------------------
Every n-gram counted ends in a word the model might be asked to predict, and a
sentence start is not one: it is where the model begins, not where it can go.
So n-grams whose last word is ``<s>`` are not counted, and ``<s>`` is not in
the prediction vocabulary. It appears only as context.

Character n-grams
-----------------
The same window slid along a word's characters instead of a sentence's words.
``<where>`` at three gives ``<wh whe her ere re>``, which is FastText's
example and the same pieces its own hashing reads; here they serve the
bag-of-n-grams document features, where character pieces survive a misspelling
that would lose a whole word.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError

SENTENCE_START = "<s>"
"""Stands before a sentence, ``order - 1`` times, so the first word has context."""

SENTENCE_END = "</s>"
"""Stands after a sentence, once; predicted like a word, so the model can stop."""

UNKNOWN_WORD = "<unk>"
"""What every word outside the vocabulary becomes, so the vocabulary is closed."""

RESERVED_WORDS: frozenset[str] = frozenset({SENTENCE_START, SENTENCE_END, UNKNOWN_WORD})
"""The markers, which a *raw corpus* may not contain as ordinary words."""

FRAMING_MARKERS: frozenset[str] = frozenset({SENTENCE_START, SENTENCE_END})
"""The two a caller may never supply, because :func:`padded` adds them itself.

Narrower than :data:`RESERVED_WORDS`, and the difference is the point. A raw
corpus is refused if it holds any of the three. By the time sentences reach
counting they have been reduced to a vocabulary, so ``<unk>`` is expected
there and refusing it would make the substitution impossible to perform. The
framing markers stay refused at both steps, because one supplied by a caller
would be counted a second time and shift every context along by a position.
"""


class NGram:
    """A run of one or more adjacent words.

    Parameters
    ----------
    words:
        The words, in order. At least one, each a non-empty string.

    Raises
    ------
    EmptyValuesError
        If there are no words.
    InvalidValuesError
        If a word is not a non-empty string.
    """

    __slots__ = ("_words",)

    def __init__(self, words: Sequence[str]) -> None:
        if isinstance(words, str) or len(words) == 0:
            raise EmptyValuesError(
                "an n-gram is a sequence of at least one word, not a string"
            )

        for word in words:
            if not isinstance(word, str) or not word:
                raise InvalidValuesError(
                    f"every word of an n-gram is a non-empty string, got {word!r}"
                )

        self._words = tuple(words)

    @property
    def words(self) -> tuple[str, ...]:
        """The words, in order."""
        return self._words

    @property
    def order(self) -> int:
        """How many words: one for a unigram, two for a bigram."""
        return len(self._words)

    @property
    def context(self) -> tuple[str, ...]:
        """Every word but the last: what the last is predicted from."""
        return self._words[:-1]

    @property
    def word(self) -> str:
        """The last word: what is predicted."""
        return self._words[-1]

    def __iter__(self) -> Iterator[str]:
        return iter(self._words)

    def __len__(self) -> int:
        return len(self._words)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NGram):
            return NotImplemented
        return self._words == other._words

    def __hash__(self) -> int:
        return hash(self._words)

    def __repr__(self) -> str:
        return f"NGram({' '.join(self._words)!r})"


def checked_order(order: int) -> int:
    """``order`` as an int of at least one.

    Raises
    ------
    InvalidValuesError
        If ``order`` is below one.
    """
    if order < 1:
        raise InvalidValuesError(f"an n-gram order is at least 1, got {order}")
    return int(order)


def padded(words: Sequence[str], order: int) -> tuple[str, ...]:
    """The sentence framed for an ``order``-gram model.

    ``order - 1`` sentence starts in front, one sentence end behind, so that
    every position from the first word to the end marker has a full context.

    Raises
    ------
    InvalidValuesError
        If ``order`` is below one.
    """
    starts = (SENTENCE_START,) * (checked_order(order) - 1)
    return (*starts, *words, SENTENCE_END)


def word_n_grams(words: Sequence[str], order: int) -> tuple[NGram, ...]:
    """Every window of ``order`` adjacent words, left to right, unpadded.

    Fewer words than the order gives no n-grams at all, which is the honest
    answer rather than a shorter one.

    Raises
    ------
    InvalidValuesError
        If ``order`` is below one.
    """
    checked = checked_order(order)
    return tuple(
        NGram(words[start : start + checked])
        for start in range(len(words) - checked + 1)
    )


def character_n_grams(
    word: str, order: int, start_marker: str = "", end_marker: str = ""
) -> tuple[str, ...]:
    """Every window of ``order`` adjacent characters of ``word``, with the
    markers on either side.

    With markers of ``<`` and ``>`` a word's first and last pieces are
    distinct from the same letters mid-word, which is what makes ``her`` in
    ``where`` a different piece from the word ``her``. Repeated pieces are
    kept, in order; a caller wanting the set takes it.

    Raises
    ------
    InvalidValuesError
        If ``order`` is below one, or ``word`` is not a string.
    """
    checked = checked_order(order)
    if not isinstance(word, str):
        raise InvalidValuesError(f"a word is a string, got {type(word).__name__}")

    wrapped = f"{start_marker}{word}{end_marker}"
    return tuple(
        wrapped[start : start + checked] for start in range(len(wrapped) - checked + 1)
    )
