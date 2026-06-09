"""The pair-merging arithmetic several tokenizers share, written once.

Byte pair encoding, WordPiece, the byte-level variant, SuperBPE, the
SentencePiece framework's BPE mode and the morpheme-constrained one all do the
same two things: while learning, find the best adjacent pair across a weighted
set of symbol sequences and replace it everywhere; while encoding, apply the
learned merges to one sequence, earliest rank first. They differ in how a word
is first spelled in symbols, in whether merges may cross a boundary, and in what
"best" means -- and in nothing else. Six copies of the loop would be six places
for the tie rule to drift, so the loop lives here and each tokenizer supplies
its spelling.

What "best" means, twice
------------------------
:attr:`PairScoring.FREQUENCY` is Sennrich's rule: the pair that occurs most
often, weighted by how often each word occurs. :attr:`PairScoring.LIKELIHOOD`
is WordPiece's: the pair whose merge most increases the likelihood of the corpus
under a unigram model of the symbols, which works out to
``count(pair) / (count(left) * count(right))``. The difference is what it
promotes. Frequency favours pairs of common symbols whether or not they belong
together (``e`` followed by ``s`` is frequent partly because both are frequent);
likelihood favours pairs that occur together *more than their parts predict*,
so a rare symbol that always appears beside one particular neighbour merges
early. On the Sennrich corpus frequency's first merge is ``e s`` at 9; under
likelihood ``9 / (17 * 9)`` loses to ``i d`` at ``3 / (3 * 3) = 1/3``, a pair
whose symbols never appear apart.

Either way a pair must still occur ``minimum_pair_frequency`` times, because a
likelihood ratio computed from one occurrence is a fact about one word. The
minimum filters the candidates before the best is chosen; it is not a test on
the winner that ends the walk. Under frequency scoring nothing distinguishes the
two, and under likelihood scoring the distinction is the whole point: on the
Sennrich corpus at a minimum of 3, the stopping-test reading learned five merges
and quit at a count-2 pair while count-9 pairs were still unmerged.

The tie rule
------------
Scores tie constantly on a small corpus and every later merge depends on which
pair won. The tie goes to the lexicographically smaller ``(left, right)`` pair,
so the same corpus gives the same vocabulary whatever order its texts arrive
in. Arbitrary, and stated, like :meth:`~oop_ml.core.tree.split.Split.beats`,
and stated once here rather than once per tokenizer.

Why encoding merges one occurrence at a time
--------------------------------------------
:func:`apply_merges` finds the lowest-ranked applicable pair and merges its
leftmost occurrence only, then looks again. Without dropout that reaches the
same answer as merging every occurrence at once, since the same pair has the
same rank and wins the next step too. With dropout the two differ: each
occurrence has to be given its own chance to be skipped, or a word with the same
pair twice would drop both or neither.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from enum import StrEnum

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.natural_language_processing.tokenization.subword.merges import (
    Merge,
    Merges,
)


class PairScoring(StrEnum):
    """How a candidate pair is ranked against the others at each merge step."""

    FREQUENCY = "frequency"
    """The weighted count of the pair. Sennrich et al. (2016)."""

    LIKELIHOOD = "likelihood"
    """``count(pair) / (count(left) count(right))``. Schuster and Nakajima's
    WordPiece, as described by Wu et al. (2016)."""


class SpelledWord:
    """One distinct word as a sequence of symbols, with how often it occurs.

    Parameters
    ----------
    symbols:
        The word's current spelling, at least one symbol, none empty.
    count:
        How many times the corpus used the word. At least one.

    Raises
    ------
    EmptyValuesError
        If there are no symbols or one of them is empty.
    InvalidValuesError
        If ``count`` is below one.
    """

    __slots__ = ("_count", "_symbols")

    def __init__(self, symbols: Sequence[str], count: int) -> None:
        if len(symbols) == 0 or any(not symbol for symbol in symbols):
            raise EmptyValuesError("a spelled word needs at least one non-empty symbol")

        if count < 1:
            raise InvalidValuesError(
                f"a word of the corpus appeared at least once, got count {count}"
            )

        self._symbols = tuple(symbols)
        self._count = int(count)

    @property
    def symbols(self) -> tuple[str, ...]:
        """The current spelling."""
        return self._symbols

    @property
    def count(self) -> int:
        """How often the corpus used this word."""
        return self._count

    def with_pair_merged(self, merge: Merge) -> SpelledWord:
        """The same word with every occurrence of the pair joined."""
        return SpelledWord(with_pair_merged(self._symbols, merge), self._count)

    def __iter__(self) -> Iterator[str]:
        return iter(self._symbols)

    def __len__(self) -> int:
        return len(self._symbols)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpelledWord):
            return NotImplemented
        return self._symbols == other._symbols and self._count == other._count

    def __hash__(self) -> int:
        return hash((self._symbols, self._count))

    def __repr__(self) -> str:
        return f"SpelledWord({list(self._symbols)!r}, count={self._count})"


def alphabet_of(spelled: Sequence[SpelledWord]) -> tuple[str, ...]:
    """Every distinct symbol across the words, in codepoint order."""
    return tuple(sorted({symbol for word in spelled for symbol in word}))


def best_pair(
    spelled: Sequence[SpelledWord],
    scoring: PairScoring,
    minimum_pair_frequency: int = 1,
) -> Merge | None:
    """The adjacent pair that ranks highest under ``scoring``, or ``None``.

    Only pairs occurring at least ``minimum_pair_frequency`` times are
    candidates, so the minimum is a filter and not merely a stopping test.
    Under frequency scoring the two readings coincide, since the winner is the
    most frequent pair and below the minimum means every pair is; under
    likelihood scoring they do not, because a pair seen twice whose symbols
    are rare can outscore one seen nine times, and a stopping test would let
    it win and then end the walk with frequent pairs still unmerged.

    The merge's score is the *count* of the pair under either rule, so that
    the minimum means the same thing whichever rule chose the pair. Ties go to
    the lexicographically smaller pair. ``None`` when no candidate is left.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    symbol_counts: Counter[str] = Counter()
    for word in spelled:
        for symbol in word:
            symbol_counts[symbol] += word.count
        for left, right in zip(word.symbols, word.symbols[1:], strict=False):
            pair_counts[(left, right)] += word.count

    candidates = {
        pair: count
        for pair, count in pair_counts.items()
        if count >= minimum_pair_frequency
    }
    if not candidates:
        return None

    def ranking(item: tuple[tuple[str, str], int]) -> tuple[float, str, str]:
        (left, right), count = item
        if scoring is PairScoring.LIKELIHOOD:
            score = count / (symbol_counts[left] * symbol_counts[right])
        else:
            score = float(count)
        return (-score, left, right)

    (left, right), count = min(candidates.items(), key=ranking)
    return Merge(left, right, count)


def with_pair_merged(symbols: Sequence[str], merge: Merge) -> tuple[str, ...]:
    """``symbols`` with every occurrence of the pair joined, left to right."""
    merged: list[str] = []
    position = 0
    while position < len(symbols):
        if (
            position + 1 < len(symbols)
            and symbols[position] == merge.left
            and symbols[position + 1] == merge.right
        ):
            merged.append(merge.merged)
            position += 2
        else:
            merged.append(symbols[position])
            position += 1
    return tuple(merged)


def learn_merges(
    spelled: Sequence[SpelledWord],
    n_merges: int,
    minimum_pair_frequency: int,
    scoring: PairScoring = PairScoring.FREQUENCY,
) -> Merges:
    """Up to ``n_merges`` merges, learned greedily over the spelled words.

    Stops early when no pair occurring at least ``minimum_pair_frequency``
    times is left, so the result may be shorter than asked.

    Parameters
    ----------
    spelled:
        Every distinct word in its starting spelling, with its count.
    n_merges:
        How many merges to learn at most.
    minimum_pair_frequency:
        A pair seen fewer times than this is never merged.
    scoring:
        What makes one pair better than another.
    """
    current = list(spelled)
    merges: list[Merge] = []
    while len(merges) < n_merges:
        best = best_pair(current, scoring, minimum_pair_frequency)
        if best is None:
            break
        current = [word.with_pair_merged(best) for word in current]
        merges.append(best)
    return Merges(merges)


def apply_merges(
    symbols: Sequence[str],
    merges: Merges,
    dropout: float = 0.0,
    generator: random.Random | None = None,
) -> tuple[str, ...]:
    """Apply learned merges to one symbol sequence, earliest rank first.

    Parameters
    ----------
    symbols:
        A word's starting spelling.
    merges:
        The learned merges; rank is the order they apply in.
    dropout:
        Probability of skipping each applicable occurrence at each step. Zero
        is ordinary, deterministic merging.
    generator:
        Draws the skips. Required when ``dropout`` is positive.

    Raises
    ------
    InvalidValuesError
        If ``dropout`` is positive and no generator was supplied.
    """
    if dropout > 0.0 and generator is None:
        raise InvalidValuesError("merge dropout needs a generator to draw from")

    current = list(symbols)
    while len(current) > 1:
        best_rank: int | None = None
        best_position = 0
        for position in range(len(current) - 1):
            rank = merges.rank_of(current[position], current[position + 1])
            if rank is None:
                continue
            if dropout > 0.0 and generator is not None and generator.random() < dropout:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_position = position

        if best_rank is None:
            break

        current[best_position : best_position + 2] = [
            current[best_position] + current[best_position + 1]
        ]
    return tuple(current)
