"""One merge a subword trainer chose, and the ordered list of all of them.

A merge is three things that mean nothing apart: the two symbols that were
adjacent, the symbol they became, and the figure that made this pair win over
every other candidate at that moment. :class:`Merge` binds them. :class:`Merges`
is the whole sequence in the order they were learned, and that order is the
model: encoding a new word applies the *earliest* applicable merge first, so a
merge's position in the list is its rank and two trainers that learned the same
pairs in a different order are different tokenizers.

The score is a ``float`` so that one class serves two trainers. Byte pair
encoding chooses by raw pair count, which is a whole number; WordPiece chooses
by likelihood gain, ``count(pair) / (count(left) count(right))``, which is not.
Storing an ``int`` for one and inventing a second class for the other would make
the two vocabularies harder to compare, and comparing them is the point of
having both.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError


class Merge:
    """Two adjacent symbols that became one, and why they were chosen.

    Parameters
    ----------
    left, right:
        The two symbols, in order. Each at least one character.
    score:
        The figure that chose this pair at the moment it was chosen: a pair
        count for byte pair encoding, a likelihood gain for WordPiece.

    Raises
    ------
    EmptyValuesError
        If either symbol is empty.
    """

    __slots__ = ("_left", "_right", "_score")

    def __init__(self, left: str, right: str, score: float) -> None:
        if not left or not right:
            raise EmptyValuesError("a merge joins two non-empty symbols")

        self._left = left
        self._right = right
        self._score = float(score)

    @property
    def left(self) -> str:
        """The first of the two symbols."""
        return self._left

    @property
    def right(self) -> str:
        """The second."""
        return self._right

    @property
    def merged(self) -> str:
        """The symbol the two became: their concatenation."""
        return self._left + self._right

    @property
    def score(self) -> float:
        """The figure that chose this pair."""
        return self._score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Merge):
            return NotImplemented
        return (
            self._left == other._left
            and self._right == other._right
            and self._score == other._score
        )

    def __hash__(self) -> int:
        return hash((self._left, self._right, self._score))

    def __repr__(self) -> str:
        return f"Merge({self._left!r} + {self._right!r}, score={self._score:g})"


class Merges:
    """Every merge a trainer learned, in the order it learned them.

    Parameters
    ----------
    merges:
        The merges in learned order. May be empty, since a corpus whose every
        word is one symbol long has nothing to merge. No pair may appear
        twice, because once merged a pair no longer exists to be merged again.

    Raises
    ------
    InvalidValuesError
        If a pair appears twice.
    """

    __slots__ = ("_merges", "_ranks")

    def __init__(self, merges: Sequence[Merge]) -> None:
        ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            pair = (merge.left, merge.right)
            if pair in ranks:
                raise InvalidValuesError(
                    f"the pair {merge.left!r} + {merge.right!r} is merged twice, "
                    f"at ranks {ranks[pair]} and {rank}"
                )
            ranks[pair] = rank

        self._merges = tuple(merges)
        self._ranks = ranks

    @property
    def n_merges(self) -> int:
        """How many merges were learned."""
        return len(self._merges)

    def rank_of(self, left: str, right: str) -> int | None:
        """Where this pair sits in the learned order, or ``None`` if never merged.

        Lower is earlier, and earlier is applied first when encoding.
        """
        return self._ranks.get((left, right))

    def __iter__(self) -> Iterator[Merge]:
        """Iterate the merges in learned order."""
        return iter(self._merges)

    def __len__(self) -> int:
        return len(self._merges)

    def __getitem__(self, rank: int) -> Merge:
        return self._merges[rank]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Merges):
            return NotImplemented
        return self._merges == other._merges

    def __hash__(self) -> int:
        return hash(self._merges)

    def __repr__(self) -> str:
        return f"Merges(n_merges={self.n_merges})"
