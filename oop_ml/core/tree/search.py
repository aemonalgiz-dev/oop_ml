"""The whole search a node performed, not just the question it settled on.

Why this exists
---------------
``_best_split`` answers "which question should this node ask" and throws away
everything it looked at to decide. That is right for fitting and useless for
inspection: the interesting part of a split is rarely the winner on its own, it
is the winner *against the field* -- that ``slept < 6.25`` scored 0.2133 while
the best cut on ``studied`` managed 0.1800, and that eleven candidates were
never eligible because they would have left a child too small.

So the search is modelled rather than discarded. A tool that wants to show how
a tree decides asks for a :class:`SplitSearch` and gets every candidate, its
gain, and the reason any of them were excluded.

Why it is a separate route from fitting
---------------------------------------
Because keeping the record is not free, and at fitting scale it is not close to
free. One node of a 5000-row, 20-feature problem considers about 100000
candidates; building an object for each costs 0.29s in pydantic and 0.05s with
``__slots__``, before a single impurity has been computed, and a tree has many
nodes.

Rather than pay that always or make the hot loop test a flag on every
iteration, the two live side by side: :meth:`~oop_ml.core.base.tree_model.
TreeModel._best_split` keeps a running winner and allocates almost nothing,
while :meth:`~oop_ml.core.base.tree_model.TreeModel.search_splits` records
everything. They are two readings of one definition, and a test asserts they
agree -- the same arrangement the neighbour models use for their threaded and
serial paths.

Nothing here is pydantic. These are constructed per candidate, and validation
per candidate is the cost this design exists to avoid.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum

from oop_ml.core.tree.split import Split


class SplitRejection(StrEnum):
    """Why a candidate split was not eligible, or that it was.

    A closed enum rather than a message, so a caller can group and count by
    reason without matching on prose -- and so that a new reason cannot be
    added without every reader of this type being made to notice.

    Attributes
    ----------
    ADMITTED:
        Eligible. Competed for the node, and may or may not have won.
    TOO_FEW_ROWS:
        Would have left one child holding fewer than ``min_samples_leaf``
        rows. Excluded before its gain mattered, because a split has to be
        ruled out before it can win rather than after.
    NO_GAIN:
        Scored exactly zero. The children have the same composition as the
        parent, so the question separates nothing.
    BELOW_MINIMUM_DECREASE:
        Scored above zero but below ``min_impurity_decrease``. A real
        separation that was judged not worth the structure it would cost.
    """

    ADMITTED = "admitted"
    TOO_FEW_ROWS = "too few rows"
    NO_GAIN = "no gain"
    BELOW_MINIMUM_DECREASE = "below minimum decrease"


class SplitCandidate:
    """One question a node considered, and what became of it.

    Parameters
    ----------
    split:
        The candidate itself, carrying its feature, threshold and gain. The
        gain is present even on a rejected candidate where one could be
        computed, because "excluded despite scoring well" is the interesting
        case and hiding the number would conceal it.
    rejection:
        Whether it was eligible, and if not, why not.
    rows_left, rows_right:
        How the node's rows divided. What ``min_samples_leaf`` is judged
        against, and the thing a reader wants beside the gain.
    """

    __slots__ = ("_rejection", "_rows_left", "_rows_right", "_split")

    def __init__(
        self,
        split: Split,
        rejection: SplitRejection,
        rows_left: int,
        rows_right: int,
    ) -> None:
        self._split = split
        self._rejection = rejection
        self._rows_left = rows_left
        self._rows_right = rows_right

    @property
    def split(self) -> Split:
        """The candidate question, with its gain."""
        return self._split

    @property
    def rejection(self) -> SplitRejection:
        """Whether it was eligible, and why not if it was not."""
        return self._rejection

    @property
    def was_admitted(self) -> bool:
        """Whether this candidate competed for the node."""
        return self._rejection is SplitRejection.ADMITTED

    @property
    def rows_left(self) -> int:
        """How many rows would have gone left."""
        return self._rows_left

    @property
    def rows_right(self) -> int:
        """How many rows would have gone right."""
        return self._rows_right

    def __repr__(self) -> str:
        return (
            f"SplitCandidate({self._split!r}, {self._rejection.value}, "
            f"{self._rows_left}/{self._rows_right})"
        )


class SplitSearch:
    """Every candidate one node considered, and the winner among them.

    Iterate it to walk the candidates in the order they were scanned --
    features in fitted order, thresholds ascending -- which is the order a
    step-by-step explanation wants to present them in.

    Parameters
    ----------
    candidates:
        Every ``(feature, threshold)`` pair the node evaluated, admitted or
        not.
    """

    __slots__ = ("_candidates",)

    def __init__(self, candidates: Sequence[SplitCandidate]) -> None:
        self._candidates = tuple(candidates)

    @property
    def best(self) -> Split | None:
        """The admitted candidate with the highest gain, or ``None``.

        ``None`` when nothing was admitted: every column constant, every split
        too lopsided, or nothing clearing the gain rules. The same answer
        ``_best_split`` gives, by the same tie-break -- an exact tie keeps the
        earlier candidate, so the winner does not depend on how the scan was
        arranged.
        """
        winner: SplitCandidate | None = None

        for candidate in self._candidates:
            if not candidate.was_admitted:
                continue
            if candidate.split.beats(None if winner is None else winner.split):
                winner = candidate

        return None if winner is None else winner.split

    @property
    def result(self) -> Split | None:
        """What the efficient path would have returned. See :meth:`best`."""
        return self.best

    @property
    def admitted(self) -> tuple[SplitCandidate, ...]:
        """The candidates that were eligible to win."""
        return tuple(
            candidate for candidate in self._candidates if candidate.was_admitted
        )

    def rejected_for(self, reason: SplitRejection) -> tuple[SplitCandidate, ...]:
        """The candidates excluded for one particular reason."""
        return tuple(
            candidate for candidate in self._candidates if candidate.rejection is reason
        )

    def for_feature(self, feature_name: str) -> tuple[SplitCandidate, ...]:
        """Every candidate that was a cut on one named feature.

        By name rather than index, because a reader looking at a scoreboard
        knows the column as ``slept`` and not as 1.
        """
        return tuple(
            candidate
            for candidate in self._candidates
            if candidate.split.feature_name == feature_name
        )

    def __iter__(self) -> Iterator[SplitCandidate]:
        return iter(self._candidates)

    def __len__(self) -> int:
        return len(self._candidates)

    def __repr__(self) -> str:
        return (
            f"SplitSearch({len(self._candidates)} candidates, "
            f"{len(self.admitted)} admitted, best={self.best!r})"
        )
