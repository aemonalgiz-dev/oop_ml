"""A held-out estimate that costs nothing, and what it costs anyway.

Drawing ``n`` rows from ``n`` with replacement misses about 36.8% of them, so
every member of an averaging ensemble carries a set of rows it never saw. Line
the samples up and read down a column: for training row ``i``, the members whose
sample missed ``i`` are the only ones entitled to an opinion about it, because
they are the only ones that did not memorise it. Averaging just those gives an
honest prediction for that row, and doing it for every row gives a held-out
score without a split and without refitting anything.

Three properties, all of which the caller should know before trusting it.

**It is ragged.** Every row is judged by a different subset of members, and the
subsets are different sizes. This is not the fitted ensemble answering; it is a
different, smaller ensemble per row.

**Some rows have no judges.** A row is in-bag for every member with probability
``(1 - 1/e)^B``, which is around 10% at five members and around 1e-20 at a
hundred. Those rows have no honest prediction available and are excluded, so
the estimate is computed over fewer rows than the training set holds. The count
is reported rather than hidden, because at small ``n_members`` it stops being
negligible.

**It is biased pessimistic.** Each row is judged by roughly ``0.368 * B``
members rather than by ``B``, so the thing being measured is a smaller ensemble
than the one that was fitted, and a smaller averaging ensemble is a worse one.
Out-of-bag error therefore overstates the error of the model actually in hand.
That direction is the useful one -- a conservative estimate is safe to act on
in a way an optimistic one is not -- but it is a bias rather than a wobble, and
it does not shrink by adding rows.
"""

from __future__ import annotations

from oop_ml.core.types import FloatArray, MaskArray


class OutOfBagEstimate:
    """Predictions for the training rows, made only by members that missed them.

    A value object rather than a bare array, because the predictions are
    meaningless without knowing which rows they cover and how many members
    stood behind each one. Handing back the vector alone would let a caller
    average a row that no member was entitled to judge.

    Parameters
    ----------
    predictions:
        ``(n_rows,)``. Entries for rows with no out-of-bag member are not
        meaningful and are masked out by ``covered``.
    covered:
        ``(n_rows,)``, true where at least one member missed the row and an
        honest prediction was therefore available.
    judges:
        ``(n_rows,)`` counts of how many members judged each row. Zero exactly
        where ``covered`` is false. Worth reading directly: a fit where the
        smallest count is 1 is resting some of its estimate on single members.
    """

    __slots__ = ("_covered", "_judges", "_predictions")

    def __init__(
        self, predictions: FloatArray, covered: MaskArray, judges: FloatArray
    ) -> None:
        self._predictions = predictions
        self._covered = covered
        self._judges = judges

    @property
    def predictions(self) -> FloatArray:
        """``(n_rows,)``, only meaningful where ``covered`` is true."""
        return self._predictions

    @property
    def covered(self) -> MaskArray:
        """``(n_rows,)``, true where at least one member missed the row."""
        return self._covered

    @property
    def judges(self) -> FloatArray:
        """``(n_rows,)`` counts of the members that judged each row."""
        return self._judges

    @property
    def covered_predictions(self) -> FloatArray:
        """Just the rows that had a judge, in row order."""
        return self._predictions[self._covered]

    @property
    def n_covered(self) -> int:
        """How many training rows the estimate could speak to."""
        return int(self._covered.sum())

    @property
    def n_uncovered(self) -> int:
        """How many rows every member happened to draw, and so cannot judge."""
        return int((~self._covered).sum())

    @property
    def mean_judges(self) -> float:
        """The average number of members behind a covered row.

        Converges on ``0.368 * n_members``, and is the number that says how
        much smaller the measured ensemble is than the fitted one.
        """
        if self.n_covered == 0:
            return 0.0

        return float(self._judges[self._covered].mean())

    def __len__(self) -> int:
        return int(self._predictions.size)

    def __repr__(self) -> str:
        return (
            f"OutOfBagEstimate({self.n_covered} covered, "
            f"{self.n_uncovered} uncovered, "
            f"{self.mean_judges:.1f} judges per row)"
        )
