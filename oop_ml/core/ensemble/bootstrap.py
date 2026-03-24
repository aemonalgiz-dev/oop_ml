"""Drawing a resample, and remembering which rows it missed.

A bootstrap sample is ``n`` rows drawn from ``n`` rows *with replacement*, so
some arrive several times and some not at all. That is the whole mechanism
behind bagging: it is what makes members fitted on the same data differ from
each other, and members that differ are the only thing an average has to work
with.

The rows it misses are not waste. Drawing ``n`` times from ``n`` rows, any
particular row is missed with probability ``(1 - 1/n)^n``, which converges to
``1/e`` -- about 36.8% -- almost immediately. Every member therefore has roughly
a third of the training set that it never saw, and those rows are a held-out set
that cost nothing to obtain. Averaging each row's prediction over only the
members that missed it gives an honest error estimate without a separate split
and without refitting anything, which is the out-of-bag estimate.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.types import IndexArray, MaskArray


class BootstrapSample:
    """One resample: which rows were drawn, and which were left out.

    Parameters
    ----------
    drawn:
        ``(n_rows,)`` positions into the training set, with repeats. Length
        matches the training set, which is what makes a member see the same
        quantity of data while seeing a different selection of it.
    n_rows:
        How many rows the training set holds, so the out-of-bag rows can be
        worked out from what is absent.
    """

    __slots__ = ("_drawn", "_n_rows")

    def __init__(self, drawn: IndexArray, n_rows: int) -> None:
        # Frozen: in_bag and the out-of-bag grid are recomputed from these
        # positions, so a write into them would silently change which rows
        # count as held out.
        self._drawn = drawn.copy()
        self._drawn.setflags(write=False)
        self._n_rows = n_rows

    @classmethod
    def draw(cls, n_rows: int, generator: np.random.Generator) -> BootstrapSample:
        """Draw ``n_rows`` positions from ``n_rows``, with replacement.

        Parameters
        ----------
        n_rows:
            The size of the training set, and of the sample.
        generator:
            Passed in rather than created, so an ensemble draws every member
            from one stream and a seeded fit is reproducible end to end.

        Returns
        -------
        BootstrapSample
            Holding the drawn positions and able to report what it missed.
        """
        drawn = generator.choice(n_rows, size=n_rows)

        return BootstrapSample(
            drawn=drawn,
            n_rows=n_rows,
        )

    @property
    def drawn(self) -> IndexArray:
        """The positions drawn, with repeats, in draw order.

        The buffer is frozen at construction: ``in_bag`` and the out-of-bag
        grid are recomputed from these positions, so a caller writing into
        them would silently change which rows count as held out.
        """
        return self._drawn

    @property
    def n_rows(self) -> int:
        """How many rows the training set holds."""
        return self._n_rows

    @property
    def in_bag(self) -> MaskArray:
        """``(n_rows,)``, true for rows this sample drew at least once."""
        seen = np.zeros(self._n_rows, dtype=bool)
        seen[self._drawn] = True

        return seen

    @property
    def out_of_bag(self) -> IndexArray:
        """Positions of the rows this sample never drew.

        Roughly 36.8% of them, and the reason bagging gets a held-out estimate
        for free. A member never saw these, so its prediction on them is an
        honest one.
        """
        return np.flatnonzero(~self.in_bag)

    @property
    def out_of_bag_share(self) -> float:
        """What fraction of the training set this member never saw.

        Converges on ``1/e`` as ``n`` grows, and is close to it by about
        twenty rows.
        """
        return float(self.out_of_bag.size / self._n_rows)

    def __len__(self) -> int:
        return int(self._drawn.size)

    def __repr__(self) -> str:
        return f"BootstrapSample({len(self)} drawn, {self.out_of_bag.size} out of bag)"
