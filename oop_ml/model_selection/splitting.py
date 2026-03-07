"""Ways of dividing a dataset into rows to learn from and rows to score on.

Theory
------
Training error is not evidence. Every model in this library minimises it by
construction, so it can only fall as the model gains freedom, whether that comes
from more polynomial terms, a smaller penalty, or more epochs. The polynomial
table makes the point bluntly, with train R^2 climbing monotonically to exactly
1.0 while test R^2 collapses.
A number that improves no matter what you do cannot tell you what to do.

The fix is to keep some rows back, fit without them, and score on them. Those
rows stand in for data the model will meet in production and has never seen.

Two ways to do it, with a real tradeoff:

* :class:`TrainTestSplitter` holds back a single block, which is cheap at one
  fit, although the estimate then depends on which rows happened to land in that
  held-out block, and on a small dataset that is a great deal of luck.
* :class:`KFold` divides the rows into ``k`` blocks and takes each in turn as
  the held-out one, so **every row is scored exactly once** while every fit
  still sees ``k - 1`` blocks. Costs ``k`` fits, and gives a far steadier
  estimate.

Shuffling and reproducibility
-----------------------------
Row order is rarely accidental, since data tends to arrive sorted by date, by
class, or by whatever the export happened to order it by. Splitting an ordered
file without shuffling can put every large value into the test half, so both
splitters shuffle by default and take a ``random_seed`` that lets a result be
reproduced exactly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from oop_ml.core.exceptions import EmptyValuesError, TooFewValuesError
from oop_ml.core.types import IndexArray
from oop_ml.model_selection.dataset import Dataset, DataSplit


class Splits:
    """The partitions a splitter produced, in order.

    A collection rather than a bare list so callers iterate the object itself
    and never reach for an index they have to interpret.

    Parameters
    ----------
    splits:
        One :class:`~oop_ml.model_selection.dataset.DataSplit` per partition.

    Raises
    ------
    EmptyValuesError
        If no splits are supplied.
    """

    __slots__ = ("_splits",)

    def __init__(self, splits: Sequence[DataSplit]) -> None:
        if not splits:
            raise EmptyValuesError("at least one split is required")

        self._splits = tuple(splits)

    @property
    def n_splits(self) -> int:
        """How many partitions there are."""
        return len(self._splits)

    @property
    def splits(self) -> Sequence[DataSplit]:
        return self._splits

    def __iter__(self) -> Iterator[DataSplit]:
        return iter(self._splits)

    def __len__(self) -> int:
        return self.n_splits

    def __repr__(self) -> str:
        return f"Splits(n_splits={self.n_splits})"


class RowShuffler(BaseModel):
    """Produces the row order a splitter cuts up.

    Pulled out so both splitters share one implementation of "shuffle unless
    told not to, and do it reproducibly when given a seed".

    Parameters
    ----------
    shuffle:
        Whether to permute the rows before splitting. On by default, because
        row order in real data is rarely meaningless.
    random_seed:
        Seed for the permutation. ``None`` means a fresh, unreproducible order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shuffle: bool = True
    random_seed: int | None = None

    def row_order(self, n_samples: int) -> IndexArray:
        """Indices ``0 .. n_samples - 1``, permuted when ``shuffle`` is set."""
        indices = np.arange(n_samples, dtype=np.intp)

        if not self.shuffle:
            return indices

        np.random.default_rng(self.random_seed).shuffle(indices)

        return indices


class TrainTestSplitter(BaseModel):
    """Hold back a single block of rows for scoring.

    Parameters
    ----------
    test_fraction:
        Proportion of rows to hold back. Must leave at least one row on each
        side.
    shuffle, random_seed:
        Passed through to :class:`RowShuffler`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    test_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    shuffle: bool = True
    random_seed: int | None = None

    def _testing_count(self, n_samples: int) -> int:
        """How many rows to hold back, with both sides guaranteed non-empty.

        ``round(self.test_fraction * n_samples)`` is the wanted share. Clamp it
        into ``1 .. n_samples - 1`` so a small dataset or an extreme fraction
        cannot produce an empty half, which ``min`` and ``max`` do in one line.

        Raises
        ------
        TooFewValuesError
            If ``n_samples`` is below two, where no clamp can help.
        """
        if n_samples < 2:
            raise TooFewValuesError(
                f"a split needs at least 2 rows to leave one on each side, "
                f"got {n_samples}"
            )

        wanted_count = round(self.test_fraction * n_samples)

        return min(max(wanted_count, 1), n_samples - 1)

    def split(self, dataset: Dataset) -> DataSplit:
        """Divide ``dataset`` into a training part and a testing part.

        Three steps:

        1. ``RowShuffler(shuffle=..., random_seed=...).row_order(n_samples)``
           for the order to cut up. Build the shuffler from this splitter's own
           ``shuffle`` and ``random_seed``.
        2. Slice that order with :meth:`_testing_count`. Plain Python slicing
           works on a numpy array: the **last** block is the testing one, so the
           unshuffled case holds back the tail.
        3. ``dataset.select_rows(...)`` on each slice, and pair them in a
           :class:`~oop_ml.model_selection.dataset.DataSplit`.

        Raises
        ------
        TooFewValuesError
            If the dataset has fewer than two rows, so no split can leave a row
            on each side.
        """
        testing_count = self._testing_count(dataset.n_samples)

        row_order = RowShuffler(
            shuffle=self.shuffle, random_seed=self.random_seed
        ).row_order(dataset.n_samples)

        # select_rows subsets the predictors and the target by the same indices,
        # which is what keeps the two halves internally aligned.
        return DataSplit(
            training=dataset.select_rows(row_order[:-testing_count]),
            testing=dataset.select_rows(row_order[-testing_count:]),
        )


class KFold(BaseModel):
    """Divide the rows into ``n_folds`` blocks, each taking a turn held out.

    Parameters
    ----------
    n_folds:
        How many blocks. Must be at least 2, and no more than the row count --
        every fold needs at least one row to score on.
    shuffle, random_seed:
        Passed through to :class:`RowShuffler`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_folds: int = Field(default=5, ge=2)
    shuffle: bool = True
    random_seed: int | None = None

    def _index_blocks(self, row_order: IndexArray) -> list[IndexArray]:
        """Cut the row order into ``n_folds`` near-equal blocks.

        ``np.array_split(row_order, self.n_folds)`` does exactly this, including
        the uneven case: it hands the extra rows to the earlier blocks, so sizes
        differ by at most one. Note that this is ``array_split`` rather than
        ``split``, since the latter refuses to divide unevenly at all.
        """
        return np.array_split(row_order, self.n_folds)

    def split(self, dataset: Dataset) -> Splits:
        """Produce one :class:`~oop_ml.model_selection.dataset.DataSplit` per fold.

        Each row appears in exactly one fold's testing half and in every other
        fold's training half. That is the property worth checking, and it is the
        one that makes the averaged score use all of the data.

        Steps:

        1. Shuffle with :class:`RowShuffler`, then cut with
           :meth:`_index_blocks`.
        2. For each block in turn: that block is the fold's **testing** indices,
           and the training indices are every *other* block joined back together
           with ``np.concatenate([...])``.
        3. ``dataset.select_rows(...)`` on each, pair them in a
           :class:`~oop_ml.model_selection.dataset.DataSplit`, and collect the lot into
           :class:`Splits`.

        Raises
        ------
        TooFewValuesError
            If there are fewer rows than folds.
        """
        if dataset.n_samples < self.n_folds:
            raise TooFewValuesError(
                f"{self.n_folds} folds need at least {self.n_folds} rows to give "
                f"each one something to score on, got {dataset.n_samples}"
            )

        blocks = self._index_blocks(
            RowShuffler(shuffle=self.shuffle, random_seed=self.random_seed).row_order(
                dataset.n_samples
            )
        )

        splits = []
        for held_out_position, testing_indices in enumerate(blocks):
            # Every block except the held-out one, joined back together. This is
            # what makes each row train in k-1 folds and test in exactly one.
            training_indices = np.concatenate(
                [
                    block
                    for position, block in enumerate(blocks)
                    if position != held_out_position
                ]
            )

            splits.append(
                DataSplit(
                    training=dataset.select_rows(training_indices),
                    testing=dataset.select_rows(testing_indices),
                )
            )

        return Splits(splits)
