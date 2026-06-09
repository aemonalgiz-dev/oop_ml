"""Targets that may not all count equally, and the rule that they must count.

Every tree in this library until now treated its rows as interchangeable: a
node's impurity was a statement about how many rows of each kind it held, and
a leaf's answer was an average over the rows that reached it. That is the right
reading almost always, and it is exactly wrong for one thing -- boosting, where
the whole method is to make the next learner care more about the rows the last
one got wrong.

So a node's targets and the weights on them are one thing rather than two.
Pairing them in an object is the alternative to threading three weight arrays
through :meth:`~oop_ml.core.tree.impurity.Impurity.gain` beside its three
columns, where nothing would stop a caller pairing the left weights with the
right rows.

What a weight means
-------------------
That the row counts that much. A row of weight 2 contributes to every sum the
way two copies of it would, so a fit on weighted rows and a fit on the
duplicated rows agree exactly, and that identity is what the spec asserts
rather than any formula being restated.

What a weight is not is a count of rows. ``n_samples`` on a node stays the
number of rows that reached it, and ``min_samples_leaf`` still counts rows,
because both are statements about the *structure* the tree grew rather than
about the importance of what it grew from. scikit-learn draws the line in the
same place and keeps a separate ``min_weight_fraction_leaf`` for the other
reading; this library has only the row-count one.

Uniform weights reduce to the unweighted formulas exactly, not nearly, which is
what let this be threaded through a tuned split search without a second code
path to keep in step with the first.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.data.column import Column
from oop_ml.core.exceptions import (
    InvalidValuesError,
    NonEqualArrayLengthError,
)
from oop_ml.core.types import FloatArray, NumericInput
from oop_ml.core.validation import ValueRole, to_float_array


class WeightedTargets:
    """One node's targets, and how much each of them counts.

    Raises
    ------
    NonEqualArrayLengthError
        If there is not exactly one weight per target.
    InvalidValuesError
        If a weight is negative or not finite, or if they total zero. A node
        whose rows all count for nothing has no impurity to measure and no
        answer to give, so it is refused rather than divided by.
    """

    __slots__ = ("_column", "_total", "_weights")

    def __init__(self, column: Column, weights: NumericInput | None = None) -> None:
        self._column = column
        self._weights = (
            _uniform(column.n_samples)
            if weights is None
            else _checked(weights, column.n_samples)
        )
        self._total = float(self._weights.sum())

        if self._total <= 0.0:
            raise InvalidValuesError(
                "the weights of these rows total zero, so nothing here counts "
                "at all. A node with no weight has no impurity to measure and "
                "no answer to give"
            )

    @classmethod
    def already_checked(cls, column: Column, weights: FloatArray) -> WeightedTargets:
        """Pair a column with weights the library itself derived.

        The subsetting constructor. Growth partitions one validated set of
        weights over and over, and the parts of a checked whole are checked, so
        re-establishing what cannot have changed is the cost this skips. Same
        reasoning as :meth:`~oop_ml.core.data.column.Column.selecting`.
        """
        pairing = object.__new__(cls)
        pairing._column = column
        pairing._weights = weights
        pairing._total = float(weights.sum())

        return pairing

    @property
    def column(self) -> Column:
        """The targets themselves."""
        return self._column

    @property
    def values(self) -> FloatArray:
        """The targets as an array, for the kernels that want one."""
        return self._column.values

    @property
    def weights(self) -> FloatArray:
        """How much each row counts, one per target, none of them negative."""
        return self._weights

    @property
    def total_weight(self) -> float:
        """What the weights add up to. The denominator of every share here.

        Stands where the row count stands in an unweighted measure, which is
        why uniform weights reduce those measures to themselves: the total is
        then the count.
        """
        return self._total

    @property
    def n_rows(self) -> int:
        """How many rows there are, which is not what they weigh."""
        return self._column.n_samples

    def selecting(self, chosen: np.ndarray) -> WeightedTargets:
        """The same pairing restricted to some rows, targets and weights alike.

        One method rather than two selections a caller has to keep aligned,
        which is the point of the pairing.
        """
        return WeightedTargets.already_checked(
            Column.selecting(self._column.values[chosen], self._column.role),
            self._weights[chosen],
        )

    def taking(self, count: int) -> WeightedTargets:
        """The first ``count`` rows, for a sweep that cuts after each in turn."""
        return WeightedTargets.already_checked(
            Column.selecting(self._column.values[:count], self._column.role),
            self._weights[:count],
        )

    def dropping(self, count: int) -> WeightedTargets:
        """Everything after the first ``count`` rows, the other side of a cut."""
        return WeightedTargets.already_checked(
            Column.selecting(self._column.values[count:], self._column.role),
            self._weights[count:],
        )

    def reordered(self, order: np.ndarray) -> WeightedTargets:
        """The same pairing under a new row order, for a sorted sweep."""
        return WeightedTargets.already_checked(
            Column.selecting(self._column.values[order], self._column.role),
            self._weights[order],
        )

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        if float(self._weights.min()) == float(self._weights.max()) == 1.0:
            return f"WeightedTargets({self.n_rows} rows, unweighted)"

        return f"WeightedTargets({self.n_rows} rows, weighing {self._total:g})"


def _uniform(n_rows: int) -> FloatArray:
    """A weight of one for every row, which is what unweighted means."""
    return np.ones(n_rows, dtype=np.float64)


def _checked(weights: NumericInput, n_rows: int) -> FloatArray:
    """Weights validated once, on the way in.

    The coercion boundary for this type, the way ``Column.__init__`` is for a
    column. Everything downstream takes the array and re-validates nothing.
    """
    values = to_float_array(weights, ValueRole.WEIGHT_VALUES)

    if values.size != n_rows:
        raise NonEqualArrayLengthError(
            f"there are {n_rows} rows and {values.size} weight(s); a weight "
            "says how much one row counts, so there has to be one each"
        )

    if bool((values < 0.0).any()):
        raise InvalidValuesError(
            "a weight of less than zero would have a row count against itself, "
            "which no impurity measure is defined for. Use zero for a row that "
            "should be ignored"
        )

    return values
