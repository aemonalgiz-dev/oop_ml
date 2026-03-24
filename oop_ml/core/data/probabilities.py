"""What a classifier believes, in the two shapes it comes in.

``probabilities: FloatArray`` was the least informative annotation in the
library. It did not say whether the array was one probability per row or one
per class per row, whether the entries were bounded, or whether anything summed
to anything. Two of those matter to the arithmetic downstream and the third is
what makes a probability a probability.

The two shapes are genuinely different objects rather than one with a flag.
:class:`Probabilities` is ``(n_rows,)``, one number per observation, and is what
a binary model produces: the chance of class 1, with the chance of class 0
implied. :class:`ProbabilityMatrix` is ``(n_rows, n_classes)``, and its rows sum
to one because the classes are exhaustive. Collapsing them into a single type
would mean either dropping the row-sum invariant, which is the only interesting
thing the matrix guarantees, or asserting it of a vector where it is false.

Both are bounded to ``[0, 1]``. That check has caught a real class of bug in
this library's history: a log-loss gradient that returns the loss instead of the
residual type-checks, has the right shape, and is silently wrong.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.types import FloatArray, array_for_protocol

ROW_SUM_TOLERANCE = 1e-9
"""Rows come from a softmax or a mean, so they land near one rather than on it."""


class Probabilities:
    """``(n_rows,)`` chances, one per observation.

    What a binary classifier believes: the chance of class 1, with the chance
    of class 0 implied by it. Bounded but not summing to anything, because
    these are separate rows rather than alternatives.

    Parameters
    ----------
    values:
        ``(n_rows,)``, each in ``[0, 1]``.

    Raises
    ------
    EmptyValuesError
        If there are no rows.
    InvalidValuesError
        If the array is not one-dimensional, or any entry falls outside
        ``[0, 1]``.
    """

    __slots__ = ("_values",)

    def __init__(self, values: FloatArray) -> None:
        if values.ndim != 1:
            raise InvalidValuesError(
                f"Probabilities are one-dimensional, got {values.ndim}"
            )
        if values.size == 0:
            raise EmptyValuesError("At least one probability is required")
        if not np.all((values >= 0.0) & (values <= 1.0)):
            raise InvalidValuesError("Every probability must lie in [0, 1]")

        # Stored as a frozen copy: the caller keeps their own array, and what
        # the bounds check just validated cannot be un-validated later --
        # neither through the caller's reference nor through a view handed
        # back out.
        values = values.copy()
        values.setflags(write=False)

        self._values = values

    @property
    def values(self) -> FloatArray:
        """``(n_rows,)``, for the arithmetic."""
        return self._values

    @property
    def n_rows(self) -> int:
        """How many observations."""
        return int(self._values.size)

    def above(self, threshold: float) -> FloatArray:
        """1.0 where the chance clears ``threshold``, 0.0 elsewhere.

        The decision a binary classifier makes, kept here so that no caller
        has to remember whether the comparison is strict.

        Raises
        ------
        InvalidValuesError
            If the threshold is outside ``[0, 1]``, where it could only ever
            answer all-one-class.
        """
        if not 0.0 <= threshold <= 1.0:
            raise InvalidValuesError(f"A threshold must lie in [0, 1], got {threshold}")

        return (self._values >= threshold).astype(np.float64)

    @property
    def shape(self) -> tuple[int, ...]:
        """The wrapped array's shape, so a caller can assert on it directly."""
        return self._values.shape

    @property
    def dtype(self):
        """The wrapped array's dtype."""
        return self._values.dtype

    def __array__(self, dtype=None, copy=None) -> FloatArray:
        """Hand numpy this wrapper, honouring the copy parameter.

        See :func:`~oop_ml.core.types.array_for_protocol` for the
        contract and the corruption it exists to prevent.
        """
        return array_for_protocol(self._values, dtype, copy)

    def __getitem__(self, index) -> FloatArray:
        return self._values[index]

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        return f"Probabilities({self.n_rows} rows)"


class ClassScores:
    """``(n_rows, n_classes)`` chances, one per class per row.

    What a multi-class classifier believes, without claiming the rows add up.
    That is the weaker of the two guarantees here and it is the honest one for
    a :class:`~oop_ml.classification.multiclass.one_vs_rest.OneVsRestClassifier`,
    whose K models were fitted separately and never asked to agree. Its rows
    genuinely do not sum to one, and its own spec asserts as much rather than
    hiding it.

    :class:`ProbabilityMatrix` is the stronger case, and a subclass rather than
    a flag, so a model that does produce a distribution says so in its return
    type and a caller needing one cannot be handed scores by mistake.

    Parameters
    ----------
    values:
        ``(n_rows, n_classes)``, each in ``[0, 1]``.

    Raises
    ------
    EmptyValuesError
        If there are no rows.
    InvalidValuesError
        If the array is not two-dimensional, or any entry falls outside
        ``[0, 1]``.
    """

    __slots__ = ("_values",)

    def __init__(self, values: FloatArray) -> None:
        if values.ndim != 2:
            raise InvalidValuesError(
                f"A probability matrix is two-dimensional, got {values.ndim}"
            )
        if values.shape[0] == 0:
            raise EmptyValuesError("At least one row is required")
        if not np.all((values >= 0.0) & (values <= 1.0)):
            raise InvalidValuesError("Every probability must lie in [0, 1]")

        # Stored as a frozen copy: the caller keeps their own array, and what
        # the bounds check just validated cannot be un-validated later --
        # neither through the caller's reference nor through a view handed
        # back out.
        values = values.copy()
        values.setflags(write=False)

        self._values = values

    @property
    def values(self) -> FloatArray:
        """``(n_rows, n_classes)``, for the arithmetic."""
        return self._values

    @property
    def n_rows(self) -> int:
        """How many observations."""
        return int(self._values.shape[0])

    @property
    def n_classes(self) -> int:
        """How many classes the row is spread across."""
        return int(self._values.shape[1])

    @property
    def most_likely(self) -> FloatArray:
        """``(n_rows,)`` class positions, as floats on the ``0 .. K-1`` scale.

        Ties go to the lowest class index, which ``argmax`` gives by taking
        the first maximum. Stated here once, so that every classifier in the
        library breaks a tie the same way instead of each inheriting whatever
        its own call happens to do.
        """
        return np.argmax(self._values, axis=1).astype(np.float64)

    def for_class(self, class_index: int) -> Probabilities:
        """The column for one class, as one probability per row.

        Raises
        ------
        InvalidValuesError
            If there is no such class.
        """
        if not 0 <= class_index < self.n_classes:
            raise InvalidValuesError(
                f"class {class_index} is outside a problem with "
                f"{self.n_classes} classes"
            )

        return Probabilities(self._values[:, class_index])

    @property
    def shape(self) -> tuple[int, ...]:
        """The wrapped array's shape, so a caller can assert on it directly."""
        return self._values.shape

    def __array__(self, dtype=None, copy=None) -> FloatArray:
        """Hand numpy this wrapper, honouring the copy parameter.

        See :func:`~oop_ml.core.types.array_for_protocol` for the
        contract and the corruption it exists to prevent.
        """
        return array_for_protocol(self._values, dtype, copy)

    def __getitem__(self, index) -> FloatArray:
        return self._values[index]

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.n_rows}x{self.n_classes})"


class ProbabilityMatrix(ClassScores):
    """Class scores that are also a distribution: every row sums to one.

    What a softmax produces, and what an averaged ensemble of distributions
    produces. The extra guarantee is worth its own type because a caller
    reasoning about a distribution -- taking an entropy, sampling from it,
    reporting one class's share as *the* remaining chance -- is wrong the
    moment the rows do not add up, and one-vs-rest scores look identical
    until they are summed.

    Raises
    ------
    InvalidValuesError
        Everything ``ClassScores`` refuses, plus a row that does not sum
        to one.
    """

    __slots__ = ()

    def __init__(self, values: FloatArray) -> None:
        super().__init__(values)

        row_sums = values.sum(axis=1)
        if not np.all(np.abs(row_sums - 1.0) <= ROW_SUM_TOLERANCE):
            worst = float(np.max(np.abs(row_sums - 1.0)))
            raise InvalidValuesError(
                f"Rows of a probability matrix sum to 1, off by up to {worst}"
            )
