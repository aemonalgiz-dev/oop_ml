"""A validated column of observations, and the one place raw input is coerced.

Every model, metric, and guard in this package ultimately wants the same thing,
which is a finite one-dimensional ``float64`` vector. I originally reached for
``to_float_array`` at each of those places, and the result was that a single
call would coerce the same input two or three times on its way through, with
every function along the path having to remember to do it.

:class:`Column` moves that from being a step inside each function to being a
property of the value itself. Coercion and validation happen exactly once, in
the constructor; from that point on the type is the proof, and anything holding
a :class:`Column` can simply use it. :meth:`Column.of` is the adapter at the
boundary. Hand it raw input and it validates, hand it a :class:`Column` and it
hands the very same object straight back, so passing a validated column down
through three layers costs nothing at all.

A column also carries its own :class:`~oop_ml.core.validation.ValueRole`, which
means a guard that fails can name the offending input without every caller
having to thread a label along behind it.

The statistics that depend only on the column live here too, namely ``mean``,
``deviations``, and ``sum_of_squared_deviations``. They are the building blocks
of least squares and of every metric we compute, and keeping that arithmetic
beside the data it describes keeps it out of the models.
"""

from __future__ import annotations

from typing import Protocol, TypeAlias, cast

import numpy as np

from oop_ml.core.types import FloatArray, NumericInput
from oop_ml.core.validation import (
    ValueRole,
    check_equal_length,
    check_has_both_classes,
    check_has_variance,
    check_is_binary,
    check_min_length,
    check_non_empty,
    to_float_array,
)


class HasColumn(Protocol):
    """Anything that can supply an already-validated :class:`Column`.

    This is structural rather than nominal, so
    :class:`~oop_ml.core.feature.Feature` satisfies it purely by exposing
    ``column``, without importing anything from here. That is what keeps the
    dependency running one way only, where ``feature`` knows about ``column``
    and never the reverse, while still allowing a feature to be handed to
    anything that wants a column.

    Deliberately not ``@runtime_checkable``. The decorator would let us write
    ``isinstance(values, HasColumn)``, although for a single-member protocol
    that call does nothing more than look for the attribute, and it does so
    through ``inspect.getattr_static`` at roughly sixty times the cost of
    ``hasattr``. :meth:`Column.of` sits on the hot path of every fit, predict
    and evaluate in the library, so it asks for the attribute directly.
    """

    @property
    def column(self) -> Column: ...


ColumnSource: TypeAlias = "Column | HasColumn | NumericInput"
"""Anything :meth:`Column.of` can turn into a column without re-validating."""


class Column:
    """An immutable, finite, one-dimensional ``float64`` vector of observations.

    Parameters
    ----------
    values:
        The observations. These are coerced to a finite one-dimensional
        ``float64`` array, then copied, and then frozen. The copy matters as
        much as the freeze; without it we would be write-protecting an array
        that the caller still holds a reference to.
    role:
        Which input this column represents. Used to name the column in any error
        it raises.

    Raises
    ------
    InvalidValuesError
        If the values are not coercible, not one-dimensional, or not finite.
    EmptyValuesError
        If there are no observations.
    """

    __slots__ = ("_role", "_values")

    def __init__(self, values: NumericInput, role: ValueRole) -> None:
        values_array = to_float_array(values, role).copy()
        check_non_empty(values_array, role)
        values_array.setflags(write=False)

        self._values = values_array
        self._role = role

    @classmethod
    def of(cls, values: ColumnSource, role: ValueRole) -> Column:
        """Return ``values`` as a column, validating it only if it is raw input.

        This is the idempotent entry point that every public boundary should
        be using. An already validated column, or anything carrying one such as
        a :class:`~oop_ml.core.feature.Feature`, comes straight back with its
        own role intact, so handing one down through nested calls never
        re-coerces or re-copies it. Only genuinely raw input pays for
        validation.
        """
        if isinstance(values, Column):
            return values

        # Structural check by attribute rather than by isinstance against the
        # protocol; see the note on HasColumn for why.
        carried_column = getattr(values, "column", None)
        if isinstance(carried_column, Column):
            return carried_column

        # Neither a Column nor anything carrying one, so what is left is raw
        # input. The cast says only that, since ``getattr`` cannot narrow the
        # type the way ``isinstance`` against the protocol used to.
        return cls(cast(NumericInput, values), role)

    @property
    def values(self) -> FloatArray:
        """The observations as a read-only ``float64`` array.

        This hands back the frozen buffer itself rather than a copy, since the
        write flag is already the protection and this sits squarely on the hot
        path; every ``feature_matrix``, ``predict``, and ``deviations`` call
        reads it.
        """
        return self._values

    @property
    def role(self) -> ValueRole:
        """Which input this column represents."""
        return self._role

    @property
    def n_samples(self) -> int:
        """Number of observations."""
        return int(self._values.size)

    @property
    def mean(self) -> float:
        """Arithmetic mean of the observations."""
        return float(np.mean(self._values))

    @property
    def deviations(self) -> FloatArray:
        """Each observation's signed distance from the column mean."""
        return self._values - self.mean

    @property
    def sum_of_squared_deviations(self) -> float:
        """``sum((value - mean) ** 2)``, meaning the column's total variation.

        The same quantity turns up in two places worth connecting: it is the
        denominator of the least-squares slope, and it is the total sum of
        squares that R^2 measures against.
        """
        return float(np.sum(self.deviations**2))

    @property
    def standard_deviation(self) -> float:
        """Population standard deviation: ``sqrt(mean((value - mean) ** 2))``.

        Population rather than sample, so we divide by ``n`` and not by
        ``n - 1``. Standardizing is a rescaling of these observations rather
        than an estimate of some wider population's spread, which makes ``n``
        the honest denominator here, and it is also what makes the standardized
        column's own variance come out at exactly 1.
        """
        return float(np.sqrt(self.sum_of_squared_deviations / self.n_samples))

    def check_has_variance(self) -> None:
        """Raise :class:`~oop_ml.core.exceptions.AllSameValuesError` if constant."""
        check_has_variance(self._values, self._role)

    def check_is_binary(self) -> None:
        """Raise :class:`~oop_ml.core.exceptions.NonBinaryLabelsError` unless 0/1.

        A fitting rule rather than a structural one, in the same way as
        :meth:`check_has_variance`, so it is asked for by the classifier that
        needs it rather than enforced on every column that happens to exist.
        """
        check_is_binary(self._values, self._role)

    def check_has_both_classes(self) -> None:
        """Raise :class:`~oop_ml.core.exceptions.SingleClassError` if one-sided."""
        check_has_both_classes(self._values, self._role)

    def check_min_length(self, minimum_length: int) -> None:
        """Raise :class:`~oop_ml.core.exceptions.TooFewValuesError` if too short."""
        check_min_length(self._values, minimum_length, self._role)

    def check_equal_length(self, other: Column) -> None:
        """Raise ``NonEqualArrayLengthError`` if ``other`` is a different length."""
        check_equal_length(self._values, other._values)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return f"Column(role={self._role.value!r}, n_samples={self.n_samples})"
