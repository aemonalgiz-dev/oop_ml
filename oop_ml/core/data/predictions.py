"""What a model answers, as something with a type rather than a shape.

``predict`` returned ``FloatArray`` everywhere, which is the array-in/array-out
habit this library exists to avoid, sitting at the one place a user actually
looks. A caller receiving it had no way to ask how many rows came back without
reading ``.shape[0]``, and no way to hand it to an evaluation without the
evaluation re-validating it.

Two things make this affordable rather than ceremonial.

It implements ``__array__``, so numpy treats it as the array it wraps.
``np.allclose(model.predict(rows), expected)``, ``predictions[0]``, and
``len(predictions)`` all keep working, because the object is genuinely
array-like rather than a box you have to open. Wrapping the return type
therefore costs the reader nothing and gains them a type that says what it is.

It exposes ``column``, which satisfies
:class:`~oop_ml.core.data.column.HasColumn`, so every evaluation in the library
accepts one directly and coerces nothing. ``RegressionEvaluation(truth,
model.predict(rows))`` now passes two validated objects rather than two arrays
it has to check.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from oop_ml.core.data.column import Column
from oop_ml.core.types import FloatArray, array_for_protocol
from oop_ml.core.validation import ValueRole


class Predictions:
    """One answer per observation, in the order they were asked about.

    A regressor's answers are quantities and a classifier's are class
    positions on the ``0 .. K-1`` scale. The difference is in what the numbers
    mean rather than in what the container guarantees, so both use this: one
    finite value per row, and the row order matching the query order.

    Parameters
    ----------
    values:
        ``(n_rows,)``. Validated by
        :class:`~oop_ml.core.data.column.Column`, so it inherits the
        one-dimensional, finite and non-empty guarantees rather than repeating
        them.

    Raises
    ------
    EmptyValuesError
        If there are no rows.
    InvalidValuesError
        If the values are not a finite one-dimensional array.
    """

    __slots__ = ("_column",)

    def __init__(self, values: FloatArray) -> None:
        self._column = Column.of(values, ValueRole.PREDICTED_VALUES)

    @classmethod
    def already_checked(cls, values: FloatArray) -> Predictions:
        """Wrap an array a model just computed, without re-validating it.

        The model's own arithmetic produced these, from inputs that were
        already coerced at the boundary, so running the full check again costs
        a copy to re-establish what cannot have changed. The same argument
        ``Column.selecting`` makes, and it matters here because ``predict``
        sits on the hot path of every score, every fold and every ensemble
        member.
        """
        predictions = cls.__new__(cls)
        predictions._column = Column.selecting(values, ValueRole.PREDICTED_VALUES)

        return predictions

    @property
    def values(self) -> FloatArray:
        """The answers as a read-only array."""
        return self._column.values

    @property
    def column(self) -> Column:
        """The answers as a column.

        What makes this satisfy ``HasColumn``, so an evaluation takes one
        straight and coerces nothing.
        """
        return self._column

    @property
    def n_rows(self) -> int:
        """How many observations were answered."""
        return self._column.n_samples

    @property
    def shape(self) -> tuple[int, ...]:
        """The wrapped array's shape, so a caller can assert on it directly."""
        return self._column.values.shape

    @property
    def dtype(self):
        """The wrapped array's dtype."""
        return self._column.values.dtype

    def __array__(self, dtype=None, copy=None) -> FloatArray:
        """Hand numpy this wrapper, honouring the copy parameter.

        See :func:`~oop_ml.core.types.array_for_protocol` for the
        contract and the corruption it exists to prevent.
        """
        return array_for_protocol(self._column.values, dtype, copy)

    def __getitem__(self, index) -> FloatArray:
        return self._column.values[index]

    def __iter__(self) -> Iterator[np.float64]:
        return iter(self._column.values)

    def __len__(self) -> int:
        return self.n_rows

    @staticmethod
    def _is_array_like(other: object) -> bool:
        """Whether ``other`` is something to compare element by element.

        Anything else is deferred to, by returning ``NotImplemented``, so that
        a comparison helper which expects to be asked -- ``pytest.approx`` is
        the one that matters -- still gets its turn. Broadcasting against it
        instead produces an array where the caller wanted a verdict, and
        ``assert`` on an array of more than one element raises rather than
        failing usefully.
        """
        return isinstance(other, Predictions | np.ndarray | list | tuple | int | float)

    def __eq__(self, other: object):  # type: ignore[override]
        """Compare element by element, the way the array it wraps would.

        A scalar answer would be inconsistent with ``__array__``: something
        numpy treats as an array has to compare like one, or
        ``model.predict(a) != model.predict(b)`` quietly returns a single
        ``True`` where the caller wanted one answer per row. That exact mistake
        broke an example.
        """
        if not self._is_array_like(other):
            return NotImplemented

        return np.asarray(self) == np.asarray(other)

    def __ne__(self, other: object):  # type: ignore[override]
        if not self._is_array_like(other):
            return NotImplemented

        return np.asarray(self) != np.asarray(other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Predictions({self.n_rows} rows)"
