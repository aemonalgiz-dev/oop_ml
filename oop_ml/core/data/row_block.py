"""Observations as rows and features as columns, with the columns still named.

The matrix a tree splits on and a neighbour model measures distances against.
It is not a design matrix: there is no intercept column, nothing here is being
solved for, and the width is exactly the number of predictors.

Every method that took this as a bare array left three things for the caller to
know from somewhere else: that it is two-dimensional, that rows are
observations rather than features, and which column is which predictor. The
third is the one the README promises the library keeps hold of, and it was
being dropped the moment a `FeatureSet` was flattened into numbers.

``column_for`` is what that promise buys. A split search asking for
``block.column_for("slept")`` cannot silently read the wrong predictor the way
``feature_matrix[:, 1]`` can once somebody reorders a feature set.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, NonUniqueFeaturesError
from oop_ml.core.types import FloatArray, IndexArray, MaskArray


class RowBlock:
    """``(n_rows, n_features)`` observations, with the feature names attached.

    Parameters
    ----------
    values:
        Rows are observations, columns are predictors, in ``feature_names``
        order.
    feature_names:
        One per column. Unique, because a name that appears twice cannot
        address a column.

    Raises
    ------
    InvalidValuesError
        If the array is not two-dimensional, or its width does not match the
        number of names.
    NonUniqueFeaturesError
        If two names are the same.
    """

    __slots__ = ("_feature_names", "_position_of", "_values")

    def __init__(self, values: FloatArray, feature_names: Sequence[str]) -> None:
        if values.ndim != 2:
            raise InvalidValuesError(
                f"A row block is two-dimensional, got {values.ndim}"
            )
        if values.shape[1] != len(feature_names):
            raise InvalidValuesError(
                f"{len(feature_names)} feature names for {values.shape[1]} columns"
            )

        position_of: dict[str, int] = {}
        for position, name in enumerate(feature_names):
            if name in position_of:
                raise NonUniqueFeaturesError(f"Duplicate feature name: {name!r}")
            position_of[name] = position

        self._values = values
        self._feature_names = tuple(feature_names)
        self._position_of = position_of

    @property
    def values(self) -> FloatArray:
        """``(n_rows, n_features)``, for the kernels that want the buffer."""
        return self._values

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The column names, in column order."""
        return self._feature_names

    @property
    def n_rows(self) -> int:
        """How many observations."""
        return int(self._values.shape[0])

    @property
    def n_features(self) -> int:
        """How many predictors."""
        return len(self._feature_names)

    def column_for(self, name: str) -> FloatArray:
        """The column called ``name``, as a ``(n_rows,)`` view.

        Raises
        ------
        InvalidValuesError
            If no column has that name.
        """
        if name not in self._position_of:
            known = ", ".join(self._feature_names)
            raise InvalidValuesError(f"No column named {name!r}. Columns: {known}")

        return self._values[:, self._position_of[name]]

    def column_at(self, position: int) -> FloatArray:
        """The column at ``position``, for the loops that walk them in order."""
        return self._values[:, position]

    def select_rows(self, rows: MaskArray | IndexArray) -> RowBlock:
        """The same columns, keeping only these rows.

        What a tree does at every split and a bootstrap does at every draw.
        The names come along, which is the point: a child block still knows
        what its columns are.
        """
        return self._sharing_columns(self._values[rows])

    def rows_between(self, start: int, stop: int) -> RowBlock:
        """The same columns, keeping rows ``start`` up to ``stop``.

        A view rather than a copy, which is why this exists beside
        ``select_rows``. The neighbour model splits its queries into blocks and
        hands each to a thread; taking those with fancy indexing copied the
        block every time and measured 5.6x slower end to end on a small
        predict. A slice does not copy.
        """
        return self._sharing_columns(self._values[start:stop])

    def _sharing_columns(self, values: FloatArray) -> RowBlock:
        """A block over ``values`` reusing this one's already-checked columns.

        Every row selection produces a block with exactly the columns of the
        one it came from, so re-running the constructor re-derives a name
        mapping that cannot have changed. A tree rebuilds that dictionary at
        every node it grows, which measured 1.5x on a fit before this existed.
        """
        block = RowBlock.__new__(RowBlock)
        block._values = values
        block._feature_names = self._feature_names
        block._position_of = self._position_of

        return block

    def row(self, position: int) -> FloatArray:
        """One observation as a ``(n_features,)`` array, for routing it."""
        return self._values[position]

    def __iter__(self):
        """Every row in turn, for the models that predict one at a time."""
        return iter(self._values)

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        named = ", ".join(self._feature_names)

        return f"RowBlock({self.n_rows}x{self.n_features}: {named})"


def rows_of(values: FloatArray, feature_names: Sequence[str]) -> RowBlock:
    """A row block over ``values``, made contiguous for row slicing.

    Trees slice rows constantly and never form ``X.T @ v``, so C order is what
    they want; a feature set assembles itself column-major for the linear
    models, which is the opposite. This is the one place that conversion
    happens.
    """
    return RowBlock(np.ascontiguousarray(values, dtype=np.float64), feature_names)
