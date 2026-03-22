"""The matrix a linear solver works on, and what it knows about itself.

``X`` is not just numbers. It has a width that has to match the coefficient
vector coming back, a column order that has to match the feature names it will
be paired with, and -- the part that has actually cost this library a bug -- a
leading ones column that is there only sometimes.

Every solver here used to take it as a bare array and consult
``self.fit_intercept`` separately to find out whether column zero was real.
Ridge did that twice, once in ``_solve`` and once in ``normal_equations``, and
the note in its docstring records what came of it::

    zeroing the [0, 0] slot of the penalty matrix unconditionally exempts a
    real predictor whenever fit_intercept is false

That bug is only writeable while the matrix and the fact about its first column
live in different objects. Put them together and ``penalty_diagonal`` can be
asked for the right answer, once, by everyone.

Column-major on purpose, for the reason
:class:`~oop_ml.core.data.feature_set.FeatureSet` gives: the solvers are
dominated by ``X.T @ v``, and at 20000x51 that product costs 4.4x more on a
row-major buffer than a column-major one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.types import FloatArray

INTERCEPT_COLUMN_NAME = "intercept"
"""What the ones column is called when the columns are named."""


class DesignMatrix:
    """``X``: the feature columns, with a leading ones column or without one.

    Parameters
    ----------
    values:
        ``(n_rows, n_columns)``. Column 0 is the ones column exactly when
        ``has_intercept`` is set.
    feature_names:
        One per predictor column, in order, excluding the intercept.
    has_intercept:
        Whether a leading ones column is present.

    Raises
    ------
    InvalidValuesError
        If the array is not two-dimensional, or if its width does not match
        the number of feature names plus the intercept.
    """

    __slots__ = ("_feature_names", "_has_intercept", "_values")

    def __init__(
        self,
        values: FloatArray,
        feature_names: Sequence[str],
        has_intercept: bool,
    ) -> None:
        if values.ndim != 2:
            raise InvalidValuesError(
                f"A design matrix is two-dimensional, got {values.ndim}"
            )

        expected = len(feature_names) + (1 if has_intercept else 0)
        if values.shape[1] != expected:
            raise InvalidValuesError(
                f"A design matrix over {len(feature_names)} features "
                f"{'with' if has_intercept else 'without'} an intercept has "
                f"{expected} columns, got {values.shape[1]}"
            )

        self._values = values
        self._feature_names = tuple(feature_names)
        self._has_intercept = has_intercept

    @property
    def values(self) -> FloatArray:
        """``(n_rows, n_columns)``, for the matmuls.

        Handed out rather than copied, since every solver reads it repeatedly
        and this is squarely on the hot path.
        """
        return self._values

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The predictor names, in column order, excluding the intercept."""
        return self._feature_names

    @property
    def has_intercept(self) -> bool:
        """Whether column 0 is the ones column."""
        return self._has_intercept

    @property
    def n_rows(self) -> int:
        """How many observations."""
        return int(self._values.shape[0])

    @property
    def n_columns(self) -> int:
        """How many unknowns a solver has to find, intercept included."""
        return int(self._values.shape[1])

    @property
    def n_features(self) -> int:
        """How many predictors, excluding the intercept."""
        return len(self._feature_names)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Every column named, so a solution can be read without counting."""
        if self._has_intercept:
            return (INTERCEPT_COLUMN_NAME, *self._feature_names)

        return self._feature_names

    def penalty_diagonal(self, penalty: float) -> FloatArray:
        """``penalty`` down the diagonal, with the intercept exempt.

        The one place that decides whether column zero is shrunk, which is the
        whole reason this class exists. A penalty on the intercept would make
        the fit depend on where the target's zero happens to sit: shift every
        target up by a hundred and a shrunk intercept cannot follow, so the
        slopes bend to absorb it. The intercept is a level, not an effect, and
        there is nothing about it to regularise.

        Asking the matrix removes the chance of getting it wrong. A caller
        writing ``matrix[0, 0] = 0.0`` for itself has to remember that the slot
        is only the intercept when there is one, and forgetting exempts a real
        predictor instead.

        Raises
        ------
        InvalidValuesError
            If the penalty is negative.
        """
        if penalty < 0.0:
            raise InvalidValuesError(f"Penalty must not be negative, got {penalty}")

        diagonal = np.eye(self.n_columns) * penalty
        if self._has_intercept:
            diagonal[0, 0] = 0.0

        return diagonal

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        return (
            f"DesignMatrix({self.n_rows}x{self.n_columns}, "
            f"intercept={self._has_intercept})"
        )
