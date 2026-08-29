"""Ridge regression that never sees the space it fits in.

Theory
------
Ridge minimises ``||y - Xw||^2 + penalty ||w||^2`` and solves it in closed form
as ``w = (X'X + penalty I)^-1 X'y``. That inverse is ``(p, p)``, so the cost is
set by the number of *features*. Expand to degree 5 over 20 features and ``p``
is 53,130, and the solve is hopeless before the memory is.

The way out is an identity worth knowing on its own::

    (X'X + penalty I)^-1 X'  =  X' (XX' + penalty I)^-1

Both sides are the same matrix. The left inverts a ``(p, p)``; the right
inverts an ``(n, n)``. So the same fit can be had at a cost set by the number of
*rows* instead, and rows are something you have a fixed number of however far
the features are expanded.

That rearrangement is what makes the trick usable. Substituting it::

    w  =  X' (XX' + penalty I)^-1 y  =  X' a      where  a = (K + penalty I)^-1 y

and ``K = XX'`` is the Gram matrix -- every pairwise inner product, and nothing
else about ``X``. So:

* **The fit** needs only ``K``, an ``(n, n)`` table of inner products.
* **The weights** are never formed. ``w = X'a`` is a combination of the training
  rows, and in the expanded space it would have 53,130 entries, so it is left
  as the coefficients ``a`` -- one per training row.
* **A prediction** is ``x'w = x'X'a = k(x, X) a``: the query's kernel values
  against the training rows, weighted by ``a``. Again only inner products.

Replace every inner product with a kernel and the model fits in whatever space
that kernel implies, at a cost that never mentions its dimension.

The dual weights are not the coefficients
------------------------------------------
This is the difference that matters when reading a fitted model.
``RidgeRegression`` learns one number per **feature**, and
``coefficients["floor_area"]`` is a sentence about floor area. Kernel ridge
learns one number per **training row**, and ``a[7]`` says how much row 7
contributes to every prediction. There is no per-feature number to report,
because in the implied space the features are not things you have names for.

So this model has no ``coefficients`` property, and that absence is the honest
one. It is also why every training row has to be kept: prediction needs the
kernel between the query and each of them, so the "model" is the rows plus
``a``, and it grows with the training set rather than with the features.

What the penalty does here
---------------------------
The same thing it does in ridge, and one thing more. ``K + penalty I`` is
better conditioned than ``K``, and with an RBF kernel on distinct rows ``K`` is
positive definite but often barely so, with eigenvalues near zero. At
``penalty = 0`` the solve is numerically hopeless and the fit interpolates every
training point exactly. The penalty is not optional here in the way it is
optional for ordinary least squares.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Regressor
from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.kernel.functions import Kernel, LinearKernel
from oop_ml.core.kernel.matrix import KernelMatrix
from oop_ml.core.types import FloatArray
from oop_ml.core.validation import ValueRole


class KernelRidgeRegression(Regressor[Sequence[Feature], Feature]):
    """Ridge regression solved through the Gram matrix rather than the columns.

    Parameters
    ----------
    kernel:
        Which space to fit in. The default is
        :class:`~oop_ml.core.kernel.functions.LinearKernel`, which makes this
        ordinary ridge regression solved the other way round -- a control worth
        having, since the two must agree.
    penalty:
        The ridge penalty, applied to the diagonal of the Gram matrix. Must be
        positive: at zero the solve is ill-conditioned for most kernels and the
        model interpolates the training rows exactly.

    Raises
    ------
    InvalidValuesError
        If ``penalty`` is not positive.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    kernel: Kernel = LinearKernel()
    penalty: float = Field(default=1.0, gt=0.0)

    _dual_weights: FloatArray | None = PrivateAttr(default=None)
    _training_rows: RowBlock | None = PrivateAttr(default=None)
    _feature_means: FloatArray | None = PrivateAttr(default=None)
    _target_mean: float = PrivateAttr(default=0.0)

    @property
    def dual_weights(self) -> FloatArray:
        """One weight per training row -- what this model learned.

        Not coefficients. There is one of these per *row*, not per feature, and
        entry ``i`` says how much training row ``i`` contributes to every
        prediction. See the module docstring for why no per-feature number
        exists to report.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._dual_weights is not None
        return self._dual_weights.copy()

    @property
    def n_training_rows(self) -> int:
        """How many rows this model has to keep in order to predict.

        Worth reading, because it is the model's size. A kernel model grows
        with the training set rather than with the features.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._training_rows is not None
        return self._training_rows.n_rows

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Learn one dual weight per training row.

        Builds the Gram matrix, centres the target, hands both to
        :meth:`_solve`, and keeps the training rows -- prediction needs them,
        which is what makes this model's size the training set's size.

        **Both** the features and the target are centred, and an intercept
        column is never added. In the implied space there is no column to add,
        and centring achieves the same thing: the fit works on deviations from
        the means, and the target's mean is added back in ``predict``.

        Centring the features as well as the target is what makes the identity
        exact rather than approximate. Measured on the quadratic fixture with a
        linear kernel: centring only the target gives 4.2841 where ridge gives
        4.3580, close enough to look like rounding and wrong for a reason.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features and target are different lengths.
        NonUniqueFeaturesError
            If two features share a name.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        names = tuple(feature.name for feature in feature_set)
        raw_rows = self._as_rows(feature_set, names)
        target_column = Column.of(target_values, ValueRole.TARGET_VALUES)

        self._feature_means = np.mean(raw_rows.values, axis=0)
        rows = rows_of(raw_rows.values - self._feature_means, names)

        self._training_rows = rows
        self._target_mean = float(np.mean(target_column.values))

        centred_target = Column.selecting(
            target_column.values - self._target_mean, ValueRole.TARGET_VALUES
        )
        self._dual_weights = self._solve(
            self.kernel.between(rows, rows), centred_target
        )
        self._mark_fitted()

        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """Weight each training row's kernel value against the query rows.

        A prediction is ``k(x, X) a + mean``: the query's kernel values against
        every training row, weighted by the dual weights, with the centring
        undone.

        Note the shape of the kernel matrix here. Fitting pairs the training
        rows with themselves and gets an ``(n, n)``; this pairs queries with
        training rows and gets ``(n_queries, n_training)``. It is not square
        and it is not symmetric, which is why ``KernelMatrix`` distinguishes
        the two rather than treating "the kernel matrix" as one thing.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        raise NotImplementedError

    def _solve(self, kernel_matrix: KernelMatrix, target_column: Column) -> FloatArray:
        """Solve ``(K + penalty I) a = y`` for the dual weights.

        The whole fit, and it is one linear solve.

        1. Take the Gram matrix's values, an ``(n, n)`` table.
        2. Add ``penalty`` to its diagonal. Every entry of the diagonal, with
           no exemption -- unlike ridge's ``penalty_diagonal``, where column
           zero may be an intercept that must not be shrunk. There are no
           columns here, only rows, and no row is an intercept.
        3. Solve against ``target_column``. Use ``numpy.linalg.solve``, not an
           explicit inverse: forming ``inv(A) @ b`` is slower and less accurate
           than solving ``A x = b``, and the identity in the module docstring
           is a statement about which matrix to *solve*, not one to invert.

        The matrix is symmetric positive definite once the penalty is added, so
        the solve cannot fail on a valid kernel. If it does, the kernel was not
        one -- which is the practical face of Mercer's condition.

        Parameters
        ----------
        kernel_matrix:
            The training Gram matrix, square. Call ``check_square()`` on it
            before relying on that.
        target_column:
            The centred target, one value per training row.

        Returns
        -------
        FloatArray
            The dual weights, shape ``(n_training_rows,)``. Do not set
            ``_fitted`` here.
        """
        raise NotImplementedError

    def _as_rows(self, feature_set: FeatureSet, names: tuple[str, ...]) -> RowBlock:
        """The features as a row block, in the given order."""
        ordered = FeatureSet.matching(names, list(feature_set))

        return rows_of(
            np.column_stack([ordered.column(name).values for name in names]), names
        )

    def _query_rows(self, input_values: Sequence[Feature]) -> RowBlock:
        """Check the features match the fit, then lay them out in its order."""
        self._check_fitted()
        assert self._training_rows is not None

        fitted = self._training_rows.feature_names
        supplied = {feature.name for feature in input_values}

        if supplied != set(fitted):
            raise InvalidValuesError(
                f"expected exactly the fitted features {sorted(fitted)}; "
                f"got {sorted(supplied)}"
            )

        assert self._feature_means is not None
        raw = self._as_rows(FeatureSet(input_values), fitted)

        return rows_of(raw.values - self._feature_means, fitted)

    @property
    def training_rows(self) -> RowBlock:
        """The rows this model kept, which prediction needs.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._training_rows is not None
        return self._training_rows

    @property
    def target_mean(self) -> float:
        """The mean subtracted before fitting, added back when predicting.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        return self._target_mean

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"KernelRidgeRegression({self.kernel!r}, unfitted)"

        return (
            f"KernelRidgeRegression({self.kernel!r}, penalty={self.penalty}, "
            f"n_training_rows={self.n_training_rows})"
        )
