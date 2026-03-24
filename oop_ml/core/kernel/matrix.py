"""The Gram matrix: every pair's kernel value, and what it is allowed to be.

A kernel model's whole view of the data is this table. Once it exists the
original rows are not consulted again during the fit, which is precisely what
makes the expanded space unnecessary -- and also what makes the table's
properties worth enforcing, since a mistake here is a mistake about the data
itself rather than about one step of an algorithm.

Two shapes, and only one of them is square
-------------------------------------------
Fitting pairs the training rows with themselves, so ``K`` is ``(n, n)``,
symmetric, and positive semi-definite. Predicting pairs queries with the
*training* rows, so it is ``(n_queries, n_training)`` and none of those
properties apply -- it is not square and symmetry is not even meaningful.

Conflating the two is the easiest mistake in this corner of the library, because
both are "the kernel matrix" in conversation. :meth:`KernelMatrix.check_square`
is what a fit calls before assuming it has the first kind.

Centring, and why it is done here rather than on the rows
----------------------------------------------------------
PCA centres its data before decomposing, because variance is measured about the
mean. Kernel PCA has to do the same thing -- in the expanded space, whose points
it cannot see. So the mean cannot be subtracted from the rows.

It can be done to the matrix instead. Centring the implied features is exactly::

    K_centred = K - 1_n K - K 1_n + 1_n K 1_n

where ``1_n`` is the ``(n, n)`` matrix with every entry ``1/n``. Each term
subtracts one of the pieces that expanding ``(phi(a) - mean) . (phi(b) - mean)``
produces, and the last one adds back the piece subtracted twice -- it is the
inclusion-exclusion of the expansion, not a formula to memorise.

That identity is the reason kernel PCA is possible at all, and it is why this
lives on the matrix rather than in the model: it is a fact about Gram matrices,
and any model needing centred implied features needs this exact expression.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.types import FloatArray, array_for_protocol

SYMMETRY_TOLERANCE = 1e-08
"""How far a square Gram matrix may stray from its own transpose.

``K(a, b)`` and ``K(b, a)`` are computed by the same expression on swapped
operands, so they agree to within floating-point reordering and no further.
"""


class KernelMatrix:
    """Kernel values for every pair of rows in two blocks.

    Parameters
    ----------
    values:
        Shape ``(n_left, n_right)``. Entry ``[i, j]`` is the kernel value
        pairing left row ``i`` with right row ``j``.

    Raises
    ------
    InvalidValuesError
        If the values are not a two-dimensional finite array. A non-finite
        entry means the kernel overflowed -- a polynomial kernel of high degree
        on unscaled features is the usual cause -- and it would poison every
        solve downstream while looking like a number.
    """

    __slots__ = ("_values",)

    def __init__(self, values: FloatArray) -> None:
        as_array = np.asarray(values, dtype=np.float64)

        if as_array.ndim != 2:
            raise InvalidValuesError(
                f"a kernel matrix pairs two blocks of rows, so it has two "
                f"dimensions; got shape {as_array.shape}"
            )

        if not np.all(np.isfinite(as_array)):
            raise InvalidValuesError(
                "a kernel matrix holds a non-finite value, which usually means "
                "the kernel overflowed on unscaled features"
            )

        self._values = as_array

    @property
    def values(self) -> FloatArray:
        """The table itself, as a copy so a caller cannot corrupt it."""
        return self._values.copy()

    @property
    def n_left(self) -> int:
        """How many rows came from the left block."""
        return int(self._values.shape[0])

    @property
    def n_right(self) -> int:
        """How many rows came from the right block."""
        return int(self._values.shape[1])

    @property
    def is_square(self) -> bool:
        """Whether this pairs a block with itself, shape-wise."""
        return self.n_left == self.n_right

    def check_square(self) -> None:
        """Raise unless this is a training matrix rather than a query one.

        What a fit calls before assuming symmetry and positive
        semi-definiteness. A ``(n_queries, n_training)`` matrix has neither, and
        the distinction is easy to lose because both are "the kernel matrix" in
        conversation.

        Raises
        ------
        InvalidValuesError
            If the matrix is not square.
        """
        if not self.is_square:
            raise InvalidValuesError(
                f"expected a training kernel matrix, which is square; got "
                f"{self.n_left} by {self.n_right}"
            )

    def check_symmetric(self) -> None:
        """Raise unless ``K(a, b)`` equals ``K(b, a)`` throughout.

        Half of Mercer's condition, and the half that can be checked cheaply.
        A square Gram matrix that is not symmetric did not come from a kernel,
        whatever it came from.

        Raises
        ------
        InvalidValuesError
            If the matrix is not square, or is square and not symmetric.
        """
        self.check_square()

        if not np.allclose(
            self._values, self._values.T, atol=SYMMETRY_TOLERANCE, rtol=0.0
        ):
            raise InvalidValuesError(
                "a kernel matrix must be symmetric; K(a, b) and K(b, a) differ "
                "by more than rounding, so this did not come from a kernel"
            )

    def centred(self) -> KernelMatrix:
        """The same pairing, with the implied features centred on their mean.

        The identity in the module docstring::

            K - 1_n K - K 1_n + 1_n K 1_n

        Returns a new matrix; nothing here mutates.

        Raises
        ------
        InvalidValuesError
            If the matrix is not square, since centring is defined against a
            block's own mean and a query matrix does not have one.
        """
        self.check_square()

        n_rows = self.n_left
        uniform = np.full((n_rows, n_rows), 1.0 / n_rows)
        values = self._values

        return KernelMatrix(
            values - uniform @ values - values @ uniform + uniform @ values @ uniform
        )

    def centred_against(self, training: KernelMatrix) -> KernelMatrix:
        """Centre a query matrix against the training block's implied mean.

        The counterpart for prediction, and the step that is easy to skip. New
        rows have to be shifted by the mean the *fit* learned, exactly as
        ``PrincipalComponentAnalysis.transform`` subtracts the fitted means
        rather than the query rows' own. Centring a query matrix against itself
        would be the same leak, in a space where nothing looks wrong.

        Parameters
        ----------
        training:
            The square matrix the fit was built from.

        Raises
        ------
        InvalidValuesError
            If ``training`` is not square, or if its size does not match this
            matrix's right-hand block.
        """
        training.check_square()

        if self.n_right != training.n_left:
            raise InvalidValuesError(
                f"this matrix pairs against {self.n_right} training rows, but "
                f"the training matrix holds {training.n_left}"
            )

        n_rows = training.n_left
        uniform_query = np.full((self.n_left, n_rows), 1.0 / n_rows)
        uniform_training = np.full((n_rows, n_rows), 1.0 / n_rows)
        trained = training.values

        return KernelMatrix(
            self._values
            - uniform_query @ trained
            - self._values @ uniform_training
            + uniform_query @ trained @ uniform_training
        )

    def __array__(self, dtype=None, copy=None) -> FloatArray:
        """Hand numpy this wrapper, honouring the copy parameter.

        See :func:`~oop_ml.core.types.array_for_protocol` for the
        contract and the corruption it exists to prevent.
        """
        return array_for_protocol(self._values, dtype, copy)

    def __repr__(self) -> str:
        return f"KernelMatrix({self.n_left} by {self.n_right})"
