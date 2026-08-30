"""Spec for the one failure that escaped the exception hierarchy.

Collinear features -- one column a linear combination of others -- make
``X.T X`` singular, and until now that escaped ``MultipleLinearRegression`` as
a bare ``numpy.linalg.LinAlgError``: the only reachable failure in the library
that did not derive from ``MLLibError``. A caller catching the library's own
errors missed it, and behind a web API it is the difference between a 422 with
a diagnosis and a 500 with a numpy traceback.

The ridge contrast is asserted alongside, because it is the teaching point:
the same data that breaks ordinary least squares fits fine under any positive
penalty, which is half of why ridge exists. At ``penalty=0`` ridge inherits
the failure and now reports it the same way.

Kernel ridge fails for a different reason with the same numpy face: its system
``K + penalty I`` can only be singular when ``K`` was not positive
semi-definite, which means the kernel was not a kernel -- Mercer's condition,
failed in practice. Its error says that, not "collinear features", because the
diagnosis a caller needs is about their kernel parameters, not their columns.
"""

import numpy as np
import pytest

from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    CollinearFeaturesError,
    InvalidValuesError,
    MLLibError,
)
from oop_ml.core.kernel.matrix import KernelMatrix
from oop_ml.core.validation import ValueRole
from oop_ml.regression.kernels.kernel_ridge_regression import KernelRidgeRegression
from oop_ml.regression.least_squares.multiple_feature_regression import (
    MultipleLinearRegression,
)
from oop_ml.regression.penalised.ridge_regression import RidgeRegression

DOUBLED_COLUMN = [
    Feature("floor_area", [1.0, 2.0, 3.0, 4.0, 5.0]),
    Feature("floor_area_copied", [2.0, 4.0, 6.0, 8.0, 10.0]),
]
TARGET = Feature("price", [3.0, 6.0, 9.0, 12.0, 15.0])


class TestOrdinaryLeastSquares:
    """The escape, closed."""

    def test_collinear_features_raise_a_typed_error(self) -> None:
        """The second column is exactly twice the first, so X.T X is singular.

        This used to escape as numpy.linalg.LinAlgError -- the one reachable
        failure outside the MLLibError hierarchy, recorded on the roadmap since
        the linear models were built.
        """
        with pytest.raises(CollinearFeaturesError):
            MultipleLinearRegression().fit(DOUBLED_COLUMN, TARGET)

    def test_the_error_routes_with_the_rest_of_the_hierarchy(self) -> None:
        """One except clause catches everything this library can raise."""
        with pytest.raises(MLLibError):
            MultipleLinearRegression().fit(DOUBLED_COLUMN, TARGET)

    def test_the_message_carries_the_diagnosis(self) -> None:
        """Collinear is the word a caller can act on; singular is not."""
        with pytest.raises(CollinearFeaturesError, match="collinear"):
            MultipleLinearRegression().fit(DOUBLED_COLUMN, TARGET)


class TestRidge:
    """The contrast that is the teaching point."""

    def test_any_positive_penalty_makes_the_same_data_solvable(self) -> None:
        """Half of why ridge exists: the penalty makes X.T X + pI invertible."""
        model = RidgeRegression(penalty=0.1).fit(DOUBLED_COLUMN, TARGET)

        assert model.score(DOUBLED_COLUMN, TARGET) > 0.99

    def test_at_penalty_zero_ridge_inherits_the_failure_and_the_error(self) -> None:
        """No penalty means no protection, and the report is the same one."""
        with pytest.raises(CollinearFeaturesError):
            RidgeRegression(penalty=0.0).fit(DOUBLED_COLUMN, TARGET)


class TestKernelRidge:
    """The same numpy face, a different diagnosis."""

    def test_a_singular_system_reports_the_kernel_not_the_columns(self) -> None:
        """``K + penalty I`` singular means the kernel failed Mercer.

        A valid kernel's Gram matrix is positive semi-definite, so adding a
        positive penalty always gives an invertible system; only an invalid
        kernel (a sigmoid with unlucky parameters) can defeat it. The crafted
        matrix here has eigenvalue -1, so at ``penalty=1`` the system is
        exactly singular -- and the error should send the caller to their
        kernel parameters, not to their columns.
        """
        crafted = KernelMatrix(np.array([[-1.0, 0.0], [0.0, 1.0]]))
        target = Column.of([1.0, 2.0], ValueRole.TARGET_VALUES)

        with pytest.raises(InvalidValuesError, match="kernel"):
            KernelRidgeRegression(penalty=1.0)._solve(crafted, target)
