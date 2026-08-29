"""Spec for the kernel functions and the Gram matrix.

Green already, since these are the vocabulary rather than a model. The test
carrying the argument is ``test_it_matches_the_explicit_expansion``, which
builds the degree-2 feature map by hand and checks the kernel returns the same
inner product. That is the trick's whole claim, and it is stated here as an
equality between two independently written expressions rather than as prose.

The oracle is written from the definition of the feature map, not from the
kernel formula, for the reason ``test/core/distance/test_distance.py`` records:
an oracle derived from the implementation tests nothing.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.data.row_block import rows_of
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.kernel.functions import (
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)
from oop_ml.core.kernel.matrix import KernelMatrix
from test.fixtures import (
    KERNEL_PAIR_DEGREE_TWO,
    KERNEL_PAIR_INNER_PRODUCT,
    KERNEL_PAIR_LEFT,
    KERNEL_PAIR_RIGHT,
)

NAMES = ("first", "second")


def block(*rows: tuple[float, float]):
    """Rows over the two fixture features."""
    return rows_of(np.array(rows, dtype=np.float64), NAMES)


def degree_two_expansion(row: tuple[float, float]) -> np.ndarray:
    """The explicit feature map for a degree-2 polynomial kernel.

    Written from the definition -- ``(x1^2, x2^2, sqrt(2) x1 x2)`` -- and not
    from the kernel formula, so that comparing the two compares two things.
    """
    first, second = row

    return np.array([first * first, second * second, np.sqrt(2.0) * first * second])


class TestTheTrick:
    """The claim the whole package rests on."""

    def test_it_matches_the_explicit_expansion(self) -> None:
        """``(a . b)^2`` equals ``phi(a) . phi(b)``, computed both ways.

        Hand-checked in the module docstring: for (1, 2) and (3, 4) the
        expansion gives 9 + 64 + 48 = 121, and the kernel gives 11^2 = 121.
        """
        expanded = float(
            degree_two_expansion(KERNEL_PAIR_LEFT)
            @ degree_two_expansion(KERNEL_PAIR_RIGHT)
        )
        kernelled = PolynomialKernel(degree=2, constant=0.0).between(
            block(KERNEL_PAIR_LEFT), block(KERNEL_PAIR_RIGHT)
        )

        assert expanded == pytest.approx(KERNEL_PAIR_DEGREE_TWO)
        assert kernelled.values[0, 0] == pytest.approx(expanded)

    def test_it_holds_for_every_pair_not_just_one(self) -> None:
        """One matching pair could be coincidence; the whole table cannot."""
        left = [(1.0, 2.0), (0.5, -1.0), (3.0, 3.0)]
        right = [(3.0, 4.0), (-2.0, 1.0)]

        expanded = np.array(
            [
                [
                    float(degree_two_expansion(one) @ degree_two_expansion(other))
                    for other in right
                ]
                for one in left
            ]
        )
        kernelled = PolynomialKernel(degree=2, constant=0.0).between(
            block(*left), block(*right)
        )

        assert np.allclose(kernelled.values, expanded)


class TestTheFourKernels:
    """What each one computes, on values checkable by hand."""

    def test_linear_is_the_inner_product(self) -> None:
        """(1, 2) . (3, 4) is 3 + 8."""
        result = LinearKernel().between(
            block(KERNEL_PAIR_LEFT), block(KERNEL_PAIR_RIGHT)
        )

        assert result.values[0, 0] == pytest.approx(KERNEL_PAIR_INNER_PRODUCT)

    def test_polynomial_at_degree_one_without_a_constant_is_linear(self) -> None:
        """The family contains the linear kernel, which is worth pinning."""
        left, right = block(KERNEL_PAIR_LEFT), block(KERNEL_PAIR_RIGHT)

        assert PolynomialKernel(degree=1, constant=0.0).between(
            left, right
        ).values == pytest.approx(LinearKernel().between(left, right).values)

    def test_radial_basis_is_one_at_zero_distance(self) -> None:
        """A point is maximally similar to itself, whatever gamma is."""
        same = block(KERNEL_PAIR_LEFT)

        assert RadialBasisKernel(gamma=3.0).between(same, same).values[
            0, 0
        ] == pytest.approx(1.0)

    def test_radial_basis_falls_towards_zero_with_distance(self) -> None:
        near = RadialBasisKernel(gamma=0.5).between(
            block((0.0, 0.0)), block((1.0, 0.0))
        )
        far = RadialBasisKernel(gamma=0.5).between(
            block((0.0, 0.0)), block((10.0, 0.0))
        )

        assert 0.0 < far.values[0, 0] < near.values[0, 0] < 1.0

    def test_radial_basis_matches_the_definition(self) -> None:
        """exp(-gamma * ||a - b||^2), computed from the gap directly."""
        gap = np.array(KERNEL_PAIR_LEFT) - np.array(KERNEL_PAIR_RIGHT)
        expected = float(np.exp(-0.5 * (gap @ gap)))

        assert RadialBasisKernel(gamma=0.5).between(
            block(KERNEL_PAIR_LEFT), block(KERNEL_PAIR_RIGHT)
        ).values[0, 0] == pytest.approx(expected)

    def test_sigmoid_is_bounded(self) -> None:
        """tanh, so it cannot leave (-1, 1) however large the inputs."""
        values = (
            SigmoidKernel(gamma=2.0)
            .between(block((100.0, 100.0)), block((100.0, 100.0)))
            .values
        )

        assert -1.0 < values[0, 0] <= 1.0


class TestKernelProperties:
    """What a Gram matrix has to look like, and what it refuses."""

    @pytest.mark.parametrize(
        "kernel",
        [
            LinearKernel(),
            PolynomialKernel(degree=3),
            RadialBasisKernel(gamma=0.5),
            SigmoidKernel(gamma=0.1),
        ],
    )
    def test_a_training_matrix_is_symmetric(self, kernel) -> None:
        """Half of Mercer's condition, and the half that is cheap to check."""
        rows = block((1.0, 2.0), (3.0, 4.0), (-1.0, 0.5))

        kernel.between(rows, rows).check_symmetric()

    @pytest.mark.parametrize(
        "kernel",
        [LinearKernel(), PolynomialKernel(degree=2), RadialBasisKernel(gamma=0.3)],
    )
    def test_a_valid_kernel_is_positive_semi_definite(self, kernel) -> None:
        """The other half, checked through the eigenvalues.

        A negative eigenvalue means no feature map exists, and then the
        optimisation problems built on the matrix are not convex.
        """
        rows = block((1.0, 2.0), (3.0, 4.0), (-1.0, 0.5), (0.0, 0.0))
        eigenvalues = np.linalg.eigvalsh(kernel.between(rows, rows).values)

        assert eigenvalues.min() > -1e-08

    def test_a_query_matrix_is_not_square(self) -> None:
        """The distinction that is easy to lose: both are "the kernel matrix"."""
        queries = block((1.0, 1.0), (2.0, 2.0))
        training = block((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

        result = LinearKernel().between(queries, training)

        assert (result.n_left, result.n_right) == (2, 3)
        assert not result.is_square

        with pytest.raises(InvalidValuesError):
            result.check_square()

    def test_pairing_across_different_features_is_rejected(self) -> None:
        """An inner product over mismatched columns is arithmetic, not a measure."""
        with pytest.raises(InvalidValuesError):
            LinearKernel().between(
                block((1.0, 2.0)),
                rows_of(np.array([[1.0, 2.0]]), ("first", "different")),
            )


class TestCentring:
    """Centring the implied features without touching them."""

    def training(self):
        return block((1.0, 2.0), (3.0, 4.0), (-1.0, 0.5), (0.0, 0.0))

    def test_centring_a_linear_kernel_matches_centring_the_rows(self) -> None:
        """The identity, checked against the thing it stands in for.

        For a linear kernel the implied space is the original one, so centring
        the matrix must equal building the matrix from centred rows. That is
        the only kernel where both routes are available, which is exactly why
        it is the one that can verify the identity.
        """
        rows = self.training()
        centred_rows = rows_of(rows.values - np.mean(rows.values, axis=0), NAMES)

        through_the_matrix = LinearKernel().between(rows, rows).centred()
        through_the_rows = LinearKernel().between(centred_rows, centred_rows)

        assert np.allclose(through_the_matrix.values, through_the_rows.values)

    def test_a_centred_matrix_has_rows_summing_to_zero(self) -> None:
        """What centring means, read off the result."""
        centred = LinearKernel().between(self.training(), self.training()).centred()

        assert np.allclose(np.sum(centred.values, axis=0), 0.0, atol=1e-10)

    def test_centring_leaves_the_original_untouched(self) -> None:
        matrix = LinearKernel().between(self.training(), self.training())
        before = matrix.values

        matrix.centred()

        assert np.allclose(matrix.values, before)

    def test_a_query_matrix_is_centred_against_the_training_one(self) -> None:
        """Query rows shift by the mean the fit learned, not their own."""
        training = self.training()
        queries = block((5.0, 5.0), (1.0, 2.0))

        trained = LinearKernel().between(training, training)
        centred = LinearKernel().between(queries, training).centred_against(trained)

        assert (centred.n_left, centred.n_right) == (2, 4)

    def test_centring_a_query_matrix_against_itself_is_refused(self) -> None:
        """It has no mean of its own to centre against, and it is not square."""
        training = self.training()
        queries = block((5.0, 5.0), (1.0, 2.0))

        with pytest.raises(InvalidValuesError):
            LinearKernel().between(queries, training).centred()

    def test_a_mismatched_training_matrix_is_refused(self) -> None:
        training = self.training()
        queries = block((5.0, 5.0))
        smaller = LinearKernel().between(block((1.0, 1.0)), block((1.0, 1.0)))

        with pytest.raises(InvalidValuesError):
            LinearKernel().between(queries, training).centred_against(smaller)


class TestTheMatrixItself:
    """Construction guards."""

    def test_a_non_finite_value_is_rejected(self) -> None:
        """What an overflowed kernel produces, caught where it is made."""
        with pytest.raises(InvalidValuesError):
            KernelMatrix(np.array([[1.0, np.inf], [0.0, 1.0]]))

    def test_a_one_dimensional_array_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            KernelMatrix(np.array([1.0, 2.0]))

    def test_an_asymmetric_square_matrix_is_rejected(self) -> None:
        """Square but not symmetric did not come from a kernel."""
        with pytest.raises(InvalidValuesError):
            KernelMatrix(np.array([[1.0, 2.0], [3.0, 1.0]])).check_symmetric()

    def test_it_reads_as_an_array(self) -> None:
        """The interop every output type here has."""
        matrix = KernelMatrix(np.array([[1.0, 0.0], [0.0, 1.0]]))

        assert np.allclose(np.asarray(matrix), np.eye(2))


class TestKernelParameters:
    """Validation at construction, where every hyperparameter here is checked."""

    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_a_non_positive_gamma_is_rejected(self, gamma: float) -> None:
        """At zero every pair scores 1 and the matrix is rank one."""
        with pytest.raises(ValidationError):
            RadialBasisKernel(gamma=gamma)

    def test_a_degree_below_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PolynomialKernel(degree=0)

    def test_a_negative_constant_is_rejected(self) -> None:
        """It can make the Gram matrix indefinite, so it stops being a kernel."""
        with pytest.raises(ValidationError):
            PolynomialKernel(constant=-1.0)

    def test_an_unknown_parameter_is_rejected(self) -> None:
        """extra="forbid", so a misspelling is not silently a default."""
        with pytest.raises(ValidationError):
            RadialBasisKernel(**{"gama": 0.5})
