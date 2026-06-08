"""Spec for the Gaussian arithmetic both backends share.

Mostly exercised through the three generative models that call it, so what is
here is the one piece that has no natural home in any of them: the rank test,
and the measurement that says why a Cholesky factorisation could not do its
job.
"""

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from oop_ml.core.gaussian import (
    LOG_TWO_PI,
    MINIMUM_CLASS_ROWS,
    is_invertible,
    normalised_from_log_scores,
)

#: Determinant exactly zero, eigenvalues exactly [0, 10], and accepted by
#: ``cho_factor``. The whole reason :func:`is_invertible` exists.
SINGULAR_THAT_FACTORISES = np.array([[8.0, 4.0], [4.0, 2.0]])


class TestIsInvertible:
    def test_a_matrix_with_spread_in_every_direction_passes(self):
        assert is_invertible(np.array([[2.0, 0.5], [0.5, 1.0]]))

    def test_the_identity_passes(self):
        assert is_invertible(np.eye(4))

    def test_an_exactly_singular_matrix_is_refused(self):
        assert not is_invertible(SINGULAR_THAT_FACTORISES)

    def test_and_a_cholesky_factorisation_would_have_accepted_it(self):
        """The measurement behind the whole function.

        LAPACK refuses a matrix with a *negative* pivot. An exactly singular
        one has a zero pivot, which is not negative, so it can pass. Catching
        ``LinAlgError`` around the solve therefore tests something adjacent to
        what a caller wanted tested, and on this matrix the two answers differ.
        """
        factored = cho_factor(SINGULAR_THAT_FACTORISES, check_finite=False)

        assert factored is not None
        assert np.linalg.det(SINGULAR_THAT_FACTORISES) == 0.0

    def test_and_the_answer_it_gives_does_not_solve_the_system(self):
        """Left to itself the factorisation does not fail loudly, which is the
        problem. It returns numbers around 1e15 that put ``A x`` back at the
        origin rather than at ``b``, and inside a fit those become ``inf``
        scores from a model that reports itself fitted."""
        wanted = np.array([1.0, 1.0])
        factor, lower = cho_factor(SINGULAR_THAT_FACTORISES, check_finite=False)

        answer = cho_solve((factor, lower), wanted, check_finite=False)

        assert np.max(np.abs(answer)) > 1e14
        assert not np.allclose(SINGULAR_THAT_FACTORISES @ answer, wanted)

    def test_the_test_is_relative_so_units_do_not_decide_it(self):
        """A well-conditioned covariance stays invertible however small the
        numbers in it are, because what is compared is the least direction
        against the largest rather than against an absolute floor."""
        well_shaped = np.array([[2.0, 0.5], [0.5, 1.0]])

        for power in (0, -6, -12, -60):
            assert is_invertible(well_shaped * (10.0**power))


class TestNormalisedFromLogScores:
    def test_rows_sum_to_one(self):
        scores = np.array([[1.0, 2.0, 3.0], [-40.0, -41.0, -42.0]])

        normalised = normalised_from_log_scores(scores)

        assert np.allclose(normalised.sum(axis=1), 1.0)

    def test_a_row_of_huge_scores_does_not_overflow(self):
        """Each row's largest is subtracted before exponentiating, which is the
        same answer and cannot overflow."""
        scores = np.array([[900.0, 899.0], [-900.0, -899.0]])

        normalised = normalised_from_log_scores(scores)

        assert np.isfinite(normalised).all()
        assert np.allclose(normalised.sum(axis=1), 1.0)

    def test_adding_a_constant_to_a_row_changes_nothing(self):
        """Which is what lets the discriminants drop every term that does not
        depend on the class."""
        scores = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])
        shifted = scores + np.array([[100.0], [-7.0]])

        assert np.allclose(
            normalised_from_log_scores(scores), normalised_from_log_scores(shifted)
        )


class TestTheConstants:
    def test_log_two_pi_is_the_number_and_not_the_expression(self):
        """Written out rather than recomputed, so this asserts the constant is
        right rather than that it equals itself."""
        assert abs(LOG_TWO_PI - 1.8378770664093453) < 1e-15

    def test_it_is_what_makes_a_standard_normal_integrate_to_one(self):
        """The reason the term is in the density at all. Trapezoidal over
        [-12, 12] is plenty for a curve this thin in the tails."""
        positions = np.linspace(-12.0, 12.0, 200_001)
        density = np.exp(-0.5 * (positions**2 + LOG_TWO_PI))

        assert np.trapezoid(density, positions) == pytest.approx(1.0, abs=1e-9)

    def test_a_class_needs_two_rows_before_it_has_any_spread(self):
        assert MINIMUM_CLASS_ROWS == 2
