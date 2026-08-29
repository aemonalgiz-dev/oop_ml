"""Spec for kernel ridge -- red until ``_solve`` and ``predict`` land.

Two tests carry the argument and they pull in opposite directions.

``test_a_linear_kernel_matches_ridge_regression`` is the control. Kernel ridge
with a linear kernel *is* ridge regression, solved through the ``(n, n)`` Gram
matrix instead of the ``(p, p)`` covariance, so the two must agree to floating
point. Where they disagree, one of them is wrong, and this is the only test here
that can tell you which half of the identity broke.

``test_a_polynomial_kernel_fits_the_curve_a_line_cannot`` is the payoff. The
fixture is ``y = first ** 2`` exactly, which no straight line reaches, and a
degree-2 kernel reproduces it. Asserting the linear ceiling alongside the
polynomial floor is what makes that a demonstration rather than a claim.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import InvalidValuesError, NotFittedError
from oop_ml.core.kernel.functions import (
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
)
from oop_ml.regression.kernels.kernel_ridge_regression import KernelRidgeRegression
from oop_ml.regression.penalised.ridge_regression import RidgeRegression
from test.fixtures import (
    QUADRATIC_CURVE,
    QUADRATIC_LINEAR_CEILING,
    QUADRATIC_POLYNOMIAL_FLOOR,
)

PENALTY = 0.01


def fitted(kernel=None, penalty: float = PENALTY) -> KernelRidgeRegression:
    """A model fitted to the quadratic fixture."""
    return KernelRidgeRegression(kernel=kernel or LinearKernel(), penalty=penalty).fit(
        QUADRATIC_CURVE.input_features, QUADRATIC_CURVE.target_feature
    )


class TestAgainstRidge:
    """The control: with a linear kernel these are the same model."""

    def test_a_linear_kernel_matches_ridge_regression(self) -> None:
        """The identity in the module docstring, asserted rather than described.

        ``(X'X + pI)^-1 X'`` and ``X'(XX' + pI)^-1`` are the same matrix, so
        the two routes are the same fit at any penalty. This is the test that
        localises a bug: if the polynomial fit is wrong and this is right, the
        kernel is at fault; if this is wrong too, the solve is.
        """
        kernelled = fitted(LinearKernel(), penalty=PENALTY)
        ordinary = RidgeRegression(penalty=PENALTY).fit(
            QUADRATIC_CURVE.input_features, QUADRATIC_CURVE.target_feature
        )

        assert np.allclose(
            np.asarray(kernelled.predict(QUADRATIC_CURVE.input_features)),
            np.asarray(ordinary.predict(QUADRATIC_CURVE.input_features)),
            atol=1e-06,
        )

    def test_the_agreement_holds_at_a_larger_penalty_too(self) -> None:
        """One penalty could be luck; the identity is penalty-free."""
        kernelled = KernelRidgeRegression(penalty=5.0).fit(
            QUADRATIC_CURVE.input_features, QUADRATIC_CURVE.target_feature
        )
        ordinary = RidgeRegression(penalty=5.0).fit(
            QUADRATIC_CURVE.input_features, QUADRATIC_CURVE.target_feature
        )

        assert np.allclose(
            np.asarray(kernelled.predict(QUADRATIC_CURVE.input_features)),
            np.asarray(ordinary.predict(QUADRATIC_CURVE.input_features)),
            atol=1e-06,
        )


class TestWhatTheKernelBuys:
    """The payoff: a curve no line reaches."""

    def test_a_polynomial_kernel_fits_the_curve_a_line_cannot(self) -> None:
        """``y = first ** 2`` exactly. Both halves asserted, so it is a contrast."""
        linear = fitted(LinearKernel()).score(
            QUADRATIC_CURVE.input_features, QUADRATIC_CURVE.target_feature
        )
        polynomial = fitted(PolynomialKernel(degree=2, constant=1.0)).score(
            QUADRATIC_CURVE.input_features, QUADRATIC_CURVE.target_feature
        )

        assert linear < QUADRATIC_LINEAR_CEILING
        assert polynomial > QUADRATIC_POLYNOMIAL_FLOOR

    def test_a_radial_kernel_also_reaches_the_curve(self) -> None:
        """A different implied space, the same conclusion."""
        score = fitted(RadialBasisKernel(gamma=0.1)).score(
            QUADRATIC_CURVE.input_features, QUADRATIC_CURVE.target_feature
        )

        assert score > QUADRATIC_LINEAR_CEILING


class TestWhatItLearns:
    """Dual weights, and how they differ from coefficients."""

    def test_it_learns_one_weight_per_training_row(self) -> None:
        """Not one per feature. That is the whole shape of the model."""
        model = fitted()

        assert model.dual_weights.shape == (QUADRATIC_CURVE.n_samples,)
        assert model.n_training_rows == QUADRATIC_CURVE.n_samples

    def test_it_has_no_coefficients_to_report(self) -> None:
        """The honest absence: in the implied space, features have no names."""
        assert not hasattr(fitted(PolynomialKernel(degree=2)), "coefficients")

    def test_the_dual_weights_satisfy_the_system_they_solved(self) -> None:
        """``(K + penalty I) a = y_centred``, checked by substituting back.

        Written from the definition rather than by re-running the solve, so it
        is an oracle rather than a copy of the implementation.
        """
        model = fitted(PolynomialKernel(degree=2))
        kernel_values = model.kernel.between(
            model.training_rows, model.training_rows
        ).values
        system = kernel_values + PENALTY * np.eye(QUADRATIC_CURVE.n_samples)
        centred_target = QUADRATIC_CURVE.target_feature.values - model.target_mean

        assert np.allclose(system @ model.dual_weights, centred_target, atol=1e-06)

    def test_it_keeps_every_training_row(self) -> None:
        """Unlike a support vector machine, which keeps a fraction.

        Prediction needs the kernel between the query and each training row, so
        the model's size is the training set's size.
        """
        assert fitted().training_rows.n_rows == QUADRATIC_CURVE.n_samples


class TestPredicting:
    """The query side, where the kernel matrix stops being square."""

    def test_it_predicts_one_value_per_row(self) -> None:
        predicted = fitted(PolynomialKernel(degree=2)).predict(
            QUADRATIC_CURVE.input_features
        )

        assert len(np.asarray(predicted)) == QUADRATIC_CURVE.n_samples

    def test_it_predicts_for_rows_the_fit_never_saw(self) -> None:
        """A single new row, which makes the query matrix (1, n)."""
        model = fitted(PolynomialKernel(degree=2, constant=1.0))
        predicted = model.predict([Feature("first", [2.0]), Feature("second", [0.0])])

        assert float(np.asarray(predicted)[0]) == pytest.approx(4.0, abs=0.5)

    def test_column_order_does_not_matter(self) -> None:
        model = fitted(PolynomialKernel(degree=2))
        first, second = QUADRATIC_CURVE.input_features

        assert np.allclose(
            np.asarray(model.predict([first, second])),
            np.asarray(model.predict([second, first])),
        )

    def test_the_centring_is_undone(self) -> None:
        """The target mean was subtracted before fitting and has to come back.

        Skip it and every prediction is off by the same constant -- the shape
        is right, the level is wrong, and R^2 goes sharply negative.
        """
        model = fitted(PolynomialKernel(degree=2, constant=1.0))
        predicted = np.asarray(model.predict(QUADRATIC_CURVE.input_features))

        assert float(np.mean(predicted)) == pytest.approx(
            float(np.mean(QUADRATIC_CURVE.target_feature.values)), abs=0.5
        )


class TestWhatItRefuses:
    """Guards, each from the MLLibError hierarchy."""

    def test_reading_the_weights_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = KernelRidgeRegression().dual_weights

    def test_predicting_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            KernelRidgeRegression().predict(QUADRATIC_CURVE.input_features)

    def test_a_non_positive_penalty_is_rejected(self) -> None:
        """At zero the solve is ill-conditioned and the fit interpolates."""
        with pytest.raises(ValidationError):
            KernelRidgeRegression(penalty=0.0)

    def test_predicting_without_every_fitted_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().predict([QUADRATIC_CURVE.input_features[0]])

    def test_predicting_with_an_unknown_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().predict(
                [
                    *QUADRATIC_CURVE.input_features,
                    Feature("extra", [1.0] * QUADRATIC_CURVE.n_samples),
                ]
            )
