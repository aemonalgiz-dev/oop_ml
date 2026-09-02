"""Spec for NewtonLogisticRegression -- red until the four stubs land.

The validation, the wiring and the singular-system guard are already in place,
so the guard tests pass now and hold the contract while the solver is written.

Three specifications here carry more weight than the rest. The first is that
this must agree with the gradient-ascent model to within floating point: they
maximise the same concave function, so they have no licence to disagree about
where its maximum is, and a test that only checked "close-ish" would let a
subtly wrong Hessian through. The second is the iteration count -- the entire
argument for the method is that it needs single digits, and a version that
converged in three hundred steps would be correct and pointless. The third is
the Hessian's own shape: symmetric and positive semi-definite by construction,
which is what makes undamped Newton safe on this objective.
"""

import numpy as np
import pytest

from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.probabilities import Probabilities
from oop_ml.core.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    SingleClassError,
    SingularHessianError,
)
from oop_ml.numpy.classification.binary.logistic_regression import LogisticRegression
from oop_ml.numpy.classification.binary.newton_logistic_regression import (
    NewtonLogisticRegression,
)
from test.fixtures import (
    OVERLAPPING_LABELS,
    OVERLAPPING_LABELS_BOUNDARY,
    OVERLAPPING_LABELS_ODDS_MULTIPLIER,
    SEPARABLE_LABELS,
)


def fitted_model(**overrides) -> NewtonLogisticRegression:
    model = NewtonLogisticRegression(**overrides)
    model.fit(OVERLAPPING_LABELS.input_features, OVERLAPPING_LABELS.target_feature)

    return model


def design_matrix_of(fixture) -> DesignMatrix:
    """``X`` with its ones column, built the way the model builds it.

    A :class:`~oop_ml.core.data.design_matrix.DesignMatrix` rather than a bare
    array, because that is what every solver takes now: the matrix carries
    whether its first column is the intercept, so nothing downstream has to
    consult ``fit_intercept`` to find out.
    """
    columns = [feature.values for feature in fixture.input_features]
    names = [feature.name for feature in fixture.input_features]

    return DesignMatrix(
        np.column_stack([np.ones(len(columns[0]))] + columns), names, True
    )


class TestConstruction:
    def test_defaults(self):
        model = NewtonLogisticRegression()

        assert model.max_iterations == 100
        assert model.tolerance == pytest.approx(1e-10)
        assert model.threshold == pytest.approx(0.5)
        assert model.fit_intercept is True

    @pytest.mark.parametrize("max_iterations", [0, -1])
    def test_non_positive_iteration_cap_is_rejected(self, max_iterations):
        with pytest.raises(ValueError):
            NewtonLogisticRegression(max_iterations=max_iterations)

    @pytest.mark.parametrize("tolerance", [0.0, -1e-8])
    def test_non_positive_tolerance_is_rejected(self, tolerance):
        with pytest.raises(ValueError):
            NewtonLogisticRegression(tolerance=tolerance)

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.2, 1.5])
    def test_threshold_outside_the_open_unit_interval_is_rejected(self, threshold):
        with pytest.raises(ValueError):
            NewtonLogisticRegression(threshold=threshold)

    def test_there_is_no_learning_rate(self):
        # The whole point: curvature sets the step length, so there is no dial.
        assert "learning_rate" not in NewtonLogisticRegression.model_fields


class TestNotFitted:
    @pytest.mark.parametrize(
        "attribute", ["coefficients", "intercept", "iterations_run", "converged"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(NewtonLogisticRegression(), attribute)

    def test_predict_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            NewtonLogisticRegression().predict(OVERLAPPING_LABELS.input_features)

    def test_predict_probability_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            NewtonLogisticRegression().predict_probability(
                OVERLAPPING_LABELS.input_features
            )


class TestValidation:
    def test_no_features_is_rejected(self):
        with pytest.raises(EmptyValuesError):
            NewtonLogisticRegression().fit([], OVERLAPPING_LABELS.target_feature)

    def test_duplicate_feature_names_are_rejected(self):
        repeated = Feature("hours", [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(NonUniqueFeaturesError):
            NewtonLogisticRegression().fit(
                [repeated, repeated], Feature("passed", [0, 1, 0, 1])
            )

    def test_misaligned_target_is_rejected(self):
        with pytest.raises(NonEqualArrayLengthError):
            NewtonLogisticRegression().fit(
                [Feature("hours", [1.0, 2.0, 3.0, 4.0])], Feature("passed", [0, 1, 0])
            )

    def test_constant_predictor_is_rejected(self):
        with pytest.raises(AllSameValuesError):
            NewtonLogisticRegression().fit(
                [Feature("hours", [2.0, 2.0, 2.0, 2.0])],
                Feature("passed", [0, 1, 0, 1]),
            )

    @pytest.mark.parametrize("labels", [[0, 1, 2, 1], [0.0, 0.5, 1.0, 1.0]])
    def test_non_binary_target_is_rejected(self, labels):
        with pytest.raises(NonBinaryLabelsError):
            NewtonLogisticRegression().fit(
                [Feature("hours", [1.0, 2.0, 3.0, 4.0])], Feature("passed", labels)
            )

    @pytest.mark.parametrize("labels", [[0, 0, 0, 0], [1, 1, 1, 1]])
    def test_single_class_target_is_rejected(self, labels):
        with pytest.raises(SingleClassError):
            NewtonLogisticRegression().fit(
                [Feature("hours", [1.0, 2.0, 3.0, 4.0])], Feature("passed", labels)
            )


class TestSingularSystemGuard:
    """The guard is wired, so these pass before the solver exists."""

    def test_singular_hessian_is_reported_as_an_mllib_error(self):
        with pytest.raises(SingularHessianError):
            NewtonLogisticRegression._solve_newton_system(
                np.zeros((2, 2)), np.array([1.0, 1.0])
            )

    def test_the_message_names_separation(self):
        with pytest.raises(SingularHessianError, match="separable"):
            NewtonLogisticRegression._solve_newton_system(np.zeros((3, 3)), np.ones(3))

    def test_a_solvable_system_is_solved(self):
        step = NewtonLogisticRegression._solve_newton_system(
            np.array([[2.0, 0.0], [0.0, 4.0]]), np.array([6.0, 8.0])
        )

        assert step == pytest.approx([3.0, 2.0])


class TestVarianceWeights:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [(0.5, 0.25), (0.0, 0.0), (1.0, 0.0), (0.2, 0.16), (0.9, 0.09)],
    )
    def test_weight_is_the_bernoulli_variance(self, probability, expected):
        weights = NewtonLogisticRegression._variance_weights(
            Probabilities(np.array([probability]))
        )

        assert weights[0] == pytest.approx(expected)

    def test_certainty_carries_no_weight_and_doubt_carries_most(self):
        weights = NewtonLogisticRegression._variance_weights(
            Probabilities(np.array([0.001, 0.5, 0.999]))
        )

        assert weights[1] == pytest.approx(0.25)
        assert weights[0] < 1e-2
        assert weights[2] < 1e-2


class TestHessian:
    def test_is_symmetric(self):
        design = design_matrix_of(OVERLAPPING_LABELS)
        weights = np.linspace(0.05, 0.25, design.n_rows)

        hessian = NewtonLogisticRegression._hessian_matrix(design, weights)

        assert hessian == pytest.approx(hessian.T)

    def test_is_positive_semi_definite(self):
        # This is what makes an undamped Newton step safe on this objective.
        design = design_matrix_of(OVERLAPPING_LABELS)
        weights = np.linspace(0.05, 0.25, design.n_rows)

        eigenvalues = np.linalg.eigvalsh(
            NewtonLogisticRegression._hessian_matrix(design, weights)
        )

        assert eigenvalues.min() > -1e-10

    def test_matches_the_definition_summed_row_by_row(self):
        # sum(w_i * outer(x_i, x_i)), which is the definition rather than a
        # rival spelling of the same matrix product. An earlier version of this
        # oracle built X.T diag(w) X, and an implementation that reached for
        # np.diag then matched it character for character -- a test comparing
        # the code to itself, which passes for no reason at all.
        design = design_matrix_of(OVERLAPPING_LABELS)
        weights = np.linspace(0.05, 0.25, design.n_rows)

        expected = sum(
            weight * np.outer(row, row)
            for weight, row in zip(weights, design.values, strict=True)
        )

        assert NewtonLogisticRegression._hessian_matrix(design, weights) == (
            pytest.approx(expected)
        )

    def test_never_materialises_the_weight_matrix(self, monkeypatch):
        # W is n x n on paper and one number per row in practice. At the
        # benchmark's 20000 rows the honest matrix is 3.2 GB to hold 20000
        # numbers, beside a design matrix of 8 MB, so building it would cost
        # the method the only thing it is for.
        def refuse(*args, **kwargs):
            raise AssertionError(
                "the n x n weight matrix must never be built; scale the rows "
                "of the design matrix instead"
            )

        monkeypatch.setattr(np, "diag", refuse)
        monkeypatch.setattr(np, "diagflat", refuse)

        design = design_matrix_of(OVERLAPPING_LABELS)
        weights = np.linspace(0.05, 0.25, design.n_rows)

        assert NewtonLogisticRegression._hessian_matrix(design, weights).shape == (
            design.n_columns,
            design.n_columns,
        )

    def test_uniform_weights_reduce_to_a_scaled_gram_matrix(self):
        design = design_matrix_of(OVERLAPPING_LABELS)
        weights = np.full(design.n_rows, 0.25)

        assert NewtonLogisticRegression._hessian_matrix(design, weights) == (
            pytest.approx(0.25 * (design.values.T @ design.values))
        )


class TestGradient:
    def test_matches_central_finite_differences(self):
        # Checked against the objective itself rather than against a restated
        # formula, which would only prove the test and the code agree.
        design = design_matrix_of(OVERLAPPING_LABELS)
        labels = OVERLAPPING_LABELS.target_feature.column
        weights = np.array([-0.7, 0.3])
        model = NewtonLogisticRegression()

        def log_likelihood(candidate):
            linear = design.values @ candidate
            return float(np.sum(labels.values * linear - np.logaddexp(0.0, linear)))

        step = 1e-6
        numerical = np.array(
            [
                (
                    log_likelihood(weights + step * basis)
                    - log_likelihood(weights - step * basis)
                )
                / (2 * step)
                for basis in np.eye(len(weights))
            ]
        )

        analytic = model._gradient(
            design, labels, model._sigmoid(design.values @ weights)
        )

        assert analytic == pytest.approx(numerical, abs=1e-6)

    def test_is_not_divided_by_the_sample_count(self):
        # Any constant scaling cancels in H^-1 g, so it is left out of both.
        # Dividing here and not in the Hessian would silently shrink the step.
        design = design_matrix_of(OVERLAPPING_LABELS)
        labels = OVERLAPPING_LABELS.target_feature.column
        model = NewtonLogisticRegression()
        probabilities = Probabilities(np.full(labels.n_samples, 0.5))

        analytic = model._gradient(design, labels, probabilities)

        assert analytic == pytest.approx(
            design.values.T @ (labels.values - probabilities.values)
        )


class TestFitting:
    def test_recovers_the_known_coefficients(self):
        model = fitted_model()

        assert model.intercept == pytest.approx(
            OVERLAPPING_LABELS.expected_intercept, abs=1e-4
        )
        assert model.coefficients["hours"] == pytest.approx(
            OVERLAPPING_LABELS.expected_weight, abs=1e-4
        )

    def test_agrees_with_gradient_ascent_to_floating_point(self):
        # Same concave objective, so the same maximum. Nothing licences a
        # disagreement beyond the tolerance each solver was asked to stop at.
        walked = LogisticRegression(
            learning_rate=0.5, max_epochs=200_000, tolerance=1e-12
        )
        walked.fit(OVERLAPPING_LABELS.input_features, OVERLAPPING_LABELS.target_feature)
        jumped = fitted_model()

        assert jumped.intercept == pytest.approx(walked.intercept, abs=1e-7)
        assert jumped.coefficients["hours"] == pytest.approx(
            walked.coefficients["hours"], abs=1e-7
        )

    def test_converges_in_single_digit_iterations(self):
        # The entire argument for the method. A correct fit that took three
        # hundred steps would have proved nothing worth proving.
        model = fitted_model()

        assert model.converged is True
        assert model.iterations_run <= 9

    def test_the_iteration_cap_is_not_what_stopped_it(self):
        model = fitted_model()

        assert model.iterations_run < model.max_iterations

    def test_fit_returns_self_for_chaining(self):
        model = NewtonLogisticRegression()

        assert (
            model.fit(
                OVERLAPPING_LABELS.input_features, OVERLAPPING_LABELS.target_feature
            )
            is model
        )

    def test_a_tighter_tolerance_does_not_need_many_more_steps(self):
        # Quadratic convergence: the digits double, so nine more of them cost
        # about one extra iteration rather than nine times as many.
        loose = fitted_model(tolerance=1e-4)
        tight = fitted_model(tolerance=1e-13)

        assert tight.iterations_run - loose.iterations_run <= 3


class TestPredictions:
    def test_probabilities_lie_in_the_unit_interval(self):
        probabilities = fitted_model().predict_probability(
            OVERLAPPING_LABELS.input_features
        )

        assert probabilities.values.min() >= 0.0
        assert probabilities.values.max() <= 1.0

    def test_predict_returns_zeros_and_ones_as_floats(self):
        labels = fitted_model().predict(OVERLAPPING_LABELS.input_features)

        assert labels.dtype == np.float64
        assert set(np.unique(labels)) <= {0.0, 1.0}

    def test_decision_boundary_matches_the_known_crossing(self):
        model = fitted_model()

        assert model.decision_boundary_at("hours") == pytest.approx(
            OVERLAPPING_LABELS_BOUNDARY, abs=1e-3
        )

    def test_odds_multiplier_matches_the_known_value(self):
        model = fitted_model()

        assert model.odds_multiplier_for("hours") == pytest.approx(
            OVERLAPPING_LABELS_ODDS_MULTIPLIER, abs=1e-3
        )


class TestSeparation:
    def test_does_not_converge_when_the_classes_do_not_overlap(self):
        model = NewtonLogisticRegression(max_iterations=20)
        model.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)

        assert model.converged is False
        assert model.iterations_run == 20

    def test_coefficients_run_away_rather_than_settling(self):
        short = NewtonLogisticRegression(max_iterations=5)
        short.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)
        long = NewtonLogisticRegression(max_iterations=20)
        long.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)

        assert abs(long.coefficients["hours"]) > 2.0 * abs(short.coefficients["hours"])

    def test_it_diverges_faster_than_gradient_ascent_does(self):
        # Quadratic convergence toward an optimum that does not exist is
        # quadratic divergence, which is the trap worth having a test for.
        jumped = NewtonLogisticRegression(max_iterations=20)
        jumped.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)
        walked = LogisticRegression(learning_rate=0.5, max_epochs=20)
        walked.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)

        assert abs(jumped.coefficients["hours"]) > abs(walked.coefficients["hours"])


class TestWithoutIntercept:
    def test_fits_a_boundary_through_the_origin(self):
        model = NewtonLogisticRegression(fit_intercept=False)
        model.fit(OVERLAPPING_LABELS.input_features, OVERLAPPING_LABELS.target_feature)

        assert model.intercept == 0.0
        assert model.converged is True
        assert "hours" in model.coefficients
