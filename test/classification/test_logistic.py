"""Spec for LogisticRegression -- red until _sigmoid, _gradient and _solve land.

The validation and the wiring are already in place, so the guard tests pass now
and hold the contract while the solver is written. Everything that needs a
fitted model is a stub away from working.

Two specifications here are worth more than the rest. The first is that the
gradient is ``X.T (y - p)``, checked against central finite differences rather
than against a formula restated in the test, since restating it would only prove
the test and the code agree with each other. The second is separation: on data
where the classes do not overlap there is no finite answer, and the model has to
say so through ``converged`` rather than hand back whatever it had reached.
"""

import numpy as np
import pytest

from oop_ml.classification.logistic_regression import LogisticRegression
from oop_ml.data.feature import Feature
from oop_ml.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    SingleClassError,
)
from test.fixtures import (
    OVERLAPPING_LABELS,
    OVERLAPPING_LABELS_BOUNDARY,
    OVERLAPPING_LABELS_ODDS_MULTIPLIER,
    SEPARABLE_LABELS,
)


def fitted_model(**overrides) -> LogisticRegression:
    settings = {"learning_rate": 0.5, "max_epochs": 20_000, "tolerance": 1e-10}
    settings.update(overrides)
    model = LogisticRegression(**settings)
    model.fit(OVERLAPPING_LABELS.input_features, OVERLAPPING_LABELS.target_feature)

    return model


class TestConstruction:
    def test_defaults(self):
        model = LogisticRegression()

        assert model.learning_rate == pytest.approx(0.1)
        assert model.threshold == pytest.approx(0.5)
        assert model.fit_intercept is True

    @pytest.mark.parametrize("learning_rate", [0.0, -0.1])
    def test_non_positive_learning_rate_is_rejected(self, learning_rate):
        with pytest.raises(ValueError):
            LogisticRegression(learning_rate=learning_rate)

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.2, 1.5])
    def test_threshold_outside_the_open_unit_interval_is_rejected(self, threshold):
        with pytest.raises(ValueError):
            LogisticRegression(threshold=threshold)

    @pytest.mark.parametrize("max_epochs", [0, -5])
    def test_non_positive_epoch_cap_is_rejected(self, max_epochs):
        with pytest.raises(ValueError):
            LogisticRegression(max_epochs=max_epochs)


class TestFitValidation:
    def test_no_features_raises(self):
        with pytest.raises(EmptyValuesError):
            LogisticRegression().fit([], OVERLAPPING_LABELS.target_feature)

    def test_duplicate_feature_names_raise(self):
        with pytest.raises(NonUniqueFeaturesError):
            LogisticRegression().fit(
                [Feature("hours", [1, 2, 3]), Feature("hours", [4, 5, 6])],
                Feature("passed", [0, 1, 0]),
            )

    def test_target_of_a_different_length_raises(self):
        with pytest.raises(NonEqualArrayLengthError):
            LogisticRegression().fit(
                [Feature("hours", [1, 2, 3])], Feature("passed", [0, 1])
            )

    def test_a_constant_predictor_raises(self):
        with pytest.raises(AllSameValuesError):
            LogisticRegression().fit(
                [Feature("hours", [2, 2, 2, 2])], Feature("passed", [0, 1, 0, 1])
            )

    @pytest.mark.parametrize(
        "labels",
        [[0, 1, 2, 1], [0, 0.5, 1, 1], [-1, 1, 0, 1]],
        ids=["three classes", "a probability", "negative label"],
    )
    def test_a_non_binary_target_raises(self, labels):
        with pytest.raises(NonBinaryLabelsError):
            LogisticRegression().fit(
                [Feature("hours", [1, 2, 3, 4])], Feature("passed", labels)
            )

    @pytest.mark.parametrize("labels", [[0, 0, 0, 0], [1, 1, 1, 1]])
    def test_a_single_class_target_raises(self, labels):
        # Valid labels, nothing to discriminate. A boundary fitted here would be
        # arbitrary rather than merely poor.
        with pytest.raises(SingleClassError):
            LogisticRegression().fit(
                [Feature("hours", [1, 2, 3, 4])], Feature("passed", labels)
            )


class TestBeforeFitting:
    @pytest.mark.parametrize(
        "attribute", ["coefficients", "intercept", "epochs_run", "converged"]
    )
    def test_reading_a_fitted_attribute_raises(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(LogisticRegression(), attribute)

    @pytest.mark.parametrize("method", ["predict", "predict_probability"])
    def test_predicting_raises(self, method):
        with pytest.raises(NotFittedError):
            getattr(LogisticRegression(), method)(OVERLAPPING_LABELS.input_features)


class TestSigmoid:
    @pytest.mark.parametrize(
        ("linear_predictor", "expected"),
        [(0.0, 0.5), (2.0, 0.880797), (-2.0, 0.119203), (1.0, 0.731059)],
    )
    def test_maps_log_odds_onto_a_probability(self, linear_predictor, expected):
        result = LogisticRegression._sigmoid(np.array([linear_predictor]))

        assert float(result[0]) == pytest.approx(expected, abs=1e-6)

    def test_is_symmetric_about_zero(self):
        values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

        forward = LogisticRegression._sigmoid(values)
        backward = LogisticRegression._sigmoid(-values)

        np.testing.assert_allclose(forward + backward, np.ones_like(values), atol=1e-12)

    @pytest.mark.parametrize("extreme", [800.0, -800.0, 1e6, -1e6])
    def test_does_not_overflow_at_the_extremes(self, extreme):
        # The naive 1 / (1 + exp(-z)) overflows here and returns nan exactly
        # when the model is most confident, which is when separated data drives
        # the linear predictor a long way out.
        result = float(LogisticRegression._sigmoid(np.array([extreme]))[0])

        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0


class TestGradient:
    def test_matches_central_finite_differences(self):
        model = LogisticRegression()
        design = np.column_stack(
            [np.ones(8), np.array(OVERLAPPING_LABELS.input_features[0].values)]
        )
        labels = np.array(OVERLAPPING_LABELS.target_feature.values)

        def log_likelihood(weights):
            probability = model._sigmoid(design @ weights)
            return float(
                np.sum(
                    labels * np.log(probability)
                    + (1 - labels) * np.log(1 - probability)
                )
            )

        for weights in [np.zeros(2), np.array([-2.0, 1.5]), np.array([1.0, -0.3])]:
            analytic = model._gradient(design, labels, weights) * len(labels)
            numeric = np.zeros(2)
            for index in range(2):
                up, down = weights.copy(), weights.copy()
                up[index] += 1e-6
                down[index] -= 1e-6
                numeric[index] = (log_likelihood(up) - log_likelihood(down)) / 2e-6

            np.testing.assert_allclose(analytic, numeric, atol=1e-5)

    def test_at_zero_weights_reduces_to_the_deviation_from_a_half(self):
        model = LogisticRegression()
        design = np.column_stack(
            [np.ones(8), np.array(OVERLAPPING_LABELS.input_features[0].values)]
        )
        labels = np.array(OVERLAPPING_LABELS.target_feature.values)

        gradient = model._gradient(design, labels, np.zeros(2)) * len(labels)

        np.testing.assert_allclose(gradient, design.T @ (labels - 0.5), atol=1e-10)


class TestFittedModel:
    def test_recovers_the_known_coefficients(self):
        model = fitted_model()

        assert model.intercept == pytest.approx(
            OVERLAPPING_LABELS.expected_intercept, abs=1e-3
        )
        assert model.coefficients["hours"] == pytest.approx(
            OVERLAPPING_LABELS.expected_weight, abs=1e-3
        )

    def test_reports_that_it_converged(self):
        model = fitted_model()

        assert model.converged is True
        assert 0 < model.epochs_run <= 20_000

    def test_probabilities_are_in_the_unit_interval_and_increase_with_hours(self):
        model = fitted_model()

        probabilities = model.predict_probability(OVERLAPPING_LABELS.input_features)

        assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
        # hours is sorted ascending in the fixture, and the slope is positive.
        assert np.all(np.diff(probabilities) > 0.0)

    def test_labels_are_the_thresholded_probabilities(self):
        model = fitted_model()

        probabilities = model.predict_probability(OVERLAPPING_LABELS.input_features)
        labels = model.predict(OVERLAPPING_LABELS.input_features)

        np.testing.assert_allclose(labels, (probabilities >= 0.5).astype(float))

    @pytest.mark.parametrize(
        ("threshold", "expected_positive_count"), [(0.1, 8), (0.5, 4), (0.9, 0)]
    )
    def test_the_threshold_moves_how_many_rows_are_called_positive(
        self, threshold, expected_positive_count
    ):
        model = fitted_model(threshold=threshold)

        labels = model.predict(OVERLAPPING_LABELS.input_features)

        assert int(np.sum(labels)) == expected_positive_count

    def test_reads_the_boundary_off_the_coefficients(self):
        model = fitted_model()

        assert model.decision_boundary_at("hours") == pytest.approx(
            OVERLAPPING_LABELS_BOUNDARY, abs=1e-3
        )

    def test_reads_the_odds_multiplier_off_the_coefficient(self):
        model = fitted_model()

        assert model.odds_multiplier_for("hours") == pytest.approx(
            OVERLAPPING_LABELS_ODDS_MULTIPLIER, abs=1e-3
        )

    def test_scores_six_of_the_eight_rows(self):
        model = fitted_model()

        evaluation = model.evaluate(
            OVERLAPPING_LABELS.input_features, OVERLAPPING_LABELS.target_feature
        )

        assert evaluation.confusion_matrix.n_samples == 8
        assert evaluation.accuracy == pytest.approx(0.75)


class TestSeparation:
    def test_a_separable_target_never_converges(self):
        # No finite maximum likelihood estimate exists, so the walk cannot
        # settle. Reporting converged = True here would be a lie, and the
        # coefficients it stopped at are not an answer.
        model = LogisticRegression(learning_rate=0.5, max_epochs=500, tolerance=1e-10)
        model.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)

        assert model.converged is False
        assert model.epochs_run == 500

    def test_the_coefficients_keep_growing_with_more_epochs(self):
        shorter = LogisticRegression(learning_rate=0.5, max_epochs=200, tolerance=1e-12)
        shorter.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)

        longer = LogisticRegression(
            learning_rate=0.5, max_epochs=2_000, tolerance=1e-12
        )
        longer.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)

        assert abs(longer.coefficients["hours"]) > abs(shorter.coefficients["hours"])

    def test_it_still_classifies_the_training_rows_perfectly(self):
        # The fit is meaningless as an estimate and still separates the data,
        # which is exactly why the training score cannot be the thing you check.
        model = LogisticRegression(learning_rate=0.5, max_epochs=500)
        model.fit(SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature)

        assert model.score(
            SEPARABLE_LABELS.input_features, SEPARABLE_LABELS.target_feature
        ) == pytest.approx(1.0)


class TestPredictContract:
    def test_features_may_arrive_in_any_order(self):
        model = LogisticRegression(learning_rate=0.5, max_epochs=5_000)
        first = Feature("hours", [1.0, 2.0, 3.0, 4.0])
        second = Feature("slept", [8.0, 5.0, 7.0, 6.0])
        model.fit([first, second], Feature("passed", [0, 0, 1, 1]))

        forward = model.predict_probability([first, second])
        reversed_order = model.predict_probability([second, first])

        np.testing.assert_allclose(forward, reversed_order)

    def test_a_missing_feature_raises(self):
        model = fitted_model()

        with pytest.raises(InvalidValuesError):
            model.predict([Feature("something_else", [1, 2, 3, 4, 5, 6, 7, 8])])
