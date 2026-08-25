"""Spec for MultinomialLogisticRegression and OneVsRestClassifier.

Two specifications carry more weight than the rest.

The first is that with two classes the softmax model must agree with the binary
one to floating point. That is not a nice-to-have: softmax on two classes *is*
the sigmoid applied to the difference of the two weight vectors, so a
disagreement means the reference-class handling is wrong, and a test that only
checked "roughly similar" would let that through.

The second is that class 0's weights are zero by construction and not by
fitting. The likelihood has a flat ridge -- adding a constant to every class's
weight for a feature changes nothing observable -- so a model that learned all
K vectors would wander along it and return a different answer every run while
scoring identically. Pinning the reference is what makes the fit reproducible.
"""

import numpy as np
import pytest

from oop_ml.classification.logistic import softmax
from oop_ml.classification.logistic_regression import LogisticRegression
from oop_ml.classification.multinomial_logistic_regression import (
    MultinomialLogisticRegression,
)
from oop_ml.classification.one_vs_rest import OneVsRestClassifier
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
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
    THREE_CLASSES,
    THREE_CLASSES_ACCURACY,
)

SETTINGS = {"learning_rate": 1.0, "max_epochs": 200_000, "tolerance": 1e-8}


def fitted(**overrides) -> MultinomialLogisticRegression:
    settings = SETTINGS | overrides
    model = MultinomialLogisticRegression(**settings)
    model.fit(THREE_CLASSES.input_features, THREE_CLASSES.target_feature)

    return model


def two_class_features() -> tuple[list[Feature], Feature]:
    """The binary fixture, relabelled as a two-class multi-class problem."""
    return OVERLAPPING_LABELS.input_features, OVERLAPPING_LABELS.target_feature


def design_matrix_of(fixture) -> np.ndarray:
    """``X`` with its ones column, built the way the model builds it."""
    columns = [feature.values for feature in fixture.input_features]

    return np.column_stack([np.ones(len(columns[0]))] + columns)


def log_likelihood(design_matrix, classes, learned_weights) -> float:
    """``sum_i log p_{y_i}``, computed independently of anything under test."""
    scores = np.column_stack(
        [np.zeros(len(classes)), design_matrix @ np.asarray(learned_weights).T]
    )
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    return float(log_probabilities[np.arange(len(classes)), classes].sum())


class TestScores:
    def test_shape_is_rows_by_all_classes(self):
        design = design_matrix_of(THREE_CLASSES)
        weights = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])

        scores = MultinomialLogisticRegression()._scores(design, weights)

        assert scores.shape == (36, 3)

    def test_the_reference_class_column_is_zero(self):
        # Class 0's score is held at zero; that is the whole of what makes the
        # remaining weights identifiable.
        design = design_matrix_of(THREE_CLASSES)
        weights = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])

        scores = MultinomialLogisticRegression()._scores(design, weights)

        assert scores[:, 0] == pytest.approx(np.zeros(36))

    def test_the_learned_columns_are_the_linear_predictors(self):
        design = design_matrix_of(THREE_CLASSES)
        weights = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])

        scores = MultinomialLogisticRegression()._scores(design, weights)

        assert scores[:, 1] == pytest.approx(design @ weights[0])
        assert scores[:, 2] == pytest.approx(design @ weights[1])


class TestIndicatorMatrix:
    def test_one_hots_the_target(self):
        indicator = MultinomialLogisticRegression._indicator_matrix(
            THREE_CLASSES.target_feature.column, 3
        )

        assert indicator.shape == (36, 3)
        assert indicator.sum(axis=1) == pytest.approx(np.ones(36))

    def test_the_one_is_in_the_class_column(self):
        indicator = MultinomialLogisticRegression._indicator_matrix(
            THREE_CLASSES.target_feature.column, 3
        )

        assert np.argmax(indicator, axis=1).tolist() == THREE_CLASSES.class_values


class TestGradient:
    def weights(self) -> np.ndarray:
        return np.array([[-0.3, 0.6, 0.2], [0.4, -0.5, 0.7]])

    def test_shape_is_learned_classes_by_parameters(self):
        # One row per class *beyond* the reference, not one per class.
        design = design_matrix_of(THREE_CLASSES)
        model = MultinomialLogisticRegression()
        probabilities = softmax(model._scores(design, self.weights()))
        indicator = model._indicator_matrix(THREE_CLASSES.target_feature.column, 3)

        assert model._gradient(design, indicator, probabilities).shape == (2, 3)

    def test_matches_central_finite_differences(self):
        # Checked against the objective rather than against a restated formula,
        # which would only prove the test and the code agree with each other.
        design = design_matrix_of(THREE_CLASSES)
        classes = np.array(THREE_CLASSES.class_values)
        model = MultinomialLogisticRegression()
        weights = self.weights()
        probabilities = softmax(model._scores(design, weights))
        indicator = model._indicator_matrix(THREE_CLASSES.target_feature.column, 3)

        step = 1e-6
        numerical = np.zeros_like(weights)
        for row in range(weights.shape[0]):
            for column in range(weights.shape[1]):
                up, down = weights.copy(), weights.copy()
                up[row, column] += step
                down[row, column] -= step
                numerical[row, column] = (
                    log_likelihood(design, classes, up)
                    - log_likelihood(design, classes, down)
                ) / (2 * step)

        analytic = model._gradient(design, indicator, probabilities)

        assert analytic == pytest.approx(numerical / len(classes), abs=1e-8)

    def test_is_averaged_over_the_rows(self):
        # Without the division the same learning_rate that converges in the
        # binary model diverges here, because the steps are n times too large.
        design = design_matrix_of(THREE_CLASSES)
        model = MultinomialLogisticRegression()
        probabilities = softmax(model._scores(design, self.weights()))
        indicator = model._indicator_matrix(THREE_CLASSES.target_feature.column, 3)

        analytic = model._gradient(design, indicator, probabilities)
        unaveraged = ((indicator - probabilities).T @ design)[1:]

        assert analytic == pytest.approx(unaveraged / 36)

    def test_the_reference_class_row_is_not_returned(self):
        # The full gradient has K rows; only the learned classes are wanted.
        design = design_matrix_of(THREE_CLASSES)
        model = MultinomialLogisticRegression()
        probabilities = softmax(model._scores(design, self.weights()))
        indicator = model._indicator_matrix(THREE_CLASSES.target_feature.column, 3)

        full = ((indicator - probabilities).T @ design) / 36

        assert model._gradient(design, indicator, probabilities) == pytest.approx(
            full[1:]
        )


class TestConstruction:
    def test_defaults(self):
        model = MultinomialLogisticRegression()

        assert model.learning_rate == pytest.approx(0.1)
        assert model.max_epochs == 10_000
        assert model.fit_intercept is True

    @pytest.mark.parametrize("learning_rate", [0.0, -0.5])
    def test_non_positive_learning_rate_is_rejected(self, learning_rate):
        with pytest.raises(ValueError):
            MultinomialLogisticRegression(learning_rate=learning_rate)

    @pytest.mark.parametrize("max_epochs", [0, -1])
    def test_non_positive_epoch_cap_is_rejected(self, max_epochs):
        with pytest.raises(ValueError):
            MultinomialLogisticRegression(max_epochs=max_epochs)


class TestNotFitted:
    @pytest.mark.parametrize(
        "attribute", ["n_classes", "epochs_run", "converged", "intercepts"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(MultinomialLogisticRegression(), attribute)

    def test_coefficients_raise_before_fit(self):
        with pytest.raises(NotFittedError):
            MultinomialLogisticRegression().coefficients_for(1)

    def test_predict_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            MultinomialLogisticRegression().predict(THREE_CLASSES.input_features)


class TestValidation:
    def test_no_features_is_rejected(self):
        with pytest.raises(EmptyValuesError):
            MultinomialLogisticRegression().fit([], THREE_CLASSES.target_feature)

    def test_duplicate_feature_names_are_rejected(self):
        repeated = Feature("first", [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(NonUniqueFeaturesError):
            MultinomialLogisticRegression().fit(
                [repeated, repeated], Feature("outcome", [0, 1, 2, 1])
            )

    def test_misaligned_target_is_rejected(self):
        with pytest.raises(NonEqualArrayLengthError):
            MultinomialLogisticRegression().fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])], Feature("outcome", [0, 1, 2])
            )

    def test_constant_predictor_is_rejected(self):
        with pytest.raises(AllSameValuesError):
            MultinomialLogisticRegression().fit(
                [Feature("first", [2.0, 2.0, 2.0, 2.0])],
                Feature("outcome", [0, 1, 2, 1]),
            )

    @pytest.mark.parametrize("classes", [[0, 1, -1, 2], [0.0, 1.5, 2.0, 1.0]])
    def test_non_class_targets_are_rejected(self, classes):
        with pytest.raises(NonBinaryLabelsError):
            MultinomialLogisticRegression().fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])], Feature("outcome", classes)
            )

    def test_a_single_class_is_rejected(self):
        with pytest.raises(SingleClassError):
            MultinomialLogisticRegression().fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])],
                Feature("outcome", [1, 1, 1, 1]),
            )

    def test_a_gap_in_the_classes_is_rejected(self):
        with pytest.raises(SingleClassError):
            MultinomialLogisticRegression().fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])],
                Feature("outcome", [0, 0, 2, 2]),
            )


class TestReferenceClass:
    def test_class_zero_has_no_weights(self):
        # Zero by construction, not by fitting: it is what makes the answer
        # unique when the likelihood has a flat ridge through it.
        model = fitted()

        assert model.coefficients_for(0)["first"] == 0.0
        assert model.coefficients_for(0)["second"] == 0.0
        assert model.intercepts[0] == 0.0

    def test_an_unknown_class_is_rejected(self):
        with pytest.raises(InvalidValuesError):
            fitted().coefficients_for(3)

    def test_the_fit_is_reproducible(self):
        # Two runs must land on the same point, which is only true because the
        # reference class pins the ridge down.
        first, second = fitted(), fitted()

        assert first.coefficients_for(2)["first"] == pytest.approx(
            second.coefficients_for(2)["first"], abs=1e-12
        )


class TestFitting:
    def test_recovers_the_known_weights(self):
        model = fitted()
        expected = THREE_CLASSES.expected_weights

        for class_index, (intercept, first, second) in enumerate(expected, start=1):
            assert model.intercepts[class_index] == pytest.approx(intercept, abs=1e-3)
            assert model.coefficients_for(class_index)["first"] == pytest.approx(
                first, abs=1e-3
            )
            assert model.coefficients_for(class_index)["second"] == pytest.approx(
                second, abs=1e-3
            )

    def test_reports_the_class_count(self):
        assert fitted().n_classes == 3

    def test_converges(self):
        model = fitted()

        assert model.converged is True
        assert 0 < model.epochs_run < model.max_epochs

    def test_exhausting_the_cap_is_reported(self):
        model = MultinomialLogisticRegression(learning_rate=1.0, max_epochs=5)
        model.fit(THREE_CLASSES.input_features, THREE_CLASSES.target_feature)

        assert model.converged is False
        assert model.epochs_run == 5

    def test_fit_returns_self_for_chaining(self):
        model = MultinomialLogisticRegression(**SETTINGS)

        assert (
            model.fit(THREE_CLASSES.input_features, THREE_CLASSES.target_feature)
            is model
        )


class TestProbabilities:
    def test_every_row_sums_to_one(self):
        # The property one-vs-rest cannot offer, and the reason softmax exists.
        probabilities = fitted().predict_probabilities(THREE_CLASSES.input_features)

        assert probabilities.sum(axis=1) == pytest.approx(np.ones(36))

    def test_the_shape_is_rows_by_classes(self):
        probabilities = fitted().predict_probabilities(THREE_CLASSES.input_features)

        assert probabilities.shape == (36, 3)

    def test_probabilities_are_in_the_unit_interval(self):
        probabilities = fitted().predict_probabilities(THREE_CLASSES.input_features)

        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0

    def test_predict_is_the_argmax_of_the_probabilities(self):
        model = fitted()
        probabilities = model.predict_probabilities(THREE_CLASSES.input_features)

        assert model.predict(THREE_CLASSES.input_features) == pytest.approx(
            np.argmax(probabilities, axis=1).astype(float)
        )

    def test_feature_order_does_not_matter(self):
        model = fitted()
        forwards = model.predict_probabilities(THREE_CLASSES.input_features)
        backwards = model.predict_probabilities(
            list(reversed(THREE_CLASSES.input_features))
        )

        assert np.allclose(forwards, backwards)

    def test_unknown_features_are_rejected(self):
        with pytest.raises(InvalidValuesError):
            fitted().predict([Feature("nonsense", [1.0] * 36)])


class TestAgreementWithTheBinaryModel:
    def test_two_classes_reduce_to_the_sigmoid(self):
        # softmax over two classes IS the sigmoid on the difference of the two
        # weight vectors, so the reference-class model must land exactly where
        # the binary one does.
        features, target = two_class_features()
        multi = MultinomialLogisticRegression(
            learning_rate=0.5, max_epochs=200_000, tolerance=1e-11
        )
        multi.fit(features, target)
        binary = LogisticRegression(
            learning_rate=0.5, max_epochs=200_000, tolerance=1e-11
        )
        binary.fit(features, target)

        assert multi.intercepts[1] == pytest.approx(binary.intercept, abs=1e-6)
        assert multi.coefficients_for(1)["hours"] == pytest.approx(
            binary.coefficients["hours"], abs=1e-6
        )

    def test_two_class_probabilities_match(self):
        features, target = two_class_features()
        multi = MultinomialLogisticRegression(
            learning_rate=0.5, max_epochs=200_000, tolerance=1e-11
        )
        multi.fit(features, target)
        binary = LogisticRegression(
            learning_rate=0.5, max_epochs=200_000, tolerance=1e-11
        )
        binary.fit(features, target)

        assert multi.predict_probabilities(features)[:, 1] == pytest.approx(
            binary.predict_probability(features), abs=1e-6
        )


class TestEvaluation:
    def test_evaluate_returns_a_multi_class_evaluation(self):
        model = fitted()
        result = model.evaluate(
            THREE_CLASSES.input_features, THREE_CLASSES.target_feature
        )

        assert result.n_classes == 3
        assert result.n_samples == 36
        assert result.accuracy == pytest.approx(THREE_CLASSES_ACCURACY, abs=1e-4)

    def test_score_is_the_accuracy(self):
        model = fitted()

        assert model.score(
            THREE_CLASSES.input_features, THREE_CLASSES.target_feature
        ) == pytest.approx(THREE_CLASSES_ACCURACY, abs=1e-4)

    def test_the_table_is_the_fitted_width_not_the_truths(self):
        # A held-out fold missing a class must still score against a K x K
        # table, or the remaining classes get silently renumbered.
        model = fitted()
        keep = [
            index
            for index, value in enumerate(THREE_CLASSES.class_values)
            if value in (0, 1)
        ]
        subset = [
            Feature(feature.name, feature.values[keep])
            for feature in THREE_CLASSES.input_features
        ]
        target = Feature("outcome", THREE_CLASSES.target_feature.values[keep])

        assert model.evaluate(subset, target).n_classes == 3


class TestOneVsRest:
    def build(self) -> OneVsRestClassifier:
        model = OneVsRestClassifier(
            binary_model=LogisticRegression(
                learning_rate=0.5, max_epochs=200_000, tolerance=1e-10
            )
        )
        model.fit(THREE_CLASSES.input_features, THREE_CLASSES.target_feature)

        return model

    def test_fits_one_model_per_class(self):
        model = self.build()

        models = [model.model_for(index) for index in range(3)]

        assert model.n_classes == 3
        assert all(
            first is not second
            for position, first in enumerate(models)
            for second in models[position + 1 :]
        )

    def test_each_model_learned_something_different(self):
        model = self.build()
        weights = [model.model_for(index).coefficients["first"] for index in range(3)]

        assert len(set(weights)) == 3

    def test_the_prototype_is_never_fitted(self):
        prototype = LogisticRegression(learning_rate=0.5, max_epochs=200_000)
        wrapper = OneVsRestClassifier(binary_model=prototype)
        wrapper.fit(THREE_CLASSES.input_features, THREE_CLASSES.target_feature)

        with pytest.raises(NotFittedError):
            _ = prototype.coefficients

    def test_probabilities_have_one_column_per_class(self):
        probabilities = self.build().predict_probabilities(THREE_CLASSES.input_features)

        assert probabilities.shape == (36, 3)

    def test_probabilities_do_not_sum_to_one(self):
        # The honest cost of the approach, asserted rather than hidden. The K
        # models were never asked to agree, so their answers do not add up.
        probabilities = self.build().predict_probabilities(THREE_CLASSES.input_features)
        totals = probabilities.sum(axis=1)

        assert not np.allclose(totals, 1.0)

    def test_predict_is_the_argmax_across_the_models(self):
        model = self.build()
        probabilities = model.predict_probabilities(THREE_CLASSES.input_features)

        assert model.predict(THREE_CLASSES.input_features) == pytest.approx(
            np.argmax(probabilities, axis=1).astype(float)
        )

    def test_it_can_be_evaluated_like_any_other_multi_class_model(self):
        result = self.build().evaluate(
            THREE_CLASSES.input_features, THREE_CLASSES.target_feature
        )

        assert result.n_classes == 3
        assert 0.0 <= result.accuracy <= 1.0

    def test_a_single_class_target_is_rejected(self):
        model = OneVsRestClassifier(binary_model=LogisticRegression())

        with pytest.raises(SingleClassError):
            model.fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0])],
                Feature("outcome", [1, 1, 1, 1]),
            )
