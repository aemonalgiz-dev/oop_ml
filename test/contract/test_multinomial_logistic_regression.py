"""The contract every backend's MultinomialLogisticRegression keeps.

Three classes strung along one feature, each overlapping its neighbour by
two rows, so the likelihood has a finite maximum and the fit has something
to settle on. The feature is centred on zero, because a fixed-step ascent
on an uncentred column crawls along the intercept's direction and the
contract should not spend seconds waiting for it. A softmax fit cannot be
written down by hand, so the contract asserts what the definition fixes:
class 0 is the reference and carries zero weights, the higher classes lean
further along the feature, every probability row is a distribution, and the
rows deep inside each class are called correctly.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.data.probabilities import ProbabilityMatrix
from oop_ml.core.exceptions import NotFittedError, SingleClassError

from .harness import provided

#: Class 0 on -5..-1, class 1 on -2..2, class 2 on 1..5: each pair overlaps.
_POSITIONS = np.array(
    [-5.0, -4.0, -3.0, -2.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]
)
_CLASSES = np.array([0.0] * 5 + [1.0] * 5 + [2.0] * 5)
FEATURES = [Feature("position", _POSITIONS)]
TARGET = Feature("band", _CLASSES)

#: Rows the classes do not overlap on, and what they are.
INTERIOR_ROWS = [0, 1, 2, 7, 12, 13, 14]
INTERIOR_CLASSES = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])

MAX_EPOCHS = 100_000

#: The maximum with every intercept held at zero, from an independent BFGS
#: minimisation of the softmax log loss in scipy: 0.711728 and 1.423456.
#: The fixture is symmetric under reflection with classes 0 and 2 swapped,
#: which is why the second is exactly twice the first, and that ratio also
#: holds with the intercepts free, at 1.583 and 3.166. It is the value that
#: tells the two fits apart, not the ratio.
SLOPES_THROUGH_THE_ORIGIN = (0.0, 0.7117, 1.4235)


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    model = MultinomialLogisticRegression(
        max_epochs=50, tolerance=1e-6, fit_intercept=False
    )

    assert model.max_epochs == 50
    assert model.tolerance == pytest.approx(1e-6)
    assert model.fit_intercept is False


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    model = MultinomialLogisticRegression(max_epochs=MAX_EPOCHS)

    assert model.fit(FEATURES, TARGET) is model
    assert model.n_classes == 3


def test_it_predicts_one_class_per_row(backend: ModuleType) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    model = MultinomialLogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    predictions = np.asarray(model.predict(FEATURES))

    assert predictions.shape == (len(_CLASSES),)
    assert np.array_equal(predictions[INTERIOR_ROWS], INTERIOR_CLASSES)


def test_its_probability_rows_are_distributions(backend: ModuleType) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    model = MultinomialLogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    probabilities = model.predict_probabilities(FEATURES)

    assert isinstance(probabilities, ProbabilityMatrix)
    assert probabilities.shape == (len(_CLASSES), 3)
    assert np.allclose(np.asarray(probabilities).sum(axis=1), 1.0)


def test_class_zero_is_the_reference_and_the_rest_lean_along_the_feature(
    backend: ModuleType,
) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    model = MultinomialLogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    assert model.coefficients_for(0)["position"] == 0.0
    assert model.intercepts[0] == 0.0
    assert model.coefficients_for(2)["position"] > model.coefficients_for(1)["position"]
    assert model.coefficients_for(1)["position"] > 0.0
    assert model.intercepts.shape == (3,)


def test_it_reports_that_it_settled(backend: ModuleType) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    model = MultinomialLogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    assert model.converged is True
    assert 1 <= model.epochs_run <= MAX_EPOCHS


def test_it_refuses_a_target_with_a_gap_in_its_classes(backend: ModuleType) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    gapped = Feature("band", np.where(_CLASSES == 1.0, 2.0, _CLASSES))

    with pytest.raises(SingleClassError):
        MultinomialLogisticRegression().fit(FEATURES, gapped)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    model = MultinomialLogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) >= 0.7


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")

    with pytest.raises(NotFittedError):
        MultinomialLogisticRegression().predict(FEATURES)


def test_without_intercepts_every_class_scores_through_the_origin(
    backend: ModuleType,
) -> None:
    MultinomialLogisticRegression = provided(backend, "MultinomialLogisticRegression")
    model = MultinomialLogisticRegression(
        fit_intercept=False, max_epochs=MAX_EPOCHS
    ).fit(FEATURES, TARGET)

    assert np.array_equal(model.intercepts, np.zeros(3))
    for class_index, slope in enumerate(SLOPES_THROUGH_THE_ORIGIN):
        assert model.coefficients_for(class_index)["position"] == pytest.approx(
            slope, abs=1e-3
        )
