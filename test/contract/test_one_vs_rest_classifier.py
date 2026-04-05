"""The contract every backend's OneVsRestClassifier keeps.

The binary model it wraps is taken from the same backend, so a scikit
one-vs-rest holds scikit logistic regressions and a numpy one holds numpy
ones. What the contract fixes is the shape of the thing: K fitted models,
one per class, each addressable and each carrying its own coefficients, and
a score matrix that is honest about not being a distribution.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.data.probabilities import ClassScores, ProbabilityMatrix
from oop_ml.core.exceptions import NotFittedError, SingleClassError

from .harness import provided

#: The multinomial contract's fixture: three overlapping runs, centred.
_POSITIONS = np.array(
    [-5.0, -4.0, -3.0, -2.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]
)
_CLASSES = np.array([0.0] * 5 + [1.0] * 5 + [2.0] * 5)
FEATURES = [Feature("position", _POSITIONS)]
TARGET = Feature("band", _CLASSES)

#: The outer classes' interior rows. The middle class against the rest is
#: not a linearly separable question, so its own rows are left to the fit.
OUTER_ROWS = [0, 1, 2, 12, 13, 14]
OUTER_CLASSES = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])

MAX_EPOCHS = 100_000

#: A prototype whose every compared field sits away from its default, so a
#: member built fresh per class rather than copied from this one differs in
#: all four. ``learning_rate`` is deliberately left alone, since a backend
#: whose engine chooses its own step refuses a configured one.
PROTOTYPE_CONFIGURATION: dict[str, object] = {
    "fit_intercept": False,
    "threshold": 0.4,
    "tolerance": 1e-06,
    "max_epochs": MAX_EPOCHS,
}


def _wrapped(backend: ModuleType):
    OneVsRestClassifier = provided(backend, "OneVsRestClassifier")
    LogisticRegression = provided(backend, "LogisticRegression")

    return OneVsRestClassifier(binary_model=LogisticRegression(max_epochs=MAX_EPOCHS))


def test_it_is_constructed_by_the_same_keyword(backend: ModuleType) -> None:
    OneVsRestClassifier = provided(backend, "OneVsRestClassifier")
    LogisticRegression = provided(backend, "LogisticRegression")
    model = OneVsRestClassifier(binary_model=LogisticRegression(threshold=0.4))

    assert model.binary_model.threshold == pytest.approx(0.4)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    model = _wrapped(backend)

    assert model.fit(FEATURES, TARGET) is model
    assert model.n_classes == 3


def test_it_predicts_one_class_per_row(backend: ModuleType) -> None:
    model = _wrapped(backend).fit(FEATURES, TARGET)

    predictions = np.asarray(model.predict(FEATURES))

    assert predictions.shape == (len(_CLASSES),)
    assert np.array_equal(predictions[OUTER_ROWS], OUTER_CLASSES)


def test_its_scores_are_bounded_and_deliberately_not_a_distribution(
    backend: ModuleType,
) -> None:
    model = _wrapped(backend).fit(FEATURES, TARGET)

    scores = model.predict_probabilities(FEATURES)

    assert isinstance(scores, ClassScores)
    assert not isinstance(scores, ProbabilityMatrix)
    assert scores.shape == (len(_CLASSES), 3)
    values = np.asarray(scores)
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_it_holds_one_fitted_binary_model_per_class(backend: ModuleType) -> None:
    model = _wrapped(backend).fit(FEATURES, TARGET)

    for class_index in range(3):
        fitted = model.model_for(class_index)
        assert fitted.is_fitted
        assert fitted is not model.binary_model
        assert isinstance(fitted.coefficients["position"], float)

    # The lowest class is left behind as position grows, the highest is not.
    assert model.model_for(0).coefficients["position"] < 0.0
    assert model.model_for(2).coefficients["position"] > 0.0


def test_each_member_is_configured_as_the_prototype(backend: ModuleType) -> None:
    """A member carries the configuration, not merely the type.

    ``binary_model`` is documented as the classifier to clone once per class,
    and without this a backend that built a default model per class, or
    overwrote a field on each copy, answers different boundaries and stays
    green. Only the configuration is compared, never the fitted intercepts,
    which the two backends reach by different solvers.
    """
    LogisticRegression = provided(backend, "LogisticRegression")
    OneVsRestClassifier = provided(backend, "OneVsRestClassifier")
    prototype = LogisticRegression(**PROTOTYPE_CONFIGURATION)

    model = OneVsRestClassifier(binary_model=prototype).fit(FEATURES, TARGET)

    for class_index in range(3):
        member = model.model_for(class_index)

        assert member.fit_intercept == prototype.fit_intercept
        assert member.threshold == pytest.approx(prototype.threshold)
        assert member.tolerance == pytest.approx(prototype.tolerance)
        assert member.max_epochs == prototype.max_epochs


def test_the_prototype_is_never_fitted(backend: ModuleType) -> None:
    model = _wrapped(backend).fit(FEATURES, TARGET)

    assert model.binary_model.is_fitted is False


def test_it_refuses_a_target_with_a_gap_in_its_classes(backend: ModuleType) -> None:
    gapped = Feature("band", np.where(_CLASSES == 1.0, 2.0, _CLASSES))

    with pytest.raises(SingleClassError):
        _wrapped(backend).fit(FEATURES, gapped)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    model = _wrapped(backend).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) >= 0.6


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    with pytest.raises(NotFittedError):
        _wrapped(backend).predict(FEATURES)
