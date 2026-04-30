"""The contract every backend's GaussianNaiveBayes keeps.

The first model here added to both backends at once rather than wrapped after
the fact, and the first classifier that models what each class looks like
instead of drawing a boundary between them.

It is also the one model whose two backends can be held to a stricter standard
than the rest. Elsewhere a contract deliberately asserts the encapsulation and
not the arithmetic, because two solvers reach the same normal equations by
different routes and would disagree in the last bits. There is no solver here:
a prior is a count, a mean is a sum over a count, and a variance is a sum of
squares. Both backends compute exactly those three, so their summaries agree
bit for bit, and this spec says so.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

#: Two clearly separated clumps with a third overlapping them, so the model has
#: something to get right and something to be uncertain about.
_FIRST = np.array([1.0, 1.4, 0.8, 1.2, 5.0, 5.4, 4.8, 5.2, 3.0, 3.2, 2.8, 3.1])
_SECOND = np.array([1.1, 0.9, 1.3, 1.0, 5.1, 4.9, 5.3, 5.0, 3.1, 2.9, 3.3, 3.0])
FEATURES = [Feature("first", _FIRST), Feature("second", _SECOND)]
TARGET = Feature("group", np.array([0.0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))


def test_it_is_constructed_by_the_same_keyword(backend: ModuleType) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")

    model = GaussianNaiveBayes(variance_smoothing=1e-8)

    assert model.variance_smoothing == pytest.approx(1e-8)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes()

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_FIRST)
    assert np.array_equal(np.asarray(predictions), np.asarray(TARGET.column.values))


def test_it_counts_the_classes_it_saw(backend: ModuleType) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")

    assert GaussianNaiveBayes().fit(FEATURES, TARGET).n_classes == 3


def test_its_priors_are_the_class_shares(backend: ModuleType) -> None:
    # Four rows of each of three classes, so each prior is exactly a third.
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    assert np.allclose(model.class_priors, 1 / 3)
    assert model.class_priors.sum() == pytest.approx(1.0)


def test_its_summaries_are_one_per_class_per_column(backend: ModuleType) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    assert model.means.shape == (3, 2)
    assert model.variances.shape == (3, 2)
    assert model.feature_names == ("first", "second")


def test_the_first_class_mean_is_the_hand_worked_average(backend: ModuleType) -> None:
    # (1.0 + 1.4 + 0.8 + 1.2) / 4 = 1.1, and (1.1 + 0.9 + 1.3 + 1.0) / 4 = 1.075.
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    assert model.means[0][0] == pytest.approx(1.1)
    assert model.means[0][1] == pytest.approx(1.075)


def test_every_variance_is_strictly_positive(backend: ModuleType) -> None:
    """The floor's whole job. A zero variance is a division by zero in the
    density, and a class whose column never varies is data rather than a bug."""
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    flat = [Feature("first", _FIRST), Feature("second", np.full(len(_FIRST), 2.0))]

    model = GaussianNaiveBayes().fit(flat, TARGET)

    assert (model.variances > 0.0).all()


def test_its_probabilities_are_one_row_per_query_summing_to_one(
    backend: ModuleType,
) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    scores = np.asarray(model.predict_probabilities(FEATURES))

    assert scores.shape == (len(_FIRST), 3)
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_the_prediction_is_the_largest_log_score(backend: ModuleType) -> None:
    """The two routes to one answer, which must not part company. Predicting
    reads the log scores directly, since normalising cannot reorder them."""
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    from_logs = np.argmax(model.log_scores(FEATURES), axis=1)

    assert np.array_equal(np.asarray(model.predict(FEATURES)), from_logs.astype(float))


def test_the_log_scores_are_unnormalised(backend: ModuleType) -> None:
    """They sum to nothing in particular, which is what makes them the honest
    quantity rather than a probability."""
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    scores = model.log_scores(FEATURES)

    assert not np.allclose(np.exp(scores).sum(axis=1), 1.0)


def test_a_log_score_may_be_positive(backend: ModuleType) -> None:
    """Because a density is not a probability.

    A tight class has most of its mass in a narrow interval, so its density
    there exceeds one and the log of it is positive. Asserting these were
    negative looked obviously right and is false on this fixture, where the
    clumps are close together.
    """
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    scores = model.log_scores(FEATURES)

    assert (scores > 0.0).any()
    assert np.isfinite(scores).all()


def test_it_matches_query_columns_by_name(backend: ModuleType) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")
    model = GaussianNaiveBayes().fit(FEATURES, TARGET)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.array_equal(
        np.asarray(model.predict(reversed_order)), np.asarray(model.predict(FEATURES))
    )


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")

    with pytest.raises(NotFittedError):
        GaussianNaiveBayes().predict(FEATURES)


def test_it_refuses_to_report_its_summaries_before_fit(backend: ModuleType) -> None:
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")

    with pytest.raises(NotFittedError):
        _ = GaussianNaiveBayes().means
