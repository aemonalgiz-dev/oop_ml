"""The contract every backend's LinearDiscriminantAnalysis keeps.

The second generative classifier here, and like the first it can be held to
its arithmetic rather than only to its encapsulation, because there is no
solver for two backends to reach by different routes. A prior is a count, a
mean is a sum over a count, and the pooled covariance is a sum of outer
products over one denominator. Measured, the two backends' priors and means
are bit-identical and their pooled covariances agree to 2.2e-16.

The denominator is the thing to pin
------------------------------------
It is ``n - K``, not ``n``, because ``K`` means have already been estimated
from these rows. The engine reports its own ``covariance_`` under the second
convention while whitening by the first, so a wrapper reading that attribute
through would be too small by a fifth on this fixture and nothing would raise.
The fixture below is arranged so the right answer is exactly ``4 / 3`` and
the wrong one is exactly ``1``.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml import Feature
from oop_ml.core.exceptions import (
    CollinearFeaturesError,
    NotFittedError,
    TooFewValuesError,
)

from .harness import provided

#: Two square clouds of four corners each, one centred on (1, 1) and one on
#: (6, 1). Every deviation from a class mean is one of (+-1, +-1), so the
#: summed outer products are 8 on each diagonal and 0 off it, and the pooled
#: covariance is 8 / (8 - 2) on the diagonal: exactly 4 / 3.
_LEFT = np.array([0.0, 2.0, 0.0, 2.0, 5.0, 7.0, 5.0, 7.0])
_RIGHT = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0])
FEATURES = [Feature("left", _LEFT), Feature("right", _RIGHT)]
TARGET = Feature("group", np.array([0.0, 0, 0, 0, 1, 1, 1, 1]))

POOLED_DIAGONAL = 4.0 / 3.0
POOLED_DIAGONAL_IF_DIVIDED_BY_N = 1.0

#: The two class means differ only in ``left``, and the covariance is a
#: multiple of the identity, so with equal priors the boundary is the
#: perpendicular bisector: ``left = 3.5``, at every value of ``right``.
BOUNDARY = 3.5


def _tilted_cloud(
    generator: np.random.Generator, count: int, offset: float
) -> np.ndarray:
    """A long thin cloud lying along the 45 degree line, offset across it.

    The two columns are strongly correlated *within* a class, which is the
    assumption naive Bayes makes and this model does not.
    """
    along = generator.normal(scale=3.0, size=count)
    across = generator.normal(scale=0.35, size=count) + offset

    return np.column_stack([along + across, along - across])


_GENERATOR = np.random.default_rng(5)
_FIRST_CLOUD = _tilted_cloud(_GENERATOR, 60, -1.0)
_SECOND_CLOUD = _tilted_cloud(_GENERATOR, 60, 1.0)
_CLOUD_ROWS = np.vstack([_FIRST_CLOUD, _SECOND_CLOUD])

#: Within-class correlation 0.98. Separable by a plane that reads the two
#: columns together, and badly separable by one that reads them apart.
CORRELATED_FEATURES = [
    Feature("left", _CLOUD_ROWS[:, 0]),
    Feature("right", _CLOUD_ROWS[:, 1]),
]
CORRELATED_TARGET = Feature("group", np.array([0.0] * 60 + [1.0] * 60))


def test_it_takes_no_hyperparameters(backend: ModuleType) -> None:
    """The only classifier here with none, and a fact about the model rather
    than an omission: the means, the pooled covariance and the priors are all
    counts and sums, so there is no starting point, no step size and nothing
    to choose."""
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")

    # Not ``model_fields_set``, which is empty on any freshly built model and
    # would pass for a class with a dozen defaulted fields.
    assert LinearDiscriminantAnalysis.model_fields == {}

    with pytest.raises(ValidationError):
        LinearDiscriminantAnalysis(shrinkage=0.5)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis()

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_LEFT)
    assert np.array_equal(np.asarray(predictions), np.asarray(TARGET.column.values))


def test_it_counts_the_classes_it_saw(backend: ModuleType) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")

    assert LinearDiscriminantAnalysis().fit(FEATURES, TARGET).n_classes == 2


def test_its_priors_are_the_class_shares(backend: ModuleType) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(FEATURES, TARGET)

    assert np.allclose(model.class_priors, 0.5)
    assert model.class_priors.sum() == pytest.approx(1.0)


def test_its_means_are_the_hand_worked_class_averages(backend: ModuleType) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(FEATURES, TARGET)

    assert model.means.shape == (2, 2)
    assert model.feature_names == ("left", "right")
    assert np.allclose(model.means, [[1.0, 1.0], [6.0, 1.0]])


def test_the_pooled_covariance_is_divided_by_the_degrees_of_freedom(
    backend: ModuleType,
) -> None:
    """``n - K``, and the fixture is built so the two conventions cannot be
    confused: the right denominator gives exactly ``4 / 3`` and dividing by
    ``n`` instead gives exactly ``1``."""
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(FEATURES, TARGET)

    pooled = model.pooled_covariance

    assert pooled.shape == (2, 2)
    assert pooled[0][0] == pytest.approx(POOLED_DIAGONAL)
    assert pooled[1][1] == pytest.approx(POOLED_DIAGONAL)
    assert pooled[0][0] != pytest.approx(POOLED_DIAGONAL_IF_DIVIDED_BY_N)
    assert pooled[0][1] == pytest.approx(0.0)


def test_the_pooled_covariance_is_symmetric(backend: ModuleType) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(CORRELATED_FEATURES, CORRELATED_TARGET)

    pooled = np.asarray(model.pooled_covariance)

    assert np.allclose(pooled, pooled.T)
    assert pooled[0][1] != pytest.approx(0.0)


def test_the_boundary_is_a_plane_at_the_midpoint(backend: ModuleType) -> None:
    """What the word linear in the name is about.

    The two classes share one covariance, so every term that depends on the
    row alone is the same in both scores and cancels. On this fixture the
    covariance is a multiple of the identity and the priors are equal, so what
    is left is the perpendicular bisector of the two means: ``left = 3.5``,
    exactly, and at any value of ``right`` however far away.
    """
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(FEATURES, TARGET)

    on_the_boundary = [
        Feature("left", np.array([BOUNDARY, BOUNDARY, BOUNDARY])),
        Feature("right", np.array([1.0, -50.0, 50.0])),
    ]
    scores = np.asarray(model.discriminant_scores(on_the_boundary))

    assert np.allclose(scores[:, 0] - scores[:, 1], 0.0, atol=1e-12)


def test_either_side_of_that_plane_is_decided_by_one_column(
    backend: ModuleType,
) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(FEATURES, TARGET)

    straddling = [
        Feature("left", np.array([BOUNDARY - 0.1, BOUNDARY + 0.1])),
        Feature("right", np.array([1.0, 1.0])),
    ]

    assert np.array_equal(np.asarray(model.predict(straddling)), [0.0, 1.0])


def test_its_probabilities_are_one_row_per_query_summing_to_one(
    backend: ModuleType,
) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(FEATURES, TARGET)

    scores = np.asarray(model.predict_probabilities(FEATURES))

    assert scores.shape == (len(_LEFT), 2)
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_the_prediction_is_the_largest_discriminant(backend: ModuleType) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(CORRELATED_FEATURES, CORRELATED_TARGET)

    from_scores = np.argmax(model.discriminant_scores(CORRELATED_FEATURES), axis=1)

    assert np.array_equal(
        np.asarray(model.predict(CORRELATED_FEATURES)), from_scores.astype(float)
    )


def test_letting_the_columns_vary_together_beats_assuming_they_do_not(
    backend: ModuleType,
) -> None:
    """The reason to prefer this over naive Bayes, on the case that shows it.

    Two long thin clouds lying along the same diagonal and offset across it,
    correlated at 0.98 within a class. Reading the columns together separates
    them exactly; reading them apart does not, because an axis-aligned view of
    either cloud is wide in both directions and the two overlap.
    """
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    GaussianNaiveBayes = provided(backend, "GaussianNaiveBayes")

    together = LinearDiscriminantAnalysis().fit(CORRELATED_FEATURES, CORRELATED_TARGET)
    apart = GaussianNaiveBayes().fit(CORRELATED_FEATURES, CORRELATED_TARGET)

    assert together.score(CORRELATED_FEATURES, CORRELATED_TARGET) == 1.0
    assert apart.score(CORRELATED_FEATURES, CORRELATED_TARGET) < 0.9


def test_it_refuses_a_column_that_says_nothing_new(backend: ModuleType) -> None:
    """A duplicated column leaves the pooled covariance singular, so
    infinitely many discriminants separate the classes equally well and no
    solver can choose among them. The engine alone would have answered, taking
    whichever its decomposition gives; both backends refuse by name."""
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    duplicated = [*FEATURES, Feature("copy_of_left", _LEFT.copy())]

    with pytest.raises(CollinearFeaturesError):
        LinearDiscriminantAnalysis().fit(duplicated, TARGET)


def test_it_refuses_when_no_row_is_left_over_to_measure_spread(
    backend: ModuleType,
) -> None:
    """One row per class leaves ``n - K`` at zero. Every class spends a degree
    of freedom on its own mean, so there is nothing left to pool."""
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    two_rows = [Feature("left", np.array([0.0, 5.0])), Feature("right", [1.0, 4.0])]

    with pytest.raises(TooFewValuesError):
        LinearDiscriminantAnalysis().fit(two_rows, Feature("group", [0.0, 1.0]))


def test_it_matches_query_columns_by_name(backend: ModuleType) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")
    model = LinearDiscriminantAnalysis().fit(FEATURES, TARGET)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.array_equal(
        np.asarray(model.predict(reversed_order)), np.asarray(model.predict(FEATURES))
    )


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")

    with pytest.raises(NotFittedError):
        LinearDiscriminantAnalysis().predict(FEATURES)


@pytest.mark.parametrize(
    "summary", ["class_priors", "means", "pooled_covariance", "n_classes"]
)
def test_it_refuses_to_report_its_summaries_before_fit(
    backend: ModuleType, summary: str
) -> None:
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")

    with pytest.raises(NotFittedError):
        getattr(LinearDiscriminantAnalysis(), summary)
