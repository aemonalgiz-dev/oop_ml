"""The contract every backend's QuadraticDiscriminantAnalysis keeps.

The third generative classifier, and the third contract here that can assert
the arithmetic rather than only the encapsulation, for the reason the other two
give: a prior is a count, a mean is a sum over a count, and a covariance is a
sum of outer products over one denominator, so there is no solver for the two
backends to reach by different routes. Measured across three shrinkages, their
priors and means come out bit-identical, their covariances agree to 2.7e-15 and
their unnormalised discriminants to 2.8e-14.

The two claims worth reading
-----------------------------
That the boundary can curve, which is the whole reason to pay for a covariance
per class, and that ``shrinkage`` really is a dial between the sample estimate
and a unit sphere. The first is asserted on a fixture the linear model cannot
do anything with, a class sitting inside another; the second at its far end,
where the discriminant is exactly a nearest-mean score tilted by the priors.
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

#: Four corners of a square about (1, 1), and four corners of a rectangle twice
#: as wide about (12, 1). Every deviation is (+-1, +-1) in the first and
#: (+-2, +-1) in the second, so the covariances are 4 / 3 on both diagonals and
#: 16 / 3 by 4 / 3 -- different matrices, which is what this model has and the
#: linear one does not. Dividing by n_k instead would give 1 and 4.
_LEFT = np.array([0.0, 2.0, 0.0, 2.0, 10.0, 14.0, 10.0, 14.0])
_RIGHT = np.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0])
FEATURES = [Feature("left", _LEFT), Feature("right", _RIGHT)]
TARGET = Feature("group", np.array([0.0, 0, 0, 0, 1, 1, 1, 1]))

TIGHT_DIAGONAL = 4.0 / 3.0
WIDE_DIAGONAL = 16.0 / 3.0

_GENERATOR = np.random.default_rng(19)
_ANGLES = _GENERATOR.uniform(0.0, 2.0 * np.pi, 70)
_RADII = 5.0 + _GENERATOR.normal(scale=0.5, size=70)
_RING = np.column_stack([_RADII * np.cos(_ANGLES), _RADII * np.sin(_ANGLES)])
_CORE = _GENERATOR.normal(scale=0.9, size=(70, 2))
_ENCLOSED_ROWS = np.vstack([_CORE, _RING])

#: A tight blob at the origin with a ring of the other class around it. No
#: plane separates a thing from what surrounds it; an ellipse does.
ENCLOSED_FEATURES = [
    Feature("left", _ENCLOSED_ROWS[:, 0]),
    Feature("right", _ENCLOSED_ROWS[:, 1]),
]
ENCLOSED_TARGET = Feature("group", np.array([0.0] * 70 + [1.0] * 70))


def test_it_is_constructed_by_the_same_keyword(backend: ModuleType) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")

    assert QuadraticDiscriminantAnalysis(shrinkage=0.25).shrinkage == pytest.approx(
        0.25
    )
    assert QuadraticDiscriminantAnalysis().shrinkage == 0.0


@pytest.mark.parametrize("outside", [-0.1, 1.1])
def test_it_refuses_a_shrinkage_that_is_not_a_share(
    backend: ModuleType, outside: float
) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")

    with pytest.raises(ValidationError):
        QuadraticDiscriminantAnalysis(shrinkage=outside)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    model = QuadraticDiscriminantAnalysis()

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    model = QuadraticDiscriminantAnalysis().fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_LEFT)
    assert np.array_equal(np.asarray(predictions), np.asarray(TARGET.column.values))


def test_it_counts_the_classes_it_saw(backend: ModuleType) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")

    assert QuadraticDiscriminantAnalysis().fit(FEATURES, TARGET).n_classes == 2


def test_its_priors_and_means_are_the_class_shares_and_averages(
    backend: ModuleType,
) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    model = QuadraticDiscriminantAnalysis().fit(FEATURES, TARGET)

    assert np.allclose(model.class_priors, 0.5)
    assert np.allclose(model.means, [[1.0, 1.0], [12.0, 1.0]])
    assert model.feature_names == ("left", "right")


def test_every_class_gets_a_covariance_of_its_own(backend: ModuleType) -> None:
    """The whole of what is bought here. The linear model has one matrix; this
    one has ``K``, and on this fixture the second class is four times as wide
    as the first in ``left`` and exactly as wide in ``right``.

    The denominator is ``n_k - 1``, per class, so the diagonals are 4 / 3 and
    16 / 3. Dividing by ``n_k`` instead would give 1 and 4.
    """
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    model = QuadraticDiscriminantAnalysis().fit(FEATURES, TARGET)

    covariances = np.asarray(model.covariances)

    assert covariances.shape == (2, 2, 2)
    assert covariances[0][0][0] == pytest.approx(TIGHT_DIAGONAL)
    assert covariances[0][1][1] == pytest.approx(TIGHT_DIAGONAL)
    assert covariances[1][0][0] == pytest.approx(WIDE_DIAGONAL)
    assert covariances[1][1][1] == pytest.approx(TIGHT_DIAGONAL)
    assert covariances[0][0][1] == pytest.approx(0.0)


def test_a_boundary_that_can_curve_separates_a_class_from_what_surrounds_it(
    backend: ModuleType,
) -> None:
    """The reason to prefer this over the linear model, on the case that shows
    it. A tight blob with a ring of the other class around it cannot be cut by
    a plane, so the linear model is barely better than guessing, while a
    boundary free to close on itself separates them nearly exactly."""
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    LinearDiscriminantAnalysis = provided(backend, "LinearDiscriminantAnalysis")

    curved = QuadraticDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)
    flat = LinearDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)

    assert curved.score(ENCLOSED_FEATURES, ENCLOSED_TARGET) > 0.95
    assert flat.score(ENCLOSED_FEATURES, ENCLOSED_TARGET) < 0.7


def test_shrinkage_all_the_way_is_a_nearest_mean_score(backend: ModuleType) -> None:
    """The far end of the dial, and it lands exactly.

    At ``shrinkage = 1`` every class's covariance is the identity, so the log
    determinant is zero for all of them and the quadratic form is the plain
    squared distance to the class mean. What is left is
    ``-0.5 ||x - mu_k||^2 + log prior_k``, which needs no matrix at all.
    """
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    model = QuadraticDiscriminantAnalysis(shrinkage=1.0).fit(FEATURES, TARGET)

    probe_left = np.array([6.0, 5.0, 7.0])
    probe_right = np.array([1.0, 1.0, 1.0])
    probe = [Feature("left", probe_left), Feature("right", probe_right)]
    block = np.column_stack([probe_left, probe_right])

    means = np.asarray(model.means)
    by_hand = np.column_stack(
        [
            -0.5 * ((block - means[label]) ** 2).sum(axis=1)
            + np.log(model.class_priors[label])
            for label in range(2)
        ]
    )

    assert np.allclose(model.discriminant_scores(probe), by_hand, atol=1e-12)
    assert np.allclose(np.asarray(model.covariances), [np.eye(2), np.eye(2)])


def test_its_probabilities_are_one_row_per_query_summing_to_one(
    backend: ModuleType,
) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    model = QuadraticDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)

    scores = np.asarray(model.predict_probabilities(ENCLOSED_FEATURES))

    assert scores.shape == (len(_ENCLOSED_ROWS), 2)
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_the_prediction_is_the_largest_discriminant(backend: ModuleType) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    model = QuadraticDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)

    from_scores = np.argmax(model.discriminant_scores(ENCLOSED_FEATURES), axis=1)

    assert np.array_equal(
        np.asarray(model.predict(ENCLOSED_FEATURES)), from_scores.astype(float)
    )


def test_a_class_that_describes_no_shape_is_refused(backend: ModuleType) -> None:
    """A duplicated column leaves every class's covariance singular, so no
    class can be inverted and there is no discriminant to compute."""
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    duplicated = [*FEATURES, Feature("copy_of_left", _LEFT.copy())]

    with pytest.raises(CollinearFeaturesError):
        QuadraticDiscriminantAnalysis().fit(duplicated, TARGET)


def test_and_any_shrinkage_at_all_rescues_it(backend: ModuleType) -> None:
    """Which is what the dial is for. ``(1 - s) S + s I`` is positive definite
    for any ``s`` above zero whatever the data did, so the fit that could not
    be made at all can be made at the cost of some of its freedom."""
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    duplicated = [*FEATURES, Feature("copy_of_left", _LEFT.copy())]

    model = QuadraticDiscriminantAnalysis(shrinkage=0.2).fit(duplicated, TARGET)

    assert model.score(duplicated, TARGET) == 1.0


def test_a_class_of_one_row_has_no_spread_to_measure(backend: ModuleType) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    lonely = [
        Feature("left", np.array([0.0, 2.0, 0.0, 10.0])),
        Feature("right", np.array([0.0, 0.0, 2.0, 5.0])),
    ]

    with pytest.raises(TooFewValuesError):
        QuadraticDiscriminantAnalysis().fit(lonely, Feature("group", [0.0, 0, 0, 1]))


def test_it_matches_query_columns_by_name(backend: ModuleType) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")
    model = QuadraticDiscriminantAnalysis().fit(FEATURES, TARGET)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.array_equal(
        np.asarray(model.predict(reversed_order)), np.asarray(model.predict(FEATURES))
    )


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")

    with pytest.raises(NotFittedError):
        QuadraticDiscriminantAnalysis().predict(FEATURES)


@pytest.mark.parametrize(
    "summary", ["class_priors", "means", "covariances", "n_classes"]
)
def test_it_refuses_to_report_its_summaries_before_fit(
    backend: ModuleType, summary: str
) -> None:
    QuadraticDiscriminantAnalysis = provided(backend, "QuadraticDiscriminantAnalysis")

    with pytest.raises(NotFittedError):
        getattr(QuadraticDiscriminantAnalysis(), summary)
