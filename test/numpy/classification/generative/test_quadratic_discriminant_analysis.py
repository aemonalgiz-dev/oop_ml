"""Spec for QuadraticDiscriminantAnalysis.

The oracle for the summaries is the same one the linear model's spec uses, a
plain Python loop over rows and column pairs, written from the definition
rather than from the implementation's broadcasting.

What this file adds is the geometry, which is the reason to pay for a
covariance per class and is asserted rather than asserted-about. A straight
line crosses a plane at most once; :class:`TestTheBoundaryCurves` walks a ray
across the fixture and counts the crossings, which is two for this model and
one for anything linear. The same count is what makes ``shrinkage`` legible:
turned all the way up it makes every class the same shape, and the boundary
straightens back to a single crossing.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    CollinearFeaturesError,
    NotFittedError,
    SingleClassError,
    TooFewValuesError,
)
from oop_ml.core.types import FloatArray
from oop_ml.numpy.classification.generative.linear_discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from oop_ml.numpy.classification.generative.quadratic_discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)

#: Two squares of the same size, one about (1, 1) and one about (6, 1). Both
#: classes have the same shape, which is the case the linear model assumes.
MATCHED_LEFT = np.array([0.0, 2.0, 0.0, 2.0, 5.0, 7.0, 5.0, 7.0])
MATCHED_RIGHT = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0])
MATCHED_FEATURES = [
    Feature("left", MATCHED_LEFT),
    Feature("right", MATCHED_RIGHT),
]
MATCHED_TARGET = Feature("group", np.array([0.0, 0, 0, 0, 1, 1, 1, 1]))

_GENERATOR = np.random.default_rng(19)
_ANGLES = _GENERATOR.uniform(0.0, 2.0 * np.pi, 70)
_RADII = 5.0 + _GENERATOR.normal(scale=0.5, size=70)
_RING = np.column_stack([_RADII * np.cos(_ANGLES), _RADII * np.sin(_ANGLES)])
_CORE = _GENERATOR.normal(scale=0.9, size=(70, 2))
ENCLOSED_ROWS = np.vstack([_CORE, _RING])

#: A tight blob with a ring of the other class around it, centred on the
#: origin, so a ray through the centre leaves one class, enters the other and
#: leaves it again.
ENCLOSED_FEATURES = [
    Feature("left", ENCLOSED_ROWS[:, 0]),
    Feature("right", ENCLOSED_ROWS[:, 1]),
]
ENCLOSED_TARGET = Feature("group", np.array([0.0] * 70 + [1.0] * 70))

#: 241 points along the horizontal through the origin, far enough out either
#: way to leave the ring behind.
RAY_POSITIONS = np.linspace(-12.0, 12.0, 241)
RAY = [
    Feature("left", RAY_POSITIONS),
    Feature("right", np.zeros_like(RAY_POSITIONS)),
]


def looped_class_covariance(
    rows: FloatArray, labels: FloatArray, label: int
) -> FloatArray:
    """One class's sample covariance, one row and one pair of columns at a time."""
    belonging = [
        rows[index] for index in range(rows.shape[0]) if int(labels[index]) == label
    ]
    n_features = rows.shape[1]
    means = [
        sum(row[column] for row in belonging) / len(belonging)
        for column in range(n_features)
    ]

    covariance = np.zeros((n_features, n_features))
    for first in range(n_features):
        for second in range(n_features):
            total = 0.0
            for row in belonging:
                total += (row[first] - means[first]) * (row[second] - means[second])
            covariance[first][second] = total / (len(belonging) - 1)

    return covariance


def crossings_along_the_ray(model: QuadraticDiscriminantAnalysis) -> int:
    """How many times the answer changes while walking a straight line.

    A plane is crossed at most once by a straight line, whatever its
    orientation, so anything above one is a boundary that is not a plane.
    """
    picked = np.asarray(model.predict(RAY))

    return int(np.count_nonzero(np.diff(picked) != 0))


class TestConstruction:
    def test_shrinkage_defaults_to_none_at_all(self):
        assert QuadraticDiscriminantAnalysis().shrinkage == 0.0

    @pytest.mark.parametrize("invalid", [-0.01, 1.01])
    def test_a_shrinkage_outside_the_unit_interval_is_rejected(self, invalid):
        with pytest.raises(ValueError):
            QuadraticDiscriminantAnalysis(shrinkage=invalid)


class TestBeforeFit:
    @pytest.mark.parametrize(
        "attribute", ["class_priors", "means", "covariances", "n_classes"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(QuadraticDiscriminantAnalysis(), attribute)


class TestThePerClassCovariances:
    @pytest.mark.parametrize("label", [0, 1])
    def test_each_matches_the_looped_definition(self, label):
        model = QuadraticDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)
        expected = looped_class_covariance(
            ENCLOSED_ROWS, np.asarray(ENCLOSED_TARGET.column.values), label
        )

        assert np.allclose(np.asarray(model.covariances)[label], expected, atol=1e-12)

    def test_each_is_symmetric(self):
        model = QuadraticDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)

        for covariance in np.asarray(model.covariances):
            assert np.allclose(covariance, covariance.T)

    def test_shrinkage_is_the_convex_combination_it_says_it_is(self):
        """``(1 - s) S + s I``, exactly, checked against the unshrunk fit."""
        raw = QuadraticDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)
        shrunk = QuadraticDiscriminantAnalysis(shrinkage=0.5).fit(
            ENCLOSED_FEATURES, ENCLOSED_TARGET
        )

        expected = 0.5 * np.asarray(raw.covariances) + 0.5 * np.array(
            [np.eye(2), np.eye(2)]
        )

        assert np.allclose(np.asarray(shrunk.covariances), expected, atol=1e-14)

    def test_full_shrinkage_leaves_every_class_a_unit_sphere(self):
        model = QuadraticDiscriminantAnalysis(shrinkage=1.0).fit(
            ENCLOSED_FEATURES, ENCLOSED_TARGET
        )

        assert np.allclose(np.asarray(model.covariances), [np.eye(2), np.eye(2)])


class TestTheBoundaryCurves:
    def test_a_straight_line_crosses_it_twice(self):
        """Which no plane allows, at any orientation. The ring class surrounds
        the other, so walking in from the far left the answer goes ring, core,
        ring again."""
        model = QuadraticDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)

        assert crossings_along_the_ray(model) == 2

    def test_and_full_shrinkage_straightens_it_back(self):
        """Every class the same shape means the quadratic term is identical in
        both scores and cancels, so what is left is a plane. The cost of that
        is the whole of this model's advantage here: 0.99 to 0.61, which is
        the linear model's own number on this fixture.
        """
        model = QuadraticDiscriminantAnalysis(shrinkage=1.0).fit(
            ENCLOSED_FEATURES, ENCLOSED_TARGET
        )
        flat = LinearDiscriminantAnalysis().fit(ENCLOSED_FEATURES, ENCLOSED_TARGET)

        assert crossings_along_the_ray(model) == 1
        assert model.score(ENCLOSED_FEATURES, ENCLOSED_TARGET) == pytest.approx(
            flat.score(ENCLOSED_FEATURES, ENCLOSED_TARGET), abs=0.02
        )


class TestItContainsTheLinearModel:
    """When the classes really do have one shape, the two models are the same
    model, and this fixture is arranged so they are the same *exactly*."""

    def test_the_per_class_covariances_equal_the_pooled_one(self):
        """Four rows per class, so each class's ``n_k - 1`` is 3 and the pooled
        ``n - K`` is 6 over twice the deviations. Both come to 4 / 3."""
        curved = QuadraticDiscriminantAnalysis().fit(MATCHED_FEATURES, MATCHED_TARGET)
        flat = LinearDiscriminantAnalysis().fit(MATCHED_FEATURES, MATCHED_TARGET)

        for covariance in np.asarray(curved.covariances):
            assert np.allclose(covariance, flat.pooled_covariance, atol=1e-14)

    def test_and_so_do_the_probabilities(self):
        curved = QuadraticDiscriminantAnalysis().fit(MATCHED_FEATURES, MATCHED_TARGET)
        flat = LinearDiscriminantAnalysis().fit(MATCHED_FEATURES, MATCHED_TARGET)

        assert np.allclose(
            np.asarray(curved.predict_probabilities(MATCHED_FEATURES)),
            np.asarray(flat.predict_probabilities(MATCHED_FEATURES)),
            atol=1e-14,
        )

    def test_the_boundary_is_the_perpendicular_bisector(self):
        """Straight, at the midpoint of the two means, and unmoved by the other
        column however far out it goes."""
        model = QuadraticDiscriminantAnalysis().fit(MATCHED_FEATURES, MATCHED_TARGET)

        on_the_line = [
            Feature("left", np.array([3.5, 3.5, 3.5])),
            Feature("right", np.array([1.0, -40.0, 40.0])),
        ]
        scores = np.asarray(model.discriminant_scores(on_the_line))

        assert np.allclose(scores[:, 0] - scores[:, 1], 0.0, atol=1e-12)


class TestWhatItRefuses:
    def test_a_class_of_one_row_has_no_deviation_to_measure(self):
        lonely = [
            Feature("left", np.array([0.0, 2.0, 0.0, 10.0])),
            Feature("right", np.array([0.0, 0.0, 2.0, 5.0])),
        ]

        with pytest.raises(TooFewValuesError):
            QuadraticDiscriminantAnalysis().fit(
                lonely, Feature("group", [0.0, 0, 0, 1])
            )

    def test_a_class_with_no_more_rows_than_columns_is_singular(self):
        """Two rows in two columns leaves a rank of one, so that class
        describes no shape in one whole direction."""
        cramped = [
            Feature("left", np.array([0.0, 2.0, 0.0, 2.0, 10.0, 14.0])),
            Feature("right", np.array([0.0, 2.0, 2.0, 0.0, 0.0, 2.0])),
        ]

        with pytest.raises(CollinearFeaturesError):
            QuadraticDiscriminantAnalysis().fit(
                cramped, Feature("group", [0.0, 0, 0, 0, 1, 1])
            )

    def test_and_shrinkage_lets_that_same_fit_through(self):
        cramped = [
            Feature("left", np.array([0.0, 2.0, 0.0, 2.0, 10.0, 14.0])),
            Feature("right", np.array([0.0, 2.0, 2.0, 0.0, 0.0, 2.0])),
        ]
        target = Feature("group", [0.0, 0, 0, 0, 1, 1])

        model = QuadraticDiscriminantAnalysis(shrinkage=0.3).fit(cramped, target)

        assert model.score(cramped, target) == 1.0

    def test_a_refused_fit_leaves_the_model_unfitted(self):
        model = QuadraticDiscriminantAnalysis()
        duplicated = [
            *MATCHED_FEATURES,
            Feature("again", MATCHED_LEFT.copy()),
        ]

        with pytest.raises(CollinearFeaturesError):
            model.fit(duplicated, MATCHED_TARGET)

        with pytest.raises(NotFittedError):
            _ = model.covariances

    def test_a_single_class_is_refused(self):
        with pytest.raises(SingleClassError):
            QuadraticDiscriminantAnalysis().fit(
                MATCHED_FEATURES, Feature("group", np.zeros(8))
            )
