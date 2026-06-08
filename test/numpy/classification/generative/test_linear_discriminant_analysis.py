"""Spec for LinearDiscriminantAnalysis.

Two oracles here, both written from definitions the implementation does not
share.

The first is the pooled covariance, built by a plain Python loop over rows and
column pairs. It is the slowest possible way to compute the matrix and that is
the point: an oracle assembled from the same broadcasting the model uses would
only prove the expression had been copied.

The second is the whole reason the model is called linear.
:func:`full_log_likelihood` scores a row under a class the honest generative
way, keeping every term including the two the model drops, and what this file
asserts is that the difference between the two scorings is *constant across
classes within a row*. That is the cancellation the module docstring derives,
and it is checkable rather than merely plausible.
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

#: Four corners of a square about (1, 1) and four about (6, 1). Every deviation
#: is one of (+-1, +-1), so the pooled covariance is 8 / 6 on each diagonal.
SQUARE_LEFT = np.array([0.0, 2.0, 0.0, 2.0, 5.0, 7.0, 5.0, 7.0])
SQUARE_RIGHT = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0])
SQUARE_FEATURES = [Feature("left", SQUARE_LEFT), Feature("right", SQUARE_RIGHT)]
SQUARE_TARGET = Feature("group", np.array([0.0, 0, 0, 0, 1, 1, 1, 1]))

_GENERATOR = np.random.default_rng(2026)
_ROWS = np.vstack(
    [
        _GENERATOR.multivariate_normal([0.0, 0.0], [[2.0, 1.4], [1.4, 1.5]], size=25),
        _GENERATOR.multivariate_normal([2.5, 1.0], [[2.0, 1.4], [1.4, 1.5]], size=18),
        _GENERATOR.multivariate_normal([1.0, 3.0], [[2.0, 1.4], [1.4, 1.5]], size=22),
    ]
)
#: Three classes drawn from one covariance and three different means, which is
#: the situation this model actually assumes, at unequal class sizes so the
#: priors are not all the same number.
DRAWN_FEATURES = [Feature("left", _ROWS[:, 0]), Feature("right", _ROWS[:, 1])]
DRAWN_TARGET = Feature("group", np.array([0.0] * 25 + [1.0] * 18 + [2.0] * 22))


def looped_pooled_covariance(
    rows: FloatArray, labels: FloatArray, n_classes: int
) -> FloatArray:
    """The definition, one row and one pair of columns at a time.

    Deliberately the slow way. An oracle written with the same broadcasting as
    the implementation is a copy of the implementation.
    """
    n_rows, n_features = rows.shape
    means = [
        [
            sum(
                rows[index][column] for index in range(n_rows) if labels[index] == label
            )
            / sum(1 for index in range(n_rows) if labels[index] == label)
            for column in range(n_features)
        ]
        for label in range(n_classes)
    ]

    pooled = np.zeros((n_features, n_features))
    for first in range(n_features):
        for second in range(n_features):
            total = 0.0
            for index in range(n_rows):
                label = int(labels[index])
                total += (rows[index][first] - means[label][first]) * (
                    rows[index][second] - means[label][second]
                )
            pooled[first][second] = total / (n_rows - n_classes)

    return pooled


def full_log_likelihood(
    rows: FloatArray, mean: FloatArray, covariance: FloatArray, prior: float
) -> FloatArray:
    """``log(prior * density)`` with nothing dropped, one value per row.

    The quantity the discriminant is a shortened form of. Written out with the
    determinant and the ``(2 pi) ** (p / 2)`` the model leaves out, so that
    what those omissions cost can be measured rather than assumed.
    """
    gaps = rows - mean
    precision = np.linalg.inv(covariance)
    quadratic = np.einsum("ij,jk,ik->i", gaps, precision, gaps)
    _, log_determinant = np.linalg.slogdet(covariance)
    constant = rows.shape[1] * np.log(2.0 * np.pi)

    return np.log(prior) - 0.5 * (quadratic + log_determinant + constant)


def drawn_matrix() -> FloatArray:
    return np.asarray(_ROWS, dtype=np.float64)


class TestBeforeFit:
    @pytest.mark.parametrize(
        "attribute", ["class_priors", "means", "pooled_covariance", "n_classes"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(LinearDiscriminantAnalysis(), attribute)

    def test_predicting_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            LinearDiscriminantAnalysis().predict(SQUARE_FEATURES)


class TestThePooledCovariance:
    def test_matches_the_looped_definition(self):
        model = LinearDiscriminantAnalysis().fit(DRAWN_FEATURES, DRAWN_TARGET)
        expected = looped_pooled_covariance(
            drawn_matrix(), np.asarray(DRAWN_TARGET.column.values), 3
        )

        assert np.allclose(model.pooled_covariance, expected, atol=1e-12)

    def test_is_symmetric(self):
        model = LinearDiscriminantAnalysis().fit(DRAWN_FEATURES, DRAWN_TARGET)
        pooled = np.asarray(model.pooled_covariance)

        assert np.allclose(pooled, pooled.T)

    def test_the_denominator_is_the_degrees_of_freedom_not_the_row_count(self):
        """``n - K``, and on this fixture that is 8 - 2, so the diagonal is
        4 / 3 where dividing by 8 would give exactly 1."""
        model = LinearDiscriminantAnalysis().fit(SQUARE_FEATURES, SQUARE_TARGET)

        assert model.pooled_covariance[0][0] == pytest.approx(4.0 / 3.0)
        assert model.pooled_covariance[1][1] == pytest.approx(4.0 / 3.0)

    def test_each_class_is_centred_on_its_own_mean(self):
        """The word *within* in within-class covariance.

        Centring every row on one overall mean instead would fold the distance
        between the class means into the matrix meant to describe the spread
        inside them. On this fixture the classes are 5 apart along ``left``,
        so that mistake inflates the first diagonal entry from 4 / 3 to
        58 / 6, seven times too large, and leaves the second alone.
        """
        model = LinearDiscriminantAnalysis().fit(SQUARE_FEATURES, SQUARE_TARGET)

        rows = np.column_stack([SQUARE_LEFT, SQUARE_RIGHT])
        centred_on_everything = rows - rows.mean(axis=0)
        wrong = centred_on_everything.T @ centred_on_everything / (8 - 2)

        assert wrong[0][0] == pytest.approx(58.0 / 6.0, abs=1e-9)
        assert model.pooled_covariance[0][0] != pytest.approx(wrong[0][0])
        assert model.pooled_covariance[1][1] == pytest.approx(wrong[1][1])


class TestTheCancellation:
    """That the discriminant is the full log likelihood with terms dropped,
    and that the dropped terms do not depend on the class."""

    def test_the_gap_to_the_full_log_likelihood_is_constant_within_a_row(self):
        model = LinearDiscriminantAnalysis().fit(DRAWN_FEATURES, DRAWN_TARGET)
        rows = drawn_matrix()

        honest = np.column_stack(
            [
                full_log_likelihood(
                    rows,
                    np.asarray(model.means)[label],
                    np.asarray(model.pooled_covariance),
                    float(model.class_priors[label]),
                )
                for label in range(3)
            ]
        )
        shortened = np.asarray(model.discriminant_scores(DRAWN_FEATURES))
        gaps = honest - shortened

        assert np.allclose(gaps.max(axis=1) - gaps.min(axis=1), 0.0, atol=1e-9)

    def test_the_dropped_terms_are_not_themselves_zero(self):
        """A guard on the guard. If the omitted terms happened to vanish, the
        test above would hold for a model that dropped nothing and would prove
        nothing about the cancellation."""
        model = LinearDiscriminantAnalysis().fit(DRAWN_FEATURES, DRAWN_TARGET)
        rows = drawn_matrix()

        honest = full_log_likelihood(
            rows,
            np.asarray(model.means)[0],
            np.asarray(model.pooled_covariance),
            float(model.class_priors[0]),
        )
        shortened = np.asarray(model.discriminant_scores(DRAWN_FEATURES))[:, 0]

        assert np.abs(honest - shortened).min() > 1.0

    def test_the_probabilities_survive_the_dropping(self):
        model = LinearDiscriminantAnalysis().fit(DRAWN_FEATURES, DRAWN_TARGET)
        rows = drawn_matrix()

        honest = np.column_stack(
            [
                full_log_likelihood(
                    rows,
                    np.asarray(model.means)[label],
                    np.asarray(model.pooled_covariance),
                    float(model.class_priors[label]),
                )
                for label in range(3)
            ]
        )
        shifted = np.exp(honest - honest.max(axis=1, keepdims=True))
        from_the_definition = shifted / shifted.sum(axis=1, keepdims=True)

        assert np.allclose(
            np.asarray(model.predict_probabilities(DRAWN_FEATURES)),
            from_the_definition,
            atol=1e-12,
        )


class TestThePriorsMoveTheBoundary:
    def test_a_rarer_class_needs_more_evidence(self):
        """The prior is a term in the discriminant, so making one class rarer
        shifts the plane toward it rather than leaving it where equal priors
        put it."""
        balanced = LinearDiscriminantAnalysis().fit(SQUARE_FEATURES, SQUARE_TARGET)

        lopsided_left = np.concatenate([SQUARE_LEFT, [0.0, 2.0, 0.0, 2.0]])
        lopsided_right = np.concatenate([SQUARE_RIGHT, [0.0, 2.0, 2.0, 0.0]])
        lopsided = LinearDiscriminantAnalysis().fit(
            [Feature("left", lopsided_left), Feature("right", lopsided_right)],
            Feature(
                "group", np.concatenate([SQUARE_TARGET.column.values, [0, 0, 0, 0]])
            ),
        )

        just_past_the_middle = [
            Feature("left", np.array([3.6])),
            Feature("right", np.array([1.0])),
        ]

        assert balanced.predict(just_past_the_middle)[0] == 1.0
        assert lopsided.predict(just_past_the_middle)[0] == 0.0


class TestWhatItRefuses:
    def test_a_duplicated_column_leaves_no_unique_discriminant(self):
        duplicated = [*SQUARE_FEATURES, Feature("again", SQUARE_LEFT.copy())]

        with pytest.raises(CollinearFeaturesError):
            LinearDiscriminantAnalysis().fit(duplicated, SQUARE_TARGET)

    def test_a_column_that_is_a_sum_of_the_others_is_refused_too(self):
        combined = [
            *SQUARE_FEATURES,
            Feature("total", SQUARE_LEFT + SQUARE_RIGHT),
        ]

        with pytest.raises(CollinearFeaturesError):
            LinearDiscriminantAnalysis().fit(combined, SQUARE_TARGET)

    def test_a_refused_fit_leaves_the_model_unfitted(self):
        """Nothing is assigned before the refusal, so a model that could not
        be fitted does not answer as though it had been."""
        model = LinearDiscriminantAnalysis()
        duplicated = [*SQUARE_FEATURES, Feature("again", SQUARE_LEFT.copy())]

        with pytest.raises(CollinearFeaturesError):
            model.fit(duplicated, SQUARE_TARGET)

        with pytest.raises(NotFittedError):
            _ = model.means

    def test_one_row_per_class_leaves_nothing_to_pool(self):
        with pytest.raises(TooFewValuesError):
            LinearDiscriminantAnalysis().fit(
                [Feature("left", [0.0, 5.0]), Feature("right", [1.0, 4.0])],
                Feature("group", [0.0, 1.0]),
            )

    def test_a_single_class_is_refused(self):
        with pytest.raises(SingleClassError):
            LinearDiscriminantAnalysis().fit(
                SQUARE_FEATURES, Feature("group", np.zeros(8))
            )
