"""Spec for the support vector machine -- red until ``_ascend`` and the decision land.

``SURROUNDED_CLASS`` is the fixture that makes the point: one class on a circle
of radius 2, the other on a circle of radius 5 around it. No straight line
separates a class from the class enclosing it, so the linear kernel is capped
and the radial one is not. Both halves are asserted, because a spec that only
showed the radial success would not be showing anything.

``TestSupportVectors`` covers the property that separates this from kernel
ridge: most multipliers come out at zero, so most training rows drop out of the
model entirely. Kernel ridge keeps all of them.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    InvalidValuesError,
    NonBinaryLabelsError,
    NotFittedError,
    SingleClassError,
)
from oop_ml.core.kernel.functions import LinearKernel, RadialBasisKernel
from oop_ml.numpy.classification.kernels.support_vector_classifier import (
    SupportVector,
    SupportVectorClassifier,
    SupportVectors,
)
from test.fixtures import (
    SURROUNDED_CLASS,
    SURROUNDED_LINEAR_CEILING,
    SURROUNDED_RADIAL_FLOOR,
)


def fitted(kernel=None, **overrides) -> SupportVectorClassifier:
    """A model fitted to the surrounded-class fixture."""
    return SupportVectorClassifier(
        kernel=kernel or RadialBasisKernel(gamma=0.1),
        learning_rate=0.01,
        max_epochs=3000,
        **overrides,
    ).fit(SURROUNDED_CLASS.input_features, SURROUNDED_CLASS.target_feature)


class TestWhatTheKernelBuys:
    """A class surrounded by another, which no line can cut out."""

    def test_a_radial_kernel_separates_what_a_linear_one_cannot(self) -> None:
        """Both halves asserted, so this is a contrast rather than a claim."""
        linear = fitted(LinearKernel()).score(
            SURROUNDED_CLASS.input_features, SURROUNDED_CLASS.target_feature
        )
        radial = fitted(RadialBasisKernel(gamma=0.1)).score(
            SURROUNDED_CLASS.input_features, SURROUNDED_CLASS.target_feature
        )

        assert linear < SURROUNDED_LINEAR_CEILING
        assert radial > SURROUNDED_RADIAL_FLOOR

    def test_the_boundary_encloses_the_inner_class(self) -> None:
        """A point at the origin is inside the inner circle, so it is class 1."""
        model = fitted(RadialBasisKernel(gamma=0.1))
        predicted = model.predict([Feature("first", [0.0]), Feature("second", [0.0])])

        assert float(np.asarray(predicted)[0]) == pytest.approx(1.0)

    def test_a_point_beyond_the_outer_ring_is_the_outer_class(self) -> None:
        model = fitted(RadialBasisKernel(gamma=0.1))
        predicted = model.predict([Feature("first", [9.0]), Feature("second", [0.0])])

        assert float(np.asarray(predicted)[0]) == pytest.approx(0.0)


class TestSupportVectors:
    """The property that makes the model small."""

    def test_only_some_training_rows_survive(self) -> None:
        """Most multipliers are zero, so most rows drop out of the boundary.

        This is what separates a support vector machine from kernel ridge,
        which keeps every row because every dual weight is non-zero.
        """
        vectors = fitted(RadialBasisKernel(gamma=0.1)).support_vectors

        assert 0 < vectors.n_vectors <= SURROUNDED_CLASS.n_samples
        assert vectors.n_training_rows == SURROUNDED_CLASS.n_samples

    def test_every_support_vector_has_a_positive_multiplier(self) -> None:
        """A row with a zero multiplier is not one, by definition."""
        for vector in fitted().support_vectors:
            assert vector.multiplier > 0.0

    def test_multipliers_stay_inside_the_box(self) -> None:
        """The projection is what enforces ``0 <= a <= C``."""
        model = fitted(capacity=2.0)
        multipliers = model.multipliers

        assert multipliers.min() >= -1e-12
        assert multipliers.max() <= 2.0 + 1e-12

    def test_labels_are_the_signed_encoding_the_dual_uses(self) -> None:
        signed = fitted().signed_labels

        assert set(np.unique(signed)) == {-1.0, 1.0}

    def test_it_reports_which_rows_the_boundary_depends_on(self) -> None:
        positions = fitted().support_vectors.positions()

        assert all(0 <= position < SURROUNDED_CLASS.n_samples for position in positions)
        assert len(set(positions)) == len(positions)

    def test_a_smaller_capacity_puts_more_vectors_at_the_cap(self) -> None:
        """The cap is the price of a violation, so a low one binds more rows."""
        tight = fitted(capacity=0.01)
        loose = fitted(capacity=10.0)

        assert tight.support_vectors.n_at_the_cap(
            0.01
        ) >= loose.support_vectors.n_at_the_cap(10.0)


class TestConvergence:
    """That the ascent runs and stops."""

    def test_it_records_how_many_steps_it_took(self) -> None:
        model = fitted()

        assert 0 < model.epochs_run <= 3000

    def test_the_same_configuration_fits_the_same_way(self) -> None:
        """No randomness anywhere: the ascent starts from zero every time."""
        first = fitted().multipliers
        second = fitted().multipliers

        assert np.allclose(first, second)

    def test_it_ascends_rather_than_descends(self) -> None:
        """The sign that is easy to get backwards.

        Descending would drive every multiplier to zero, leaving no support
        vectors at all and a model that says the same thing about every row --
        useless-looking rather than obviously wrong.
        """
        model = fitted()

        assert model.support_vectors.n_vectors > 0
        assert model.multipliers.max() > 0.0


class TestPredicting:
    """Decisions, and the score that is not a probability."""

    def test_it_predicts_zero_or_one(self) -> None:
        predicted = np.asarray(fitted().predict(SURROUNDED_CLASS.input_features))

        assert set(np.unique(predicted)).issubset({0.0, 1.0})

    def test_decision_values_are_signed(self) -> None:
        """Positive is the +1 side, and both signs appear on a real fit."""
        values = fitted().decision_values(SURROUNDED_CLASS.input_features)

        assert values.min() < 0.0 < values.max()

    def test_the_prediction_is_the_sign_of_the_decision(self) -> None:
        model = fitted()
        values = model.decision_values(SURROUNDED_CLASS.input_features)
        predicted = np.asarray(model.predict(SURROUNDED_CLASS.input_features))

        assert np.array_equal(predicted, np.where(values >= 0.0, 1.0, 0.0))

    def test_the_score_is_bounded_but_is_not_calibrated(self) -> None:
        """The frame needs a bounded number; nothing here makes it a probability.

        It is monotonic in the decision value, so ranking is meaningful and
        0.9 still does not mean nine times in ten.

        Monotonicity is asserted pairwise rather than through ``argsort``. On
        this fixture the rows are symmetric, so many decision values are
        exactly equal and their order within a tie is arbitrary -- comparing
        two ``argsort`` results would be asserting how the sort broke ties
        rather than that the mapping preserves order.
        """
        model = fitted()
        scores = np.asarray(model.predict_probability(SURROUNDED_CLASS.input_features))
        values = model.decision_values(SURROUNDED_CLASS.input_features)

        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

        order = np.argsort(values)
        assert np.all(np.diff(scores[order]) >= -1e-12)

    def test_column_order_does_not_matter(self) -> None:
        model = fitted()
        first, second = SURROUNDED_CLASS.input_features

        assert np.allclose(
            model.decision_values([first, second]),
            model.decision_values([second, first]),
        )


class TestTheIntercept:
    """The absorbed constant, which for a while existed only in prose.

    The module docstring justifies dropping the dual's equality constraint by
    absorbing the intercept -- adding a constant to every kernel value, which
    is appending an always-one feature. The first implementation never added
    it, so the boundary was forced through the origin in the implied space:
    with a linear kernel on x = [1, 2, 3, 4], y = [0, 0, 1, 1] -- trivially
    separable with any offset line -- it predicted all ones and scored 0.5.
    """

    def test_a_separable_offset_boundary_is_found(self) -> None:
        """One feature, split at 2.5. No through-origin line can do this.

        Capacity 10 rather than the default 1, because the absorbed offset is
        shrunk like any other weight -- a documented cost of the trick -- and
        four points at C=1 cannot push it to the midpoint.
        """
        model = SupportVectorClassifier(
            kernel=LinearKernel(), capacity=10.0, learning_rate=0.01, max_epochs=20000
        ).fit(
            [Feature("first", [1.0, 2.0, 3.0, 4.0])],
            Feature("outcome", [0.0, 0.0, 1.0, 1.0]),
        )

        predicted = np.asarray(model.predict([Feature("first", [1.0, 2.0, 3.0, 4.0])]))

        assert list(predicted) == [0.0, 0.0, 1.0, 1.0]

    def test_the_offset_shifts_the_decision_values(self) -> None:
        """Through the origin, the decision at x=0 would be exactly zero."""
        model = SupportVectorClassifier(
            kernel=LinearKernel(), capacity=10.0, learning_rate=0.01, max_epochs=20000
        ).fit(
            [Feature("first", [1.0, 2.0, 3.0, 4.0])],
            Feature("outcome", [0.0, 0.0, 1.0, 1.0]),
        )

        at_zero = model.decision_values([Feature("first", [0.0])])

        assert float(at_zero[0]) < 0.0


class TestTheValueObjects:
    """What a support vector is allowed to be."""

    def test_a_zero_multiplier_is_not_a_support_vector(self) -> None:
        with pytest.raises(InvalidValuesError):
            SupportVector(0, 0.0, 1.0)

    def test_a_label_outside_the_signed_encoding_is_rejected(self) -> None:
        """The dual is written in -1/+1, not 0/1."""
        with pytest.raises(InvalidValuesError):
            SupportVector(0, 0.5, 0.0)

    def test_a_repeated_training_row_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            SupportVectors(
                [SupportVector(3, 0.5, 1.0), SupportVector(3, 0.2, -1.0)], 10
            )

    def test_it_reports_its_share_of_the_training_set(self) -> None:
        vectors = SupportVectors(
            [SupportVector(0, 0.5, 1.0), SupportVector(4, 0.2, -1.0)], 10
        )

        assert vectors.share_of_training_rows == pytest.approx(0.2)

    def test_a_multiplier_at_the_cap_is_recognised(self) -> None:
        """Those are the rows the soft margin gave up on."""
        assert SupportVector(0, 1.0, 1.0).is_at_the_cap(1.0)
        assert not SupportVector(0, 0.5, 1.0).is_at_the_cap(1.0)


class TestWhatItRefuses:
    """Guards, each from the MLLibError hierarchy."""

    def test_reading_the_vectors_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = SupportVectorClassifier().support_vectors

    def test_predicting_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            SupportVectorClassifier().predict(SURROUNDED_CLASS.input_features)

    def test_a_target_with_one_class_is_rejected(self) -> None:
        """There is no corridor to widen between a class and nothing."""
        with pytest.raises(SingleClassError):
            SupportVectorClassifier().fit(
                [Feature("first", [1.0, 2.0, 3.0]), Feature("second", [1.0, 1.0, 1.0])],
                Feature("outcome", [1.0, 1.0, 1.0]),
            )

    def test_a_non_binary_target_is_rejected(self) -> None:
        """A margin between two classes; a third cannot be silently recoded.

        The first version checked only that both 0 and 1 appeared, so a
        three-class target passed and every 2 was folded into class 1 by the
        signed-label conversion -- a model that fits, predicts, and answers a
        question nobody asked.
        """
        with pytest.raises(NonBinaryLabelsError):
            SupportVectorClassifier().fit(
                [Feature("first", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                Feature("outcome", [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            )

    def test_a_non_positive_capacity_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SupportVectorClassifier(capacity=0.0)

    def test_predicting_without_every_fitted_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().predict([SURROUNDED_CLASS.input_features[0]])
