"""Spec for the five losses -- red until each ``_measure_aligned`` lands.

Two things here are worth reading before writing any of them.

The first is the gradient check, run over every loss by the same
parametrized test. Its oracle is the definition of a derivative: nudge one raw
output, measure how the loss actually moved, compare against the claim. It is
built out of forward evaluations alone and knows nothing about how any gradient
was derived, so a formula that agrees with it is right rather than merely
plausible. Absolute error and Huber are checked away from their kinks, where a
derivative genuinely exists.

The second is the shared-gradient test. Three of the five have *the same*
gradient, ``(prediction - truth) / n_rows``, and asserting it in one place
records the canonical-link fact the module docstring explains. A version of
softmax cross-entropy that got its Jacobian wrong would pass a shape test, pass
a value test, and fail this.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.logistic import stable_logistic, stable_softmax
from oop_ml.core.network.loss import (
    AbsoluteError,
    BinaryCrossEntropy,
    HuberError,
    Loss,
    LossMeasurement,
    SoftmaxCrossEntropy,
    SquaredError,
)

REGRESSION_LOSSES = [SquaredError(), AbsoluteError(), HuberError(threshold=1.0)]
EVERY_LOSS = [
    *REGRESSION_LOSSES,
    BinaryCrossEntropy(),
    SoftmaxCrossEntropy(),
]


def sample_for(loss: Loss) -> tuple[np.ndarray, np.ndarray]:
    """Raw outputs and targets of the shape that loss expects.

    Deliberately away from every kink, so a finite difference is measuring a
    derivative that exists.
    """
    generator = np.random.default_rng(5)
    if isinstance(loss, SoftmaxCrossEntropy):
        outputs = generator.normal(size=(4, 3))
        return outputs, SoftmaxCrossEntropy.one_hot(generator.integers(0, 3, 4), 3)
    if isinstance(loss, BinaryCrossEntropy):
        outputs = generator.normal(size=(4, 1))
        return outputs, (generator.random((4, 1)) > 0.5).astype(float)
    outputs = generator.normal(size=(4, 2)) * 2.0
    return outputs, generator.normal(size=(4, 2)) * 2.0


class TestEveryLoss:
    @pytest.mark.parametrize("loss", EVERY_LOSS)
    def test_its_gradient_matches_a_finite_difference(self, loss: Loss) -> None:
        """The oracle is the definition of a derivative, not any formula."""
        outputs, targets = sample_for(loss)
        claimed = loss.measure(outputs, targets).gradient

        step = 1e-6
        measured = np.empty_like(outputs)
        for row in range(outputs.shape[0]):
            for column in range(outputs.shape[1]):
                up, down = outputs.copy(), outputs.copy()
                up[row, column] += step
                down[row, column] -= step
                measured[row, column] = (
                    loss.measure(up, targets).value - loss.measure(down, targets).value
                ) / (2.0 * step)

        assert np.allclose(claimed, measured, atol=1e-7)

    @pytest.mark.parametrize("loss", EVERY_LOSS)
    def test_its_gradient_is_shaped_like_its_outputs(self, loss: Loss) -> None:
        outputs, targets = sample_for(loss)

        assert loss.measure(outputs, targets).gradient.shape == outputs.shape

    @pytest.mark.parametrize("loss", EVERY_LOSS)
    def test_it_answers_a_python_float(self, loss: Loss) -> None:
        outputs, targets = sample_for(loss)

        assert type(loss.measure(outputs, targets).value) is float

    @pytest.mark.parametrize("loss", EVERY_LOSS)
    def test_it_is_never_negative(self, loss: Loss) -> None:
        outputs, targets = sample_for(loss)

        assert loss.measure(outputs, targets).value >= 0.0

    @pytest.mark.parametrize("loss", EVERY_LOSS)
    def test_mismatched_blocks_are_refused(self, loss: Loss) -> None:
        with pytest.raises(ShapeMismatchError):
            loss.measure(np.zeros((2, 3)), np.zeros((2, 4)))

    @pytest.mark.parametrize("loss", EVERY_LOSS)
    def test_it_describes_itself(self, loss: Loss) -> None:
        assert isinstance(loss.description, str)
        assert loss.description.strip()

    @pytest.mark.parametrize("loss", REGRESSION_LOSSES)
    def test_a_perfect_answer_costs_nothing(self, loss: Loss) -> None:
        truth = np.array([[1.0, -2.0], [3.0, 0.5]])

        assert loss.measure(truth, truth).value == pytest.approx(0.0)

    @pytest.mark.parametrize("loss", EVERY_LOSS)
    def test_it_divides_by_rows_rather_than_summing(self, loss: Loss) -> None:
        """Doubling the batch with identical rows must not double the loss."""
        outputs, targets = sample_for(loss)
        doubled = loss.measure(
            np.vstack([outputs, outputs]), np.vstack([targets, targets])
        )

        assert doubled.value == pytest.approx(
            loss.measure(outputs, targets).value, rel=1e-9
        )


class TestTheSharedGradient:
    """Three different losses, one derivative, which is the canonical link."""

    def test_squared_error_is_prediction_minus_truth(self) -> None:
        outputs = np.array([[2.0, -1.0], [0.5, 3.0]])
        targets = np.array([[1.0, 1.0], [0.0, 4.0]])

        gradient = SquaredError().measure(outputs, targets).gradient

        assert np.allclose(gradient, (outputs - targets) / 2)

    def test_binary_log_loss_is_squashed_prediction_minus_truth(self) -> None:
        outputs = np.array([[0.4], [-1.2], [3.0], [0.0]])
        targets = np.array([[1.0], [0.0], [1.0], [0.0]])

        gradient = BinaryCrossEntropy().measure(outputs, targets).gradient

        assert np.allclose(gradient, (stable_logistic(outputs) - targets) / 4)

    def test_softmax_log_loss_is_squashed_prediction_minus_truth(self) -> None:
        outputs = np.array([[1.0, 0.0, -1.0], [2.0, 2.0, 0.0]])
        targets = SoftmaxCrossEntropy.one_hot(np.array([0, 2]), 3)

        gradient = SoftmaxCrossEntropy().measure(outputs, targets).gradient

        assert np.allclose(gradient, (stable_softmax(outputs) - targets) / 2)

    def test_absolute_error_deliberately_breaks_the_pattern(self) -> None:
        """Its pull does not grow with the miss, which is the whole point."""
        outputs = np.array([[1.0, 1.0, 1.0, 100.0]])
        targets = np.zeros((1, 4))

        gradient = AbsoluteError().measure(outputs, targets).gradient

        assert np.allclose(gradient, [[1.0, 1.0, 1.0, 1.0]])

    def test_and_squared_error_lets_the_outlier_dominate(self) -> None:
        outputs = np.array([[1.0, 1.0, 1.0, 100.0]])
        targets = np.zeros((1, 4))

        gradient = SquaredError().measure(outputs, targets).gradient

        assert np.allclose(gradient, [[1.0, 1.0, 1.0, 100.0]])


class TestSquaredError:
    def test_the_half_is_in_the_value(self) -> None:
        """A miss of 2 across one row costs 2, not 4."""
        assert SquaredError().measure(
            np.array([[2.0]]), np.array([[0.0]])
        ).value == pytest.approx(2.0)

    def test_it_sums_over_outputs_and_divides_by_rows(self) -> None:
        outputs = np.array([[3.0, 4.0]])
        targets = np.zeros((1, 2))

        assert SquaredError().measure(outputs, targets).value == pytest.approx(12.5)


class TestAbsoluteError:
    def test_it_is_the_mean_absolute_miss_per_row(self) -> None:
        outputs = np.array([[3.0, -4.0], [1.0, 0.0]])
        targets = np.zeros((2, 2))

        assert AbsoluteError().measure(outputs, targets).value == pytest.approx(4.0)

    def test_an_exact_tie_pulls_in_neither_direction(self) -> None:
        """No derivative exists at the kink; this library answers zero."""
        gradient = AbsoluteError().measure(np.zeros((1, 1)), np.zeros((1, 1))).gradient

        assert float(gradient[0][0]) == 0.0


class TestHuberError:
    def test_inside_the_threshold_it_is_squared_error(self) -> None:
        outputs = np.array([[0.5]])
        targets = np.zeros((1, 1))

        assert HuberError(threshold=1.0).measure(
            outputs, targets
        ).value == pytest.approx(0.125)

    def test_outside_the_threshold_the_pull_stops_growing(self) -> None:
        far = HuberError(threshold=1.0).measure(np.array([[100.0]]), np.zeros((1, 1)))

        assert float(far.gradient[0][0]) == pytest.approx(1.0)

    def test_the_two_pieces_meet_at_the_threshold(self) -> None:
        """The ``- 0.5*d`` term exists only to make this true."""
        loss = HuberError(threshold=2.0)
        below = loss.measure(np.array([[2.0 - 1e-9]]), np.zeros((1, 1))).value
        above = loss.measure(np.array([[2.0 + 1e-9]]), np.zeros((1, 1))).value

        assert below == pytest.approx(2.0, abs=1e-6)
        assert above == pytest.approx(below, abs=1e-6)

    def test_its_slope_is_continuous_at_the_threshold_too(self) -> None:
        loss = HuberError(threshold=2.0)
        below = loss.measure(np.array([[2.0 - 1e-9]]), np.zeros((1, 1))).gradient
        above = loss.measure(np.array([[2.0 + 1e-9]]), np.zeros((1, 1))).gradient

        assert float(below[0][0]) == pytest.approx(2.0, abs=1e-6)
        assert float(above[0][0]) == pytest.approx(2.0, abs=1e-6)

    def test_a_large_threshold_behaves_like_squared_error(self) -> None:
        outputs = np.array([[1.5, -0.5]])
        targets = np.zeros((1, 2))

        assert HuberError(threshold=1000.0).measure(
            outputs, targets
        ).value == pytest.approx(SquaredError().measure(outputs, targets).value)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_threshold_that_is_not_positive_and_finite_is_refused(
        self, bad: float
    ) -> None:
        with pytest.raises(InvalidValuesError):
            HuberError(threshold=bad)

    def test_two_of_the_same_threshold_are_equal(self) -> None:
        assert HuberError(threshold=2.0) == HuberError(threshold=2.0)
        assert HuberError(threshold=2.0) != HuberError(threshold=3.0)
        assert HuberError(threshold=2.0) != SquaredError()


class TestBinaryCrossEntropy:
    def test_an_undecided_answer_costs_the_logarithm_of_two(self) -> None:
        assert BinaryCrossEntropy().measure(
            np.zeros((1, 1)), np.array([[1.0]])
        ).value == pytest.approx(float(np.log(2.0)), abs=1e-9)

    def test_a_confident_right_answer_costs_almost_nothing(self) -> None:
        assert (
            BinaryCrossEntropy().measure(np.array([[20.0]]), np.array([[1.0]])).value
            < 1e-8
        )

    def test_a_confidently_wrong_row_stays_finite(self) -> None:
        """The sigmoid saturates and ``log(0)`` would be ``-inf``."""
        measurement = BinaryCrossEntropy().measure(
            np.array([[-800.0]]), np.array([[1.0]])
        )

        assert np.isfinite(measurement.value)
        assert np.all(np.isfinite(measurement.gradient))


class TestSoftmaxCrossEntropy:
    def test_an_undecided_answer_costs_the_logarithm_of_the_class_count(
        self,
    ) -> None:
        assert SoftmaxCrossEntropy().measure(
            np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]])
        ).value == pytest.approx(float(np.log(3.0)), abs=1e-9)

    def test_a_confident_wrong_answer_costs_a_great_deal(self) -> None:
        assert (
            SoftmaxCrossEntropy()
            .measure(np.array([[10.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
            .value
            > 9.0
        )

    def test_a_confidently_wrong_row_stays_finite(self) -> None:
        measurement = SoftmaxCrossEntropy().measure(
            np.array([[-800.0, 800.0]]), np.array([[1.0, 0.0]])
        )

        assert np.isfinite(measurement.value)

    def test_one_hot_puts_a_single_one_in_each_row(self) -> None:
        block = SoftmaxCrossEntropy.one_hot(np.array([0, 2, 1]), 3)

        assert np.allclose(block, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])


class TestLossMeasurement:
    def test_it_freezes_its_gradient(self) -> None:
        measurement = LossMeasurement(value=1.0, gradient=np.zeros((2, 2)))

        with pytest.raises(ValueError):
            measurement.gradient[0][0] = 1.0

    def test_mutating_the_source_does_not_reach_it(self) -> None:
        gradient = np.zeros((2, 2))
        measurement = LossMeasurement(value=1.0, gradient=gradient)

        gradient[0][0] = 99.0

        assert measurement.gradient[0][0] == 0.0


class TestTheClipIsSymmetric:
    """A mirrored mistake must cost the same, which a lopsided clip breaks.

    ``tiny`` is the tempting floor and it cannot be paired: ``1 - tiny`` is not
    representable and rounds to 1.0, so the upper bound is forced to
    ``1 - eps``. Using both left the same error costing 708.4 one way and 36.0
    the other. The gradients stayed mirror images throughout, since they read
    the unclipped probabilities, so nothing but the reported loss was wrong,
    and the reported loss is exactly what a training curve plots.
    """

    def test_a_mirrored_mistake_costs_the_same(self) -> None:
        loss = BinaryCrossEntropy()

        toward_zero = loss.measure(np.array([[-800.0]]), np.array([[1.0]])).value
        toward_one = loss.measure(np.array([[800.0]]), np.array([[0.0]])).value

        assert toward_zero == pytest.approx(toward_one)

    def test_a_mirrored_mistake_pulls_the_same(self) -> None:
        loss = BinaryCrossEntropy()

        toward_zero = loss.measure(np.array([[-800.0]]), np.array([[1.0]])).gradient
        toward_one = loss.measure(np.array([[800.0]]), np.array([[0.0]])).gradient

        assert float(toward_zero[0][0]) == pytest.approx(-float(toward_one[0][0]))
