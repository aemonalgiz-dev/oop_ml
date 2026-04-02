"""Spec for the loss and the backward pass -- red until three bodies land.

The one that matters is the gradient check, and it is worth understanding why
it is the only test here that could not be fooled.

Backpropagation has a nasty property: almost every wrong version still runs.
Transpose a matrix and the shapes may still line up. Forget the activation's
slope and every number is finite and plausible. Drop the row averaging and
training still descends, just at a different rate. None of that raises, and
none of it shows up in a shape test.

So the oracle is the definition of a derivative itself. Nudge one weight by a
hair, measure how the loss actually moved, and compare against what the
backward pass claims. That number is computed from forward passes alone and
knows nothing about how the gradient was derived, so a gradient that agrees
with it to eight decimal places is right for the same reason a proof is right.

It is also how every serious implementation is checked, and it is the reason
this file is worth reading before writing the bodies.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import ShapeMismatchError
from oop_ml.core.network.activation import HyperbolicTangent, Identity
from oop_ml.core.network.gradient import BackwardPass, LayerGradient
from oop_ml.core.network.layer import DenseLayer
from oop_ml.core.network.loss import LossMeasurement, SoftmaxCrossEntropy
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.stack import LayerStack


def small_network(seed: int = 0) -> LayerStack:
    """Four inputs, a three-wide bent middle, three classes.

    The website's shape in miniature, and small enough that a gradient check
    over every parameter finishes instantly.
    """
    generator = np.random.default_rng(seed)
    hidden = DenseLayer(
        [
            Neuron(
                generator.normal(size=4) * 0.5,
                bias=float(generator.normal()) * 0.5,
                activation=HyperbolicTangent(),
            )
            for _ in range(3)
        ]
    )
    output = DenseLayer(
        [
            Neuron(
                generator.normal(size=3) * 0.5,
                bias=float(generator.normal()) * 0.5,
                activation=Identity(),
            )
            for _ in range(3)
        ]
    )
    return LayerStack([hidden, output])


def dense_at(stack: LayerStack, position: int) -> DenseLayer:
    """The layer at ``position``, narrowed to the dense layer this test built.

    A stack holds ``Layer`` rather than ``DenseLayer``, and has done since
    convolution and pooling arrived, so indexing one yields something with no
    weight matrix as far as the type checker is concerned. Only a dense layer
    has weights to nudge, and every stack in this file is built out of them, so
    the narrowing is a fact being stated rather than a hope. Saying it here once
    keeps the assertion out of the finite-difference loops, where it would
    obscure the arithmetic being checked.
    """
    layer = stack[position]
    assert isinstance(layer, DenseLayer)
    return layer


def dense_layers(stack: LayerStack) -> list[DenseLayer]:
    """Every layer in order, narrowed the same way."""
    return [dense_at(stack, position) for position in range(len(stack))]


def sample_batch(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    rows = generator.normal(size=(6, 4))
    positions = generator.integers(0, 3, size=6)
    return rows, SoftmaxCrossEntropy.one_hot(positions, 3)


class TestSoftmaxCrossEntropy:
    def test_a_confident_right_answer_costs_almost_nothing(self) -> None:
        logits = np.array([[10.0, 0.0, 0.0]])
        targets = np.array([[1.0, 0.0, 0.0]])

        assert SoftmaxCrossEntropy().measure(logits, targets).value < 1e-4

    def test_a_confident_wrong_answer_costs_a_great_deal(self) -> None:
        logits = np.array([[10.0, 0.0, 0.0]])
        targets = np.array([[0.0, 0.0, 1.0]])

        assert SoftmaxCrossEntropy().measure(logits, targets).value > 9.0

    def test_an_undecided_answer_costs_the_logarithm_of_the_class_count(
        self,
    ) -> None:
        """Three equal logits give each class 1/3, and -log(1/3) is 1.0986."""
        logits = np.zeros((1, 3))
        targets = np.array([[1.0, 0.0, 0.0]])

        measurement = SoftmaxCrossEntropy().measure(logits, targets)

        assert measurement.value == pytest.approx(float(np.log(3.0)), abs=1e-9)

    def test_it_averages_over_rows_rather_than_summing(self) -> None:
        """Otherwise the step size would secretly depend on the batch size."""
        logits = np.zeros((4, 3))
        targets = SoftmaxCrossEntropy.one_hot(np.zeros(4), 3)

        assert SoftmaxCrossEntropy().measure(logits, targets).value == pytest.approx(
            float(np.log(3.0)), abs=1e-9
        )

    def test_a_confidently_wrong_row_stays_finite(self) -> None:
        """``log(0)`` is ``-inf``; the floor is what stops it poisoning the mean."""
        logits = np.array([[-800.0, 800.0]])
        targets = np.array([[1.0, 0.0]])

        measurement = SoftmaxCrossEntropy().measure(logits, targets)

        assert np.isfinite(measurement.value)
        assert np.all(np.isfinite(measurement.gradient))

    def test_its_gradient_matches_a_finite_difference(self) -> None:
        """The oracle is the definition of a derivative, not the formula."""
        generator = np.random.default_rng(3)
        logits = generator.normal(size=(4, 3))
        targets = SoftmaxCrossEntropy.one_hot(generator.integers(0, 3, 4), 3)
        loss = SoftmaxCrossEntropy()

        claimed = loss.measure(logits, targets).gradient
        step = 1e-6
        measured = np.empty_like(logits)
        for row in range(logits.shape[0]):
            for column in range(logits.shape[1]):
                up, down = logits.copy(), logits.copy()
                up[row, column] += step
                down[row, column] -= step
                measured[row, column] = (
                    loss.measure(up, targets).value - loss.measure(down, targets).value
                ) / (2.0 * step)

        assert np.allclose(claimed, measured, atol=1e-8)

    def test_mismatched_blocks_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            SoftmaxCrossEntropy().measure(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_one_hot_puts_a_single_one_in_each_row(self) -> None:
        block = SoftmaxCrossEntropy.one_hot(np.array([0, 2, 1]), 3)

        assert np.allclose(block, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])


class TestTheGradientCheck:
    """Nudge each parameter, watch the loss, compare against the claim.

    This is the test the whole backward pass exists to pass. It knows nothing
    about the chain rule; it only runs the network forward many times.
    """

    def numeric_gradient(
        self, stack: LayerStack, rows: np.ndarray, targets: np.ndarray
    ) -> list[np.ndarray]:
        """How the loss really moves when each weight is nudged."""
        loss = SoftmaxCrossEntropy()
        step = 1e-6

        def loss_of(candidate: LayerStack) -> float:
            answers = candidate.respond_to(rows).outputs
            return loss.measure(answers, targets).value

        measured = []
        for index, layer in enumerate(dense_layers(stack)):
            block = np.empty_like(layer.weight_matrix)
            for neuron in range(layer.weight_matrix.shape[0]):
                for weight in range(layer.weight_matrix.shape[1]):
                    nudged = []
                    for direction in (+step, -step):
                        weights = np.array(layer.weight_matrix, copy=True)
                        weights[neuron, weight] += direction
                        rebuilt = list(stack)
                        rebuilt[index] = layer.with_parameters(
                            weights, layer.bias_vector
                        )
                        nudged.append(loss_of(LayerStack(rebuilt)))
                    block[neuron, weight] = (nudged[0] - nudged[1]) / (2.0 * step)
            measured.append(block)
        return measured

    def test_every_weight_gradient_matches_a_finite_difference(self) -> None:
        stack = small_network()
        rows, targets = sample_batch()

        claimed = stack.backward_pass(rows, targets, SoftmaxCrossEntropy())
        measured = self.numeric_gradient(stack, rows, targets)

        for gradient, expected in zip(claimed, measured, strict=True):
            assert gradient is not None
            assert np.allclose(gradient.weights, expected, atol=1e-7)

    def test_every_bias_gradient_matches_a_finite_difference(self) -> None:
        stack = small_network()
        rows, targets = sample_batch()
        loss = SoftmaxCrossEntropy()
        step = 1e-6

        claimed = stack.backward_pass(rows, targets, loss)

        for index, (layer, gradient) in enumerate(
            zip(dense_layers(stack), claimed, strict=True)
        ):
            assert gradient is not None
            measured = np.empty_like(layer.bias_vector)
            for neuron in range(layer.bias_vector.size):
                moved = []
                for direction in (+step, -step):
                    biases = np.array(layer.bias_vector, copy=True)
                    biases[neuron] += direction
                    rebuilt = list(stack)
                    rebuilt[index] = layer.with_parameters(layer.weight_matrix, biases)
                    answers = LayerStack(rebuilt).respond_to(rows).outputs
                    moved.append(loss.measure(answers, targets).value)
                measured[neuron] = (moved[0] - moved[1]) / (2.0 * step)

            assert np.allclose(gradient.biases, measured, atol=1e-7)

    def test_the_check_also_holds_on_a_three_layer_network(self) -> None:
        """Blame has to travel through a middle layer, not just one join."""
        generator = np.random.default_rng(7)

        def bent(n_inputs: int, n_neurons: int, bend: object) -> DenseLayer:
            return DenseLayer(
                [
                    Neuron(
                        generator.normal(size=n_inputs) * 0.5,
                        bias=float(generator.normal()) * 0.5,
                        activation=bend,  # type: ignore[arg-type]
                    )
                    for _ in range(n_neurons)
                ]
            )

        stack = LayerStack(
            [
                bent(4, 5, HyperbolicTangent()),
                bent(5, 3, HyperbolicTangent()),
                bent(3, 3, Identity()),
            ]
        )
        rows, targets = sample_batch()

        claimed = stack.backward_pass(rows, targets, SoftmaxCrossEntropy())
        measured = self.numeric_gradient(stack, rows, targets)

        for gradient, expected in zip(claimed, measured, strict=True):
            assert gradient is not None
            assert np.allclose(gradient.weights, expected, atol=1e-7)


class TestBackwardPassShape:
    def test_one_gradient_per_layer_in_layer_order(self) -> None:
        stack = small_network()
        rows, targets = sample_batch()

        backward = stack.backward_pass(rows, targets, SoftmaxCrossEntropy())

        assert len(backward) == 2
        for layer, gradient in zip(dense_layers(stack), backward, strict=True):
            assert gradient is not None
            assert gradient.weights.shape == layer.weight_matrix.shape
            assert gradient.biases.shape == layer.bias_vector.shape

    def test_it_carries_the_loss_it_measured(self) -> None:
        stack = small_network()
        rows, targets = sample_batch()

        backward = stack.backward_pass(rows, targets, SoftmaxCrossEntropy())
        directly = SoftmaxCrossEntropy().measure(
            stack.respond_to(rows).outputs, targets
        )

        assert backward.loss == pytest.approx(directly.value)

    def test_targets_of_the_wrong_width_are_refused(self) -> None:
        stack = small_network()
        rows, _ = sample_batch()

        with pytest.raises(ShapeMismatchError):
            stack.backward_pass(rows, np.zeros((6, 5)), SoftmaxCrossEntropy())


class TestStepping:
    def test_a_step_moves_against_the_slope(self) -> None:
        stack = small_network()
        rows, targets = sample_batch()
        backward = stack.backward_pass(rows, targets, SoftmaxCrossEntropy())

        stepped = stack.stepped_by(backward, learning_rate=0.1)

        for before, gradient, after in zip(
            dense_layers(stack), backward, dense_layers(stepped), strict=True
        ):
            assert gradient is not None
            assert np.allclose(
                after.weight_matrix,
                before.weight_matrix - 0.1 * gradient.weights,
            )

    def test_a_step_lowers_the_loss(self) -> None:
        """The point of the whole exercise, on one small step."""
        stack = small_network()
        rows, targets = sample_batch()
        loss = SoftmaxCrossEntropy()

        before = stack.backward_pass(rows, targets, loss).loss
        after = (
            stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.1
            )
            .backward_pass(rows, targets, loss)
            .loss
        )

        assert after < before

    def test_the_original_stack_is_untouched(self) -> None:
        """Training does not mutate, which is what keeps the caches honest."""
        stack = small_network()
        rows, targets = sample_batch()
        before = np.array(dense_at(stack, 0).weight_matrix, copy=True)

        stack.stepped_by(
            stack.backward_pass(rows, targets, SoftmaxCrossEntropy()),
            learning_rate=0.5,
        )

        assert np.allclose(dense_at(stack, 0).weight_matrix, before)

    def test_repeated_steps_train(self) -> None:
        """Twenty steps on one batch have to visibly reduce the loss."""
        stack = small_network()
        rows, targets = sample_batch()
        loss = SoftmaxCrossEntropy()

        first = stack.backward_pass(rows, targets, loss).loss
        for _ in range(20):
            stack = stack.stepped_by(
                stack.backward_pass(rows, targets, loss), learning_rate=0.5
            )
        last = stack.backward_pass(rows, targets, loss).loss

        assert last < first * 0.9


class TestGradientValueObjects:
    def test_a_layer_gradient_refuses_disagreeing_blocks(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerGradient(weights=np.zeros((3, 4)), biases=np.zeros(2))

    def test_a_backward_pass_refuses_to_be_empty(self) -> None:
        from oop_ml.core.exceptions import EmptyValuesError

        with pytest.raises(EmptyValuesError):
            BackwardPass(loss=1.0, gradients=[])

    def test_the_largest_movement_is_the_biggest_slope_anywhere(self) -> None:
        first = LayerGradient(weights=np.array([[1.0, -5.0]]), biases=np.array([2.0]))
        second = LayerGradient(weights=np.array([[0.5]]), biases=np.array([-0.25]))

        assert BackwardPass(loss=0.0, gradients=[first, second]).largest_movement == 5.0

    def test_a_loss_measurement_freezes_its_gradient(self) -> None:
        measurement = LossMeasurement(value=1.0, gradient=np.zeros((2, 2)))

        with pytest.raises(ValueError):
            measurement.gradient[0][0] = 1.0


class TestGradientsAreAddressableByLayer:
    """Gradient k belongs to layer k, so asking for it by number is fair."""

    def test_a_backward_pass_answers_by_position(self) -> None:
        stack = small_network()
        rows, targets = sample_batch()

        backward = stack.backward_pass(rows, targets, SoftmaxCrossEntropy())

        for index in range(len(backward)):
            gradient = backward[index]
            assert gradient is not None
            assert gradient.weights.shape == dense_at(stack, index).weight_matrix.shape

    def test_it_counts_from_the_end_too(self) -> None:
        stack = small_network()
        rows, targets = sample_batch()

        backward = stack.backward_pass(rows, targets, SoftmaxCrossEntropy())

        last = backward[-1]
        assert last is not None
        assert last.weights.shape == dense_at(stack, -1).weight_matrix.shape

    def test_it_still_hands_out_no_container(self) -> None:
        stack = small_network()
        rows, targets = sample_batch()

        backward = stack.backward_pass(rows, targets, SoftmaxCrossEntropy())

        assert not hasattr(backward, "gradients")
