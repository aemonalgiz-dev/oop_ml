"""Spec for the skip connection, and the one claim it exists to make.

The claim is about the way back rather than the way up. What this layer passes
down is what the inner layer passes down *plus* what arrived, so the slope has a
route home that is multiplied by one instead of by the inner layer's weights. A
version that forgot the addition would still run, still descend, and would be
exactly the deep chain this layer was introduced to rescue, so the addition is
asserted directly and then again by finite difference.

The finite difference is the load-bearing one, for the reason every layer in
this package gives: it runs the layer forward many times and knows nothing about
how the backward step was written. It is taken on both halves, the inner layer's
weights and the block that arrived, because only the second sees the skip.

The refusal at construction is the other half of the module. A residual only
means anything when the inner layer answers with what it reads, and that is an
integer comparison available before any data exists, so it is made there rather
than discovered in an addition that happens to broadcast. ``Flatten((2, 2))``
reads ``(2, 2)`` and answers ``(4,)``, which holds the same four numbers, and is
the case a count comparison would have let through.

``SpyLayer`` is here for two things a real layer cannot show. It records which
:class:`PassPurpose` reached it, so the forwarding of that keyword is checked
rather than assumed, and it states a coercion rule of its own, which is how the
delegation of ``_read_block`` is asserted. No layer in the library today both
overrides that method and answers with what it reads, so the test double is the
only way to hold the wrapper to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import HyperbolicTangent
from oop_ml.core.network.flatten import Flatten
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import DenseLayer, Layer, LayerResponse
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.residual import Residual, ResidualResponse
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray

#: The step a central difference takes on either side.
NUDGE = 1e-6

WIDTH = 3


def spread_weights() -> np.ndarray:
    """A weight block no two entries of which coincide."""
    return (np.arange(WIDTH * WIDTH, dtype=np.float64) / 9.0 - 0.4).reshape(
        WIDTH, WIDTH
    )


def spread_biases() -> np.ndarray:
    """Biases no two of which coincide."""
    return np.array([0.1, -0.2, 0.3])


def dense_inner(weights: np.ndarray, biases: np.ndarray) -> DenseLayer:
    """A square dense layer, which is the smallest thing a residual can wrap."""
    return DenseLayer(
        [
            Neuron(row, bias=float(bias), activation=HyperbolicTangent())
            for row, bias in zip(weights, biases, strict=True)
        ]
    )


def residual_of(weights: np.ndarray, biases: np.ndarray) -> Residual:
    """The fixture layer, carrying these parameters inside."""
    return Residual(dense_inner(weights, biases))


def rows_block(n_rows: int = 2) -> np.ndarray:
    """Rows of measurements, every value distinct so nothing cancels."""
    total = n_rows * WIDTH
    return (np.arange(total, dtype=np.float64) / 5.0 - 0.6).reshape(n_rows, WIDTH)


def summary_of(layer: Residual, block: np.ndarray, weights: np.ndarray) -> float:
    """A scalar reading of the answer whose slope is ``weights`` exactly."""
    return float(np.sum(weights * layer.respond_to(block).outputs))


def measured_weight_slopes(
    weights: np.ndarray, biases: np.ndarray, block: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every inner weight's slope, by nudging it and rebuilding the residual."""
    measured = np.zeros_like(weights)
    for row in range(weights.shape[0]):
        for column in range(weights.shape[1]):
            raised = np.array(weights, copy=True)
            lowered = np.array(weights, copy=True)
            raised[row, column] += NUDGE
            lowered[row, column] -= NUDGE
            measured[row, column] = (
                summary_of(residual_of(raised, biases), block, arriving)
                - summary_of(residual_of(lowered, biases), block, arriving)
            ) / (2.0 * NUDGE)
    return measured


def measured_input_slopes(
    layer: Residual, block: np.ndarray, arriving: np.ndarray
) -> np.ndarray:
    """Every input value's slope, which is the half that sees the skip."""
    measured = np.zeros_like(block)
    for index in np.ndindex(block.shape):
        raised = np.array(block, copy=True)
        lowered = np.array(block, copy=True)
        raised[index] += NUDGE
        lowered[index] -= NUDGE
        measured[index] = (
            summary_of(layer, raised, arriving) - summary_of(layer, lowered, arriving)
        ) / (2.0 * NUDGE)
    return measured


class SpyLayer(Layer):
    """An identity layer that records what reached it and refuses fractions.

    Two things no layer in the library can currently show a residual doing. It
    keeps every :class:`PassPurpose` it was handed, so the forwarding of that
    keyword is a checked claim, and it overrides ``_read_block`` with a rule of
    its own, which is how the wrapper's delegation of that method is asserted.
    """

    def __init__(self, width: int = WIDTH) -> None:
        self._shape = LayerShape(n_inputs=width, n_outputs=width)
        self.purposes: list[PassPurpose] = []

    @property
    def shape(self) -> LayerShape:
        return self._shape

    def _read_block(self, inputs: object) -> FloatArray:
        block = np.array(inputs, dtype=np.float64, copy=True)
        if not np.array_equal(block, np.floor(block)):
            raise InvalidValuesError("this layer reads whole numbers only")
        return block

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        self.purposes.append(purpose)
        return LayerResponse.already_checked(
            inputs=inputs, scores=inputs, outputs=np.array(inputs, copy=True)
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        return LayerCorrection(passed_down=np.zeros_like(arriving), gradient=None)

    def stepped_by(self, gradient: LayerGradient | None, learning_rate: float) -> Layer:
        return self


class TestConstruction:
    def test_it_wears_the_shape_of_what_it_holds(self) -> None:
        layer = residual_of(spread_weights(), spread_biases())

        assert layer.shape == LayerShape(n_inputs=WIDTH, n_outputs=WIDTH)

    def test_it_moves_no_join(self) -> None:
        layer = residual_of(spread_weights(), spread_biases())

        assert layer.shape.reads == layer.shape.answers

    def test_it_keeps_the_layer_it_was_given(self) -> None:
        inner = dense_inner(spread_weights(), spread_biases())

        assert Residual(inner).inner is inner

    def test_an_inner_layer_that_changes_the_width_is_refused(self) -> None:
        """The addition would broadcast or fail; neither is what was meant."""
        narrower = DenseLayer(
            [Neuron(np.ones(WIDTH), bias=0.0, activation=HyperbolicTangent())]
        )

        with pytest.raises(ShapeMismatchError):
            Residual(narrower)

    def test_an_inner_layer_that_changes_the_arrangement_is_refused(self) -> None:
        """Flatten((2, 2)) holds the same four numbers and is not the same shape."""
        with pytest.raises(ShapeMismatchError):
            Residual(Flatten((2, 2)))

    def test_an_inner_layer_that_flattens_nothing_is_accepted(self) -> None:
        """Flatten(4) reads and answers (4,), so wrapping it is sound if useless."""
        assert Residual(Flatten(4)).shape == LayerShape(n_inputs=4, n_outputs=4)

    def test_it_names_what_it_holds_in_its_repr(self) -> None:
        inner = Flatten(4)

        assert repr(Residual(inner)) == f"Residual(inner={inner!r})"


class TestTheForwardPass:
    def test_it_answers_the_inner_layer_plus_what_it_read(self) -> None:
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        block = rows_block()

        response = layer.respond_to(block)

        inner_outputs = dense_inner(weights, biases).respond_to(block).outputs
        assert np.allclose(response.outputs, inner_outputs + block)

    def test_wrapping_an_identity_doubles_the_block(self) -> None:
        """The clearest reading of what the addition is: x + x."""
        response = Residual(Flatten(4)).respond_to(np.array([[1.0, 2.0, 3.0, 4.0]]))

        assert np.allclose(response.outputs, [[2.0, 4.0, 6.0, 8.0]])

    def test_it_carries_the_inner_response(self) -> None:
        """The backward step needs it and cannot rebuild it from the answer."""
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        block = rows_block()

        response = layer.respond_to(block)

        assert isinstance(response, ResidualResponse)
        assert np.allclose(
            response.inner_response.outputs,
            dense_inner(weights, biases).respond_to(block).outputs,
        )

    def test_it_keeps_the_block_it_read(self) -> None:
        block = rows_block()

        response = residual_of(spread_weights(), spread_biases()).respond_to(block)

        assert np.allclose(response.inputs, block)

    def test_the_answer_is_frozen(self) -> None:
        response = residual_of(spread_weights(), spread_biases()).respond_to(
            rows_block()
        )

        with pytest.raises(ValueError):
            response.outputs[0, 0] = 99.0

    def test_the_purpose_reaches_the_inner_layer(self) -> None:
        spy = SpyLayer()
        layer = Residual(spy)

        layer.respond_to(np.zeros((2, WIDTH)), PassPurpose.TRAINING)

        assert spy.purposes == [PassPurpose.TRAINING]

    def test_the_default_purpose_is_forwarded_too(self) -> None:
        spy = SpyLayer()

        Residual(spy).respond_to(np.zeros((2, WIDTH)))

        assert spy.purposes == [PassPurpose.PREDICTING]

    def test_a_wrongly_shaped_block_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            residual_of(spread_weights(), spread_biases()).respond_to(
                np.zeros((2, WIDTH + 1))
            )

    def test_a_block_of_no_rows_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            residual_of(spread_weights(), spread_biases()).respond_to(
                np.zeros((0, WIDTH))
            )

    def test_a_non_finite_entry_is_refused(self) -> None:
        block = rows_block()
        block[1, 2] = np.inf

        with pytest.raises(InvalidValuesError):
            residual_of(spread_weights(), spread_biases()).respond_to(block)


class TestTheDelegatedCoercion:
    """The wrapper never invents a reading rule; it uses the one inside it."""

    def test_the_inner_layers_refusal_is_the_wrappers_refusal(self) -> None:
        with pytest.raises(InvalidValuesError):
            Residual(SpyLayer()).respond_to(np.array([[0.5, 1.0, 2.0]]))

    def test_what_the_inner_layer_accepts_the_wrapper_accepts(self) -> None:
        response = Residual(SpyLayer()).respond_to(np.array([[1.0, 2.0, 3.0]]))

        assert np.allclose(response.outputs, [[2.0, 4.0, 6.0]])


class TestTheBackwardPass:
    def test_the_gradient_is_the_inner_layers_own(self) -> None:
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        block = rows_block()
        arriving = np.random.default_rng(23).normal(size=(2, WIDTH))
        response = layer.respond_to(block)

        correction = layer.correction_for(response, arriving)

        inner = dense_inner(weights, biases)
        inner_correction = inner.correction_for(inner.respond_to(block), arriving)
        assert correction.gradient is not None
        assert inner_correction.gradient is not None
        assert np.allclose(
            correction.gradient.weights, inner_correction.gradient.weights
        )
        assert np.allclose(correction.gradient.biases, inner_correction.gradient.biases)

    def test_it_adds_the_arriving_block_to_what_the_inner_layer_passes_down(
        self,
    ) -> None:
        """The route multiplied by one, asserted as a number."""
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        block = rows_block()
        arriving = np.random.default_rng(29).normal(size=(2, WIDTH))
        response = layer.respond_to(block)

        correction = layer.correction_for(response, arriving)

        inner = dense_inner(weights, biases)
        inner_correction = inner.correction_for(inner.respond_to(block), arriving)
        assert np.allclose(
            correction.passed_down, inner_correction.passed_down + arriving
        )

    def test_around_a_layer_that_passes_nothing_down_the_skip_is_all_of_it(
        self,
    ) -> None:
        """Which is exactly the point: the slope cannot be extinguished here."""
        spy = SpyLayer()
        layer = Residual(spy)
        response = layer.respond_to(np.zeros((2, WIDTH)))
        arriving = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        correction = layer.correction_for(response, arriving)

        assert np.allclose(correction.passed_down, arriving)

    def test_a_response_from_elsewhere_is_refused(self) -> None:
        """It carries no inner response, and guessing one is not available."""
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        inner = dense_inner(weights, biases)
        borrowed = inner.respond_to(rows_block())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(borrowed, np.ones((2, WIDTH)))

    def test_arriving_slopes_of_the_wrong_width_are_refused(self) -> None:
        layer = residual_of(spread_weights(), spread_biases())
        response = layer.respond_to(rows_block())

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((2, WIDTH + 1)))

    def test_arriving_slopes_for_the_wrong_number_of_rows_are_refused(self) -> None:
        layer = residual_of(spread_weights(), spread_biases())
        response = layer.respond_to(rows_block(2))

        with pytest.raises(ShapeMismatchError):
            layer.correction_for(response, np.ones((3, WIDTH)))

    def test_slopes_that_are_not_numbers_are_refused(self) -> None:
        layer = residual_of(spread_weights(), spread_biases())
        response = layer.respond_to(rows_block())

        with pytest.raises(InvalidValuesError):
            layer.correction_for(response, [[1.0, 2.0], [3.0]])  # type: ignore[arg-type]


class TestTheResponse:
    def test_two_responses_of_the_same_route_are_equal(self) -> None:
        layer = residual_of(spread_weights(), spread_biases())
        block = rows_block()

        assert layer.respond_to(block) == layer.respond_to(block)

    def test_a_plain_response_is_not_one_of_these(self) -> None:
        weights, biases = spread_weights(), spread_biases()
        block = rows_block()
        wrapped = residual_of(weights, biases).respond_to(block)
        plain = dense_inner(weights, biases).respond_to(block)

        assert wrapped != plain
        assert plain != wrapped

    def test_it_says_how_many_rows_went_through(self) -> None:
        response = residual_of(spread_weights(), spread_biases()).respond_to(
            rows_block(3)
        )

        assert repr(response) == "ResidualResponse(n_rows=3)"


class TestTheGradientCheck:
    """Both halves, because only the second one sees the skip."""

    def test_the_inner_weight_slopes_match_a_finite_difference(self) -> None:
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        block = rows_block()
        arriving = np.random.default_rng(31).normal(size=(2, WIDTH))
        response = layer.respond_to(block)

        claimed = layer.correction_for(response, arriving).gradient
        measured = measured_weight_slopes(weights, biases, block, arriving)

        assert claimed is not None
        assert np.allclose(claimed.weights, measured, atol=1e-7)

    def test_the_slopes_passed_down_match_a_finite_difference(self) -> None:
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        block = rows_block()
        arriving = np.random.default_rng(37).normal(size=(2, WIDTH))
        response = layer.respond_to(block)

        claimed = layer.correction_for(response, arriving).passed_down
        measured = measured_input_slopes(layer, block, arriving)

        assert claimed.shape == block.shape
        assert np.allclose(claimed, measured, atol=1e-7)

    def test_dropping_the_skip_would_fail_that_check(self) -> None:
        """What the previous test is actually discriminating against.

        The inner layer's own slopes are what a residual missing its addition
        would report, and on this fixture they differ from the truth by roughly
        one everywhere, since the skip contributes exactly the arriving block.
        """
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        block = rows_block()
        arriving = np.random.default_rng(41).normal(size=(2, WIDTH))
        inner = dense_inner(weights, biases)

        without_skip = inner.correction_for(inner.respond_to(block), arriving)
        measured = measured_input_slopes(layer, block, arriving)

        assert not np.allclose(without_skip.passed_down, measured, atol=1e-7)


class TestStepping:
    def test_a_step_answers_with_a_residual_around_the_stepped_layer(self) -> None:
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        block = rows_block()
        response = layer.respond_to(block)
        correction = layer.correction_for(response, np.ones((2, WIDTH)))

        stepped = layer.stepped_by(correction.gradient, learning_rate=0.1)

        assert isinstance(stepped, Residual)
        assert isinstance(stepped.inner, DenseLayer)
        assert not np.allclose(stepped.inner.weight_matrix, weights)

    def test_a_step_does_not_touch_the_layer_it_came_from(self) -> None:
        weights, biases = spread_weights(), spread_biases()
        layer = residual_of(weights, biases)
        response = layer.respond_to(rows_block())
        correction = layer.correction_for(response, np.ones((2, WIDTH)))

        layer.stepped_by(correction.gradient, learning_rate=0.1)

        inner = layer.inner
        assert isinstance(inner, DenseLayer)
        assert np.allclose(inner.weight_matrix, weights)

    def test_a_step_keeps_the_shape(self) -> None:
        layer = residual_of(spread_weights(), spread_biases())
        response = layer.respond_to(rows_block())
        correction = layer.correction_for(response, np.ones((2, WIDTH)))

        assert (
            layer.stepped_by(correction.gradient, learning_rate=0.1).shape
            == layer.shape
        )

    def test_around_a_layer_with_nothing_to_learn_a_step_changes_nothing(self) -> None:
        layer = Residual(Flatten(4))

        stepped = layer.stepped_by(None, learning_rate=0.1)

        assert isinstance(stepped, Residual)
        assert stepped.inner is layer.inner
