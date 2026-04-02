"""Spec for the stack -- red until ``_response_for`` lands.

Two of these carry the weight.

The first is that a bad chain is refused at construction, with no data anywhere
in the test. That is the claim this whole package was built to make, and until
this class existed the library could only answer the question, not act on it.
The refusal has to name which join broke, because an error that says only "the
shapes disagree" sends a reader to look at every layer.

The second is that a good chain is never re-checked. The interior widths are
settled once, so the test walks a three-layer stack and asserts the answer
equals the layers applied by hand in sequence, which is an oracle built from
what a stack *means* rather than from how the body is likely to be spelled.
"""

import numpy as np
import pytest

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import Identity, RectifiedLinear
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import DenseLayer, Layer, LayerResponse
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import LayerStack, StackResponse
from oop_ml.core.types import FloatArray

TWO_ROWS = np.array([[1.0, 2.0], [3.0, 4.0]])


def widening_layer() -> DenseLayer:
    """Two inputs to three neurons, each answering something distinguishable."""
    return DenseLayer(
        [
            Neuron([1.0, 0.0], bias=0.0, activation=Identity()),
            Neuron([0.0, 1.0], bias=0.0, activation=Identity()),
            Neuron([1.0, 1.0], bias=0.0, activation=Identity()),
        ]
    )


def neuron_counts(stack: LayerStack) -> list[int]:
    """How many neurons each layer holds, for a stack built out of dense layers.

    A stack holds ``Layer`` rather than ``DenseLayer``, and has done since
    convolution and pooling arrived, so a length is not something every layer
    in one has. Every stack in this file is built out of dense layers, so
    narrowing here states a fact rather than assuming one, and it keeps the
    assertion out of the tests that are about ordering.
    """
    counts = []
    for layer in stack:
        assert isinstance(layer, DenseLayer)
        counts.append(len(layer))
    return counts


def narrowing_layer() -> DenseLayer:
    """Three inputs to one neuron, so a two-layer stack is 2 -> 3 -> 1."""
    return DenseLayer([Neuron([1.0, 1.0, 1.0], bias=0.0, activation=Identity())])


class ReshapingLayer(Layer):
    """A layer that only rearranges, so that a stack can be built out of shapes.

    Every other layer in this file is dense, where the arrangement and the
    element count are the same number and therefore cannot be told apart. This
    one exists so that they can be, and it is written here rather than borrowed
    from :class:`~oop_ml.core.network.convolution.Conv2d` on purpose: the claims
    below are about ``LayerStack``, and a stack test that could fail because a
    convolution changed is a test of two things.

    It carries no parameters and applies no bend, so ``scores`` and ``outputs``
    are one block and the backward step is the forward reshape run the other
    way.
    """

    __slots__ = ("_shape",)

    def __init__(self, reads: tuple[int, ...], answers: tuple[int, ...]) -> None:
        self._shape = LayerShape(n_inputs=reads, n_outputs=answers)

    @property
    def shape(self) -> LayerShape:
        return self._shape

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        outputs = inputs.reshape(inputs.shape[0], *self._shape.answers)
        return LayerResponse.already_checked(
            inputs=inputs, scores=outputs, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        block = self._checked_arriving(response, arriving)
        return LayerCorrection(
            passed_down=block.reshape(block.shape[0], *self._shape.reads),
            gradient=None,
        )

    def stepped_by(self, gradient: LayerGradient | None, learning_rate: float) -> Layer:
        return self


def picture_layer() -> ReshapingLayer:
    """Reads ``(1, 2, 4)`` and answers ``(2, 2, 2)``, both of them eight numbers.

    Eight values arranged two different ways, neither of which is ``(8,)``. That
    is the whole point of it: every other layer in this file is dense, where the
    arrangement and the count are one number, so nothing else here can tell a
    stack that reports arrangements from one that reports counts.
    """
    return ReshapingLayer(reads=(1, 2, 4), answers=(2, 2, 2))


def flattening_layer() -> ReshapingLayer:
    """The bridge ``(2, 2, 2) -> (8,)`` that a picture needs before a row."""
    return ReshapingLayer(reads=(2, 2, 2), answers=(8,))


def reading_layer(n_inputs: int = 8) -> DenseLayer:
    """Two neurons reading a row of ``n_inputs`` numbers."""
    return DenseLayer(
        [
            Neuron([1.0] * n_inputs, bias=0.0, activation=Identity()),
            Neuron([0.5] * n_inputs, bias=1.0, activation=Identity()),
        ]
    )


class TestItReportsArrangementsAndNotCounts:
    """A stack's own shape has to be one its own first layer would accept.

    Every test in this file used to build stacks out of dense layers, whose
    arrangement and element count are the same number, so nothing here could
    tell the two apart. That is why the bug below survived: ``LayerStack`` built
    its shape from ``n_inputs`` and ``n_outputs``, which are products over the
    extents, and a stack beginning with a convolution therefore reported that it
    read ``(784,)`` where its first layer read ``(1, 28, 28)``.

    Every join inside it was still checked correctly, because ``follows``
    compares extents and never consulted the collapsed figure. Only the stack's
    report about itself was wrong, which is the quiet kind: a caller who sizes a
    block from ``stack.shape.reads`` is refused by the stack's own first layer,
    with a message naming a shape they did not choose.

    The round trip in the first test is the claim worth making, because it is
    the thing a caller actually does.
    """

    def test_a_block_built_from_its_own_reported_shape_is_accepted(self) -> None:
        stack = LayerStack([picture_layer(), flattening_layer(), reading_layer()])

        answered = stack.respond_to(np.zeros((2, *stack.shape.reads)))

        assert answered.outputs.shape == (2, *stack.shape.answers)

    def test_it_reports_the_first_layer_s_arrangement(self) -> None:
        stack = LayerStack([picture_layer(), flattening_layer(), reading_layer()])

        assert stack.shape.reads == (1, 2, 4)
        assert stack.shape.reads != (stack.shape.n_inputs,)

    def test_it_reports_the_last_layer_s_arrangement(self) -> None:
        stack = LayerStack([picture_layer(), flattening_layer(), reading_layer()])

        assert stack.shape.answers == (2,)

    def test_its_repr_names_the_arrangement(self) -> None:
        """A stack reading pictures must not describe itself in counts."""
        stack = LayerStack([picture_layer(), flattening_layer(), reading_layer()])

        assert "(1, 2, 4)" in repr(stack)


class TestAJoinThatAgreesOnCountAndNotOnArrangement:
    """The case this whole package exists to catch, and its message.

    A convolution answering ``(2, 2, 2)`` and a dense layer reading ``(8,)``
    hold the same eight numbers and do not join, because one is a picture and
    the other is a row. Refusing is correct and a ``Flatten`` is what is
    missing.

    The message is the part that was wrong. Reporting the two element counts
    printed a refusal saying the two agreed, which sends a reader looking for a
    bug in the library rather than for the missing layer.
    """

    def test_the_join_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerStack([picture_layer(), reading_layer(n_inputs=8)])

    def test_the_message_names_both_arrangements(self) -> None:
        with pytest.raises(ShapeMismatchError) as raised:
            LayerStack([picture_layer(), reading_layer(n_inputs=8)])

        message = str(raised.value)
        assert "(2, 2, 2)" in message
        assert "(8,)" in message

    def test_the_message_says_the_widths_agree_and_the_arrangements_do_not(
        self,
    ) -> None:
        """Otherwise the refusal reads as though the library is broken."""
        with pytest.raises(ShapeMismatchError) as raised:
            LayerStack([picture_layer(), reading_layer(n_inputs=8)])

        assert "arrangement" in str(raised.value)


class TestConstruction:
    def test_its_shape_is_the_two_ends(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])

        assert stack.shape == LayerShape(n_inputs=2, n_outputs=1)

    def test_a_stack_with_no_layers_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            LayerStack([])

    def test_one_layer_is_a_stack(self) -> None:
        stack = LayerStack([widening_layer()])

        assert len(stack) == 1
        assert stack.shape == LayerShape(n_inputs=2, n_outputs=3)

    def test_it_is_iterable_rather_than_handing_out_its_container(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])

        assert not hasattr(stack, "layers")
        assert neuron_counts(stack) == [3, 1]

    def test_mutating_the_caller_s_list_does_not_reach_the_stack(self) -> None:
        supplied = [widening_layer(), narrowing_layer()]
        stack = LayerStack(supplied)

        supplied.reverse()

        assert neuron_counts(stack) == [3, 1]


class TestABadChainIsRefusedBeforeAnyData:
    """The claim the package exists to make, with no rows in sight."""

    def test_a_mismatched_join_raises_at_construction(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerStack(
                [
                    widening_layer(),
                    DenseLayer([Neuron([1.0, 1.0], bias=0.0, activation=Identity())]),
                ]
            )

    def test_the_refusal_names_the_join_and_both_widths(self) -> None:
        """An error saying only that shapes disagree sends a reader everywhere."""
        with pytest.raises(ShapeMismatchError) as raised:
            LayerStack(
                [
                    widening_layer(),
                    DenseLayer([Neuron([1.0, 1.0], bias=0.0, activation=Identity())]),
                ]
            )

        message = str(raised.value)
        assert "3" in message
        assert "2" in message

    def test_a_break_deep_in_a_long_chain_is_still_refused(self) -> None:
        good = DenseLayer([Neuron([1.0], bias=0.0, activation=Identity())])
        with pytest.raises(ShapeMismatchError):
            LayerStack([widening_layer(), narrowing_layer(), narrowing_layer()])

        assert good.shape == LayerShape(n_inputs=1, n_outputs=1)

    def test_a_sound_chain_of_three_is_accepted(self) -> None:
        stack = LayerStack(
            [
                widening_layer(),
                narrowing_layer(),
                DenseLayer([Neuron([2.0], bias=1.0, activation=Identity())]),
            ]
        )

        assert len(stack) == 3
        assert stack.shape == LayerShape(n_inputs=2, n_outputs=1)


class TestResponding:
    def test_it_equals_the_layers_applied_in_sequence(self) -> None:
        """The oracle is what a stack means, not how its body is spelled."""
        first, second = widening_layer(), narrowing_layer()
        stack = LayerStack([first, second])

        by_hand = second.respond_to(first.respond_to(TWO_ROWS).outputs).outputs

        assert np.allclose(stack.respond_to(TWO_ROWS).outputs, by_hand)

    def test_the_answer_is_the_last_layer_s_outputs(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])
        response = stack.respond_to(TWO_ROWS)

        assert response.outputs.shape == (2, 1)
        assert np.allclose(response.outputs, [[6.0], [14.0]])

    def test_every_layer_s_response_is_kept(self) -> None:
        """A backward pass needs the interior scores, not only the answer."""
        stack = LayerStack([widening_layer(), narrowing_layer()])
        response = stack.respond_to(TWO_ROWS)

        assert len(response) == 2
        assert [one.n_neurons for one in response] == [3, 1]

    def test_the_responses_come_back_bottom_to_top(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])
        widths = [one.n_neurons for one in stack.respond_to(TWO_ROWS)]

        assert widths == [3, 1]

    def test_a_block_of_the_wrong_width_is_refused(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])

        with pytest.raises(ShapeMismatchError):
            stack.respond_to(np.array([[1.0, 2.0, 3.0]]))

    def test_a_block_with_no_rows_is_refused(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])

        with pytest.raises(EmptyValuesError):
            stack.respond_to(np.empty((0, 2)))

    def test_a_non_finite_entry_is_refused(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])

        with pytest.raises(InvalidValuesError):
            stack.respond_to(np.array([[1.0, float("nan")]]))

    def test_a_bend_survives_the_whole_pass(self) -> None:
        """A stack that dropped its activations answers the linear collapse."""
        rectified = DenseLayer(
            [
                Neuron([1.0, 1.0], bias=-10.0, activation=RectifiedLinear()),
                Neuron([1.0, 1.0], bias=0.0, activation=RectifiedLinear()),
            ]
        )
        stack = LayerStack(
            [
                rectified,
                DenseLayer([Neuron([1.0, 1.0], bias=0.0, activation=Identity())]),
            ]
        )

        assert np.allclose(stack.respond_to(TWO_ROWS).outputs, [[3.0], [7.0]])


class TestStackResponse:
    def test_it_refuses_an_empty_sequence(self) -> None:
        with pytest.raises(EmptyValuesError):
            StackResponse([])

    def test_its_outputs_are_the_last_response_s(self) -> None:
        first = LayerResponse(
            inputs=np.zeros((1, 2)), scores=np.zeros((1, 2)), outputs=np.ones((1, 2))
        )
        last = LayerResponse(
            inputs=np.zeros((1, 2)),
            scores=np.zeros((1, 1)),
            outputs=np.full((1, 1), 7.0),
        )

        assert np.allclose(StackResponse([first, last]).outputs, [[7.0]])

    def test_mutating_the_caller_s_list_does_not_reach_it(self) -> None:
        first = LayerResponse(
            inputs=np.zeros((1, 2)), scores=np.zeros((1, 2)), outputs=np.ones((1, 2))
        )
        last = LayerResponse(
            inputs=np.zeros((1, 2)),
            scores=np.zeros((1, 1)),
            outputs=np.full((1, 1), 7.0),
        )
        supplied = [first, last]
        response = StackResponse(supplied)

        supplied.reverse()

        assert np.allclose(response.outputs, [[7.0]])


class TestPositionalAccess:
    """A backward pass pairs layer k with responses k and k-1, so both index."""

    def test_a_stack_answers_for_a_layer_by_position(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])

        assert neuron_counts(stack) == [3, 1]
        assert stack[-1] is stack[1]

    def test_a_response_answers_for_a_layer_by_position(self) -> None:
        stack = LayerStack([widening_layer(), narrowing_layer()])
        forward = stack.respond_to(TWO_ROWS)

        assert forward[0].n_neurons == 3
        assert forward[1].n_neurons == 1

    def test_the_block_a_layer_read_is_the_previous_response(self) -> None:
        """The pairing a backward pass needs, stated as a test."""
        stack = LayerStack([widening_layer(), narrowing_layer()])
        forward = stack.respond_to(TWO_ROWS)

        read_by_second = forward[0].outputs
        alone = stack[1].respond_to(read_by_second)

        assert np.allclose(alone.outputs, forward[1].outputs)

    def test_neither_hands_out_a_container(self) -> None:
        stack = LayerStack([widening_layer()])

        assert not hasattr(stack, "layers")
        assert not hasattr(stack.respond_to(TWO_ROWS), "responses")
