"""Spec for why a forward pass is happening, and for who is obliged to say.

Two claims carry this file, and neither is about the enum.

The first is that the parameter is *threaded*. A purpose that stops at the
first layer is worse than no purpose at all, because the network then trains
with its bottom layer perturbed and its top layers not, which is a state no
sound network is ever in and which nothing would raise about. So the spy layer
below records what it was told and the tests read that record from several
depths in a stack.

The second is that :meth:`LayerStack.backward_pass` states
:attr:`PassPurpose.TRAINING` for itself rather than accepting whatever the
default happens to be. That is the line which lets an ordinary training loop
never think about this parameter at all, and it is one keyword long, so it is
exactly the kind of thing a later edit removes without noticing. Reading it back
off a spy is the only way to notice.

The rest is the negative claim, and it is worth having written down: every layer
that existed before this parameter did must answer identically under both
purposes. That is what makes adding the parameter a safe change rather than a
behavioural one, and it is asserted per layer type rather than argued once.

The default is deliberately the safe direction. Forgetting to say "training"
costs a slightly slower descent; forgetting to say "predicting" would make every
answer a model gives depend on a coin toss. The tests pin the default rather
than merely the two members, since a flipped default is a one-character change
that no other test in this package would catch.
"""

import numpy as np
import pytest

from oop_ml.core.network.activation import Identity, RectifiedLinear
from oop_ml.core.network.convolution import Conv2d
from oop_ml.core.network.flatten import Flatten
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import DenseLayer, Layer, LayerResponse
from oop_ml.core.network.loss import SquaredError
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.pooling import AveragePool2d, MaxPool2d
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import LayerStack
from oop_ml.core.types import FloatArray

TWO_ROWS = np.array([[1.0, 2.0], [3.0, 4.0]])


class RecordingLayer(Layer):
    """An identity layer that remembers every purpose it was handed.

    A spy rather than a mock. It answers exactly as an identity would, so a
    stack built from these behaves like a stack, and the only thing it adds is
    a list of what it was told. That matters because the claim under test is
    about a value being *passed on*, and the only witness to a value being
    passed on is the thing that receives it.
    """

    __slots__ = ("_shape", "purposes")

    def __init__(self, width: int) -> None:
        self._shape = LayerShape(n_inputs=width, n_outputs=width)
        self.purposes: list[PassPurpose] = []

    @property
    def shape(self) -> LayerShape:
        return self._shape

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        self.purposes.append(purpose)
        return LayerResponse.already_checked(
            inputs=inputs, scores=inputs, outputs=inputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        return LayerCorrection(
            passed_down=self._checked_arriving(response, arriving), gradient=None
        )

    def stepped_by(self, gradient: LayerGradient | None, learning_rate: float) -> Layer:
        return self


def dense_layer(n_inputs: int = 2, n_neurons: int = 2) -> DenseLayer:
    """A small dense layer with distinguishable neurons."""
    return DenseLayer(
        [
            Neuron([1.0 + index] * n_inputs, bias=float(index), activation=Identity())
            for index in range(n_neurons)
        ]
    )


class TestTheEnumIsClosed:
    def test_it_has_exactly_two_members(self) -> None:
        """A closed set, which is the whole argument for it not being a bool."""
        assert set(PassPurpose) == {PassPurpose.PREDICTING, PassPurpose.TRAINING}

    def test_the_members_read_as_themselves(self) -> None:
        assert PassPurpose.PREDICTING == "predicting"
        assert PassPurpose.TRAINING == "training"

    def test_an_unknown_purpose_is_refused(self) -> None:
        with pytest.raises(ValueError):
            PassPurpose("inferring")


class TestTheDefaultIsTheSafeDirection:
    """A flipped default would make every prediction a coin toss."""

    def test_a_layer_defaults_to_predicting(self) -> None:
        layer = RecordingLayer(width=2)

        layer.respond_to(TWO_ROWS)

        assert layer.purposes == [PassPurpose.PREDICTING]

    def test_a_stack_defaults_to_predicting(self) -> None:
        layer = RecordingLayer(width=2)

        LayerStack([layer]).respond_to(TWO_ROWS)

        assert layer.purposes == [PassPurpose.PREDICTING]


class TestThePurposeReachesEveryLayer:
    """A purpose that stops partway leaves the network in two states at once."""

    @pytest.mark.parametrize("purpose", [PassPurpose.PREDICTING, PassPurpose.TRAINING])
    def test_every_layer_in_a_stack_is_told(self, purpose: PassPurpose) -> None:
        layers = [RecordingLayer(width=2) for _ in range(4)]

        LayerStack(layers).respond_to(TWO_ROWS, purpose)

        assert [layer.purposes for layer in layers] == [[purpose]] * 4

    def test_the_deepest_layer_is_told_too(self) -> None:
        """Threading that stops at the first join is the failure worth naming."""
        deepest = RecordingLayer(width=2)
        stack = LayerStack([RecordingLayer(width=2), RecordingLayer(width=2), deepest])

        stack.respond_to(TWO_ROWS, PassPurpose.TRAINING)

        assert deepest.purposes == [PassPurpose.TRAINING]

    def test_a_second_pass_records_a_second_purpose(self) -> None:
        """The spy accumulates, so a stale first reading cannot pass for a fresh one."""
        layer = RecordingLayer(width=2)
        stack = LayerStack([layer])

        stack.respond_to(TWO_ROWS, PassPurpose.PREDICTING)
        stack.respond_to(TWO_ROWS, PassPurpose.TRAINING)

        assert layer.purposes == [PassPurpose.PREDICTING, PassPurpose.TRAINING]


class TestABackwardPassStatesItsOwnPurpose:
    """One keyword, which an edit removes silently and nothing else would catch."""

    def test_it_says_training_rather_than_taking_the_default(self) -> None:
        spy = RecordingLayer(width=2)
        stack = LayerStack([spy, dense_layer(n_inputs=2, n_neurons=1)])

        stack.backward_pass(TWO_ROWS, np.array([[1.0], [2.0]]), SquaredError())

        assert spy.purposes == [PassPurpose.TRAINING]

    def test_every_layer_beneath_hears_it_too(self) -> None:
        spies = [RecordingLayer(width=2) for _ in range(3)]
        stack = LayerStack([*spies, dense_layer(n_inputs=2, n_neurons=1)])

        stack.backward_pass(TWO_ROWS, np.array([[1.0], [2.0]]), SquaredError())

        assert [spy.purposes for spy in spies] == [[PassPurpose.TRAINING]] * 3


class TestTheLayersThatCameFirstIgnoreIt:
    """Adding the parameter had to be a safe change, asserted per layer type.

    Each of these existed before ``PassPurpose`` did, and none of them has any
    business behaving differently on the two paths. Asserting it once per type
    rather than arguing it once in prose is what makes the claim survive a later
    edit to any one of them.
    """

    def test_a_dense_layer_answers_the_same(self) -> None:
        layer = dense_layer()

        predicting = layer.respond_to(TWO_ROWS, PassPurpose.PREDICTING)
        training = layer.respond_to(TWO_ROWS, PassPurpose.TRAINING)

        assert np.array_equal(predicting.outputs, training.outputs)
        assert np.array_equal(predicting.scores, training.scores)

    def test_a_convolution_answers_the_same(self) -> None:
        layer = Conv2d(
            reads=(1, 5, 5),
            n_filters=2,
            kernel_size=3,
            activation=RectifiedLinear(),
            random_seed=0,
        )
        pictures = np.arange(50.0).reshape(2, 1, 5, 5) / 50.0

        predicting = layer.respond_to(pictures, PassPurpose.PREDICTING)
        training = layer.respond_to(pictures, PassPurpose.TRAINING)

        assert np.array_equal(predicting.outputs, training.outputs)

    @pytest.mark.parametrize("pooling", [MaxPool2d, AveragePool2d])
    def test_a_pooling_layer_answers_the_same(
        self, pooling: type[MaxPool2d] | type[AveragePool2d]
    ) -> None:
        layer = pooling(reads=(2, 4, 4), window=2, stride=2)
        pictures = np.arange(64.0).reshape(2, 2, 4, 4)

        predicting = layer.respond_to(pictures, PassPurpose.PREDICTING)
        training = layer.respond_to(pictures, PassPurpose.TRAINING)

        assert np.array_equal(predicting.outputs, training.outputs)

    def test_a_flattening_answers_the_same(self) -> None:
        layer = Flatten((2, 3, 3))
        pictures = np.arange(36.0).reshape(2, 2, 3, 3)

        predicting = layer.respond_to(pictures, PassPurpose.PREDICTING)
        training = layer.respond_to(pictures, PassPurpose.TRAINING)

        assert np.array_equal(predicting.outputs, training.outputs)

    def test_a_whole_stack_of_them_answers_the_same(self) -> None:
        """The composition, since a difference could hide in any one join."""
        stack = LayerStack(
            [
                Conv2d(
                    reads=(1, 6, 6),
                    n_filters=2,
                    kernel_size=3,
                    activation=RectifiedLinear(),
                    random_seed=1,
                ),
                MaxPool2d(reads=(2, 4, 4), window=2, stride=2),
                Flatten((2, 2, 2)),
                dense_layer(n_inputs=8, n_neurons=3),
            ]
        )
        pictures = np.arange(72.0).reshape(2, 1, 6, 6) / 72.0

        predicting = stack.respond_to(pictures, PassPurpose.PREDICTING)
        training = stack.respond_to(pictures, PassPurpose.TRAINING)

        assert np.array_equal(predicting.outputs, training.outputs)
