"""Spec for the torch stack, and the two claims that make it worth having.

The first is the one this backend exists for. A chain of layers that cannot
work is refused at construction, naming the seam, before a tensor has been
allocated. The test says so in two halves: the refusal happens before any
translation is attempted, which a spy on ``translation_of`` can see, and raw
torch does not make the refusal at all, which the test demonstrates by
building the same broken chain as an ``nn.Sequential`` and watching it
construct happily and fail only when data arrives.

The second is the agreement between autograd and the hand-written backward
passes of the numpy backend. Every layer this backend translates has a
``correction_for`` written by hand in ``core/network``, and here each is held
to automatic differentiation on the same block, the same loss and the same
weights, block for block. Agreement at 1e-10 is not a thing that happens by
accident, and it is the strongest evidence either backend has that its
backward pass is right. It is also what lets ``stepped_by`` be the numpy
layers' own step applied to a torch gradient, which the stepping tests then
hold to the numpy step to the last bits.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import (
    Activation,
    HyperbolicTangent,
    Identity,
    RectifiedLinear,
    Sigmoid,
)
from oop_ml.core.network.convolution import Conv2d
from oop_ml.core.network.dropout import Dropout
from oop_ml.core.network.embedding import Embedding
from oop_ml.core.network.flatten import Flatten
from oop_ml.core.network.gradient import BackwardPass
from oop_ml.core.network.layer import DenseLayer, Layer
from oop_ml.core.network.loss import (
    AbsoluteError,
    BinaryCrossEntropy,
    HuberError,
    Loss,
    SoftmaxCrossEntropy,
    SquaredError,
)
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.normalisation import BatchNormalization, BatchStatistics
from oop_ml.core.network.pooling import AveragePool2d, MaxPool2d
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.residual import Residual
from oop_ml.core.network.row_normalisation import (
    LayerNormalization,
    RMSNormalization,
)
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.stack import LayerStack as NumpyStack
from oop_ml.core.network.stack import StackResponse
from oop_ml.core.network.upsampling import NearestUpsample2d
from oop_ml.core.types import FloatArray
from oop_ml.pytorch import stack as stack_module
from oop_ml.pytorch.plumbing import as_array, as_tensor
from oop_ml.pytorch.stack import LayerStack

LAST_BITS = 1e-12
AUTOGRAD_AGREEMENT = 1e-10


def dense(reads: int, writes: int, activation: Activation, seed: int = 0) -> DenseLayer:
    generator = np.random.default_rng(seed)
    return DenseLayer(
        [
            Neuron(
                generator.normal(size=reads) * 0.4,
                bias=float(generator.normal() * 0.1),
                activation=activation,
            )
            for _ in range(writes)
        ]
    )


def rows(seed: int, *extents: int) -> FloatArray:
    return np.random.default_rng(seed).normal(size=extents)


def dense_at(stack: NumpyStack | LayerStack, position: int) -> DenseLayer:
    """The layer at ``position``, narrowed to the dense layer the fixture put there."""
    layer = stack[position]
    assert isinstance(layer, DenseLayer)
    return layer


def convolution_at(stack: NumpyStack | LayerStack, position: int) -> Conv2d:
    """The layer at ``position``, narrowed to the convolution the fixture put there."""
    layer = stack[position]
    assert isinstance(layer, Conv2d)
    return layer


def pictures_stack() -> list[Layer]:
    """Convolution, pooling, flattening, dense. The classic small network."""
    return [
        Conv2d(
            reads=(1, 8, 8),
            n_filters=2,
            kernel_size=3,
            activation=RectifiedLinear(),
            random_seed=1,
        ),
        MaxPool2d(reads=(2, 6, 6), window=2, stride=2),
        Flatten((2, 3, 3)),
        dense(18, 3, Identity(), 2),
    ]


def normalised_stack(normalisation: Layer) -> list[Layer]:
    return [
        dense(4, 3, HyperbolicTangent(), 3),
        normalisation,
        dense(3, 2, Identity(), 4),
    ]


def embedding_stack() -> list[Layer]:
    return [
        Embedding(n_positions=6, n_dimensions=3, n_steps=4, random_seed=5),
        Flatten((4, 3)),
        dense(12, 1, Identity(), 6),
    ]


def residual_stack() -> list[Layer]:
    return [
        dense(4, 4, HyperbolicTangent(), 7),
        Residual(dense(4, 4, HyperbolicTangent(), 8)),
        dense(4, 2, Identity(), 9),
    ]


def upsampling_stack() -> list[Layer]:
    return [
        Conv2d(
            reads=(1, 6, 6),
            n_filters=2,
            kernel_size=3,
            activation=HyperbolicTangent(),
            random_seed=10,
        ),
        AveragePool2d(reads=(2, 4, 4), window=2, stride=2),
        NearestUpsample2d(reads=(2, 2, 2), factor=2),
        Flatten((2, 4, 4)),
        dense(32, 2, Identity(), 11),
    ]


def regression_targets(seed: int, n_rows: int, width: int) -> FloatArray:
    return np.random.default_rng(seed).normal(size=(n_rows, width))


def class_targets(seed: int, n_rows: int, n_classes: int) -> FloatArray:
    return SoftmaxCrossEntropy.one_hot(
        np.random.default_rng(seed).integers(0, n_classes, size=n_rows), n_classes
    )


def binary_targets(seed: int, n_rows: int) -> FloatArray:
    return (
        np.random.default_rng(seed).integers(0, 2, size=(n_rows, 1)).astype(np.float64)
    )


#: (label, layers, inputs, targets, loss), one per translated family.
BACKWARD_CASES: list[
    tuple[str, Callable[[], list[Layer]], FloatArray, FloatArray, Loss]
] = [
    (
        "pictures under softmax cross entropy",
        pictures_stack,
        rows(1, 5, 1, 8, 8),
        class_targets(1, 5, 3),
        SoftmaxCrossEntropy(),
    ),
    (
        "batch normalisation under squared error",
        lambda: normalised_stack(BatchNormalization(n_features=3)),
        rows(2, 6, 4),
        regression_targets(2, 6, 2),
        SquaredError(),
    ),
    (
        "layer normalisation under huber",
        lambda: normalised_stack(LayerNormalization(n_features=3)),
        rows(3, 6, 4),
        regression_targets(3, 6, 2),
        HuberError(threshold=0.5),
    ),
    (
        "rms normalisation under absolute error",
        lambda: normalised_stack(RMSNormalization(n_features=3)),
        rows(4, 6, 4),
        regression_targets(4, 6, 2),
        AbsoluteError(),
    ),
    (
        "embedding under binary cross entropy",
        embedding_stack,
        np.random.default_rng(5).integers(0, 6, size=(6, 4)).astype(np.float64),
        binary_targets(5, 6),
        BinaryCrossEntropy(),
    ),
    (
        "residual under squared error",
        residual_stack,
        rows(6, 6, 4),
        regression_targets(6, 6, 2),
        SquaredError(),
    ),
    (
        "upsampling under squared error",
        upsampling_stack,
        rows(7, 4, 1, 6, 6),
        regression_targets(7, 4, 2),
        SquaredError(),
    ),
]
BACKWARD_IDS = [case[0] for case in BACKWARD_CASES]


class TestABadChainIsRefusedBeforeAnyTensorExists:
    """The claim the backend exists to make."""

    def test_a_mismatched_join_is_refused_at_construction(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerStack([dense(2, 3, Identity()), dense(2, 1, Identity())])

    def test_the_refusal_reads_exactly_as_the_numpy_stack_s(self) -> None:
        """One sentence for one mistake, whichever backend meets it."""
        layers = [dense(2, 3, Identity()), dense(2, 1, Identity())]

        with pytest.raises(ShapeMismatchError) as from_torch:
            LayerStack(layers)
        with pytest.raises(ShapeMismatchError) as from_numpy:
            NumpyStack(layers)

        assert str(from_torch.value) == str(from_numpy.value)
        assert "(3,)" in str(from_torch.value)
        assert "(2,)" in str(from_torch.value)

    def test_a_join_agreeing_on_count_and_not_arrangement_says_so(self) -> None:
        picture = Conv2d(
            reads=(1, 4, 4), n_filters=2, kernel_size=3, activation=Identity()
        )

        with pytest.raises(ShapeMismatchError) as raised:
            LayerStack([picture, dense(8, 1, Identity())])

        assert "(2, 2, 2)" in str(raised.value)
        assert "(8,)" in str(raised.value)
        assert "arrangement" in str(raised.value)

    def test_the_joins_are_checked_before_a_single_translation_is_attempted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A spy on the translation, which a bad chain must never reach. This is
        what 'before any tensor exists' means operationally: no module is
        built, so nothing has been allocated when the refusal is raised."""
        attempted: list[Layer] = []

        def spy(layer: Layer) -> None:
            attempted.append(layer)
            raise AssertionError("a bad chain reached the translation")

        monkeypatch.setattr(stack_module, "translation_of", spy)

        with pytest.raises(ShapeMismatchError):
            LayerStack([dense(2, 3, Identity()), dense(2, 1, Identity())])

        assert attempted == []

    def test_raw_torch_makes_no_such_refusal(self) -> None:
        """The contrast, demonstrated rather than described. The same broken
        chain as an nn.Sequential constructs without complaint and fails only
        once data arrives, from inside the multiply, naming shapes rather
        than layers."""
        raw = nn.Sequential(nn.Linear(2, 3), nn.Linear(2, 1))

        with pytest.raises(RuntimeError) as raised:
            raw(torch.zeros(4, 2))

        assert "shapes cannot be multiplied" in str(raised.value)

    def test_an_untranslatable_layer_is_refused_at_construction(self) -> None:
        mixed = DenseLayer(
            [
                Neuron([1.0, 2.0], bias=0.0, activation=Identity()),
                Neuron([1.0, 2.0], bias=0.0, activation=Sigmoid()),
            ]
        )

        with pytest.raises(InvalidValuesError):
            LayerStack([dense(3, 2, Identity()), mixed])

    def test_no_layers_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            LayerStack([])


class TestItKeepsTheNumpyStacksInterface:
    def test_its_shape_is_the_two_ends_as_arrangements(self) -> None:
        stack = LayerStack(pictures_stack())

        assert stack.shape == LayerShape(n_inputs=(1, 8, 8), n_outputs=(3,))

    def test_it_hands_out_the_numpy_layers_and_not_modules(self) -> None:
        stack = LayerStack(pictures_stack())

        assert len(stack) == 4
        assert isinstance(stack[0], Conv2d)
        assert isinstance(stack[-1], DenseLayer)
        assert [type(layer).__name__ for layer in stack] == [
            "Conv2d",
            "MaxPool2d",
            "Flatten",
            "DenseLayer",
        ]

    def test_its_module_is_a_torch_module_of_the_same_depth(self) -> None:
        stack = LayerStack(pictures_stack())

        assert isinstance(stack.module, nn.Sequential)
        assert len(stack.module) == 4

    def test_its_repr_names_the_arrangement(self) -> None:
        assert "reads=(1, 8, 8)" in repr(LayerStack(pictures_stack()))


class TestRespondingAgreesWithTheNumpyStack:
    @pytest.mark.parametrize(
        ("label", "build", "inputs", "targets", "loss"),
        BACKWARD_CASES,
        ids=BACKWARD_IDS,
    )
    def test_every_layer_s_outputs_agree_while_predicting(
        self,
        label: str,
        build: Callable[[], list[Layer]],
        inputs: FloatArray,
        targets: FloatArray,
        loss: Loss,
    ) -> None:
        layers = build()
        expected = NumpyStack(layers).respond_to(inputs)
        answered = LayerStack(layers).respond_to(inputs)

        assert isinstance(answered, StackResponse)
        assert len(answered) == len(expected)
        for mine, theirs in zip(answered, expected, strict=True):
            assert np.max(np.abs(mine.outputs - theirs.outputs)) < LAST_BITS

    def test_a_batch_normalisation_stack_agrees_while_training(self) -> None:
        layers = normalised_stack(BatchNormalization(n_features=3))
        block = rows(8, 6, 4)

        expected = NumpyStack(layers).respond_to(block, PassPurpose.TRAINING)
        answered = LayerStack(layers).respond_to(block, PassPurpose.TRAINING)

        assert np.max(np.abs(answered.outputs - expected.outputs)) < LAST_BITS

    def test_the_answer_is_the_last_layer_s_outputs(self) -> None:
        stack = LayerStack(pictures_stack())
        block = rows(9, 3, 1, 8, 8)

        assert stack.respond_to(block).outputs.shape == (3, 3)


class TestTheBackwardPassAgreesWithAutograd:
    """Hand-written derivations held to a differentiator that never read them."""

    @pytest.mark.parametrize(
        ("label", "build", "inputs", "targets", "loss"),
        BACKWARD_CASES,
        ids=BACKWARD_IDS,
    )
    def test_the_loss_agrees(
        self,
        label: str,
        build: Callable[[], list[Layer]],
        inputs: FloatArray,
        targets: FloatArray,
        loss: Loss,
    ) -> None:
        layers = build()

        by_hand = NumpyStack(layers).backward_pass(inputs, targets, loss)
        by_autograd = LayerStack(layers).backward_pass(inputs, targets, loss)

        assert by_autograd.loss == pytest.approx(by_hand.loss, abs=LAST_BITS)

    @pytest.mark.parametrize(
        ("label", "build", "inputs", "targets", "loss"),
        BACKWARD_CASES,
        ids=BACKWARD_IDS,
    )
    def test_every_layer_s_gradient_agrees_block_for_block(
        self,
        label: str,
        build: Callable[[], list[Layer]],
        inputs: FloatArray,
        targets: FloatArray,
        loss: Loss,
    ) -> None:
        layers = build()

        by_hand = NumpyStack(layers).backward_pass(inputs, targets, loss)
        by_autograd = LayerStack(layers).backward_pass(inputs, targets, loss)

        assert isinstance(by_autograd, BackwardPass)
        assert len(by_autograd) == len(by_hand)
        for position, (mine, theirs) in enumerate(
            zip(by_autograd, by_hand, strict=True)
        ):
            if theirs is None:
                assert mine is None, f"layer {position} has nothing to learn in numpy"
                continue
            assert mine is not None, f"layer {position} learns in numpy and not here"
            assert mine.weights.shape == theirs.weights.shape
            assert mine.biases.shape == theirs.biases.shape
            assert np.max(np.abs(mine.weights - theirs.weights)) < AUTOGRAD_AGREEMENT
            assert np.max(np.abs(mine.biases - theirs.biases)) < AUTOGRAD_AGREEMENT

    def test_a_batch_normalisation_gradient_carries_the_batch_s_statistics(
        self,
    ) -> None:
        """So that the numpy layer's own stepped_by can take it, which requires one."""
        layers = normalised_stack(BatchNormalization(n_features=3))
        block = rows(10, 6, 4)

        by_hand = NumpyStack(layers).backward_pass(
            block, regression_targets(10, 6, 2), SquaredError()
        )
        by_autograd = LayerStack(layers).backward_pass(
            block, regression_targets(10, 6, 2), SquaredError()
        )

        mine, theirs = by_autograd[1], by_hand[1]
        assert isinstance(mine, BatchStatistics)
        assert isinstance(theirs, BatchStatistics)
        assert np.max(np.abs(mine.batch_mean - theirs.batch_mean)) < LAST_BITS
        assert np.max(np.abs(mine.batch_variance - theirs.batch_variance)) < LAST_BITS


class TestTheBackwardPassIsPure:
    def test_a_batch_normalisation_s_running_figures_are_untouched(self) -> None:
        """torch advances them on every training forward pass; the numpy stack
        never does, and this stack rewrites them from the description so that
        two consecutive backward passes read the same figures."""
        layers = normalised_stack(
            BatchNormalization(n_features=3, running_mean=np.array([0.2, -0.1, 0.4]))
        )
        stack = LayerStack(layers)
        block = rows(11, 6, 4)
        before = stack.respond_to(block).outputs

        stack.backward_pass(block, regression_targets(11, 6, 2), SquaredError())
        first = stack.backward_pass(block, regression_targets(11, 6, 2), SquaredError())
        second = stack.backward_pass(
            block, regression_targets(11, 6, 2), SquaredError()
        )

        assert np.array_equal(stack.respond_to(block).outputs, before)
        assert first.loss == second.loss

    def test_the_descriptions_are_untouched(self) -> None:
        layers = pictures_stack()
        stack = LayerStack(layers)
        weights_before = np.array(dense_at(stack, 3).weight_matrix, copy=True)

        stack.backward_pass(
            rows(12, 4, 1, 8, 8), class_targets(12, 4, 3), SoftmaxCrossEntropy()
        )

        assert np.array_equal(dense_at(stack, 3).weight_matrix, weights_before)


class TestStepping:
    def test_the_step_is_the_numpy_layers_own(self) -> None:
        """A torch gradient arranged as a numpy one, applied by numpy's stepped_by,
        so the two stacks step to the same place to the last bits."""
        layers = pictures_stack()
        block, targets = rows(13, 5, 1, 8, 8), class_targets(13, 5, 3)

        numpy_stepped = NumpyStack(layers).stepped_by(
            NumpyStack(layers).backward_pass(block, targets, SoftmaxCrossEntropy()), 0.1
        )
        torch_stepped = LayerStack(layers).stepped_by(
            LayerStack(layers).backward_pass(block, targets, SoftmaxCrossEntropy()), 0.1
        )

        assert isinstance(torch_stepped, LayerStack)
        assert (
            np.max(
                np.abs(
                    dense_at(torch_stepped, 3).weight_matrix
                    - dense_at(numpy_stepped, 3).weight_matrix
                )
            )
            < AUTOGRAD_AGREEMENT
        )
        assert (
            np.max(
                np.abs(
                    convolution_at(torch_stepped, 0).kernels
                    - convolution_at(numpy_stepped, 0).kernels
                )
            )
            < AUTOGRAD_AGREEMENT
        )

    def test_the_stepped_stack_s_layers_are_numpy_layers_a_numpy_stack_accepts(
        self,
    ) -> None:
        """The interchange claim, made as a round trip."""
        stack = LayerStack(pictures_stack())
        block, targets = rows(14, 5, 1, 8, 8), class_targets(14, 5, 3)

        stepped = stack.stepped_by(
            stack.backward_pass(block, targets, SoftmaxCrossEntropy()), 0.1
        )
        handed_over = NumpyStack(list(stepped))

        assert (
            np.max(
                np.abs(
                    handed_over.respond_to(block).outputs
                    - stepped.respond_to(block).outputs
                )
            )
            < LAST_BITS
        )

    def test_the_original_stack_is_untouched(self) -> None:
        stack = LayerStack(pictures_stack())
        block, targets = rows(15, 5, 1, 8, 8), class_targets(15, 5, 3)
        before = stack.respond_to(block).outputs

        stack.stepped_by(
            stack.backward_pass(block, targets, SoftmaxCrossEntropy()), 0.5
        )

        assert np.array_equal(stack.respond_to(block).outputs, before)

    def test_repeated_steps_train(self) -> None:
        stack = LayerStack(pictures_stack())
        block, targets = rows(16, 8, 1, 8, 8), class_targets(16, 8, 3)
        loss = SoftmaxCrossEntropy()

        first = stack.backward_pass(block, targets, loss).loss
        for _ in range(25):
            stack = stack.stepped_by(stack.backward_pass(block, targets, loss), 0.1)
        last = stack.backward_pass(block, targets, loss).loss

        assert last < 0.5 * first

    def test_a_batch_normalisation_stack_steps_its_running_figures(self) -> None:
        """Through numpy's stepped_by, which is where they move in this library."""
        layers = normalised_stack(BatchNormalization(n_features=3))
        stack = LayerStack(layers)
        block = rows(17, 6, 4)

        stepped = stack.stepped_by(
            stack.backward_pass(block, regression_targets(17, 6, 2), SquaredError()),
            0.1,
        )
        moved = stepped[1]

        assert isinstance(moved, BatchNormalization)
        assert not np.array_equal(moved.running_mean, np.zeros(3))


class TestTheModuleIsTorchsOwn:
    def test_a_torch_optimiser_trains_it(self) -> None:
        """The other training path, torch's in-place one, for a caller who
        wants it. What it trains is what the stack answers with."""
        stack = LayerStack(pictures_stack())
        block, targets = (
            as_tensor(rows(18, 8, 1, 8, 8)),
            as_tensor(class_targets(18, 8, 3)),
        )
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(stack.module.parameters(), lr=0.1)

        stack.module.train()
        first = float(criterion(stack.module(block), targets).detach())
        for _ in range(25):
            optimiser.zero_grad()
            criterion(stack.module(block), targets).backward()
            optimiser.step()
        last = float(criterion(stack.module(block), targets).detach())

        assert last < 0.5 * first
        assert not np.allclose(
            stack.respond_to(as_array(block)).outputs,
            NumpyStack(list(stack)).respond_to(as_array(block)).outputs,
        )


class TestThePurpose:
    def test_predicting_passes_dropout_through(self) -> None:
        stack = LayerStack(
            [dense(3, 4, Identity(), 19), Dropout(reads=4, drop_probability=0.9)]
        )
        block = rows(19, 5, 3)

        assert np.array_equal(
            stack.respond_to(block).outputs,
            stack.respond_to(block, PassPurpose.PREDICTING).outputs,
        )
        assert np.array_equal(
            stack.respond_to(block)[1].outputs, stack.respond_to(block)[0].outputs
        )

    def test_training_drops(self) -> None:
        stack = LayerStack(
            [dense(3, 50, Identity(), 20), Dropout(reads=50, drop_probability=0.5)]
        )
        block = rows(20, 5, 3)
        torch.manual_seed(1)

        dropped = stack.respond_to(block, PassPurpose.TRAINING)[1].outputs

        assert np.sum(dropped == 0.0) > 50


class TestRefusedBlocks:
    def test_the_wrong_arrangement_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerStack(pictures_stack()).respond_to(rows(21, 3, 64))

    def test_a_block_with_no_rows_is_refused(self) -> None:
        with pytest.raises(EmptyValuesError):
            LayerStack(pictures_stack()).respond_to(np.empty((0, 1, 8, 8)))

    def test_a_one_dimensional_block_is_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerStack([dense(2, 1, Identity())]).respond_to(np.array([1.0, 2.0]))

    def test_a_non_finite_entry_is_refused(self) -> None:
        block = rows(22, 3, 1, 8, 8)
        block[0, 0, 0, 0] = np.nan

        with pytest.raises(InvalidValuesError):
            LayerStack(pictures_stack()).respond_to(block)

    def test_targets_of_the_wrong_shape_are_refused(self) -> None:
        with pytest.raises(ShapeMismatchError):
            LayerStack(pictures_stack()).backward_pass(
                rows(23, 4, 1, 8, 8), class_targets(23, 4, 5), SoftmaxCrossEntropy()
            )

    def test_an_embedding_stack_refuses_a_fractional_position(self) -> None:
        with pytest.raises(InvalidValuesError):
            LayerStack(embedding_stack()).respond_to(np.array([[0.0, 1.5, 2.0, 3.0]]))
