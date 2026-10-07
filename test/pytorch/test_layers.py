"""Spec for the layer translations, held to the numpy layers they came from.

One claim carries the file: a translation carries its layer's weights across,
so that handing one block to both produces the same answer. That is what
separates a translation from a module of the right shape, and it is asserted
to the last bits on every translated layer, scores and outputs both.

The second claim is the harness's rule applied to layers. Every concrete layer
the library ships is either translated or declined by name, never both and
never neither, so that a new layer added to ``core/network`` turns up here as
a red test rather than as a layer this backend quietly cannot build.

The batch normalisation tests are about two numbers that mean opposite things
in the two libraries. ``momentum`` is handed over inverted, and the test reads
the module's own attribute to see that it was. torch then updates its running
variance with the unbiased batch variance where this library uses the biased
one, and the test pins that divergence by predicting both from the definition
and asserting torch's buffer holds the unbiased one. Neither matters to this
backend's own stepping, which rewrites the buffers from the description, and
both matter to anyone training the module directly.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

import oop_ml.core.network as network_package
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.network.activation import (
    Activation,
    HyperbolicTangent,
    Identity,
    RectifiedLinear,
    Sigmoid,
)
from oop_ml.core.network.attention import MultiHeadAttention
from oop_ml.core.network.convolution import Conv2d
from oop_ml.core.network.dropout import Dropout
from oop_ml.core.network.embedding import Embedding
from oop_ml.core.network.flatten import Flatten
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import DenseLayer, Layer, LayerResponse
from oop_ml.core.network.neuron import Neuron
from oop_ml.core.network.normalisation import BatchNormalization
from oop_ml.core.network.pooling import AveragePool2d, MaxPool2d
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.residual import Residual
from oop_ml.core.network.row_normalisation import (
    LayerNormalization,
    RMSNormalization,
)
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.network.upsampling import NearestUpsample2d
from oop_ml.core.types import FloatArray
from oop_ml.pytorch.layers import (
    NOT_TRANSLATED,
    TRANSLATED,
    Translation,
    torch_activation_for,
    torch_module_for,
    translation_of,
)
from oop_ml.pytorch.plumbing import as_array, as_tensor

LAST_BITS = 1e-12


def dense(reads: int, writes: int, activation: Activation, seed: int = 0) -> DenseLayer:
    """A seeded dense layer, built the way every caller in this library builds one."""
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


def positions(seed: int, n_rows: int, n_steps: int, n_positions: int) -> FloatArray:
    return (
        np.random.default_rng(seed)
        .integers(0, n_positions, size=(n_rows, n_steps))
        .astype(np.float64)
    )


def torch_response(
    translation: Translation, block: FloatArray, purpose: PassPurpose
) -> tuple[FloatArray, FloatArray]:
    """Scores and outputs through the translation, as arrays, under one purpose."""
    translation.module.train(purpose == PassPurpose.TRAINING)
    with torch.no_grad():
        response = translation.respond(translation.read(block))
    translation.restore_state()
    return as_array(response.scores), as_array(response.outputs)


def first_of(module: nn.Module) -> nn.Module:
    """The first child of a sequential translation, which is its standardiser."""
    assert isinstance(module, nn.Sequential)
    return module[0]


#: (layer, block) pairs, one per translated kind and per variation worth having.
AGREEMENT_CASES: list[tuple[str, Layer, FloatArray]] = [
    ("dense identity", dense(4, 3, Identity(), 1), rows(1, 6, 4)),
    ("dense relu", dense(4, 3, RectifiedLinear(), 2), rows(2, 6, 4)),
    ("dense sigmoid", dense(4, 3, Sigmoid(), 3), rows(3, 6, 4)),
    ("dense tanh", dense(4, 3, HyperbolicTangent(), 4), rows(4, 6, 4)),
    (
        "convolution",
        Conv2d(
            reads=(2, 7, 7),
            n_filters=3,
            kernel_size=3,
            activation=RectifiedLinear(),
            random_seed=5,
        ),
        rows(5, 4, 2, 7, 7),
    ),
    (
        "convolution strided and padded",
        Conv2d(
            reads=(1, 8, 8),
            n_filters=2,
            kernel_size=3,
            activation=HyperbolicTangent(),
            stride=2,
            padding=1,
            random_seed=6,
        ),
        rows(6, 3, 1, 8, 8),
    ),
    (
        "max pooling",
        MaxPool2d(reads=(2, 6, 6), window=2, stride=2),
        rows(7, 3, 2, 6, 6),
    ),
    (
        "max pooling overlapping",
        MaxPool2d(reads=(2, 5, 5), window=3, stride=1),
        rows(8, 3, 2, 5, 5),
    ),
    (
        "average pooling",
        AveragePool2d(reads=(2, 6, 6), window=2, stride=2),
        rows(9, 3, 2, 6, 6),
    ),
    (
        "average pooling overlapping",
        AveragePool2d(reads=(2, 5, 5), window=2, stride=1),
        rows(10, 3, 2, 5, 5),
    ),
    ("flatten", Flatten((2, 3, 3)), rows(11, 4, 2, 3, 3)),
    ("upsampling", NearestUpsample2d(reads=(2, 3, 3), factor=2), rows(12, 3, 2, 3, 3)),
    ("dropout", Dropout(reads=5, drop_probability=0.4, random_seed=13), rows(13, 6, 5)),
    (
        "batch normalisation with running figures",
        BatchNormalization(
            n_features=3,
            scale=np.array([1.5, 0.5, 2.0]),
            shift=np.array([0.1, -0.2, 0.3]),
            running_mean=np.array([0.4, -0.3, 1.1]),
            running_variance=np.array([2.0, 0.5, 1.3]),
        ),
        rows(14, 6, 3),
    ),
    (
        "layer normalisation",
        LayerNormalization(
            n_features=5,
            scale=np.array([1.0, 2.0, 0.5, 1.5, 0.8]),
            shift=np.array([0.0, 0.1, -0.1, 0.2, 0.3]),
        ),
        rows(15, 6, 5),
    ),
    (
        "rms normalisation",
        RMSNormalization(
            n_features=5,
            scale=np.array([1.0, 2.0, 0.5, 1.5, 0.8]),
            shift=np.array([0.0, 0.1, -0.1, 0.2, 0.3]),
        ),
        rows(16, 6, 5),
    ),
    (
        "embedding",
        Embedding(n_positions=7, n_dimensions=3, n_steps=4, random_seed=17),
        positions(17, 5, 4, 7),
    ),
    ("residual", Residual(dense(4, 4, HyperbolicTangent(), 18)), rows(18, 6, 4)),
]
AGREEMENT_IDS = [case[0] for case in AGREEMENT_CASES]


class TestEveryTranslationAnswersAsItsLayerDoes:
    """The weights were carried across, so the two compute the same function."""

    @pytest.mark.parametrize(
        ("label", "layer", "block"), AGREEMENT_CASES, ids=AGREEMENT_IDS
    )
    def test_the_outputs_agree_while_predicting(
        self, label: str, layer: Layer, block: FloatArray
    ) -> None:
        expected = layer.respond_to(block, PassPurpose.PREDICTING)
        _, outputs = torch_response(
            translation_of(layer), block, PassPurpose.PREDICTING
        )

        assert outputs.shape == expected.outputs.shape
        assert np.max(np.abs(outputs - expected.outputs)) < LAST_BITS

    @pytest.mark.parametrize(
        ("label", "layer", "block"), AGREEMENT_CASES, ids=AGREEMENT_IDS
    )
    def test_the_scores_agree_too(
        self, label: str, layer: Layer, block: FloatArray
    ) -> None:
        """The pre-activation block, which is what a backward pass reads."""
        expected = layer.respond_to(block, PassPurpose.PREDICTING)
        scores, _ = torch_response(translation_of(layer), block, PassPurpose.PREDICTING)

        assert np.max(np.abs(scores - expected.scores)) < LAST_BITS

    @pytest.mark.parametrize(
        ("label", "layer", "block"), AGREEMENT_CASES, ids=AGREEMENT_IDS
    )
    def test_the_module_alone_agrees(
        self, label: str, layer: Layer, block: FloatArray
    ) -> None:
        """torch_module_for, which is what a torch user calls, with no stack around it."""
        expected = layer.respond_to(block, PassPurpose.PREDICTING)
        module = torch_module_for(layer).eval()
        translation = translation_of(layer)

        with torch.no_grad():
            outputs = as_array(module(translation.read(block)))

        assert np.max(np.abs(outputs - expected.outputs)) < LAST_BITS

    def test_batch_normalisation_agrees_while_training_too(self) -> None:
        """By the batch's own statistics, which both backends compute the same way."""
        layer = BatchNormalization(n_features=3, scale=np.array([1.5, 0.5, 2.0]))
        block = rows(19, 8, 3)
        expected = layer.respond_to(block, PassPurpose.TRAINING)
        scores, outputs = torch_response(
            translation_of(layer), block, PassPurpose.TRAINING
        )

        assert np.max(np.abs(outputs - expected.outputs)) < LAST_BITS
        assert np.max(np.abs(scores - expected.scores)) < LAST_BITS


class TestEveryLayerIsTranslatedOrDeclinedAndNeverBoth:
    """The harness's rule, applied to layers rather than models."""

    @staticmethod
    def concrete_layers() -> set[type[Layer]]:
        found: set[type[Layer]] = set()
        pending = [Layer]
        while pending:
            parent = pending.pop()
            for child in parent.__subclasses__():
                if child.__module__.startswith(network_package.__name__):
                    if not inspect.isabstract(child):
                        found.add(child)
                    pending.append(child)
        return found

    def test_nothing_is_forgotten(self) -> None:
        unaccounted = self.concrete_layers() - set(TRANSLATED) - set(NOT_TRANSLATED)

        assert unaccounted == set()

    def test_nothing_is_both(self) -> None:
        assert set(TRANSLATED) & set(NOT_TRANSLATED) == set()

    def test_the_library_ships_the_layers_this_file_claims_to_cover(self) -> None:
        """A guard on the guard, so the two sets above cannot be empty and pass."""
        assert len(self.concrete_layers()) >= 15

    def test_a_declined_layer_is_refused_by_name_with_its_reason(self) -> None:
        """Walks the MRO, so a subclass of a declined layer is declined too."""
        declined_type = MultiHeadAttention
        assert declined_type in NOT_TRANSLATED

        class Narrower(MultiHeadAttention):
            def __init__(self) -> None:
                pass

        with pytest.raises(InvalidValuesError) as raised:
            translation_of(Narrower())

        assert declined_type.__name__ in str(raised.value) or "Narrower" in str(
            raised.value
        )
        assert NOT_TRANSLATED[declined_type][:40] in str(raised.value)

    def test_a_layer_kind_the_backend_never_heard_of_is_refused(self) -> None:
        class Unknown(Layer):
            __slots__ = ("_shape",)

            def __init__(self) -> None:
                self._shape = LayerShape(n_inputs=2, n_outputs=2)

            @property
            def shape(self) -> LayerShape:
                return self._shape

            def _response_for(
                self, inputs: FloatArray, purpose: PassPurpose
            ) -> LayerResponse:
                return LayerResponse.already_checked(
                    inputs=inputs, scores=inputs, outputs=inputs
                )

            def correction_for(
                self, response: LayerResponse, arriving: FloatArray
            ) -> LayerCorrection:
                return LayerCorrection(
                    passed_down=self._checked_arriving(response, arriving)
                )

            def stepped_by(
                self, gradient: LayerGradient | None, learning_rate: float
            ) -> Layer:
                return self

        with pytest.raises(InvalidValuesError) as raised:
            translation_of(Unknown())

        assert "Unknown" in str(raised.value)


class TestADenseLayerWithMixedBendsIsRefused:
    def test_it_is_refused_at_translation(self) -> None:
        mixed = DenseLayer(
            [
                Neuron([1.0, 2.0], bias=0.0, activation=Identity()),
                Neuron([1.0, 2.0], bias=0.0, activation=Sigmoid()),
            ]
        )

        with pytest.raises(InvalidValuesError) as raised:
            translation_of(mixed)

        assert "Identity" in str(raised.value)
        assert "Sigmoid" in str(raised.value)

    def test_a_uniform_one_is_not(self) -> None:
        translation_of(dense(2, 2, Sigmoid()))


class TestTheActivations:
    @pytest.mark.parametrize(
        ("activation", "expected"),
        [
            (Identity(), nn.Identity),
            (RectifiedLinear(), nn.ReLU),
            (Sigmoid(), nn.Sigmoid),
            (HyperbolicTangent(), nn.Tanh),
        ],
        ids=["identity", "relu", "sigmoid", "tanh"],
    )
    def test_each_becomes_its_module(
        self, activation: Activation, expected: type[nn.Module]
    ) -> None:
        assert isinstance(torch_activation_for(activation), expected)

    def test_one_the_backend_never_heard_of_is_refused(self) -> None:
        class Bent(Activation):
            __slots__ = ()

            def of(self, scores: FloatArray) -> FloatArray:
                return scores

            def derivative_at(self, scores: FloatArray) -> FloatArray:
                return np.ones_like(scores)

            @property
            def description(self) -> str:
                return "bent"

        with pytest.raises(InvalidValuesError):
            torch_activation_for(Bent())


class TestBatchNormalisationsTwoInvertedNumbers:
    def test_the_momentum_is_handed_over_inverted(self) -> None:
        """This library keeps `momentum` of the running figure; torch keeps `1 - momentum`."""
        layer = BatchNormalization(n_features=3, momentum=0.8)
        standardiser = first_of(translation_of(layer).module)

        assert isinstance(standardiser, nn.BatchNorm1d)
        assert standardiser.momentum == pytest.approx(1.0 - 0.8)

    def test_torch_updates_its_running_variance_with_the_unbiased_estimate(
        self,
    ) -> None:
        """Pinned as a difference, not hidden. With the momentum translated,
        one training pass on the raw module leaves a running variance the
        numpy layer would not have, by n / (n - 1). This backend's own
        stepping never meets it, because the buffers are rewritten from the
        description; a caller training the module directly does."""
        momentum = 0.9
        layer = BatchNormalization(n_features=2, momentum=momentum)
        block = rows(20, 5, 2)
        module = torch_module_for(layer)
        standardiser = first_of(module)
        assert isinstance(standardiser, nn.BatchNorm1d)

        module.train()
        with torch.no_grad():
            module(as_tensor(block))

        biased = block.var(axis=0)
        unbiased = block.var(axis=0, ddof=1)
        numpy_would_hold = momentum * layer.running_variance + (1 - momentum) * biased
        torch_holds = momentum * layer.running_variance + (1 - momentum) * unbiased
        assert standardiser.running_var is not None

        assert np.allclose(as_array(standardiser.running_var), torch_holds)
        assert not np.allclose(as_array(standardiser.running_var), numpy_would_hold)

    def test_restoring_state_puts_the_description_s_figures_back(self) -> None:
        layer = BatchNormalization(n_features=2, running_mean=np.array([0.3, -0.7]))
        translation = translation_of(layer)
        standardiser = first_of(translation.module)
        assert isinstance(standardiser, nn.BatchNorm1d)

        translation.module.train()
        with torch.no_grad():
            translation.respond(translation.read(rows(21, 5, 2)))
        assert standardiser.running_mean is not None
        moved = as_array(standardiser.running_mean)

        translation.restore_state()

        assert not np.allclose(moved, layer.running_mean)
        assert np.array_equal(as_array(standardiser.running_mean), layer.running_mean)


class TestTheEmbeddingReadsPositions:
    def test_whole_numbers_in_range_are_read_as_integers(self) -> None:
        translation = translation_of(
            Embedding(n_positions=5, n_dimensions=2, n_steps=3)
        )

        read = translation.read(np.array([[0.0, 4.0, 2.0]]))

        assert read.dtype == torch.int64
        assert read.tolist() == [[0, 4, 2]]

    def test_a_fraction_is_refused_rather_than_rounded(self) -> None:
        translation = translation_of(
            Embedding(n_positions=5, n_dimensions=2, n_steps=3)
        )

        with pytest.raises(InvalidValuesError) as raised:
            translation.read(np.array([[0.0, 2.9999, 2.0]]))

        assert "2.9999" in str(raised.value)

    @pytest.mark.parametrize("position", [-1.0, 5.0], ids=["below", "past the end"])
    def test_a_position_outside_the_table_is_refused_rather_than_wrapped(
        self, position: float
    ) -> None:
        translation = translation_of(
            Embedding(n_positions=5, n_dimensions=2, n_steps=3)
        )

        with pytest.raises(InvalidValuesError):
            translation.read(np.array([[0.0, position, 2.0]]))


class TestDropoutWhileTraining:
    def test_survivors_are_scaled_and_the_rest_are_zero(self) -> None:
        drop_probability = 0.4
        layer = Dropout(reads=200, drop_probability=drop_probability)
        block = np.ones((50, 200))
        torch.manual_seed(0)

        _, outputs = torch_response(translation_of(layer), block, PassPurpose.TRAINING)

        survivors = outputs[outputs != 0.0]
        assert np.allclose(survivors, 1.0 / (1.0 - drop_probability))
        assert 0.5 < survivors.size / outputs.size < 0.7
        assert outputs.mean() == pytest.approx(1.0, abs=0.03)

    def test_predicting_is_a_pass_through(self) -> None:
        layer = Dropout(reads=5, drop_probability=0.9)
        block = rows(22, 4, 5)

        _, outputs = torch_response(
            translation_of(layer), block, PassPurpose.PREDICTING
        )

        assert np.array_equal(outputs, block)


class TestTheTranslationKeepsItsLayer:
    @pytest.mark.parametrize(
        ("label", "layer", "block"), AGREEMENT_CASES, ids=AGREEMENT_IDS
    )
    def test_the_description_is_handed_back_unchanged(
        self, label: str, layer: Layer, block: FloatArray
    ) -> None:
        assert translation_of(layer).layer is layer

    @pytest.mark.parametrize(
        ("label", "layer", "block"), AGREEMENT_CASES, ids=AGREEMENT_IDS
    )
    def test_the_module_is_float64(
        self, label: str, layer: Layer, block: FloatArray
    ) -> None:
        """The numpy backend's precision, so the agreement is to the last bits."""
        for parameter in translation_of(layer).module.parameters():
            assert parameter.dtype == torch.float64
