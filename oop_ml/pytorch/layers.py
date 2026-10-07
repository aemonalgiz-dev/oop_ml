"""Each of this library's layers, as the torch module that does the same thing.

A description and an implementation were one object, and this is the seam
--------------------------------------------------------------------------
A :class:`~oop_ml.core.network.layer.DenseLayer` is two things at once. It is
a *description* of a layer, so many inputs to so many neurons under one bend,
and it is a numpy *implementation* holding the weights and doing the
arithmetic. The scikit backend met the same conflation in a kernel, where
``RadialBasisKernel(gamma=0.5)`` is a description and
``engine_kernel_parameters`` is the translation that hands it to an engine,
and the answer here is the same shape. A translation reads the description
off the numpy layer, builds the torch module that does what it describes, and
copies the learned numbers across so the two agree to the last bits.

That last part is what makes the translation checkable rather than merely
plausible. A module built to the right shape with fresh random weights would
pass every structural test and prove nothing about whether it computes the
same function. Carrying the weights over lets ``test/pytorch/test_layers.py``
hand one block to both and assert they answer identically, which it does for
every layer here at 1e-12 or better.

What a translation is obliged to answer
----------------------------------------
Four things, so that a stack built from translations can do everything a numpy
stack does. The module itself, for a caller who wants torch's own training
loop. A :meth:`Translation.respond` that runs the forward pass and hands back
the pre-activation scores alongside the outputs, which is what keeps a torch
:class:`~oop_ml.core.network.stack.StackResponse` shaped like a numpy one. A
:meth:`Translation.read_gradient` that collects what autograd left on the
parameters and arranges it the way the numpy layer's own ``correction_for``
arranges its gradient, so that the two backends' backward passes compare
block for block and the numpy layer's own ``stepped_by`` can take the torch
gradient unchanged. And :meth:`Translation.restore_state`, for the one layer
whose torch twin mutates itself during a forward pass.

The two traps the translation closes rather than inherits
----------------------------------------------------------
Batch normalisation's ``momentum`` means opposite things in the two
libraries. This library keeps ``momentum`` of the running figure and takes
``1 - momentum`` from the batch; torch does the reverse. The translation
hands torch ``1 - momentum``, and a test asserts the module carries that
number, because the two spellings produce running statistics that drift apart
without any forward pass disagreeing. torch also updates its running variance
with the *unbiased* batch variance where this library uses the biased one, so
even with the momentum translated the two running variances part by a factor
of ``n / (n - 1)`` per step. Neither matters to this backend's own stepping,
since the running figures live on the numpy description and the torch buffers
are rewritten from it after every training pass, and both are recorded here
for a caller who trains the module directly.

The second is that a stock ``LayerNorm`` and ``BatchNorm1d`` fuse the affine
into the normalisation, which would leave no pre-activation block to report as
the response's ``scores``. Each is therefore built without its affine and
followed by a :class:`PerFeatureAffine` holding the scale and shift as the
module's own parameters, so the normalised block is a tensor a caller can see
and the gradient reads off two named parameters rather than being fished out
of a fused one.

What is refused, by name
-------------------------
A dense layer whose neurons carry different bends. The numpy layer permits it,
because nothing in the mathematics forbids it, and torch bends a whole block
with one module, so the honest translation is none. Five layer kinds whose
torch twin exists but whose weight layout differs enough that the translation
has not yet been written and checked; those are in :data:`NOT_TRANSLATED` with
the reason, and the stack refuses them at construction rather than
approximating. Every translation here is ``float64``, for the reason
:mod:`oop_ml.pytorch.plumbing` gives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

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
from oop_ml.core.network.gradient import LayerGradient
from oop_ml.core.network.layer import DenseLayer, Layer
from oop_ml.core.network.normalisation import BatchNormalization, BatchStatistics
from oop_ml.core.network.pooling import AveragePool2d, MaxPool2d
from oop_ml.core.network.positions import PositionalEncoding
from oop_ml.core.network.radial_basis import RadialBasisLayer
from oop_ml.core.network.recurrent import (
    GatedRecurrentUnit,
    LongShortTermMemory,
    SimpleRecurrent,
)
from oop_ml.core.network.residual import Residual
from oop_ml.core.network.row_normalisation import (
    LayerNormalization,
    RMSNormalization,
    WithinRowNormalization,
)
from oop_ml.core.network.upsampling import NearestUpsample2d
from oop_ml.core.network.weight_normalisation import WeightNormalization
from oop_ml.core.types import FloatArray
from oop_ml.pytorch.plumbing import (
    TENSOR_DTYPE,
    as_array,
    as_positions,
    as_tensor,
    copy_into,
    required,
    slope_of,
)

#: Layers with a torch twin whose translation has not been written and checked.
#: Refused at construction by name, which is the harness's rule for a backend:
#: it may decline, it may not forget.
NOT_TRANSLATED: dict[type[Layer], str] = {
    MultiHeadAttention: (
        "torch.nn.MultiheadAttention is the twin, and it packs the query, key "
        "and value projections into one in_proj_weight where this library holds "
        "them as separate blocks per head. The translation is a reshaping of "
        "weights that has not been written and checked against the forward "
        "pass, so the layer is refused rather than approximated"
    ),
    SimpleRecurrent: (
        "torch.nn.RNN is the twin, and it carries one weight block per "
        "direction packed with its own gate ordering where this library holds "
        "the recurrent and input weights separately. Not yet translated and "
        "checked, so refused rather than approximated"
    ),
    LongShortTermMemory: (
        "torch.nn.LSTM is the twin, and it packs its four gates into one weight "
        "block in the order input, forget, cell, output. Matching this "
        "library's gates to that order is a reshaping that has not been written "
        "and checked, so the layer is refused rather than approximated"
    ),
    GatedRecurrentUnit: (
        "torch.nn.GRU is the twin, and it packs its three gates into one weight "
        "block in the order reset, update, new. Matching this library's gates to "
        "that order is a reshaping that has not been written and checked, so the "
        "layer is refused rather than approximated"
    ),
    PositionalEncoding: (
        "torch has no stock positional encoding module. The custom module is "
        "straightforward and has not yet been written, so the layer is refused "
        "by name rather than silently passed through"
    ),
    RadialBasisLayer: (
        "torch has no radial basis module. The custom module is straightforward "
        "and has not yet been written, so the layer is refused by name rather "
        "than silently passed through"
    ),
    WeightNormalization: (
        "torch.nn.utils.parametrizations.weight_norm is the twin, and it is a "
        "parametrisation of another module rather than a module of its own, "
        "which is a different shape from this library's layer. Not yet "
        "translated, so refused rather than approximated"
    ),
}


class TensorResponse:
    """What one translation answered, before any of it leaves torch.

    The tensor-shaped twin of
    :class:`~oop_ml.core.network.layer.LayerResponse`, holding the
    pre-activation scores and the outputs, and nothing else. It exists so that
    a stack can thread tensors from one translation to the next without
    leaving the gradient tape, and only convert to arrays once at the end.
    """

    __slots__ = ("_outputs", "_scores")

    def __init__(self, scores: torch.Tensor, outputs: torch.Tensor) -> None:
        self._scores = scores
        self._outputs = outputs

    @property
    def scores(self) -> torch.Tensor:
        """Before any bend, or the outputs themselves where there is no bend."""
        return self._scores

    @property
    def outputs(self) -> torch.Tensor:
        """What the next translation reads."""
        return self._outputs


class PerFeatureAffine(nn.Module):
    """``block * scale + shift`` over the last axis, with both as parameters.

    The affine every normalisation layer here ends with, kept as a module of
    its own so that the normalised block before it is a tensor a caller can
    see and the gradient reads off two named parameters.
    """

    def __init__(self, scale: FloatArray, shift: FloatArray) -> None:
        super().__init__()
        self.scale = nn.Parameter(as_tensor(scale))
        self.shift = nn.Parameter(as_tensor(shift))

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        return block * self.scale + self.shift


class RootMeanSquareStandardiser(nn.Module):
    """Divide each row by the root of its mean square, with no centring.

    RMS normalisation's standardiser. torch gained ``nn.RMSNorm`` in 2.4, and
    this backend's floor is below that, so the three-line module is written
    here. It is the same arithmetic the numpy layer does with ``centres``
    switched off.
    """

    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        spread = block.pow(2).mean(dim=-1, keepdim=True)
        return block * torch.rsqrt(spread + self.epsilon)


class ResidualModule(nn.Module):
    """``block + inner(block)``, the skip connection as a module."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        return block + self.inner(block)


def torch_activation_for(activation: Activation) -> nn.Module:
    """The module that bends the way this activation bends.

    Raises
    ------
    InvalidValuesError
        If the activation is one this backend has no module for. The four the
        library ships all translate.
    """
    if isinstance(activation, Identity):
        return nn.Identity()
    if isinstance(activation, RectifiedLinear):
        return nn.ReLU()
    if isinstance(activation, Sigmoid):
        return nn.Sigmoid()
    if isinstance(activation, HyperbolicTangent):
        return nn.Tanh()
    raise InvalidValuesError(
        f"the torch backend has no module for {type(activation).__name__}; the "
        "four activations this library ships all translate, so this is one the "
        "backend has not been told about"
    )


class Translation(ABC):
    """A numpy layer and the torch module that does what it describes.

    Parameters
    ----------
    layer:
        The description. It is kept, because the stack hands it out through
        ``__getitem__`` exactly as a numpy stack would, and because its own
        ``stepped_by`` is how a torch gradient becomes the next description.
    """

    __slots__ = ("_layer",)

    def __init__(self, layer: Layer) -> None:
        self._layer = layer

    @property
    def layer(self) -> Layer:
        """The numpy description this translation was built from."""
        return self._layer

    @property
    @abstractmethod
    def module(self) -> nn.Module:
        """The torch module, for a caller who wants torch's own training loop."""

    def read(self, block: FloatArray) -> torch.Tensor:
        """Turn the block a stack was handed into the tensor this layer reads.

        Every layer but one reads measurements, so the default is a ``float64``
        tensor. The embedding overrides it, for the reason the numpy embedding
        overrides ``_read_block``: it reads positions, and the checks that mean
        something for a position are not finiteness.
        """
        return as_tensor(block)

    @abstractmethod
    def respond(self, block: torch.Tensor) -> TensorResponse:
        """The forward pass, keeping the scores and the outputs apart."""

    def read_gradient(self) -> LayerGradient | None:
        """What autograd left on the parameters, in the numpy layer's arrangement.

        ``None`` for a layer with nothing to learn, which is the same honest
        answer the numpy layer gives rather than a block of zeros.
        """
        return None

    def restore_state(self) -> None:
        """Undo anything a training forward pass did to the module's buffers.

        A no-op for every layer whose torch twin is pure. Batch normalisation's
        is not, and its override is what keeps a stack's ``backward_pass`` as
        free of side effects as the numpy one.
        """
        return None


class DenseTranslation(Translation):
    """A dense layer as ``Linear`` followed by one bend."""

    __slots__ = ("_activation", "_linear", "_module")

    def __init__(self, layer: DenseLayer) -> None:
        super().__init__(layer)

        bends = {neuron.activation for neuron in layer}
        if len(bends) != 1:
            raise InvalidValuesError(
                "torch bends a whole layer with one module, so a dense layer "
                "whose neurons carry different activations has no translation; "
                f"this one carries {sorted(type(bend).__name__ for bend in bends)}"
            )

        n_outputs, n_inputs = layer.weight_matrix.shape
        self._linear = nn.Linear(n_inputs, n_outputs, dtype=TENSOR_DTYPE)
        copy_into(self._linear.weight, layer.weight_matrix)
        copy_into(required(self._linear.bias, "bias"), layer.bias_vector)

        self._activation = torch_activation_for(bends.pop())
        self._module = nn.Sequential(self._linear, self._activation)

    @property
    def module(self) -> nn.Module:
        return self._module

    def respond(self, block: torch.Tensor) -> TensorResponse:
        scores = self._linear(block)
        return TensorResponse(scores=scores, outputs=self._activation(scores))

    def read_gradient(self) -> LayerGradient | None:
        return LayerGradient(
            weights=slope_of(self._linear.weight, "a dense layer's weights"),
            biases=slope_of(
                required(self._linear.bias, "bias"), "a dense layer's biases"
            ),
        )


class ConvolutionTranslation(Translation):
    """A convolution as ``Conv2d`` followed by one bend.

    torch's ``Conv2d`` is a cross-correlation, as the numpy layer's sweep is,
    so the kernels copy across without being flipped. The gradient comes back
    as ``(n_filters, channels, kernel, kernel)`` and is reshaped to the
    ``(n_filters, channels * kernel * kernel)`` the numpy layer reports, which
    is a view of the same numbers in the same order.
    """

    __slots__ = ("_activation", "_convolution", "_module", "_n_filters")

    def __init__(self, layer: Conv2d) -> None:
        super().__init__(layer)

        channels = layer.shape.reads[0]
        self._n_filters = layer.n_filters
        self._convolution = nn.Conv2d(
            channels,
            layer.n_filters,
            layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dtype=TENSOR_DTYPE,
        )
        copy_into(self._convolution.weight, layer.kernels)
        copy_into(required(self._convolution.bias, "bias"), layer.bias_vector)

        self._activation = torch_activation_for(layer.activation)
        self._module = nn.Sequential(self._convolution, self._activation)

    @property
    def module(self) -> nn.Module:
        return self._module

    def respond(self, block: torch.Tensor) -> TensorResponse:
        scores = self._convolution(block)
        return TensorResponse(scores=scores, outputs=self._activation(scores))

    def read_gradient(self) -> LayerGradient | None:
        kernel_slopes = slope_of(self._convolution.weight, "a convolution's kernels")
        return LayerGradient(
            weights=kernel_slopes.reshape(self._n_filters, -1),
            biases=slope_of(
                required(self._convolution.bias, "bias"), "a convolution's biases"
            ),
        )


class PoolingTranslation(Translation):
    """A pooling layer as ``MaxPool2d`` or ``AvgPool2d``, nothing to learn.

    The output extents agree, since both floor ``(extent - window) / stride``.
    What can differ is the backward pass on an exact tie inside a window,
    where this library routes the whole arriving value to the first maximum in
    row-major order and torch routes it to whichever its kernel found first.
    Ties in float64 data are confined to constant regions and hand-built
    fixtures, as the numpy module says, and the agreement tests use data with
    none.
    """

    __slots__ = ("_module",)

    def __init__(self, layer: MaxPool2d | AveragePool2d) -> None:
        super().__init__(layer)
        if isinstance(layer, MaxPool2d):
            self._module: nn.Module = nn.MaxPool2d(
                kernel_size=layer.window, stride=layer.stride
            )
        else:
            self._module = nn.AvgPool2d(kernel_size=layer.window, stride=layer.stride)

    @property
    def module(self) -> nn.Module:
        return self._module

    def respond(self, block: torch.Tensor) -> TensorResponse:
        outputs = self._module(block)
        return TensorResponse(scores=outputs, outputs=outputs)


class FlattenTranslation(Translation):
    """A flattening as ``Flatten`` from the second axis, nothing to learn."""

    __slots__ = ("_module",)

    def __init__(self, layer: Flatten) -> None:
        super().__init__(layer)
        self._module = nn.Flatten(start_dim=1)

    @property
    def module(self) -> nn.Module:
        return self._module

    def respond(self, block: torch.Tensor) -> TensorResponse:
        outputs = self._module(block)
        return TensorResponse(scores=outputs, outputs=outputs)


class UpsamplingTranslation(Translation):
    """A nearest-neighbour upsampling as ``Upsample``, nothing to learn."""

    __slots__ = ("_module",)

    def __init__(self, layer: NearestUpsample2d) -> None:
        super().__init__(layer)
        self._module = nn.Upsample(scale_factor=float(layer.factor), mode="nearest")

    @property
    def module(self) -> nn.Module:
        return self._module

    def respond(self, block: torch.Tensor) -> TensorResponse:
        outputs = self._module(block)
        return TensorResponse(scores=outputs, outputs=outputs)


class DropoutTranslation(Translation):
    """A dropout as ``Dropout``, which is inverted, as this library's is.

    Both scale the survivors by ``1 / (1 - p)`` while training and pass the
    block through untouched while predicting, so the two agree exactly under
    :attr:`~oop_ml.core.network.purpose.PassPurpose.PREDICTING`. Under
    training they draw from different generators and cannot agree on a mask,
    only on its statistics, which is what the tests assert.
    """

    __slots__ = ("_module",)

    def __init__(self, layer: Dropout) -> None:
        super().__init__(layer)
        self._module = nn.Dropout(p=layer.drop_probability)

    @property
    def module(self) -> nn.Module:
        return self._module

    def respond(self, block: torch.Tensor) -> TensorResponse:
        outputs = self._module(block)
        return TensorResponse(scores=outputs, outputs=outputs)


class BatchNormalisationTranslation(Translation):
    """Batch normalisation as ``BatchNorm1d`` without its affine, then the affine.

    The momentum is handed over inverted and the running figures are copied
    across, so a prediction pass agrees with the numpy layer exactly. A
    training pass agrees too, since both standardise by the batch's own biased
    variance; what torch then does to its running buffers is undone by
    :meth:`restore_state`, because the running figures belong to the numpy
    description and advance in its ``stepped_by``, not here.

    The batch mean and variance are taken off the block as it arrives during a
    training pass, so that :meth:`read_gradient` can answer with a
    :class:`~oop_ml.core.network.normalisation.BatchStatistics` rather than a
    plain gradient, which is what the numpy layer's ``stepped_by`` requires.
    """

    __slots__ = (
        "_affine",
        "_batch_mean",
        "_batch_variance",
        "_module",
        "_standardiser",
    )

    def __init__(self, layer: BatchNormalization) -> None:
        super().__init__(layer)
        self._standardiser = nn.BatchNorm1d(
            layer.n_features,
            eps=layer.epsilon,
            momentum=1.0 - layer.momentum,
            affine=False,
            dtype=TENSOR_DTYPE,
        )
        self._affine = PerFeatureAffine(layer.scale, layer.shift)
        self._module = nn.Sequential(self._standardiser, self._affine)
        self._batch_mean: FloatArray | None = None
        self._batch_variance: FloatArray | None = None
        self.restore_state()

    @property
    def module(self) -> nn.Module:
        return self._module

    def respond(self, block: torch.Tensor) -> TensorResponse:
        if self._standardiser.training:
            self._batch_mean = as_array(block.mean(dim=0))
            self._batch_variance = as_array(block.var(dim=0, unbiased=False))
        scores = self._standardiser(block)
        return TensorResponse(scores=scores, outputs=self._affine(scores))

    def read_gradient(self) -> LayerGradient | None:
        if self._batch_mean is None or self._batch_variance is None:
            raise RuntimeError(
                "a batch normalisation gradient carries the batch's statistics, "
                "and no training pass has run to measure them"
            )
        return BatchStatistics(
            weights=slope_of(self._affine.scale, "a batch normalisation's scale")[
                :, None
            ],
            biases=slope_of(self._affine.shift, "a batch normalisation's shift"),
            batch_mean=self._batch_mean,
            batch_variance=self._batch_variance,
        )

    def restore_state(self) -> None:
        layer = self._layer
        if not isinstance(layer, BatchNormalization):
            raise RuntimeError("a batch normalisation translation holds another layer")
        copy_into(
            required(self._standardiser.running_mean, "running_mean"),
            layer.running_mean,
        )
        copy_into(
            required(self._standardiser.running_var, "running_var"),
            layer.running_variance,
        )


class RowNormalisationTranslation(Translation):
    """Layer or RMS normalisation as a standardiser without affine, then the affine.

    One translation for the family, because the family is one base and a
    boolean. ``centres`` picks a stock ``LayerNorm`` with its affine switched
    off, or the three-line root-mean-square standardiser, and the affine after
    either is the same module.
    """

    __slots__ = ("_affine", "_module", "_standardiser")

    def __init__(self, layer: WithinRowNormalization) -> None:
        super().__init__(layer)
        if layer.centres:
            self._standardiser: nn.Module = nn.LayerNorm(
                layer.n_features,
                eps=layer.epsilon,
                elementwise_affine=False,
                dtype=TENSOR_DTYPE,
            )
        else:
            self._standardiser = RootMeanSquareStandardiser(layer.epsilon)
        self._affine = PerFeatureAffine(layer.scale, layer.shift)
        self._module = nn.Sequential(self._standardiser, self._affine)

    @property
    def module(self) -> nn.Module:
        return self._module

    def respond(self, block: torch.Tensor) -> TensorResponse:
        scores = self._standardiser(block)
        return TensorResponse(scores=scores, outputs=self._affine(scores))

    def read_gradient(self) -> LayerGradient | None:
        return LayerGradient(
            weights=slope_of(self._affine.scale, "a normalisation's scale")[:, None],
            biases=slope_of(self._affine.shift, "a normalisation's shift"),
        )


class EmbeddingTranslation(Translation):
    """An embedding as ``Embedding``, reading positions rather than measurements.

    The override of :meth:`read` is the point of the layer. A position is a
    whole number naming a row of the table, and the two checks that mean
    something for one are integrality and range. Both refuse rather than round
    or clip, for the reason the numpy embedding gives: a caller who meant
    position two and wrote 2.9999 has a bug upstream, and rounding answers
    plausibly while hiding it.
    """

    __slots__ = ("_embedding", "_n_positions")

    def __init__(self, layer: Embedding) -> None:
        super().__init__(layer)
        self._n_positions = layer.n_positions
        self._embedding = nn.Embedding(
            layer.n_positions, layer.n_dimensions, dtype=TENSOR_DTYPE
        )
        copy_into(self._embedding.weight, layer.table)

    @property
    def module(self) -> nn.Module:
        return self._embedding

    def read(self, block: FloatArray) -> torch.Tensor:
        flat = block.reshape(-1)
        fractional = flat[flat != np.floor(flat)]
        if fractional.size > 0:
            raise InvalidValuesError(
                "a position names a row of the table, so it is a whole number; "
                f"got {float(fractional[0])}"
            )
        outside = flat[(flat < 0) | (flat >= self._n_positions)]
        if outside.size > 0:
            raise InvalidValuesError(
                f"a position names a row of a table with {self._n_positions} "
                f"rows, so it lies in [0, {self._n_positions}); got "
                f"{int(outside[0])}"
            )
        return as_positions(block)

    def respond(self, block: torch.Tensor) -> TensorResponse:
        outputs = self._embedding(block)
        return TensorResponse(scores=outputs, outputs=outputs)

    def read_gradient(self) -> LayerGradient | None:
        return LayerGradient(
            weights=slope_of(self._embedding.weight, "an embedding's table"),
            biases=np.zeros(self._n_positions),
        )


class ResidualTranslation(Translation):
    """A residual as the inner layer's module with the block added back.

    Every question is answered by the inner translation. The module wraps it,
    the response adds the block to what the inner answered, the gradient is
    the inner's unchanged because the addition has no parameters, and
    restoring state is the inner's to do.
    """

    __slots__ = ("_inner", "_module")

    def __init__(self, layer: Residual) -> None:
        super().__init__(layer)
        self._inner = translation_of(layer.inner)
        self._module = ResidualModule(self._inner.module)

    @property
    def module(self) -> nn.Module:
        return self._module

    def read(self, block: FloatArray) -> torch.Tensor:
        return self._inner.read(block)

    def respond(self, block: torch.Tensor) -> TensorResponse:
        outputs = block + self._inner.respond(block).outputs
        return TensorResponse(scores=outputs, outputs=outputs)

    def read_gradient(self) -> LayerGradient | None:
        return self._inner.read_gradient()

    def restore_state(self) -> None:
        self._inner.restore_state()


#: Every layer kind this backend translates. Read by the test that holds every
#: concrete layer the library ships to exactly one of this and
#: :data:`NOT_TRANSLATED`, so that a new layer under ``core/network`` turns up
#: as a red test rather than as one this backend quietly cannot build.
TRANSLATED: tuple[type[Layer], ...] = (
    DenseLayer,
    Conv2d,
    MaxPool2d,
    AveragePool2d,
    Flatten,
    NearestUpsample2d,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    RMSNormalization,
    Embedding,
    Residual,
)


def translation_of(layer: Layer) -> Translation:
    """The translation for this layer, built and carrying its weights.

    Parameters
    ----------
    layer:
        Any of this library's layers.

    Returns
    -------
    Translation
        Holding the layer and the torch module that does what it describes.

    Raises
    ------
    InvalidValuesError
        If the layer is in :data:`NOT_TRANSLATED`, carrying that entry's
        reason; if it is a layer kind this backend has never heard of; or if it
        is a dense layer whose neurons carry different bends.
    """
    for candidate in type(layer).__mro__:
        reason = NOT_TRANSLATED.get(candidate)
        if reason is not None:
            raise InvalidValuesError(
                f"the torch backend does not translate {type(layer).__name__}: {reason}"
            )

    if isinstance(layer, DenseLayer):
        return DenseTranslation(layer)
    if isinstance(layer, Conv2d):
        return ConvolutionTranslation(layer)
    if isinstance(layer, MaxPool2d | AveragePool2d):
        return PoolingTranslation(layer)
    if isinstance(layer, Flatten):
        return FlattenTranslation(layer)
    if isinstance(layer, NearestUpsample2d):
        return UpsamplingTranslation(layer)
    if isinstance(layer, Dropout):
        return DropoutTranslation(layer)
    if isinstance(layer, BatchNormalization):
        return BatchNormalisationTranslation(layer)
    if isinstance(layer, LayerNormalization | RMSNormalization):
        return RowNormalisationTranslation(layer)
    if isinstance(layer, Embedding):
        return EmbeddingTranslation(layer)
    if isinstance(layer, Residual):
        return ResidualTranslation(layer)

    raise InvalidValuesError(
        f"the torch backend has no translation for {type(layer).__name__}, which "
        "is a layer kind it has not been told about; every layer the library "
        "ships is either translated or declined by name in NOT_TRANSLATED"
    )


def torch_module_for(layer: Layer) -> nn.Module:
    """The torch module that does what this layer describes, weights included.

    The convenience over :func:`translation_of` for a caller who wants the
    module and nothing else, which is the third translation function of its
    kind in this library after ``engine_kernel_parameters`` and
    ``metric_engine_parameters``.
    """
    return translation_of(layer).module
