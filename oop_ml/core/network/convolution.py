"""One kernel swept across a picture, and the three facts that buys.

Why a picture is not a row
--------------------------
A dense layer reads a row, so the first thing it does to a 28x28 image is
forget that pixel 0 and pixel 1 are neighbours while pixel 0 and pixel 28 are
neighbours in the other direction. Shuffle the 784 numbers with a fixed
permutation and the layer trains exactly as well, which is another way of
saying it was never told the geometry and has to rediscover it from data.

A convolution is told. It reads a window rather than a row and sweeps one set
of weights across every position, and three things follow at once:

* **Locality.** A kernel entry only ever multiplies a value inside its own
  window, so a filter is a statement about a neighbourhood.
* **Weight sharing.** The same numbers answer at every position, so a filter
  that has learned an edge has learned it everywhere rather than once per
  location.
* **Translation equivariance.** Move the input by one pixel and the answer
  moves by one position. That is a theorem about the arithmetic, not a hope
  about the training, and there is a test that pins it.

The price is paid in parameters, and the number is worth writing down. Turning
``1x28x28`` into ``8x26x26``, counted rather than estimated::

    convolution, 8 filters of 3x3    8*1*3*3 + 8         =        80
    a dense layer of the same width  5408*784 + 5408     = 4,245,280

53,066 times fewer parameters for an answer of exactly the same size. That is
not a saving, it is the difference between something trainable on a laptop and
something that is not.

Why the output extent is computed and refused at construction
--------------------------------------------------------------
A dense layer's output width is a number somebody chose. A convolution's is
arithmetic::

    extent_out = (extent_in - kernel_size + 2 * padding) // stride + 1

Every term is known before any data arrives, so the answer is too, and a
configuration whose answer would be smaller than one pixel is refused here with
the arithmetic spelled out in the message. Discovering that four layers later,
inside a training run, is the failure :class:`ShapeMismatchError` exists to
prevent.

Why padding is a parameter and not a detail
--------------------------------------------
Without it the border is under-read. Counted on a 28x28 input under a 3x3
kernel at stride 1, a corner pixel is covered by 1 window and a central pixel
by 9, and the answer shrinks by 2 in each direction every layer, so a stack of
fourteen such layers has nothing left to read. One row of zeros on each side
makes the answer the same size as the input and lifts the corner's coverage
from 1 window to 4. Zeros specifically, because a zero contributes nothing to
any sum it enters, so the padding adds no evidence of its own; anything else
would invent pixels the picture never had.

Why the gradient block is two-dimensional
------------------------------------------
:class:`~oop_ml.core.network.gradient.LayerGradient` carries ``weights`` as
``(n_neurons, n_inputs)`` and ``biases`` as ``(n_neurons,)``. This layer's
weights are ``(n_filters, channels, kernel_size, kernel_size)``, which is four
dimensions, so :meth:`Conv2d.correction_for` reshapes them to
``(n_filters, channels * kernel_size * kernel_size)`` on the way out and
:meth:`Conv2d.stepped_by` reshapes them back on the way in. The reshape is a
view of the same numbers in the same order, so nothing is rearranged and
nothing is lost.

That is not a wart, it is the interface working as designed. A gradient block
means whatever the layer that produced it says it means, and the only object
that can say is the layer, which is precisely why ``correction_for`` and
``stepped_by`` belong here rather than in the stack. An earlier interface had
the stack pull each layer's weight matrix out and do the arithmetic itself,
and it could describe exactly one kind of layer. Nothing outside this class
ever has to unflatten anything: the block goes out of ``correction_for`` and
comes back into ``stepped_by`` without anybody in between reading it.

Why a view and a contraction, and where the loops went
------------------------------------------------------
The definition is seven nested loops for the forward pass and the same nesting
again for each of the three backward gradients, and that is how this layer was
first written, because it is verifiable by reading. It was also slow enough to
decide what the website could show: a network of one convolution, one pooling
layer and one dense layer took 30 s to train for 60 epochs on 90 pictures of
12x12, so no page could train one inside a request.

The rewrite lays every window out with ``sliding_window_view``, which is a view
of the padded block and copies nothing, takes the stride by slicing, and then
the sum over channels and kernel positions is one ``einsum`` over three shared
axes, which numpy hands to a matrix multiply. The kernel gradient is the same
contraction with the blame in place of the kernels. The blame passed down is
the only part that still loops, over the ``kernel_size ** 2`` offsets inside a
window: for one fixed offset the windows land on distinct input positions, so a
sliced ``+=`` is exact, and it is the loop over offsets that adds up the
windows that overlap. Measured, median of five, before and after::

    (90, 1, 12, 12)  4 filters 3x3            forward  123 ms -> 0.63 ms
                                              backward 319 ms -> 1.37 ms
    (32, 3, 16, 16)  8 filters 3x3, padding 1 forward  936 ms -> 1.86 ms
                                              backward 2031 ms -> 3.58 ms
    (16, 4, 15, 15)  6 filters 5x5, stride 2  forward  142 ms -> 0.17 ms
                                              backward 329 ms -> 0.52 ms

Between 195x and 830x, and the larger ratios are where the loops had more
kernel entries to walk. The loops are not thrown away. They live in
``test/core/network/test_sweep_agreement.py``, written from the definition
with no views and no einsum, and every configuration here, padding, a stride
that does not divide the extent, several channels, a one-by-one kernel, agrees
with them to 1e-12 on the scores and all three gradients.

The two places convolution is usually got wrong are worth naming, because both
run happily while training something slightly other than what was asked for.
The first is the backward pass through the kernel: the gradient for kernel
entry ``(i, j)`` sums over every output position, since the same number was
used at every one of them, and a version that forgets to accumulate trains on
the last window alone. The second is the blame passed downward: an input pixel
covered by several windows receives a contribution from each, so the
accumulation runs over output positions rather than input ones. Both are
settled by the finite-difference check in the spec, which agrees with this
implementation to a largest absolute discrepancy of 1.7e-09 across the kernels,
the biases and the inputs, on both the padded and the strided fixture.

Why the kernels start where they do
-------------------------------------
He initialisation: normal draws scaled by ``sqrt(2 / fan_in)``, where
``fan_in = channels * kernel_size * kernel_size`` is how many numbers each
output position sums. Unscaled draws make the score spread grow like the square
root of that count, which is what saturates a bend before the first step is
taken. Measured on ``fan_in = 18`` over unit-normal inputs, averaged over 500
kernel draws, the scores come out with a standard deviation of 4.22 unscaled --
next to the ``sqrt(18) = 4.24`` that predicts it -- and 1.41 He-scaled, which
is the ``sqrt(2)`` the scale is chosen to produce. A tanh saturates well before
four, so the unscaled layer starts with most of its answers already flat and
most of its gradient already gone.

The biases start at zero, which pairs with that scale rather than contradicting
it. A bias drawn at random is a shift the filter has no evidence for yet, and
the gradient supplies one the moment there is any.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from math import sqrt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from oop_ml.core.exceptions import (
    InvalidValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.activation import Activation
from oop_ml.core.network.blocks import as_block
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray

#: How many extents describe one picture: channels, height, width.
SPATIAL_EXTENTS = 3


def _checked_setting(value: int, least: int, role: str) -> int:
    """Read one configured whole number, refusing anything below ``least``.

    Any whole number is accepted rather than only a builtin ``int``, because an
    extent read off a numpy computation is a ``numpy.int64`` and is an extent by
    every standard except ``isinstance``. ``bool`` is excluded ahead of that,
    since ``True`` indexes as 1 and would otherwise slip through as a stride of
    one by accident. This is the rule
    :func:`~oop_ml.core.network.shape._as_extents` already applies to a shape,
    restated here because a stride and a padding are not extents and a padding's
    floor is zero rather than one.

    Parameters
    ----------
    value:
        The configured number.
    least:
        The smallest value that means anything. One for a count or an extent,
        zero for a padding.
    role:
        The parameter's name, for the message.

    Returns
    -------
    int
        The same number, as a builtin ``int``.

    Raises
    ------
    InvalidValuesError
        If the value is a bool, is not a whole number, or is below ``least``.
    """
    if isinstance(value, bool):
        raise InvalidValuesError(f"{role} must be a whole number, not a bool")
    try:
        whole = operator.index(value)
    except TypeError:
        raise InvalidValuesError(f"{role} must be a whole number") from None
    if whole < least:
        raise InvalidValuesError(f"{role} must be at least {least}, got {whole}")
    return whole


def _output_extent(
    extent: int, kernel_size: int, stride: int, padding: int, role: str
) -> int:
    """How far the answer reaches along one axis, or a refusal naming the sum.

    Parameters
    ----------
    extent:
        The input's extent along this axis.
    kernel_size:
        The window's extent, the same along both axes.
    stride:
        How far the window moves between positions.
    padding:
        How many zeros are added at each end of this axis.
    role:
        ``"height"`` or ``"width"``, for the message.

    Returns
    -------
    int
        ``(extent - kernel_size + 2 * padding) // stride + 1``, at least one.

    Raises
    ------
    ShapeMismatchError
        If that arithmetic comes out below one. The window is wider than what
        it has to read, so there is no position to place it in, and the message
        spells the sum out because every term in it was chosen by the caller.
    """
    reach = (extent - kernel_size + 2 * padding) // stride + 1
    if reach < 1:
        raise ShapeMismatchError(
            f"this convolution's answer would have {role} "
            f"({extent} - {kernel_size} + 2 * {padding}) // {stride} + 1 = {reach}, "
            "so the window does not fit over what it reads"
        )
    return reach


class Conv2d(Layer):
    """A bank of filters swept across a picture, each answering its own channel.

    Parameters
    ----------
    reads:
        ``(channels, height, width)``, the extents of one input picture.
    n_filters:
        How many filters, which is how many channels the answer has. Each one
        is a separate set of weights reading every input channel at once.
    kernel_size:
        The window's extent, the same along both spatial axes.
    activation:
        The bend applied to every score. One for the whole layer, since a
        filter answers at many positions and there is no per-position identity
        for a bend to belong to.
    stride:
        How far the window moves between positions, along both axes.
    padding:
        How many rows and columns of zeros are added at each edge before the
        sweep. Zero leaves the answer smaller than the input.
    random_seed:
        Which draw the kernels start from. Fixed by default, so two layers
        built the same way are the same layer and a spec can rely on it.

    Raises
    ------
    ShapeMismatchError
        If ``reads`` does not name exactly three extents, or if the configured
        window, stride and padding leave no room for a single output position.
    InvalidValuesError
        If any extent, ``n_filters``, ``kernel_size`` or ``stride`` is not a
        whole number of at least one, or ``padding`` is not a whole number of
        at least zero.

    Notes
    -----
    The kernels are ``(n_filters, channels, kernel_size, kernel_size)`` and the
    biases are ``(n_filters,)``, one per filter rather than one per output
    position, which is the weight sharing stated as a shape.
    """

    __slots__ = (
        "_activation",
        "_biases",
        "_channels",
        "_height",
        "_kernel_size",
        "_kernels",
        "_n_filters",
        "_padding",
        "_shape",
        "_stride",
        "_width",
    )

    def __init__(
        self,
        reads: Sequence[int],
        n_filters: int,
        kernel_size: int,
        activation: Activation,
        stride: int = 1,
        padding: int = 0,
        random_seed: int = 0,
    ) -> None:
        # Guarded, because ``tuple(5)`` raises a bare TypeError from builtins
        # and every failure in this library is one of its own. A bare integer is
        # the natural mistake here -- it is what a dense layer's width looks
        # like -- and it deserves the message that says so rather than
        # "'int' object is not iterable".
        try:
            extents = tuple(reads)
        except TypeError:
            raise ShapeMismatchError(
                "a convolution reads (channels, height, width), got "
                f"{reads!r}, which is not a sequence of extents"
            ) from None
        if len(extents) != SPATIAL_EXTENTS:
            raise ShapeMismatchError(
                "a convolution reads (channels, height, width), got "
                f"{len(extents)} extents"
            )

        channels = _checked_setting(extents[0], 1, "channels")
        height = _checked_setting(extents[1], 1, "height")
        width = _checked_setting(extents[2], 1, "width")
        checked_n_filters = _checked_setting(n_filters, 1, "n_filters")
        checked_kernel_size = _checked_setting(kernel_size, 1, "kernel_size")
        checked_stride = _checked_setting(stride, 1, "stride")
        checked_padding = _checked_setting(padding, 0, "padding")

        output_height = _output_extent(
            height, checked_kernel_size, checked_stride, checked_padding, "height"
        )
        output_width = _output_extent(
            width, checked_kernel_size, checked_stride, checked_padding, "width"
        )

        fan_in = channels * checked_kernel_size * checked_kernel_size
        generator = np.random.default_rng(random_seed)
        kernels = generator.normal(
            size=(
                checked_n_filters,
                channels,
                checked_kernel_size,
                checked_kernel_size,
            )
        ) * sqrt(2.0 / fan_in)

        self._configure(
            shape=LayerShape(
                n_inputs=(channels, height, width),
                n_outputs=(checked_n_filters, output_height, output_width),
            ),
            activation=activation,
            kernel_size=checked_kernel_size,
            stride=checked_stride,
            padding=checked_padding,
            kernels=kernels,
            biases=np.zeros(checked_n_filters, dtype=np.float64),
        )

    def _configure(
        self,
        shape: LayerShape,
        activation: Activation,
        kernel_size: int,
        stride: int,
        padding: int,
        kernels: FloatArray,
        biases: FloatArray,
    ) -> None:
        """Fill every slot from an already validated configuration.

        Separated from ``__init__`` because there are two ways to arrive at a
        layer and only one of them draws random numbers.
        :meth:`with_parameters` builds the next layer of a training run from
        the last one's geometry and a corrected set of weights, and rerunning
        the constructor to throw its draw away would be both wasteful and a lie
        about where those numbers came from. It is the same argument
        :meth:`LayerResponse.already_checked` makes: one validating entry point,
        and a private route for values this class produced itself.

        Parameters
        ----------
        shape:
            What the layer reads and answers with, both already computed.
        activation:
            The bend for every score.
        kernel_size:
            The window's extent.
        stride:
            How far the window moves between positions.
        padding:
            How many zeros are added at each edge.
        kernels:
            ``(n_filters, channels, kernel_size, kernel_size)``.
        biases:
            ``(n_filters,)``.
        """
        self._shape = shape
        self._activation = activation
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._channels, self._height, self._width = shape.reads
        self._n_filters = shape.answers[0]

        kernel_block = np.array(kernels, dtype=np.float64, copy=True)
        bias_block = np.array(biases, dtype=np.float64, copy=True)
        kernel_block.setflags(write=False)
        bias_block.setflags(write=False)
        self._kernels = kernel_block
        self._biases = bias_block

    @property
    def shape(self) -> LayerShape:
        """``(channels, height, width)`` in, ``(n_filters, height, width)`` out.

        Both sides carry their arrangement rather than a count, which is what
        lets a stack refuse a dense layer placed directly above a convolution
        instead of letting the element counts agree by accident.
        """
        return self._shape

    @property
    def kernels(self) -> FloatArray:
        """``(n_filters, channels, kernel_size, kernel_size)``, frozen.

        The learned parameters in the arrangement they are used in, rather than
        the flattened block a
        :class:`~oop_ml.core.network.gradient.LayerGradient` carries.
        """
        return self._kernels

    @property
    def bias_vector(self) -> FloatArray:
        """``(n_filters,)``, one bias per filter, frozen.

        One per filter and not one per output position: the whole of weight
        sharing, visible as a shape.
        """
        return self._biases

    @property
    def n_filters(self) -> int:
        """How many filters, which is how many channels the answer has."""
        return self._n_filters

    @property
    def kernel_size(self) -> int:
        """The window's extent, the same along both spatial axes."""
        return self._kernel_size

    @property
    def stride(self) -> int:
        """How far the window moves between positions."""
        return self._stride

    @property
    def padding(self) -> int:
        """How many rows and columns of zeros are added at each edge."""
        return self._padding

    @property
    def activation(self) -> Activation:
        """The bend applied to every score in the block."""
        return self._activation

    def with_parameters(self, kernels: FloatArray, biases: FloatArray) -> Conv2d:
        """A new layer of the same geometry, carrying these parameters.

        Training does not mutate. A step builds the next layer from this one's
        geometry and the corrected numbers, which is what keeps every array
        frozen and every answer reproducible.

        Parameters
        ----------
        kernels:
            ``(n_filters, channels, kernel_size, kernel_size)``.
        biases:
            ``(n_filters,)``.

        Returns
        -------
        Conv2d
            A new layer, same shape, same bend, same stride and padding.

        Raises
        ------
        ShapeMismatchError
            If either block does not match this layer's geometry.
        InvalidValuesError
            If either block cannot be read as a float array.
        """
        kernel_block = as_block(kernels, "kernels")
        bias_block = as_block(biases, "biases")

        expected = (
            self._n_filters,
            self._channels,
            self._kernel_size,
            self._kernel_size,
        )
        if kernel_block.shape != expected:
            raise ShapeMismatchError(
                f"this layer's kernels are {expected}, got {kernel_block.shape}"
            )
        if bias_block.shape != (self._n_filters,):
            raise ShapeMismatchError(
                f"this layer's biases are ({self._n_filters},), got {bias_block.shape}"
            )

        replacement = Conv2d.__new__(Conv2d)
        replacement._configure(
            shape=self._shape,
            activation=self._activation,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            kernels=kernel_block,
            biases=bias_block,
        )
        return replacement

    def _padded(self, inputs: FloatArray) -> FloatArray:
        """The block with its zero border in place, ready to be swept.

        Padding is this layer's own business, so the border is built here on
        both passes rather than carried on the response. It is a function of
        the block and two configured integers, and storing a derived value on a
        response is how two copies of one fact start to disagree.

        Parameters
        ----------
        inputs:
            ``(n_rows, channels, height, width)``.

        Returns
        -------
        FloatArray
            ``(n_rows, channels, height + 2 * padding, width + 2 * padding)``,
            zero everywhere outside the original picture. With no padding
            configured this is the block itself, since a border of nothing is
            the block, and both passes only ever read it.
        """
        if self._padding == 0:
            return inputs

        bordered = np.zeros(
            (
                inputs.shape[0],
                self._channels,
                self._height + 2 * self._padding,
                self._width + 2 * self._padding,
            ),
            dtype=np.float64,
        )
        bordered[
            :,
            :,
            self._padding : self._padding + self._height,
            self._padding : self._padding + self._width,
        ] = inputs
        return bordered

    def _windows_of(self, padded: FloatArray) -> FloatArray:
        """Every window the kernel visits, as a view rather than a copy.

        Parameters
        ----------
        padded:
            The bordered block, as :meth:`_padded` returns it.

        Returns
        -------
        FloatArray
            ``(n_rows, channels, out_height, out_width, kernel_size,
            kernel_size)``, where entry ``[n, c, y, x, i, j]`` is the padded
            value kernel entry ``(i, j)`` multiplies when the window sits at
            output position ``(y, x)``. It shares memory with ``padded``, so it
            is read and never written.

        Notes
        -----
        Every window of every step is laid out first and the stride is then
        taken by slicing, which is where the output extent formula comes from:
        ``extent - kernel_size + 1`` positions exist, and a stride of ``s``
        keeps every ``s``-th of them, ``(extent - kernel_size) // s + 1``.
        """
        every_window = sliding_window_view(
            padded, (self._kernel_size, self._kernel_size), axis=(2, 3)
        )
        return every_window[:, :, :: self._stride, :: self._stride]

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """The forward pass, given a block already known to be the right shape.

        Parameters
        ----------
        inputs:
            ``(n_rows, channels, height, width)``, already validated by
            :meth:`respond_to`: four-dimensional, at least one row, this
            layer's extents, every value finite. Nothing here re-checks any of
            that.
        purpose:
            Ignored. A convolution slides the same kernels either way.

        Returns
        -------
        LayerResponse
            ``scores`` is the pre-activation block: for each row, filter and
            output position, the sum over channels and kernel positions of
            kernel times padded input, plus that filter's bias. ``outputs`` is
            that block bent by this layer's activation. Under
            :class:`~oop_ml.core.network.activation.Identity` the bend is the
            identity function, so the two are the same block rather than two
            equal ones, which is correct and worth knowing before comparing
            them by identity.

        Notes
        -----
        The definition is seven nested loops, over rows, filters, both output
        axes, channels and both kernel axes, and the last three of those are a
        sum. Laying every window out as a view turns that sum into one
        contraction over three shared axes, which numpy hands to a matrix
        multiply. See the module docstring for what that bought, and for the
        loop version the tests still hold this against.
        """
        windows = self._windows_of(self._padded(inputs))
        scores = (
            np.einsum("ncyxij,fcij->nfyx", windows, self._kernels, optimize=True)
            + self._biases[np.newaxis, :, np.newaxis, np.newaxis]
        )

        outputs = self._activation.of(scores)

        return LayerResponse(inputs=inputs, scores=scores, outputs=outputs)

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """This layer's backward step: three gradients from one blame block.

        The bend's slope converts the blame at the outputs into blame at the
        scores, and everything else is bookkeeping over windows::

            delta        = arriving * activation.derivative_at(scores)
            dL/dkernel   = sum over rows and output positions of
                           delta * the padded input under that kernel entry
            dL/dbias     = delta summed over rows and both output axes
            passed_down  = sum over output positions of delta * kernel,
                           accumulated at the input positions the window
                           covered, then stripped of its padding

        Parameters
        ----------
        response:
            What this layer did on the way up, including the block it read.
        arriving:
            ``(n_rows, n_filters, height, width)`` of the answer's extents, the
            slope of the loss at this layer's outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is ``(n_rows, channels, height, width)``, the
            arriving block for the layer beneath. ``gradient`` carries the
            kernel slopes reshaped to
            ``(n_filters, channels * kernel_size * kernel_size)`` and one bias
            slope per filter, because
            :class:`~oop_ml.core.network.gradient.LayerGradient` is
            two-dimensional. The reshape is a view of the same numbers in the
            same order, and :meth:`stepped_by` is the only thing that undoes
            it. That is deliberate: the layer that produced a gradient block is
            the only object that can say what its axes mean, which is why the
            backward step belongs here and not in the stack.

        Raises
        ------
        ShapeMismatchError
            If the arriving block does not match the scores this layer formed.
            Every axis is forced, so a disagreement is a transposition
            somewhere above rather than a pair of results.
        InvalidValuesError
            If the arriving block cannot be read as a float array.
        """
        arriving = self._checked_arriving(response, arriving)

        _, out_height, out_width = self._shape.answers
        padded = self._padded(response.inputs)

        delta = arriving * self._activation.derivative_at(response.scores)

        # One kernel entry multiplied a value at every output position of every
        # row, so its slope is a sum over all of them. The contraction below
        # runs over rows and both output axes at once, which is weight sharing
        # seen from the back; a version that kept only one position would
        # train on the last window alone while running perfectly.
        kernel_gradient = np.einsum(
            "nfyx,ncyxij->fcij", delta, self._windows_of(padded), optimize=True
        )
        bias_gradient = delta.sum(axis=(0, 2, 3))

        # An input value covered by several windows was used several times and
        # is owed a contribution from each. Every window's contribution is
        # formed at once, then added back one kernel offset at a time: for a
        # fixed offset (i, j) the windows land on distinct input positions, so
        # a sliced += is exact, and the overlap between offsets is what the
        # loop over offsets accumulates. That keeps the loop kernel_size ** 2
        # long whatever the size of the picture.
        contributions = np.einsum(
            "nfyx,fcij->ncyxij", delta, self._kernels, optimize=True
        )
        padded_blame = np.zeros_like(padded)
        last_row = self._stride * (out_height - 1) + 1
        last_column = self._stride * (out_width - 1) + 1
        for kernel_row in range(self._kernel_size):
            for kernel_column in range(self._kernel_size):
                padded_blame[
                    :,
                    :,
                    kernel_row : kernel_row + last_row : self._stride,
                    kernel_column : kernel_column + last_column : self._stride,
                ] += contributions[:, :, :, :, kernel_row, kernel_column]

        # The border was invented by _padded and belongs to nobody, so the
        # blame that landed on it is dropped rather than passed down.
        passed_down = padded_blame[
            :,
            :,
            self._padding : self._padding + self._height,
            self._padding : self._padding + self._width,
        ]

        return LayerCorrection(
            passed_down=passed_down,
            gradient=LayerGradient(
                weights=kernel_gradient.reshape(self._n_filters, -1),
                biases=bias_gradient,
            ),
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> Conv2d:
        """A new convolution with every parameter moved against its slope.

        Downhill is against the slope, so this subtracts. The kernel block
        arrives flattened to ``(n_filters, channels * kernel_size ** 2)``,
        exactly as :meth:`correction_for` sent it, and is reshaped back here.
        Those two methods are the only places in the library that know what
        that flattening means, which is the point of the gradient belonging to
        the layer.

        Parameters
        ----------
        gradient:
            What this layer's parameters want to change by.
        learning_rate:
            How far to move.

        Returns
        -------
        Conv2d
            A new layer, same geometry and same bend, moved parameters.

        Raises
        ------
        ShapeMismatchError
            If no gradient is supplied, since a convolution has parameters and
            has nothing to answer with when asked to step by nothing; or if the
            gradient's flattened block does not describe this layer's kernels.
        """
        if gradient is None:
            raise ShapeMismatchError(
                "a convolution has parameters and needs a gradient to step by"
            )

        expected = (
            self._n_filters,
            self._channels * self._kernel_size * self._kernel_size,
        )
        if gradient.weights.shape != expected:
            raise ShapeMismatchError(
                f"this layer's kernel gradient is {expected} once flattened, got "
                f"{gradient.weights.shape}"
            )

        kernel_slopes = gradient.weights.reshape(
            self._n_filters, self._channels, self._kernel_size, self._kernel_size
        )
        return self.with_parameters(
            self._kernels - learning_rate * kernel_slopes,
            self._biases - learning_rate * gradient.biases,
        )

    def __repr__(self) -> str:
        return (
            f"Conv2d(reads={self._shape.reads!r}, "
            f"n_filters={self._n_filters!r}, "
            f"kernel_size={self._kernel_size!r}, "
            f"stride={self._stride!r}, "
            f"padding={self._padding!r})"
        )
