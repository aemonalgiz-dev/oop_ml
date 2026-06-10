"""The swept layers against their definitions, written as the loops they are.

Convolution and pooling were first written as plain nested loops, which is the
definition and can be checked by reading, and were then rewritten as a strided
view and one contraction, which is two to three orders of magnitude quicker and
cannot be checked by reading. The loops moved here rather than being thrown
away. A fast path and a slow path with nothing between them are two
implementations; this is what is between them.

The oracles below are written from the definition and share nothing with the
layers but the configuration: no views, no einsum, one output position at a
time. Every configuration a layer can take is represented at least once,
including the ones the fast path handles by slicing arithmetic that a reader
would most easily get wrong: padding, a stride that does not divide the
extent, overlapping windows, several channels, and a tie inside a window.
"""

import numpy as np
import pytest

from oop_ml import (
    AveragePool2d,
    Conv2d,
    HyperbolicTangent,
    Identity,
    MaxPool2d,
    RectifiedLinear,
)

AGREEMENT = 1e-12


def definitional_convolution_scores(layer: Conv2d, inputs: np.ndarray) -> np.ndarray:
    """Every output position's sum, one position and one term at a time."""
    n_rows = inputs.shape[0]
    n_filters, channels, side, _ = layer.kernels.shape
    padding = layer.padding
    stride = layer.stride
    padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, out_height, out_width = layer.shape.answers
    scores = np.zeros((n_rows, n_filters, out_height, out_width))
    for row in range(n_rows):
        for filter_index in range(n_filters):
            for out_row in range(out_height):
                for out_column in range(out_width):
                    total = float(layer.bias_vector[filter_index])
                    for channel in range(channels):
                        for kernel_row in range(side):
                            for kernel_column in range(side):
                                total += float(
                                    layer.kernels[
                                        filter_index, channel, kernel_row, kernel_column
                                    ]
                                    * padded[
                                        row,
                                        channel,
                                        out_row * stride + kernel_row,
                                        out_column * stride + kernel_column,
                                    ]
                                )
                    scores[row, filter_index, out_row, out_column] = total
    return scores


def definitional_convolution_backward(
    layer: Conv2d, inputs: np.ndarray, delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three gradients, accumulated one output position at a time.

    Returns the kernel slopes, the bias slopes and the blame passed down, in
    that order. A tuple of three different things, which the library refuses
    everywhere; an oracle in a test is the one place the positional pairing is
    read three lines later by the same author.
    """
    n_rows = inputs.shape[0]
    n_filters, channels, side, _ = layer.kernels.shape
    padding = layer.padding
    stride = layer.stride
    padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    _, out_height, out_width = layer.shape.answers

    kernel_slopes = np.zeros_like(layer.kernels)
    bias_slopes = np.zeros(n_filters)
    padded_blame = np.zeros_like(padded)
    for row in range(n_rows):
        for filter_index in range(n_filters):
            for out_row in range(out_height):
                for out_column in range(out_width):
                    blame = float(delta[row, filter_index, out_row, out_column])
                    bias_slopes[filter_index] += blame
                    for channel in range(channels):
                        for kernel_row in range(side):
                            for kernel_column in range(side):
                                position = (
                                    row,
                                    channel,
                                    out_row * stride + kernel_row,
                                    out_column * stride + kernel_column,
                                )
                                kernel_slopes[
                                    filter_index, channel, kernel_row, kernel_column
                                ] += blame * padded[position]
                                padded_blame[position] += (
                                    blame
                                    * layer.kernels[
                                        filter_index, channel, kernel_row, kernel_column
                                    ]
                                )
    height, width = inputs.shape[2], inputs.shape[3]
    passed_down = padded_blame[
        :, :, padding : padding + height, padding : padding + width
    ]
    return kernel_slopes, bias_slopes, passed_down


CONVOLUTIONS = [
    pytest.param((2, 1, 6, 6), 2, 3, 1, 0, Identity(), id="plain"),
    pytest.param((3, 2, 7, 7), 3, 3, 1, 1, RectifiedLinear(), id="padded-two-channels"),
    pytest.param(
        (2, 3, 9, 8), 2, 3, 2, 0, HyperbolicTangent(), id="stride-not-dividing"
    ),
    pytest.param(
        (1, 2, 11, 11),
        4,
        5,
        3,
        2,
        RectifiedLinear(),
        id="wide-kernel-padded-stride-three",
    ),
    pytest.param((4, 1, 5, 5), 1, 1, 1, 0, Identity(), id="one-by-one-kernel"),
    pytest.param(
        (2, 2, 4, 6), 2, 4, 1, 0, HyperbolicTangent(), id="kernel-equals-height"
    ),
]


@pytest.mark.parametrize(
    ("block_shape", "n_filters", "kernel_size", "stride", "padding", "activation"),
    CONVOLUTIONS,
)
class TestConvolutionAgreesWithItsDefinition:
    @staticmethod
    def _layer_and_block(
        block_shape, n_filters, kernel_size, stride, padding, activation
    ):
        layer = Conv2d(
            reads=block_shape[1:],
            n_filters=n_filters,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            padding=padding,
            random_seed=11,
        )
        # Biases start at zero, which would let a sweep that forgot them agree.
        generator = np.random.default_rng(5)
        layer = layer.with_parameters(
            layer.kernels, generator.normal(size=layer.bias_vector.shape)
        )
        return layer, generator.normal(size=block_shape)

    def test_the_scores_agree(
        self, block_shape, n_filters, kernel_size, stride, padding, activation
    ):
        layer, block = self._layer_and_block(
            block_shape, n_filters, kernel_size, stride, padding, activation
        )
        swept = np.asarray(layer.respond_to(block).scores)
        assert (
            np.max(np.abs(swept - definitional_convolution_scores(layer, block)))
            < AGREEMENT
        )

    def test_the_three_gradients_agree(
        self, block_shape, n_filters, kernel_size, stride, padding, activation
    ):
        layer, block = self._layer_and_block(
            block_shape, n_filters, kernel_size, stride, padding, activation
        )
        response = layer.respond_to(block)
        arriving = np.random.default_rng(9).normal(size=response.outputs.shape)
        correction = layer.correction_for(response, arriving)

        delta = arriving * activation.derivative_at(np.asarray(response.scores))
        kernel_slopes, bias_slopes, passed_down = definitional_convolution_backward(
            layer, block, delta
        )
        assert correction.gradient is not None
        flattened = np.asarray(correction.gradient.weights).reshape(kernel_slopes.shape)
        assert np.max(np.abs(flattened - kernel_slopes)) < AGREEMENT
        assert (
            np.max(np.abs(np.asarray(correction.gradient.biases) - bias_slopes))
            < AGREEMENT
        )
        assert (
            np.max(np.abs(np.asarray(correction.passed_down) - passed_down)) < AGREEMENT
        )


def definitional_pooling(
    inputs: np.ndarray, window: int, stride: int, keep_largest: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The answer and the blame passed down for an arriving block of ones.

    Written with a scan for the winner rather than ``argmax``, keeping the
    first strictly greater value in row-major order, so the tie convention is
    checked against a statement of it rather than against the same function.
    """
    n_rows, channels, height, width = inputs.shape
    out_height = (height - window) // stride + 1
    out_width = (width - window) // stride + 1
    answers = np.zeros((n_rows, channels, out_height, out_width))
    blame = np.zeros_like(inputs)
    for row in range(n_rows):
        for channel in range(channels):
            for out_row in range(out_height):
                for out_column in range(out_width):
                    top = out_row * stride
                    left = out_column * stride
                    if keep_largest:
                        best = top, left
                        for inner_row in range(top, top + window):
                            for inner_column in range(left, left + window):
                                if (
                                    inputs[row, channel, inner_row, inner_column]
                                    > inputs[(row, channel, *best)]
                                ):
                                    best = inner_row, inner_column
                        answers[row, channel, out_row, out_column] = inputs[
                            (row, channel, *best)
                        ]
                        blame[(row, channel, *best)] += 1.0
                    else:
                        total = 0.0
                        for inner_row in range(top, top + window):
                            for inner_column in range(left, left + window):
                                total += inputs[row, channel, inner_row, inner_column]
                                blame[row, channel, inner_row, inner_column] += 1.0 / (
                                    window * window
                                )
                        answers[row, channel, out_row, out_column] = total / (
                            window * window
                        )
    return answers, blame


POOLINGS = [
    pytest.param((2, 3, 8, 8), 2, 2, id="disjoint"),
    pytest.param((2, 2, 7, 9), 2, 1, id="overlapping"),
    pytest.param((1, 3, 11, 10), 3, 2, id="overlapping-stride-not-dividing"),
    pytest.param((3, 1, 9, 9), 3, 3, id="three-disjoint"),
    pytest.param((2, 2, 6, 6), 4, 1, id="wide-overlap"),
    pytest.param((1, 1, 5, 5), 1, 1, id="window-of-one"),
]


@pytest.mark.parametrize(("block_shape", "window", "stride"), POOLINGS)
@pytest.mark.parametrize("keep_largest", [True, False], ids=["max", "average"])
class TestPoolingAgreesWithItsDefinition:
    def test_the_answers_and_the_blame_agree(
        self, block_shape, window, stride, keep_largest
    ):
        kind = MaxPool2d if keep_largest else AveragePool2d
        layer = kind(reads=block_shape[1:], window=window, stride=stride)
        # Rounded to one decimal so ties inside windows are common, which is
        # where the two routes could disagree about the winner.
        block = np.round(np.random.default_rng(3).normal(size=block_shape), 1)
        response = layer.respond_to(block)
        correction = layer.correction_for(response, np.ones(response.outputs.shape))

        answers, blame = definitional_pooling(block, window, stride, keep_largest)
        assert np.max(np.abs(np.asarray(response.outputs) - answers)) < AGREEMENT
        assert np.max(np.abs(np.asarray(correction.passed_down) - blame)) < AGREEMENT


def test_the_tie_fixture_really_holds_ties():
    """The rounding above is only worth something if ties actually occur."""
    block = np.round(np.random.default_rng(3).normal(size=(2, 3, 8, 8)), 1)
    windows = (
        block.reshape(2, 3, 4, 2, 4, 2)
        .transpose(0, 1, 2, 4, 3, 5)
        .reshape(2, 3, 4, 4, 4)
    )
    largest = windows.max(axis=-1, keepdims=True)
    tied = int(((windows == largest).sum(axis=-1) > 1).sum())
    assert tied > 0
