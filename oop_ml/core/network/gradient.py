"""What one backward pass worked out, layer by layer.

The shape of the answer
-----------------------
A backward pass produces exactly one thing per layer: how the loss would move
if that layer's weights moved. So the gradient of a whole network is the same
shape as the network -- one block per layer, each matching that layer's own
weight matrix and bias vector -- and :class:`BackwardPass` is that shape given
a name rather than a list of loose arrays a caller has to keep aligned with the
layers by hand.

Why the loss travels with it
----------------------------
The forward pass has to run before the backward one can, and the loss falls out
of that same forward pass. Handing it back separately would mean either running
the network twice or returning a bare pair, so it rides on the object that
already exists. A training loop wants both on every epoch, one to step with and
one to record.

Why the gradients are frozen
----------------------------
The same reason every learned array here is. A step reads them to build the
next set of weights and nothing should be able to edit them in between, least
of all by accident through a view a caller kept.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from oop_ml.core.exceptions import EmptyValuesError, ShapeMismatchError
from oop_ml.core.types import FloatArray


class LayerGradient:
    """How the loss would move if one layer's parameters moved.

    Parameters
    ----------
    weights:
        ``(n_neurons, n_inputs)``, matching the layer's own weight matrix, so
        entry ``[j, i]`` is the slope for neuron ``j``'s weight on input ``i``.
    biases:
        ``(n_neurons,)``, one slope per neuron.

    Raises
    ------
    ShapeMismatchError
        If the weight block is not two-dimensional, the bias block is not
        one-dimensional, or they disagree about how many neurons there are.
    """

    __slots__ = ("_biases", "_weights")

    def __init__(self, weights: FloatArray, biases: FloatArray) -> None:
        weight_block = np.array(weights, dtype=np.float64, copy=True)
        bias_block = np.array(biases, dtype=np.float64, copy=True)

        if weight_block.ndim != 2:
            raise ShapeMismatchError(
                "a layer's weight gradient is (n_neurons, n_inputs), got "
                f"{weight_block.ndim} dimensions"
            )
        if bias_block.ndim != 1:
            raise ShapeMismatchError(
                "a layer's bias gradient is (n_neurons,), got "
                f"{bias_block.ndim} dimensions"
            )
        if weight_block.shape[0] != bias_block.shape[0]:
            raise ShapeMismatchError(
                f"weight gradient names {weight_block.shape[0]} neurons and bias "
                f"gradient names {bias_block.shape[0]}"
            )

        weight_block.setflags(write=False)
        bias_block.setflags(write=False)
        self._weights = weight_block
        self._biases = bias_block

    @property
    def weights(self) -> FloatArray:
        """``(n_neurons, n_inputs)``, aligned with the layer's weight matrix."""
        return self._weights

    @property
    def biases(self) -> FloatArray:
        """``(n_neurons,)``, one slope per neuron."""
        return self._biases

    @property
    def n_neurons(self) -> int:
        """How many neurons this gradient describes."""
        return int(self._weights.shape[0])

    @property
    def largest_movement(self) -> float:
        """The biggest single slope, which is what a convergence check reads."""
        return float(
            max(
                np.max(np.abs(self._weights)),
                np.max(np.abs(self._biases)),
            )
        )

    def __repr__(self) -> str:
        return f"LayerGradient(shape={self._weights.shape!r})"


class LayerCorrection:
    """One layer's share of the blame: what to change, and what to pass down.

    Two things a backward step produces together, and the reason the stack no
    longer needs to know how any particular layer works. It used to pull a
    layer's weight matrix and activation slopes out and do the arithmetic
    itself, which meant every layer had to *have* a weight matrix. A dropout
    layer has none, a pooling layer has none, and a flattening layer has
    nothing at all, so that interface could only ever describe dense layers.

    Asking the layer for its own correction instead lets each answer in its own
    terms, and the stack becomes a walk that threads one block downward.

    Parameters
    ----------
    passed_down:
        The slope of the loss at *this layer's inputs*, which is the arriving
        block for the layer beneath. Every layer produces one, since every
        layer sits on something.
    gradient:
        What this layer's own parameters want to change by, or ``None`` when it
        has no parameters. Dropout, pooling and flattening all answer ``None``
        honestly, where a zero-filled gradient would be a small lie about
        having something to learn.
    """

    __slots__ = ("_gradient", "_passed_down")

    def __init__(
        self, passed_down: FloatArray, gradient: LayerGradient | None = None
    ) -> None:
        frozen = np.array(passed_down, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        self._passed_down = frozen
        self._gradient = gradient

    @property
    def passed_down(self) -> FloatArray:
        """The arriving block for the layer beneath this one."""
        return self._passed_down

    @property
    def gradient(self) -> LayerGradient | None:
        """What this layer wants to change, or ``None`` if it learns nothing."""
        return self._gradient

    @property
    def learns(self) -> bool:
        """Whether this layer has parameters at all."""
        return self._gradient is not None

    def __repr__(self) -> str:
        return f"LayerCorrection(learns={self.learns!r})"


class BackwardPass:
    """One run of backpropagation: what it cost, and what to do about it.

    Parameters
    ----------
    loss:
        The averaged loss the forward pass measured.
    gradients:
        One entry per layer, bottom to top, in the same order the stack holds
        its layers. ``None`` where a layer has no parameters to learn.

    Raises
    ------
    EmptyValuesError
        If no gradients are supplied. A stack holds at least one layer.
    """

    __slots__ = ("_gradients", "_loss")

    def __init__(self, loss: float, gradients: Sequence[LayerGradient | None]) -> None:
        if not gradients:
            raise EmptyValuesError("a backward pass produces one gradient per layer")

        self._loss = float(loss)
        self._gradients = tuple(gradients)

    @property
    def loss(self) -> float:
        """The averaged loss, measured on the way forward."""
        return self._loss

    @property
    def largest_movement(self) -> float:
        """The biggest slope anywhere in the network.

        The convergence reading. When no parameter anywhere wants to move by
        more than a whisker, further epochs are buying nothing.

        Layers that learn nothing are skipped rather than counted as zero. A
        network of nothing but dropout and pooling has no parameters at all,
        and the honest answer for it is zero movement.
        """
        movements = [
            gradient.largest_movement
            for gradient in self._gradients
            if gradient is not None
        ]
        return max(movements) if movements else 0.0

    def __len__(self) -> int:
        """How many layers were corrected."""
        return len(self._gradients)

    def __getitem__(self, position: int) -> LayerGradient | None:
        """The gradient belonging to the layer at ``position``.

        Positional rather than named, because position *is* the key here. The
        layers are an ordered chain and the gradient at index ``k`` is layer
        ``k``'s, so asking for one by number is asking a real question rather
        than reaching into a list. That is the same argument
        :class:`~oop_ml.core.data.coefficients.Coefficients` makes for indexing
        by feature name.
        """
        return self._gradients[position]

    def __iter__(self) -> Iterator[LayerGradient | None]:
        """Each layer's gradient, bottom to top."""
        return iter(self._gradients)

    def __repr__(self) -> str:
        return f"BackwardPass(loss={self._loss!r}, n_layers={len(self._gradients)!r})"
