"""Many neurons reading one shared input, which is the whole of a layer.

What is actually new here
-------------------------
Not the unit. A layer holds the neurons you already have and adds one rule:
they all read the same thing. Every neuron gets the identical row and answers
with its own number, so ``m`` neurons over ``n`` inputs turn a row of ``n``
numbers into a row of ``m``.

That single rule is what changes the arithmetic underneath. One neuron owns a
weight *vector*; a layer owns a weight *matrix*, one row of it per neuron, and
the forward pass stops being a dot product and becomes a matrix multiply. None
of that is visible in the objects, which is deliberate: the matrix is an
optimisation waiting to happen, not the definition.

Why the widths cannot disagree
------------------------------
Every neuron in a layer is handed the same row, so a neuron expecting a
different number of inputs is not a slightly odd member, it is unfeedable. The
constructor refuses such a group, which is what makes :class:`LayerShape`
well defined for a layer that exists at all: ``n_inputs`` is the width they
agree on, and ``n_outputs`` is simply how many of them there are.

This is the :class:`~oop_ml.core.data.feature_set.FeatureSet` pattern. A rule
spanning several values belongs to an object that enforces it once, rather
than to a guard every caller has to remember, and an object whose type carries
the guarantee cannot be handed around in a broken state.

Why the output width owes nothing to the input width
-----------------------------------------------------
A layer of three neurons answers with three numbers whether it read four
inputs or four hundred. Widening it changes what the *next* layer must expect
and nothing about what this one accepts, which is why a network's shape is a
chain of separate equalities rather than one number carried throughout.

On the activations not having to match
--------------------------------------
Each neuron carries its own bend, so a layer may in principle mix them. Nothing
here forbids it, because nothing in the mathematics does, and a uniform layer
is a fact about ordinary practice rather than a law. It is worth knowing that
the uniform case is the one a vectorised forward pass can exploit, since one
bend applied to a whole matrix is a single call where a mixed layer needs one
per column.

Two routes through the same definition
---------------------------------------
:meth:`DenseLayer.respond_to` is the matrix multiply, which is what a layer's
weights have been all along, and it runs 294x to 2161x the neuron-at-a-time
loop it replaced. :meth:`DenseLayer.neuron_responses` is that loop, kept because a
step-by-step rendering wants one neuron's arithmetic rather than two blocks,
and it is the observed route in the sense
:mod:`oop_ml.core.observation` sets out. A test asserts the two agree, since a
fast path and a slow path with nothing between them are two implementations
rather than one calculation read two ways.

The agreement is ``approx`` and not exact. BLAS sums the products in a
different order than a Python loop does, so the two routes differ in the last
bits, which is the reassociation this library's memory-layout note already
records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy as np

from oop_ml.core.data.column import ColumnSource
from oop_ml.core.exceptions import (
    EmptyValuesError,
    ShapeMismatchError,
)
from oop_ml.core.network.blocks import as_block, check_finite
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.neuron import Neuron, NeuronResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


class LayerResponse:
    """Every neuron's score and output, for a whole block of rows.

    The plural of :class:`~oop_ml.core.network.neuron.NeuronResponse`, and it
    exists for the same reason. A forward pass reads the outputs; a backward
    pass needs the scores as well, because the slope of each bend is taken at
    the score that produced it, and recovering those later would mean running
    the whole layer again.

    ``scores`` and ``outputs`` share a shape and are aligned, so the same
    position in one describes the same unit on the same row as in the other.
    For a dense layer that shape is ``(n_rows, n_neurons)``; for a convolution
    it is ``(n_rows, n_filters, height, width)``. Only the leading axis is
    fixed, and it is always the rows, which is the same rule
    :meth:`Layer.respond_to` applies to what comes in.

    A layer with no bend, such as pooling, hands the same block as both. That
    is honest rather than lazy: there is no pre-activation value to keep,
    because nothing was activated.

    ``inputs`` is the block the layer read to produce them, kept because a
    backward pass needs it and because the forward pass is the only thing that
    knows it. Without it the backward walk has to rebuild the list of what each
    layer read by shifting the outputs down one and pushing the caller's block
    on the front, which is reconstructing information that existed a moment
    earlier.

    It is a copy rather than a reference, and not because it needs to be:
    :meth:`DenseLayer.respond_to` coerces every block it is handed through
    ``as_block``, which copies, so each layer's ``inputs`` is already a
    private array by the time it arrives here. Skipping that copy when the
    block is one this library produced is the ``already_checked`` trick again
    and is an optimisation to measure, not a correction.

    Parameters
    ----------
    inputs:
        ``(n_rows, n_inputs)``, the block the layer read.
    scores:
        ``(n_rows, n_neurons)``, the weighted sums plus biases, before any
        bend.
    outputs:
        ``(n_rows, n_neurons)``, each score bent by its own neuron's
        activation.

    Raises
    ------
    ShapeMismatchError
        If the two blocks are not the same shape, or either is not
        two-dimensional. They describe the same neurons on the same rows, so a
        disagreement between them is not a pair of results, it is a bug.
    InvalidValuesError
        If either block cannot be read as a float array at all.

    Notes
    -----
    Non-finite entries are *not* refused here, and that is a decision rather
    than an omission. Two of these are built per layer per call, and what would
    be caught is an overflow *produced* by the layer rather than handed to it,
    which a finite block times finite weights can still do. Each layer's
    entry point scans what it is given, so such a value is refused at the next
    layer, one join later than it began. The alternative is scanning every
    block twice, once on the way out and once on the way in.
    """

    __slots__ = ("_inputs", "_outputs", "_scores")

    @classmethod
    def already_checked(
        cls, inputs: FloatArray, scores: FloatArray, outputs: FloatArray
    ) -> LayerResponse:
        """Wrap two blocks a layer just built, skipping the copy and the checks.

        The :meth:`~oop_ml.core.data.column.Column.selecting` pattern. A forward
        pass allocates both blocks itself, fills every entry, and shares neither
        with anyone, so copying them to guard against a mutation that cannot
        happen is pure cost on the hot path. The checking constructor stays the
        boundary for every other caller.

        They are still frozen, because the *response* travels outward even
        though the arrays it holds were never anyone else's.
        """
        response = cls.__new__(cls)
        inputs.setflags(write=False)
        scores.setflags(write=False)
        outputs.setflags(write=False)
        response._inputs = inputs
        response._scores = scores
        response._outputs = outputs
        return response

    def __init__(
        self, inputs: FloatArray, scores: FloatArray, outputs: FloatArray
    ) -> None:
        frozen_inputs = as_block(inputs, "inputs")
        frozen_scores = as_block(scores, "scores")
        frozen_outputs = as_block(outputs, "outputs")

        if frozen_scores.ndim < 2 or frozen_outputs.ndim < 2:
            raise ShapeMismatchError(
                "a layer's scores and outputs both begin with the rows, got "
                f"{frozen_scores.ndim} and {frozen_outputs.ndim} dimensions"
            )
        if frozen_scores.shape != frozen_outputs.shape:
            raise ShapeMismatchError(
                f"scores {frozen_scores.shape} and outputs {frozen_outputs.shape} "
                "describe the same neurons on the same rows and must be the same "
                "shape"
            )

        if frozen_inputs.ndim < 2:
            raise ShapeMismatchError(
                "a layer reads a block whose first axis is the rows, got "
                f"{frozen_inputs.ndim} dimensions"
            )
        if frozen_inputs.shape[0] != frozen_scores.shape[0]:
            raise ShapeMismatchError(
                f"the block read has {frozen_inputs.shape[0]} rows and the "
                f"answers have {frozen_scores.shape[0]}"
            )

        frozen_inputs.setflags(write=False)
        frozen_scores.setflags(write=False)
        frozen_outputs.setflags(write=False)

        self._inputs = frozen_inputs
        self._scores = frozen_scores
        self._outputs = frozen_outputs

    @property
    def inputs(self) -> FloatArray:
        """``(n_rows, n_inputs)``, the block this layer read.

        A backward pass multiplies its correction by exactly this, and keeping
        it here is what lets that walk be a plain reversed loop rather than a
        reconstruction of what each layer was handed.
        """
        return self._inputs

    @property
    def scores(self) -> FloatArray:
        """``(n_rows, n_neurons)``, before any bend."""
        return self._scores

    @property
    def outputs(self) -> FloatArray:
        """``(n_rows, n_neurons)``, what the next layer reads."""
        return self._outputs

    @property
    def n_rows(self) -> int:
        """How many rows went through."""
        return int(self._scores.shape[0])

    @property
    def n_neurons(self) -> int:
        """How many neurons answered."""
        return int(self._scores.shape[1])

    def __eq__(self, other: object) -> bool:
        # ``type(self) is type(other)`` rather than ``isinstance``, which is the
        # rule Pool2d uses for the same reason. A subclass exists because it
        # carries something more -- a dropout mask, a batch's statistics -- and
        # that something is exactly what a plain response does not have, so the
        # two are not interchangeable however well their outputs match. Under
        # ``isinstance`` the subclass could not fix this for itself either: its
        # own ``__eq__`` would answer NotImplemented and Python would fall back
        # to this one, which would say they were equal.
        if not isinstance(other, LayerResponse) or type(self) is not type(other):
            return NotImplemented
        return bool(
            np.array_equal(self._scores, other._scores)
            and np.array_equal(self._outputs, other._outputs)
        )

    def __repr__(self) -> str:
        return f"LayerResponse(n_rows={self.n_rows!r}, n_neurons={self.n_neurons!r})"


class Layer(ABC):
    """Anything a network can stack: it reads a block and answers a block.

    Every layer here answers three questions and nothing more. What shape do I
    read and answer with. What do I do to a block going up. And, given the
    blame arriving from above, what do my own parameters want and what does the
    layer beneath deserve.

    Why the interface is this and not the obvious one
    -------------------------------------------------
    An earlier version had the stack pull ``weight_matrix`` and ``slopes_at``
    out of each layer and do the backward arithmetic itself. That works for
    exactly one kind of layer. A dropout layer has no weight matrix, a pooling
    layer has none, a flattening layer has no parameters at all, and none of
    the three could be expressed.

    So the backward step belongs to the layer, which is the only thing that
    knows how it works, and :class:`~oop_ml.core.network.gradient.LayerCorrection`
    is what it hands back. The stack becomes a walk that threads one block
    downward and knows nothing about weights.

    What a subclass supplies
    ------------------------
    :meth:`_response_for`, :meth:`correction_for` and :meth:`stepped_by`. The
    public :meth:`respond_to` is a template that settles the shape agreement
    once, so no subclass re-checks it.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def shape(self) -> LayerShape:
        """What this layer reads and what it answers with."""

    def respond_to(
        self, inputs: FloatArray, purpose: PassPurpose = PassPurpose.PREDICTING
    ) -> LayerResponse:
        """Push a block through this layer.

        The shape agreement is established here, once for the whole block,
        which is the payoff of a layer knowing its own shape. Nothing
        downstream re-checks it.

        One rule covers every layer type: the block's leading axis is the
        rows, and every axis after it must match :attr:`shape` exactly. That is
        a dense layer's ``(n_rows, n_inputs)`` and a convolution's
        ``(n_rows, channels, height, width)`` without either being a special
        case, and it is why this lives on the base rather than being written
        once per layer.

        Parameters
        ----------
        inputs:
            ``(n_rows, *shape.reads)``, one entry per observation.
        purpose:
            Why the pass is happening. Most layers ignore it entirely; see
            :mod:`oop_ml.core.network.purpose` for the two that cannot. The
            default answers deterministically, which is the safe direction to
            forget in.

        Returns
        -------
        LayerResponse
            What this layer read, the scores it formed, and what it answered.

        Raises
        ------
        ShapeMismatchError
            If the block's arrangement past the first axis is not what this
            layer reads.
        EmptyValuesError
            If the block holds no rows.
        InvalidValuesError
            If the block cannot be read as a float array, or any entry in it is
            not finite.
        """
        block = as_block(inputs, "inputs")
        check_finite(block, "inputs")

        if block.ndim < 2:
            raise ShapeMismatchError(
                "a layer reads a block whose first axis is the rows, got "
                f"{block.ndim} dimensions"
            )
        if block.shape[0] == 0:
            raise EmptyValuesError("a layer needs at least one row to respond to")
        if block.shape[1:] != self.shape.reads:
            raise ShapeMismatchError(
                f"this layer reads {self.shape.reads}, got a block arranged "
                f"{block.shape[1:]}"
            )

        return self._response_for(block, purpose)

    @abstractmethod
    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """The forward pass, given a block already known to be the right shape.

        ``purpose`` is passed to every layer rather than only to the two that
        read it, so that the signature is the same everywhere and a layer that
        starts caring later does not change the interface for the rest.
        """

    def _checked_arriving(
        self, response: LayerResponse, arriving: FloatArray
    ) -> FloatArray:
        """Read the arriving block, refusing one that cannot belong here.

        Every layer's backward step needs the same assurances, so the check
        lives here rather than being written once per layer. The response has
        to have come from a layer of this shape, the arriving block describes
        this layer's own outputs so its arrangement must be what this layer
        answers with, and the two must agree on how many rows there are.

        The first of those is the easiest to overlook and the worst to miss.
        A response paired with the wrong layer carries real numbers of a
        plausible shape, and the blame would be routed to arbitrary places
        without anything raising.

        What this cannot catch, named rather than left to be discovered
        ---------------------------------------------------------------
        The check reads shapes, so two layers of the same shape are
        indistinguishable to it, and same-shaped does not mean interchangeable.
        ``MaxPool2d(reads=(1, 4, 4), window=3, stride=1)`` and
        ``MaxPool2d(reads=(1, 4, 4), window=2, stride=2)`` both read
        ``(1, 4, 4)`` and answer ``(1, 2, 2)``, so a response from either is
        accepted by the other and the blame goes to positions that never won
        those windows. Nothing raises, and the numbers are plausible.

        Closing it would mean a response carrying which layer produced it,
        which is a field on the type every layer shares for a case no
        :class:`~oop_ml.core.network.stack.LayerStack` can reach: a stack pairs
        layer ``k`` with response ``k`` by construction, since it zips its own
        reversed layers against its own reversed responses. The exposure is to
        a caller pairing the two by hand, which is the observed route rather
        than the training one. So it is documented rather than defended, and
        this paragraph is the defence.

        Raises
        ------
        ShapeMismatchError
            If the arrangement or the row count does not match.
        InvalidValuesError
            If the block cannot be read as a float array.
        """
        if response.inputs.shape[1:] != self.shape.reads:
            raise ShapeMismatchError(
                f"this layer reads {self.shape.reads}, but the response came "
                f"from one reading {response.inputs.shape[1:]}"
            )

        block = as_block(arriving, "arriving")

        if block.ndim < 2 or block.shape[1:] != self.shape.answers:
            raise ShapeMismatchError(
                f"this layer answers with {self.shape.answers}, so the arriving "
                f"block must be arranged that way, got {block.shape[1:]}"
            )
        if block.shape[0] != response.outputs.shape[0]:
            raise ShapeMismatchError(
                f"the response holds {response.outputs.shape[0]} rows and the "
                f"arriving block holds {block.shape[0]}"
            )

        return block

    @abstractmethod
    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """This layer's own backward step.

        Parameters
        ----------
        response:
            What this layer did on the way up, including the block it read.
        arriving:
            ``(n_rows, *shape.answers)``, the slope of the loss at this layer's
            outputs. See :meth:`~oop_ml.core.network.stack.LayerStack.backward_pass`
            for what that means.

        Returns
        -------
        LayerCorrection
            What this layer's parameters want, if it has any, and the arriving
            block for the layer beneath it.
        """

    @abstractmethod
    def stepped_by(self, gradient: LayerGradient | None, learning_rate: float) -> Layer:
        """A new layer of the same shape, with its parameters nudged.

        Training does not mutate. A layer with nothing to learn answers with
        itself, which is correct rather than a shortcut: it is unchanged, and
        it is immutable, so there is nothing to copy.
        """


class DenseLayer(Layer):
    """A group of neurons that all read the same input.

    Parameters
    ----------
    neurons:
        At least one, every one of them reading the same number of inputs. The
        order is the order their answers come back in, and it is the order the
        next layer's weights will be aligned to.

    Raises
    ------
    EmptyValuesError
        If no neurons are supplied. A layer that answers nothing has no width
        for the next layer to agree with.
    ShapeMismatchError
        If the neurons do not all read the same number of inputs. They are
        handed the identical row, so one expecting a different width could
        never be fed at all.
    """

    __slots__ = ("_biases", "_neurons", "_shape", "_shared_activation", "_weights")

    def __init__(self, neurons: Sequence[Neuron]) -> None:
        if not neurons:
            raise EmptyValuesError("a layer needs at least one neuron")

        widths = {neuron.n_inputs for neuron in neurons}
        if len(widths) != 1:
            raise ShapeMismatchError(
                "every neuron in a layer reads the same row, so they must agree "
                f"on its width, got {sorted(widths)}"
            )

        self._neurons = tuple(neurons)

        # Assembled once rather than per call. Safe only because a neuron's
        # weights are frozen and this tuple cannot be reordered, so no update
        # can make the cache disagree with the neurons it was built from.
        self._weights = np.stack([neuron.weights.values for neuron in self._neurons])
        self._biases = np.array(
            [neuron.bias for neuron in self._neurons], dtype=np.float64
        )
        self._weights.setflags(write=False)
        self._biases.setflags(write=False)

        bends = {neuron.activation for neuron in self._neurons}
        self._shared_activation = bends.pop() if len(bends) == 1 else None

        self._shape = LayerShape(
            n_inputs=next(iter(widths)), n_outputs=len(self._neurons)
        )

    @property
    def shape(self) -> LayerShape:
        """The width this layer reads and the width it answers with."""
        return self._shape

    @property
    def weight_matrix(self) -> FloatArray:
        """``(n_neurons, n_inputs)``, one row per neuron, in neuron order.

        The same numbers the neurons hold, assembled once at construction. A
        backward pass needs them as a matrix to pass the correction downward,
        and a saved document or a diagram wants the same view.
        """
        return self._weights

    @property
    def bias_vector(self) -> FloatArray:
        """``(n_neurons,)``, one bias per neuron, in neuron order."""
        return self._biases

    def slopes_at(self, scores: FloatArray) -> FloatArray:
        """Each neuron's activation slope at the scores it formed.

        The backward pass needs the bend's derivative taken where the forward
        pass evaluated it, and a layer is the only thing that knows which bend
        belongs to which column.

        Parameters
        ----------
        scores:
            ``(n_rows, n_neurons)``, as returned by :meth:`respond_to`.

        Returns
        -------
        FloatArray
            The same shape, each entry the slope of that neuron's bend at that
            score.
        """
        if self._shared_activation is not None:
            return self._shared_activation.derivative_at(scores)

        slopes = np.empty_like(scores)
        for index, neuron in enumerate(self._neurons):
            slopes[:, index] = neuron.activation.derivative_at(scores[:, index])
        return slopes

    def with_parameters(self, weights: FloatArray, biases: FloatArray) -> DenseLayer:
        """A new layer of the same shape and bends, carrying these parameters.

        Training does not mutate. A step builds the next layer from the last
        one's bends and the corrected numbers, which is what keeps every
        neuron's weights frozen and therefore keeps the cached matrix honest.
        Measured, rebuilding a whole stack costs three forward passes at
        website scale and a quarter of one at ten thousand parameters, so the
        immutability is affordable at both ends.

        Parameters
        ----------
        weights:
            ``(n_neurons, n_inputs)``, matching this layer's shape.
        biases:
            ``(n_neurons,)``.

        Returns
        -------
        DenseLayer
            A new layer, same neuron order and same activations.

        Raises
        ------
        ShapeMismatchError
            If either block does not match this layer's shape.
        """
        weight_block = as_block(weights, "weights")
        bias_block = as_block(biases, "biases")

        expected = (self._shape.n_outputs, self._shape.n_inputs)
        if weight_block.shape != expected:
            raise ShapeMismatchError(
                f"this layer's weights are {expected}, got {weight_block.shape}"
            )
        if bias_block.shape != (self._shape.n_outputs,):
            raise ShapeMismatchError(
                f"this layer's biases are ({self._shape.n_outputs},), got "
                f"{bias_block.shape}"
            )

        return DenseLayer(
            [
                Neuron(row, bias=float(bias), activation=neuron.activation)
                for row, bias, neuron in zip(
                    weight_block, bias_block, self._neurons, strict=True
                )
            ]
        )

    def __len__(self) -> int:
        """How many neurons the layer holds, which is its output width."""
        return len(self._neurons)

    def __getitem__(self, position: int) -> Neuron:
        """The neuron at ``position``, which is the column it answers in."""
        return self._neurons[position]

    def __iter__(self) -> Iterator[Neuron]:
        """The neurons, in the order their answers come back in.

        The layer is iterable rather than exposing its container, so that
        nothing outside it can reorder the neurons and silently transpose the
        meaning of every column it answers with.
        """
        return iter(self._neurons)

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """The forward pass, given a block already known to be the right width.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_inputs)``, aligned with every neuron's weights.
        purpose:
            Ignored. A dense layer does the same arithmetic either way.

        Returns
        -------
        LayerResponse
            Scores and outputs, both ``(n_rows, n_neurons)``, where entry
            ``[row, neuron]`` is what that neuron did with that row.

        Notes
        -----
        The definition is a loop: for each row, ask each neuron. This is that
        loop written as the matrix multiply it always was, because a layer's
        weights *are* a matrix, one row of it per neuron, and asking all of
        them at once is one BLAS call rather than ``n_rows * n_neurons`` Python
        round trips.

        Measured against the neuron loop, on the same inputs::

            rows x inputs -> neurons      loop        matmul
                 32 x  8  ->   8       1.435 ms     0.005 ms      294x
                256 x 32  ->  64     142.515 ms     0.066 ms     2161x
               1024 x 64  -> 128    1056.437 ms     1.860 ms      568x

        Three things make that safe. The matrix is assembled once in
        :meth:`__init__` rather than per call, which is sound only because a
        neuron's weights are frozen and the neuron tuple cannot be reordered,
        so the cache can never go stale. The blocks are wrapped by
        :meth:`LayerResponse.already_checked`, since this method allocated them
        and has shared them with nobody. And the answers agree with the neurons
        asked one at a time to within floating point but *not* bit for bit,
        because BLAS sums the products in a different order than a Python loop
        does, which is why the agreement test says ``approx``.

        A uniform layer bends the whole block in one call. A mixed one goes
        column by column, since an elementwise function applied to the whole
        matrix cannot vary per neuron, and that is why the shared bend is
        worked out once at construction rather than per call.

        The neuron-at-a-time reading has not gone away. It is
        :meth:`neuron_responses`, kept as the observed route, and a test
        asserts the two agree.
        """
        scores = inputs @ self._weights.T + self._biases

        if self._shared_activation is not None:
            outputs = self._shared_activation.of(scores)
        else:
            outputs = np.empty_like(scores)
            for index, neuron in enumerate(self._neurons):
                outputs[:, index] = neuron.activation.of(scores[:, index])

        return LayerResponse.already_checked(
            inputs=inputs, scores=scores, outputs=outputs
        )

    def neuron_responses(self, inputs: ColumnSource) -> tuple[NeuronResponse, ...]:
        """Every neuron's own answer for one row, in order. The observed route.

        :meth:`respond_to` multiplies a matrix and hands back two blocks, which
        is the right thing for a fit and the wrong thing for anybody who wants
        to show what a single neuron did. This walks the same definition one
        neuron at a time, so each answer arrives as the
        :class:`~oop_ml.core.network.neuron.NeuronResponse` that neuron built,
        with its own score and its own output.

        Free to be slower, because it is called deliberately on one row sized
        for looking at rather than inside a training loop.

        Parameters
        ----------
        inputs:
            One row, ``(n_inputs,)``, or anything a neuron accepts as a row.

        Returns
        -------
        tuple of NeuronResponse
            One per neuron, in the order the layer holds them, which is the
            order :meth:`respond_to` puts them in columns.

        Raises
        ------
        NonEqualArrayLengthError
            If the row does not carry one value per input, by way of the
            neurons.
        InvalidValuesError
            If any value in the row is not finite, likewise.
        """
        return tuple(neuron.respond_to(inputs) for neuron in self._neurons)

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """This layer's backward step: the four lines, owned by the layer.

        ``arriving`` is the slope of the loss at this layer's outputs. The
        bend's slope converts that to a slope at the scores, which is what the
        weights actually act on, and everything else follows::

            delta       = arriving * slopes_at(scores)
            dL/dweights = delta.T @ what this layer read
            dL/dbiases  = delta.sum(axis=0)
            passed_down = delta @ weight_matrix

        Every shape is forced. If a product does not conform, a transpose is on
        the wrong side.
        """
        arriving = self._checked_arriving(response, arriving)

        delta = arriving * self.slopes_at(response.scores)

        return LayerCorrection(
            passed_down=delta @ self._weights,
            gradient=LayerGradient(
                weights=delta.T @ response.inputs, biases=delta.sum(axis=0)
            ),
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> DenseLayer:
        """A new dense layer with every parameter moved against its slope.

        Downhill is *against* the slope, so this subtracts. The layer is
        rebuilt rather than mutated, which is what keeps every neuron's weights
        frozen and this layer's cached matrix honest.
        """
        if gradient is None:
            raise ShapeMismatchError(
                "a dense layer has parameters and needs a gradient to step by"
            )

        return self.with_parameters(
            self._weights - learning_rate * gradient.weights,
            self._biases - learning_rate * gradient.biases,
        )

    def __repr__(self) -> str:
        return f"DenseLayer(n_inputs={self._shape.n_inputs!r}, n_neurons={len(self)!r})"
