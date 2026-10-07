"""Layers that read a sequence one step at a time, carrying what they read.

What is actually new here
-------------------------
Every layer before this one reads a row and answers, and two rows never meet.
A sequence is not like that. The third word of a sentence means something
different after "not" than after "very", so a layer that reads sequences has to
carry something from the step before into the step after, and that carried
thing is the state.

So the arithmetic gains a loop it did not have. A dense layer applies its
weights once per row; a recurrent layer applies the *same* weights once per
step, threading a state through, and the answer at step ``t`` depends on every
step before it. That single change is the whole family, and it is also where
every difficulty comes from.

Why time is an extent rather than a free dimension
---------------------------------------------------
This package's promise is that a network which cannot work is refused at
construction, in integer comparisons, before a row is read. Keeping it means a
recurrent layer has to know its step count up front, so a sequence layer reads
``(n_steps, n_features)`` and time is an extent like a picture's height::

    recurrent    (12, 3)   -> (8,)        answering once, at the end
    recurrent    (12, 3)   -> (12, 8)     answering at every step

The cost is real and is stated rather than hidden. Sequences of unequal length
have to be padded or cut to one length before they reach the layer, where an
implementation that discovered the step count from the block would take them as
they came. That is the trade this package has already made twice, and it is the
same one: an arrangement settled in advance is an arrangement that can be
checked in advance.

Why one weight matrix and not two
----------------------------------
Written the usual way, a recurrent cell looks like it holds two matrices, one
reading the input and one reading the previous state::

    h_t  =  tanh( W_input @ x_t  +  W_state @ h_(t-1)  +  b )

Those two products are one product against a longer row. Lay the input and the
previous state end to end and the pair collapses::

    combined  =  [ x_t , h_(t-1) ]              (n_features + n_units,)
    h_t       =  tanh( W @ combined  +  b )     W is (n_units, n_features + n_units)

That is not a trick to save a line. It means a recurrent layer's parameters are
one weight block and one bias block, of exactly the arrangement
:class:`~oop_ml.core.network.gradient.LayerGradient` was written for, so the
gradient object holds a recurrent layer unchanged and nothing in the backward
walk had to grow to accommodate this family. A layer with several gates stacks
them down the rows of that one matrix, so a long short-term memory's four gates
make it ``(4 * n_units, n_features + n_units)`` and the gate a row belongs to is
which quarter it falls in.

Why the weights' gradient is a sum and not a value
---------------------------------------------------
This is the one genuinely new piece of arithmetic in the module, and it follows
from the weights being reused rather than from anything about gates.

A dense layer's weight appears once in the calculation, so its slope is one
term. A recurrent layer's weight appears at *every step*, so the chain rule
gives one term per step and the slope is their sum. Twelve steps means twelve
contributions added into one block, and dropping the accumulation leaves a
gradient that describes the last step alone, which trains, descends, and is not
the gradient of the thing being minimised.

The walk backward therefore runs from the last step to the first, and carries
two things with it rather than one. At each step it holds what the loss wants
from that step's state, which is the sum of what arrived from above at that step
and what the *next* step wants from the state it was handed. That second half is
what makes it a walk through time rather than a loop.

Why the state's slope shrinks, and what the gates are for
----------------------------------------------------------
Going back one step multiplies the state's slope by the state-to-state part of
the weights and by the slope of the bend. Do that twelve times and the factor is
that quantity to the twelfth power. Anything below one vanishes and anything
above one explodes, so a plain recurrent layer learns almost nothing about what
happened long ago, and the failure is silent: the fit converges, and it
converged using only the recent past.

The two gated cells here exist to answer that, and they answer it differently.
A long short-term memory adds a second carried quantity, the cell, whose update
is an addition rather than a multiplication, so a slope travelling back along it
is multiplied by the forget gate rather than by weights and a bend. A gated
recurrent unit keeps one carried quantity and interpolates between keeping it
and replacing it, which gives the same additive route with fewer parameters.
Both are the same idea, that a path backward through time should be able to
avoid being multiplied by anything much smaller than one.

Why the response carries a trace
---------------------------------
:class:`RecurrentResponse` keeps every step's state, gate values and inputs,
because the backward walk needs all of them and only the forward pass knows
them. Recovering them would mean running the sequence again, and a gate value
is not recoverable from the answer at all. It is the same argument
:class:`~oop_ml.core.network.dropout.DropoutResponse` makes about its mask.

Why a base and three subclasses
--------------------------------
The loop over steps, the state threading, the accumulation of the weight
gradient over steps, the two output modes and every shape check are identical
across the three, and only the arithmetic inside one step differs. That is the
shape :class:`~oop_ml.core.network.pooling.Pool2d` already uses, where a base
owns the sweep and a subclass supplies the one function that differs. A
subclass here supplies how many gates it holds, how one step advances, and how
one step is walked back, and nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import sqrt

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray

SEQUENCE_EXTENTS = 2
"""A sequence layer reads ``(n_steps, n_features)``, which is two extents."""


def _checked_setting(value: int, lowest: int, name: str) -> int:
    """Read one whole-number setting, refusing anything below its floor.

    Parameters
    ----------
    value:
        What the caller supplied.
    lowest:
        The smallest value that means anything for this setting.
    name:
        What to call it in the refusal.

    Returns
    -------
    int
        The setting, as an ordinary integer.

    Raises
    ------
    InvalidValuesError
        If the value is not a whole number, or is below its floor.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidValuesError(f"{name} is a whole number, got {value!r}")
    if value < lowest:
        raise InvalidValuesError(f"{name} is at least {lowest}, got {value}")
    return value


def _logistic(scores: FloatArray) -> FloatArray:
    """The logistic curve, written so a large negative score cannot overflow.

    ``exp`` of a large positive number is an infinity, so the two halves are
    computed differently and stitched together, which is the same guard
    :mod:`oop_ml.core.logistic` applies for the same reason. It is repeated here
    rather than imported because that module answers for a whole model's scores
    and this is one block inside one step.
    """
    positive = scores >= 0.0
    answer = np.empty_like(scores)
    answer[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exponential = np.exp(scores[~positive])
    answer[~positive] = exponential / (1.0 + exponential)
    return answer


class CellStep:
    """What one step of a recurrent cell produced, kept for the walk back.

    A step produces several things that belong together and are useless apart,
    which is the pairing this library models rather than returns as a tuple.

    Parameters
    ----------
    state:
        ``(n_rows, n_units)``, what this step hands to the next one and what
        the layer answers with at this step.
    cell:
        ``(n_rows, n_units)``, the second carried quantity, for the one cell
        that has one. The others carry a block of zeros here, which costs an
        allocation per step and keeps the base's loop free of a branch on which
        subclass it is threading.
    gates:
        ``(n_rows, n_gates * n_units)``, every gate's value after its own bend.
        The backward step reads these and could not recompute them from the
        state, which is the reason a response carries a trace at all.
    previous_state:
        ``(n_rows, n_units)``, the state this step was handed.
    previous_cell:
        ``(n_rows, n_units)``, the cell this step was handed.
    inputs:
        ``(n_rows, n_features)``, the block this step read.
    """

    __slots__ = (
        "_cell",
        "_gates",
        "_inputs",
        "_previous_cell",
        "_previous_state",
        "_state",
    )

    def __init__(
        self,
        state: FloatArray,
        cell: FloatArray,
        gates: FloatArray,
        previous_state: FloatArray,
        previous_cell: FloatArray,
        inputs: FloatArray,
    ) -> None:
        self._state = state
        self._cell = cell
        self._gates = gates
        self._previous_state = previous_state
        self._previous_cell = previous_cell
        self._inputs = inputs

    @property
    def state(self) -> FloatArray:
        """``(n_rows, n_units)``, what this step hands forward."""
        return self._state

    @property
    def cell(self) -> FloatArray:
        """``(n_rows, n_units)``, the second carried quantity."""
        return self._cell

    @property
    def gates(self) -> FloatArray:
        """``(n_rows, n_gates * n_units)``, every gate after its bend."""
        return self._gates

    @property
    def previous_state(self) -> FloatArray:
        """``(n_rows, n_units)``, the state handed to this step."""
        return self._previous_state

    @property
    def previous_cell(self) -> FloatArray:
        """``(n_rows, n_units)``, the cell handed to this step."""
        return self._previous_cell

    @property
    def inputs(self) -> FloatArray:
        """``(n_rows, n_features)``, the block this step read."""
        return self._inputs

    @property
    def combined(self) -> FloatArray:
        """``(n_rows, n_features + n_units)``, the row the weights multiplied.

        The input and the previous state laid end to end, which is the form the
        module docstring's collapse of two matrices into one produces. Built on
        demand rather than stored, since the two halves are already here and the
        concatenation is what the backward step wants rather than what the
        forward one kept.
        """
        return np.concatenate((self._inputs, self._previous_state), axis=1)

    def __repr__(self) -> str:
        return f"CellStep(state={self._state.shape!r})"


class CellCorrection:
    """What walking one step backward worked out, which is five things.

    All five come out of the same arithmetic and none is usable without the
    others, so they travel as one object rather than as a tuple whose order a
    caller would have to remember.

    Parameters
    ----------
    weight_slopes:
        ``(n_gates * n_units, n_features + n_units)``, this step's contribution
        to the weight gradient. The base adds it to the running total, because
        the same weights are used at every step.
    bias_slopes:
        ``(n_gates * n_units,)``, this step's contribution to the bias
        gradient.
    input_slopes:
        ``(n_rows, n_features)``, what the loss wants from the block this step
        read. These are the layer's answer to the layer beneath it, at this
        step.
    state_slopes:
        ``(n_rows, n_units)``, what the loss wants from the state this step was
        handed, which is what the previous step must add to whatever arrives at
        it from above.
    cell_slopes:
        ``(n_rows, n_units)``, the same for the second carried quantity.
    """

    __slots__ = (
        "_bias_slopes",
        "_cell_slopes",
        "_input_slopes",
        "_state_slopes",
        "_weight_slopes",
    )

    def __init__(
        self,
        weight_slopes: FloatArray,
        bias_slopes: FloatArray,
        input_slopes: FloatArray,
        state_slopes: FloatArray,
        cell_slopes: FloatArray,
    ) -> None:
        self._weight_slopes = weight_slopes
        self._bias_slopes = bias_slopes
        self._input_slopes = input_slopes
        self._state_slopes = state_slopes
        self._cell_slopes = cell_slopes

    @property
    def weight_slopes(self) -> FloatArray:
        """``(n_gates * n_units, n_features + n_units)``, this step's share."""
        return self._weight_slopes

    @property
    def bias_slopes(self) -> FloatArray:
        """``(n_gates * n_units,)``, this step's share."""
        return self._bias_slopes

    @property
    def input_slopes(self) -> FloatArray:
        """``(n_rows, n_features)``, for the layer beneath, at this step."""
        return self._input_slopes

    @property
    def state_slopes(self) -> FloatArray:
        """``(n_rows, n_units)``, for the step before this one."""
        return self._state_slopes

    @property
    def cell_slopes(self) -> FloatArray:
        """``(n_rows, n_units)``, for the step before this one."""
        return self._cell_slopes

    def __repr__(self) -> str:
        return f"CellCorrection(weights={self._weight_slopes.shape!r})"


class RecurrentResponse(LayerResponse):
    """A layer response that also remembers every step of the sequence.

    The backward walk needs each step's gate values, the state it was handed
    and the block it read, and the forward pass is the only thing that knows
    them. A gate value in particular cannot be recovered from the layer's
    answer by any means, so keeping it is not an optimisation.

    ``scores`` and ``outputs`` hold whatever the layer answers with, which is
    the last state or every state depending on how the layer was built. The
    two are the same block, because a recurrent layer's bends are applied
    inside its step rather than to its answer, so there is no pre-activation
    value distinct from the answer. That is the same honesty
    :class:`~oop_ml.core.network.pooling.Pool2d` practises when it hands the
    same block as both.
    """

    __slots__ = ("_steps",)

    @classmethod
    def recording(
        cls, inputs: FloatArray, outputs: FloatArray, steps: Sequence[CellStep]
    ) -> RecurrentResponse:
        """Wrap a forward pass this layer just built, along with its trace.

        The :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked`
        pattern, sound for the same reason: the caller allocated every block,
        filled it, and has shared none of them.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_steps, n_features)``, the sequence block read.
        outputs:
            What the layer answered with, used as the scores as well.
        steps:
            One record per step, in the order they were taken.
        """
        response = cls.__new__(cls)
        inputs.setflags(write=False)
        outputs.setflags(write=False)
        response._inputs = inputs
        response._scores = outputs
        response._outputs = outputs
        response._steps = tuple(steps)
        return response

    @property
    def steps(self) -> tuple[CellStep, ...]:
        """One record per step, in the order the sequence was read."""
        return self._steps

    def __eq__(self, other: object) -> bool:
        # The trace is why this class exists, so two responses that took
        # different routes to the same answer are not the same response. The
        # base settles the type check and the blocks; this adds the trace.
        #
        # The base answers NotImplemented when the two types differ, and
        # ``bool(NotImplemented)`` is True, so coercing the base's answer would
        # report two different types equal, which is the failure this override
        # exists to prevent. It is guarded explicitly.
        if not isinstance(other, RecurrentResponse):
            return NotImplemented
        settled = super().__eq__(other)
        if settled is NotImplemented or settled is False:
            return settled
        if len(self._steps) != len(other._steps):
            return False
        return all(
            bool(np.array_equal(mine.gates, theirs.gates))
            and bool(np.array_equal(mine.state, theirs.state))
            for mine, theirs in zip(self._steps, other._steps, strict=True)
        )

    def __hash__(self) -> int:
        return NotImplemented

    def __repr__(self) -> str:
        return f"RecurrentResponse(n_steps={len(self._steps)!r})"


class RecurrentLayer(Layer, ABC):
    """A layer that reads a sequence, threading a state from step to step.

    The base owns everything that does not depend on which cell this is: the
    geometry, the loop over steps, the two output modes, the accumulation of
    the weight gradient across steps, and every refusal. A subclass supplies
    how many gates it holds and the arithmetic of one step in each direction.

    Parameters
    ----------
    reads:
        ``(n_steps, n_features)``. A sequence of that many steps, each holding
        that many measurements.
    n_units:
        How wide the state is, which is also how wide the layer answers.
    answers_every_step:
        Whether the layer answers with the state at every step, arranged
        ``(n_steps, n_units)``, or with the last state alone, ``(n_units,)``.
        The first is what a second recurrent layer above this one needs to
        read; the second is what a classifier reading one answer per sequence
        needs. It defaults to the last state, because a chain that ends in a
        dense layer is the common case and the shape check catches the mistake
        either way.
    random_seed:
        Fixes the starting weights.

    Raises
    ------
    ShapeMismatchError
        If ``reads`` is not a sequence of two extents. A bare integer is the
        natural mistake, since it is what a dense layer's width looks like, and
        it earns a message that says so rather than one from builtins.
    InvalidValuesError
        If any setting is not a whole number, or is below its floor.
    """

    def __init__(
        self,
        reads: Sequence[int],
        n_units: int,
        answers_every_step: bool = False,
        random_seed: int = 0,
    ) -> None:
        try:
            extents = tuple(reads)
        except TypeError:
            raise ShapeMismatchError(
                "a recurrent layer reads (n_steps, n_features), got "
                f"{reads!r}, which is not a sequence of extents"
            ) from None
        if len(extents) != SEQUENCE_EXTENTS:
            raise ShapeMismatchError(
                "a recurrent layer reads (n_steps, n_features), got "
                f"{len(extents)} extents"
            )

        n_steps = _checked_setting(extents[0], 1, "n_steps")
        n_features = _checked_setting(extents[1], 1, "n_features")
        checked_units = _checked_setting(n_units, 1, "n_units")

        gate_rows = self.n_gates * checked_units
        combined_width = n_features + checked_units
        generator = np.random.default_rng(random_seed)
        weights = generator.normal(size=(gate_rows, combined_width)) * sqrt(
            1.0 / combined_width
        )

        self._configure(
            shape=LayerShape(
                (n_steps, n_features),
                (n_steps, checked_units) if answers_every_step else (checked_units,),
            ),
            n_units=checked_units,
            answers_every_step=answers_every_step,
            weights=weights,
            biases=self._starting_biases(checked_units),
        )

    def _starting_biases(self, n_units: int) -> FloatArray:
        """What the biases are before anything is learned, which is nothing.

        Overridden by the one cell for which zero is a bad place to start. See
        :meth:`LongShortTermMemory._starting_biases` for the measurement that
        made it worth a method rather than a constant.
        """
        return np.zeros(self.n_gates * n_units)

    def _configure(
        self,
        shape: LayerShape,
        n_units: int,
        answers_every_step: bool,
        weights: FloatArray,
        biases: FloatArray,
    ) -> None:
        """Fill every slot from an already validated configuration.

        Separated from ``__init__`` for the reason
        :meth:`~oop_ml.core.network.convolution.Conv2d._configure` gives: there
        are two ways to arrive at a layer and only one of them draws random
        numbers. A training step builds the next layer from the last one's
        geometry and a corrected set of parameters, and rerunning the
        constructor would throw a draw away and misreport where the numbers
        came from.
        """
        self._shape = shape
        self._n_units = n_units
        self._answers_every_step = answers_every_step
        self._n_steps, self._n_features = shape.reads

        weight_block = np.array(weights, dtype=np.float64, copy=True)
        bias_block = np.array(biases, dtype=np.float64, copy=True)
        weight_block.setflags(write=False)
        bias_block.setflags(write=False)
        self._weights = weight_block
        self._biases = bias_block

    def with_parameters(
        self, weights: FloatArray, biases: FloatArray
    ) -> RecurrentLayer:
        """The same layer with different parameters, and nothing else changed.

        Parameters
        ----------
        weights:
            ``(n_gates * n_units, n_features + n_units)``.
        biases:
            ``(n_gates * n_units,)``.

        Returns
        -------
        RecurrentLayer
            A layer of this exact type and geometry, carrying these parameters.

        Raises
        ------
        ShapeMismatchError
            If either block is not the arrangement this layer's parameters have.
        """
        gate_rows = self.n_gates * self._n_units
        expected_weights = (gate_rows, self._n_features + self._n_units)
        weight_block = np.asarray(weights, dtype=np.float64)
        bias_block = np.asarray(biases, dtype=np.float64)

        if weight_block.shape != expected_weights:
            raise ShapeMismatchError(
                f"this layer's weights are {expected_weights}, got {weight_block.shape}"
            )
        if bias_block.shape != (gate_rows,):
            raise ShapeMismatchError(
                f"this layer's biases are {(gate_rows,)}, got {bias_block.shape}"
            )

        replacement = self.__class__.__new__(self.__class__)
        replacement._configure(
            shape=self._shape,
            n_units=self._n_units,
            answers_every_step=self._answers_every_step,
            weights=weight_block,
            biases=bias_block,
        )
        return replacement

    @property
    def shape(self) -> LayerShape:
        """``(n_steps, n_features)`` in, and out either ``(n_units,)`` or
        ``(n_steps, n_units)``.

        Both sides carry their arrangement rather than a count, so a chain that
        feeds a sequence answer into a layer expecting a row is refused before
        any data arrives rather than agreeing by accident on the element count.
        """
        return self._shape

    @property
    def n_units(self) -> int:
        """How wide the state is."""
        return self._n_units

    @property
    def n_steps(self) -> int:
        """How many steps this layer reads."""
        return self._n_steps

    @property
    def answers_every_step(self) -> bool:
        """Whether the answer is every state or only the last one."""
        return self._answers_every_step

    @property
    def weights(self) -> FloatArray:
        """``(n_gates * n_units, n_features + n_units)``, frozen.

        Every gate stacked down the rows, each reading the input and the
        previous state laid end to end. Frozen, because a caller who edited it
        would change a layer that has already answered.
        """
        return self._weights

    @property
    def biases(self) -> FloatArray:
        """``(n_gates * n_units,)``, frozen, aligned with the weight rows."""
        return self._biases

    @property
    @abstractmethod
    def n_gates(self) -> int:
        """How many gates this cell holds, which sets the weight block's height.

        Read during construction, before any slot is filled, so it must depend
        on the class rather than on any instance state.
        """

    @abstractmethod
    def _advance(
        self,
        inputs_at_step: FloatArray,
        previous_state: FloatArray,
        previous_cell: FloatArray,
    ) -> CellStep:
        """One step forward, given what the step reads and what it is handed."""

    @abstractmethod
    def _retreat(
        self, step: CellStep, state_slopes: FloatArray, cell_slopes: FloatArray
    ) -> CellCorrection:
        """One step backward, given what the loss wants from this step's answer."""

    def respond_to(
        self, inputs: FloatArray, purpose: PassPurpose = PassPurpose.PREDICTING
    ) -> RecurrentResponse:
        """Push a sequence block through this layer.

        Overridden only to narrow what comes back. Every layer here answers with
        a :class:`~oop_ml.core.network.layer.LayerResponse`, and this family
        answers with one that also carries its trace, which a caller reading the
        steps needs to be able to see without asking. The base still owns every
        check, and this method adds none.

        The narrowing runs *after* the call rather than before it, so a block of
        the wrong arrangement is refused by this library's own typed error
        rather than turning into an assertion. That ordering is the same trap
        recorded three times over in the neighbour models.
        """
        response = super().respond_to(inputs, purpose)
        assert isinstance(response, RecurrentResponse)
        return response

    def _response_for(
        self, inputs: FloatArray, purpose: PassPurpose
    ) -> RecurrentResponse:
        """Run the sequence, threading the state from step to step.

        The state and the cell both start at zero, which is the ordinary
        convention and is a statement that nothing has been read yet. A learned
        starting state is a real alternative and is not offered here, because it
        is a parameter of a different arrangement than the weight block and
        would be the first thing in this package to need a gradient object that
        holds three blocks.
        """
        # The loop is the layer. Everything else in this module is arithmetic
        # inside one turn of it.
        #
        #   state and cell both begin as zeros, (n_rows, self._n_units). Zero is
        #   a statement that nothing has been read yet.
        #
        #   for position in range(self._n_steps):
        #       step = self._advance(inputs[:, position, :], state, cell)
        #       keep the step, then carry step.state and step.cell into the
        #       next turn.
        #
        # The answer is every step's state stacked on axis 1 when this layer
        # answers at every step, and a copy of the last step's state when it does
        # not. Axis 1 rather than 0 because the rows stay leading, which is the
        # one rule respond_to holds every layer to.
        #
        # Then RecurrentResponse.recording(inputs, outputs, steps), which freezes
        # the blocks and keeps the trace the backward walk cannot do without.
        cell_steps: list[CellStep] = []
        state = np.zeros(shape=(inputs.shape[0], self.n_units))
        cell = np.zeros(shape=(inputs.shape[0], self.n_units))

        for position in range(self._n_steps):
            cell_step = self._advance(inputs[:, position], state, cell)
            cell_steps.append(cell_step)
            state = cell_step.state
            cell = cell_step.cell

        if self._answers_every_step:
            outputs = np.stack([step.state for step in cell_steps], axis=1)
        else:
            outputs = np.array(cell_steps[-1].state, copy=True)

        return RecurrentResponse.recording(inputs, outputs, cell_steps)

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """Walk the sequence backward, summing the weight gradient over steps.

        The walk runs from the last step to the first and carries what the loss
        wants from each step's state. At step ``t`` that is the sum of two
        things: whatever arrived from the layer above at step ``t``, which is
        the whole arriving block when the layer answers only at the end and one
        slice of it when it answers at every step, and whatever step ``t + 1``
        decided it wanted from the state it was handed. The second term is the
        walk through time, and it is the term that vanishes over long sequences.

        The weight and bias gradients are running totals rather than values,
        because the same parameters were used at every step. See the module
        docstring for why a gradient that keeps only the last step's term is
        wrong in a way that still trains.

        Parameters
        ----------
        response:
            What this layer did on the way up, carrying its trace.
        arriving:
            ``(n_rows, *shape.answers)``, the slope of the loss at this layer's
            answer.

        Returns
        -------
        LayerCorrection
            The summed parameter slopes, and a block arranged
            ``(n_rows, n_steps, n_features)`` for the layer beneath.

        Raises
        ------
        ShapeMismatchError
            If the arriving block is not what this layer answers with, if the
            row counts disagree, or if the response did not come from a
            recurrent layer and so carries no trace.
        """
        # Two guards, then the walk.
        #
        #   block = self._checked_arriving(response, arriving)
        #   then refuse a response that is not a RecurrentResponse, since a
        #   plain one carries no trace and the walk cannot start at all.
        #
        # Three running totals, all zeros to begin:
        #   weight_total   like self._weights
        #   bias_total     like self._biases
        #   passed_down    (n_rows, self._n_steps, self._n_features)
        #
        # And two carried blocks, both (n_rows, self._n_units) of zeros:
        #   state_slopes, cell_slopes
        #
        # for position in reversed(range(self._n_steps)):
        #
        #   1. Add what arrived from above at this step. When the layer answers
        #      at every step that is block[:, position, :] on every turn. When it
        #      answers once, the whole block belongs to the last step alone and
        #      nothing arrives from above at any other.
        #
        #   2. correction = self._retreat(response.steps[position],
        #                                 state_slopes, cell_slopes)
        #
        #   3. ADD the parameter slopes into the totals, never assign. The same
        #      weights were used at every step, so their slope is a sum with one
        #      term per step. This single line is what the module docstring
        #      spends a whole section on, and getting it wrong leaves a gradient
        #      that describes the last step alone, which still trains.
        #
        #   4. passed_down[:, position, :] = correction.input_slopes
        #
        #   5. state_slopes = correction.state_slopes
        #      cell_slopes  = correction.cell_slopes
        #      This is the step through time. What step t + 1 wanted from the
        #      state it was handed becomes what arrives at step t.
        #
        # Answer LayerCorrection(passed_down=..., gradient=LayerGradient(...)).
        block = self._checked_arriving(response, arriving)
        if not isinstance(response, RecurrentResponse):
            raise ShapeMismatchError(
                "a recurrent layer's backward step needs the trace its own "
                "forward pass recorded, and this response carries none"
            )

        n_rows = block.shape[0]
        weights_total = np.zeros(shape=self._weights.shape)
        bias_total = np.zeros(shape=self._biases.shape)
        passed_down = np.zeros(shape=(n_rows, self._n_steps, self._n_features))

        state_slopes = np.zeros(shape=(n_rows, self._n_units))
        cell_slopes = np.zeros(shape=(n_rows, self._n_units))

        for position in reversed(range(self._n_steps)):
            if self._answers_every_step:
                state_slopes = state_slopes + block[:, position, :]
            elif position == self._n_steps - 1:
                state_slopes = state_slopes + block

            correction = self._retreat(
                response.steps[position], state_slopes, cell_slopes
            )

            weights_total += correction.weight_slopes
            bias_total += correction.bias_slopes

            passed_down[:, position, :] = correction.input_slopes
            state_slopes = correction.state_slopes
            cell_slopes = correction.cell_slopes

        return LayerCorrection(
            passed_down=passed_down,
            gradient=LayerGradient(weights_total, bias_total),
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> RecurrentLayer:
        """A new layer of the same geometry, with its parameters nudged.

        Downhill is against the slope, so this subtracts. Training does not
        mutate, so the old layer is unchanged and still answers as it did.

        Parameters
        ----------
        gradient:
            What this layer's parameters want to change by.
        learning_rate:
            How far to move.

        Returns
        -------
        RecurrentLayer
            A new layer of this type, same geometry, moved parameters.

        Raises
        ------
        ShapeMismatchError
            If no gradient is supplied, since a recurrent layer has parameters
            and has nothing to answer with when asked to step by nothing, or if
            the gradient does not describe this layer's parameters.
        """
        if gradient is None:
            raise ShapeMismatchError(
                "a recurrent layer has parameters and needs a gradient to step by"
            )
        if gradient.weights.shape != self._weights.shape:
            raise ShapeMismatchError(
                f"this layer's weights are {self._weights.shape}, and the "
                f"gradient describes {gradient.weights.shape}"
            )

        return self.with_parameters(
            self._weights - learning_rate * gradient.weights,
            self._biases - learning_rate * gradient.biases,
        )

    def _gate_slice(self, position: int) -> slice:
        """Which rows of the weight block belong to gate ``position``."""
        return slice(position * self._n_units, (position + 1) * self._n_units)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(reads={self._shape.reads!r}, "
            f"n_units={self._n_units!r}, "
            f"answers_every_step={self._answers_every_step!r})"
        )


class SimpleRecurrent(RecurrentLayer):
    """One gate, and the state replaced outright at every step.

    The plainest thing that carries anything::

        h_t  =  tanh( W @ [ x_t , h_(t-1) ]  +  b )

    It is here for two reasons. It is the definition the other two are
    variations on, so a reader meeting gates should have met this first, and it
    is the layer whose gradient measurably vanishes, which is the argument for
    the other two rather than a claim about them.
    """

    @property
    def n_gates(self) -> int:
        """One. There is no gate at all, and the whole matrix is the one bend."""
        return 1

    def _advance(
        self,
        inputs_at_step: FloatArray,
        previous_state: FloatArray,
        previous_cell: FloatArray,
    ) -> CellStep:
        #   h_t  =  tanh( W @ [ x_t , h_(t-1) ] + b )
        #
        # combined is the step's input and the carried state laid end to end
        # along axis 1, so (n_rows, n_features + n_units), and then
        #
        #   state = np.tanh(combined @ self._weights.T + self._biases)
        #
        # There are no gates here, so the whole weight block is the one bend and
        # `gates` on the record is the state again. The cell is carried through
        # untouched, since this cell has no second quantity, and passing
        # previous_cell straight along is what keeps the base's loop free of a
        # branch on which subclass it is threading.
        #
        # Answer a CellStep carrying state, cell, gates, previous_state,
        # previous_cell and inputs.
        combined = np.concatenate((inputs_at_step, previous_state), axis=1)
        state = np.tanh(combined @ self._weights.T + self._biases)

        return CellStep(
            state=state,
            cell=previous_cell,
            gates=state,
            previous_state=previous_state,
            previous_cell=previous_cell,
            inputs=inputs_at_step,
        )

    def _retreat(
        self, step: CellStep, state_slopes: FloatArray, cell_slopes: FloatArray
    ) -> CellCorrection:
        # state_slopes is what the loss wants from this step's state, already
        # including anything the step after wanted from it.
        #
        # The bend first. tanh' = 1 - tanh^2, and tanh(score) is the state
        # itself, so the score's slope needs no stored score:
        #
        #   score_slopes = state_slopes * (1.0 - step.state ** 2)
        #
        # Then the linear part. The score was combined @ W.T + b, so
        #
        #   d W          =  score_slopes.T @ combined
        #   d b          =  score_slopes.sum(axis=0)
        #   d combined   =  score_slopes @ self._weights
        #
        # combined was the input then the state, so the first self._n_features
        # columns of d combined are what the loss wants from this step's input
        # and the rest are what it wants from the state handed in.
        # step.combined rebuilds that row for you.
        #
        # cell_slopes pass straight back out, untouched, as the cell did forward.
        score_slopes = state_slopes * (1.0 - step.state**2)
        through_weights = score_slopes @ self._weights

        return CellCorrection(
            weight_slopes=score_slopes.T @ step.combined,
            bias_slopes=score_slopes.sum(axis=0),
            input_slopes=through_weights[:, : self._n_features],
            state_slopes=through_weights[:, self._n_features :],
            cell_slopes=cell_slopes,
        )


class LongShortTermMemory(RecurrentLayer):
    """Four gates and a second carried quantity, so a slope can travel far.

    The cell is the point. It is updated by an addition rather than by a matrix
    multiply and a bend::

        c_t  =  forget * c_(t-1)  +  keep * candidate
        h_t  =  show * tanh( c_t )

    so a slope walking back along the cell is multiplied by the forget gate and
    by nothing else. A forget gate near one is a route through time that does
    not shrink, which is exactly what the plain layer above lacks, and it is
    learned rather than set.

    The four gates are stacked down the weight block in the order keep, forget,
    candidate, show. Three of them are squashed to between zero and one because
    they are used as proportions; the candidate is squashed to between minus one
    and one because it is a quantity rather than a proportion.
    """

    KEEP = 0
    FORGET = 1
    CANDIDATE = 2
    SHOW = 3

    FORGET_STARTS_AT = 1.0
    """How far open the forget gate is before anything is learned.

    Not zero, and the difference is the whole point of the cell rather than a
    detail of its start. A bias of zero squashes to a gate of one half, so the
    cell is multiplied by a half at every step and a slope walking back over
    twenty-five steps is divided by roughly thirty-three million. Measured on
    twenty-five steps of a drawn sequence, with the share of blame reaching the
    first step against the share reaching the last:

        forget bias 0.0     5.9e-05, 4.2e-05, 1.2e-04, 1.8e-05, 1.3e-04
        forget bias 1.0     3.3e-02, 3.8e-02, 5.7e-02, 9.4e-03, 5.9e-02
        forget bias 2.0     0.699,   1.032,   1.098,   0.688,   0.293

    over five seeds. At zero the cell is not merely worse than it should be, it
    is *worse than the plain layer above*, which keeps 1.0e-01 to 2.0e-05 on the
    same sequences, so the reason this family exists does not hold at all. At
    one it holds on 18 of 20 seeds. A gate that starts open remembers by default
    and learns when to forget, which is the right way round.
    """

    @property
    def n_gates(self) -> int:
        """Four: keep, forget, candidate and show."""
        return 4

    def _starting_biases(self, n_units: int) -> FloatArray:
        """Zero everywhere but the forget band, which starts open.

        See :attr:`FORGET_STARTS_AT` for the measurement. This is the one place
        in the family where the starting point is a decision rather than an
        absence of one.
        """
        biases = np.zeros(self.n_gates * n_units)
        biases[self.FORGET * n_units : (self.FORGET + 1) * n_units] = (
            self.FORGET_STARTS_AT
        )
        return biases

    def _advance(
        self,
        inputs_at_step: FloatArray,
        previous_state: FloatArray,
        previous_cell: FloatArray,
    ) -> CellStep:
        # combined = [ x_t , h_(t-1) ] along axis 1, then
        #   scores = combined @ self._weights.T + self._biases
        # which is (n_rows, 4 * n_units).
        #
        # Cut the scores into four bands with self._gate_slice(...) and bend
        # each one. Three are proportions and take the logistic; the candidate is
        # a quantity and takes tanh:
        #
        #   keep      = _logistic(scores[:, self._gate_slice(self.KEEP)])
        #   forget    = _logistic(scores[:, self._gate_slice(self.FORGET)])
        #   candidate = np.tanh(scores[:, self._gate_slice(self.CANDIDATE)])
        #   show      = _logistic(scores[:, self._gate_slice(self.SHOW)])
        #
        # Then the two lines that are the whole cell:
        #
        #   cell  = forget * previous_cell + keep * candidate
        #   state = show * np.tanh(cell)
        #
        # The first is an addition, and that is the point of the family. A slope
        # walking back along the cell is multiplied by forget and by nothing
        # else, where the plain layer multiplies by a weight matrix and a bend.
        #
        # gates on the record is the four bands concatenated back together in
        # band order, so _retreat can cut them out again with the same slices.
        combined = np.concatenate((inputs_at_step, previous_state), axis=1)
        scores = combined @ self._weights.T + self._biases
        keep = _logistic(scores[:, self._gate_slice(self.KEEP)])
        forget = _logistic(scores[:, self._gate_slice(self.FORGET)])
        candidate = np.tanh(scores[:, self._gate_slice(self.CANDIDATE)])
        show = _logistic(scores[:, self._gate_slice(self.SHOW)])
        cell = forget * previous_cell + keep * candidate
        state = show * np.tanh(cell)
        return CellStep(
            state=state,
            cell=cell,
            gates=np.concatenate((keep, forget, candidate, show), axis=1),
            previous_state=previous_state,
            previous_cell=previous_cell,
            inputs=inputs_at_step,
        )

    def _retreat(
        self, step: CellStep, state_slopes: FloatArray, cell_slopes: FloatArray
    ) -> CellCorrection:
        # Cut the four gates back out of step.gates with the same slices.
        #
        # bent_cell = np.tanh(step.cell). Two routes reach this step's cell:
        # along the cell from the step after, which is cell_slopes, and through
        # this step's own answer, since state = show * tanh(cell):
        #
        #   total_cell_slopes = cell_slopes + state_slopes * show * (1 - bent_cell ** 2)
        #
        # Now each gate's score, remembering logistic' = g (1 - g) and
        # tanh' = 1 - t ** 2, and reading off which line of the forward pass each
        # gate appeared in:
        #
        #   show        state = show * bent_cell
        #     show_scores      = state_slopes * bent_cell * show * (1 - show)
        #   keep        cell = ... + keep * candidate
        #     keep_scores      = total_cell_slopes * candidate * keep * (1 - keep)
        #   forget      cell = forget * previous_cell + ...
        #     forget_scores    = (total_cell_slopes * step.previous_cell
        #                         * forget * (1 - forget))
        #   candidate   cell = ... + keep * candidate
        #     candidate_scores = total_cell_slopes * keep * (1 - candidate ** 2)
        #
        # Concatenate those four in band order, then the same three lines the
        # plain cell uses: d W from score_slopes.T @ step.combined, d b from the
        # column sums, and d combined from score_slopes @ self._weights split at
        # self._n_features.
        #
        # The cell handed back to the step before is
        #
        #   total_cell_slopes * forget
        #
        # which is the additive route: one gate per step rather than a matrix and
        # a bend, which is the whole reason this cell remembers further.
        keep = step.gates[:, self._gate_slice(self.KEEP)]
        forget = step.gates[:, self._gate_slice(self.FORGET)]
        candidate = step.gates[:, self._gate_slice(self.CANDIDATE)]
        show = step.gates[:, self._gate_slice(self.SHOW)]
        activated_cell = np.tanh(step.cell)
        total_cell_slopes = cell_slopes + state_slopes * show * (
            1.0 - activated_cell**2
        )
        keep_scores = total_cell_slopes * candidate * keep * (1.0 - keep)
        forget_scores = total_cell_slopes * step.previous_cell * forget * (1.0 - forget)
        candidate_scores = total_cell_slopes * keep * (1.0 - candidate**2)
        show_scores = state_slopes * activated_cell * show * (1.0 - show)
        score_slopes = np.concatenate(
            (keep_scores, forget_scores, candidate_scores, show_scores), axis=1
        )
        through_weights = score_slopes @ self._weights
        return CellCorrection(
            weight_slopes=score_slopes.T @ step.combined,
            bias_slopes=score_slopes.sum(axis=0),
            input_slopes=through_weights[:, : self._n_features],
            state_slopes=through_weights[:, self._n_features :],
            cell_slopes=total_cell_slopes * forget,
        )


class GatedRecurrentUnit(RecurrentLayer):
    """Three gates, one carried quantity, and the same additive route.

    The interpolation is the whole of it::

        h_t  =  ( 1 - update ) * candidate  +  update * h_(t-1)

    An update gate near one hands the previous state through untouched, so the
    slope walking back is multiplied by that gate and by nothing else, which is
    the long short-term memory's trick with one carried quantity instead of two
    and three gates instead of four.

    The one place this cell is not a stack of independent gates is its
    candidate, which reads the previous state *after* the reset gate has scaled
    it::

        candidate  =  tanh( W_candidate @ [ x_t , reset * h_(t-1) ]  +  b )

    So the step takes two matrix multiplies rather than one, since the reset
    gate has to be known before the candidate's row can be assembled. The weight
    block is still one block, and which rows are used against which row is the
    only thing that differs.
    """

    RESET = 0
    UPDATE = 1
    CANDIDATE = 2

    @property
    def n_gates(self) -> int:
        """Three: reset, update and candidate."""
        return 3

    def _advance(
        self,
        inputs_at_step: FloatArray,
        previous_state: FloatArray,
        previous_cell: FloatArray,
    ) -> CellStep:
        combined = np.concatenate((inputs_at_step, previous_state), axis=1)
        paired_rows = slice(0, 2 * self._n_units)
        paired_scores = (
            combined @ self._weights[paired_rows].T + self._biases[paired_rows]
        )

        reset = _logistic(paired_scores[:, self._gate_slice(self.RESET)])
        update = _logistic(paired_scores[:, self._gate_slice(self.UPDATE)])

        candidate_rows = self._gate_slice(self.CANDIDATE)
        reset_combined = np.concatenate(
            (inputs_at_step, reset * previous_state), axis=1
        )
        candidate = np.tanh(
            reset_combined @ self._weights[candidate_rows].T
            + self._biases[candidate_rows]
        )

        state = (1.0 - update) * candidate + update * previous_state

        return CellStep(
            state=state,
            cell=previous_cell,
            gates=np.concatenate((reset, update, candidate), axis=1),
            previous_state=previous_state,
            previous_cell=previous_cell,
            inputs=inputs_at_step,
        )

    def _retreat(
        self, step: CellStep, state_slopes: FloatArray, cell_slopes: FloatArray
    ) -> CellCorrection:
        reset = step.gates[:, self._gate_slice(self.RESET)]
        update = step.gates[:, self._gate_slice(self.UPDATE)]
        candidate = step.gates[:, self._gate_slice(self.CANDIDATE)]
        previous_state = step.previous_state

        candidate_scores = state_slopes * (1.0 - update) * (1.0 - candidate**2)
        update_scores = (
            state_slopes * (previous_state - candidate) * update * (1.0 - update)
        )

        candidate_rows = self._gate_slice(self.CANDIDATE)
        through_candidate = candidate_scores @ self._weights[candidate_rows]
        # The candidate read the reset-scaled state, so what the loss wants from
        # the reset gate arrives entirely through that half of its row.
        reset_scores = (
            through_candidate[:, self._n_features :]
            * previous_state
            * reset
            * (1.0 - reset)
        )

        paired_scores = np.concatenate((reset_scores, update_scores), axis=1)
        paired_rows = slice(0, 2 * self._n_units)
        through_paired = paired_scores @ self._weights[paired_rows]

        reset_combined = np.concatenate((step.inputs, reset * previous_state), axis=1)
        weight_slopes = np.zeros_like(self._weights)
        weight_slopes[paired_rows] = paired_scores.T @ step.combined
        weight_slopes[candidate_rows] = candidate_scores.T @ reset_combined

        bias_slopes = np.concatenate(
            (paired_scores.sum(axis=0), candidate_scores.sum(axis=0))
        )

        # Three routes reach the previous state: through the two paired gates,
        # through the candidate's reset-scaled half, and the interpolation's own
        # untouched term.
        state_from_here = (
            through_paired[:, self._n_features :]
            + through_candidate[:, self._n_features :] * reset
            + state_slopes * update
        )
        input_from_here = (
            through_paired[:, : self._n_features]
            + through_candidate[:, : self._n_features]
        )

        return CellCorrection(
            weight_slopes=weight_slopes,
            bias_slopes=bias_slopes,
            input_slopes=input_from_here,
            state_slopes=state_from_here,
            cell_slopes=cell_slopes,
        )
