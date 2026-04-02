"""One neuron, which is the logistic model this library already has.

Fit ``LogisticRegression`` on the study-hours crowd and it settles at weight
1.282004 and bias -5.768909. Ask it for a probability at five hours, then ask a
hand-written sigmoid for the same thing::

    library  predict_probability(5)   0.655004269521
    by hand  sigmoid(w * 5 + b)       0.655004269521

Identical to twelve places, because they are the same calculation. A neuron
with a sigmoid activation *is* logistic regression: weights, a bias, a squash.
Nothing about the unit is new, and everything difficult about networks comes
from feeding one into another.

What one of them cannot do
--------------------------
Write the score at the four corners of exclusive-or, where the answer is true
when exactly one input is on::

    z(0,0) = b                    z(1,0) = w1 + b
    z(0,1) = w2 + b               z(1,1) = w1 + w2 + b

    z(0,1) + z(1,0)  =  w1 + w2 + 2b  =  z(0,0) + z(1,1)

The two true corners and the two false corners carry the same score total, for
every weight and bias there is. Exclusive-or asks for both true corners above a
threshold and both false corners below it, which makes the left total exceed
twice the threshold while the right total falls short of it -- and those two
totals are the same number.

The argument never touches the activation, only the assumption that it is
monotone. So this is a ceiling on the unit rather than a weakness of any
particular bend, and no cleverer squash escapes it. Composition does, which is
what a layer is for.

Why the weights are positional and carry no names
-------------------------------------------------
:class:`~oop_ml.core.data.coefficients.Coefficients` binds every weight to the
feature it was learned for, and that is right for a linear model, whose inputs
are named columns forever. It is wrong here past the first layer. A hidden
neuron reads the outputs of the neurons beneath it, and those live in a space
the network invented, whose coordinates have no names to bind to. The same
distinction :mod:`oop_ml.core.kernel.components` draws when it refuses to reuse
the principal-component vocabulary: a direction in a learned space is not a
feature, and giving it a feature's name would be inventing a fact.

So a neuron's weights are a :class:`~oop_ml.core.data.column.Column`, ordered
to match whatever feeds it, and the layer above is what remembers the order.

Where the input width lives
---------------------------
A neuron reading ``n`` inputs has exactly ``n`` weights. Not "should have": the
dot product is undefined otherwise, so the count of weights *is* the input
width, carried by the neuron rather than declared beside it. A ``Column`` cannot
be empty, so a neuron with no inputs -- which would be a constant wearing a
neuron's name -- is unrepresentable rather than rejected.

That is the first link in the chain that makes a network's shape decidable
before any data arrives.
"""

from __future__ import annotations

import math

import numpy as np

from oop_ml.core.data.column import Column, ColumnSource
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.network.activation import Activation
from oop_ml.core.types import FloatArray
from oop_ml.core.validation import ValueRole


class NeuronResponse:
    """What one neuron did with one row: the score it formed, and its answer.

    Two numbers that must travel together. A forward pass reads the output; a
    backward pass needs the score as well, because the slope of the bend is
    taken at the score and recomputing it later means running the dot product
    twice. Returning them as a pair of floats would make every caller remember
    which came first, which is the tuple this library does not write.

    Parameters
    ----------
    score:
        ``weights . inputs + bias``, before the activation.
    output:
        The activation applied to that score, which is what the next layer
        reads.
    """

    __slots__ = ("_output", "_score")

    def __init__(self, score: float, output: float) -> None:
        self._score = float(score)
        self._output = float(output)

    @property
    def score(self) -> float:
        """The weighted sum plus the bias, before the bend."""
        return self._score

    @property
    def output(self) -> float:
        """The bent score, which is what the next layer receives."""
        return self._output

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NeuronResponse):
            return NotImplemented
        return self._score == other._score and self._output == other._output

    def __hash__(self) -> int:
        return hash((self._score, self._output))

    def __repr__(self) -> str:
        return f"NeuronResponse(score={self._score!r}, output={self._output!r})"


class Neuron:
    """A weight per input, one bias, and the bend applied to their sum.

    Parameters
    ----------
    weights:
        One weight per input, in the order the inputs arrive. The count is the
        neuron's input width, so this cannot be empty.
    bias:
        The constant added to the weighted sum, which shifts where the bend
        happens without reference to any input.
    activation:
        The bend applied to the sum. See
        :mod:`oop_ml.core.network.activation` for why softmax is not one of
        these.

    Raises
    ------
    EmptyValuesError
        If no weights are supplied, by way of the column that holds them. A
        neuron with no inputs is a constant.
    InvalidValuesError
        If any weight is not finite, again by way of the column, or if the bias
        is not finite. A non-finite parameter makes every downstream comparison
        false rather than raising anywhere near the cause.
    """

    __slots__ = ("_activation", "_bias", "_weights")

    def __init__(
        self, weights: ColumnSource, bias: float, activation: Activation
    ) -> None:
        checked = Column.of(weights, ValueRole.WEIGHT_VALUES)

        frozen = checked.values.copy()
        frozen.setflags(write=False)

        bias_value = float(bias)
        if not math.isfinite(bias_value):
            raise InvalidValuesError("bias must be finite")

        self._weights = Column.selecting(frozen, ValueRole.WEIGHT_VALUES)
        self._bias = bias_value
        self._activation = activation

    @property
    def weights(self) -> Column:
        """One weight per input, in the order the inputs arrive."""
        return self._weights

    @property
    def bias(self) -> float:
        """The constant added to the weighted sum."""
        return self._bias

    @property
    def activation(self) -> Activation:
        """The bend applied to the sum."""
        return self._activation

    @property
    def n_inputs(self) -> int:
        """How many inputs this neuron reads, which is its weight count."""
        return len(self._weights)

    def respond_to(self, inputs: ColumnSource) -> NeuronResponse:
        """Score one row and bend the score.

        The length agreement is established here, once, so that
        :meth:`_response_for` may take the dot product without checking
        anything. A layer whose shape was settled at construction knows the
        agreement already and will have a cheaper route to the same arithmetic;
        this is the boundary, where the caller's row has not been vouched for.

        Parameters
        ----------
        inputs:
            One value per weight, in the same order. Passing a ``Column``
            costs nothing, since ``Column.of`` is idempotent.

        Returns
        -------
        NeuronResponse
            The score and the output, paired.

        Raises
        ------
        NonEqualArrayLengthError
            If the row does not carry exactly one value per weight.
        EmptyValuesError
            If the row is empty, by way of the column that holds it.
        InvalidValuesError
            If any value in the row is not finite, likewise.
        """
        row = Column.of(inputs, ValueRole.INPUT_VALUES)
        self._weights.check_equal_length(row)

        return self._response_for(row.values)

    def _response_for(self, inputs: FloatArray) -> NeuronResponse:
        """The arithmetic, given a row already known to be the right length.

        Parameters
        ----------
        inputs:
            ``(n_inputs,)``, aligned with :attr:`weights`.

        Returns
        -------
        NeuronResponse
            The weighted sum plus the bias, and that sum bent by
            :attr:`activation`.

        Notes
        -----
        The activation works on arrays and there is exactly one number here, so
        the score has to reach it wrapped. The wrapper is a *zero-dimensional*
        array rather than a one-element one, and that choice removes an index
        from the far side: ``float`` accepts a 0-d array and refuses anything
        wider, so a one-element wrapper has to be unwrapped by hand::

            float(activation.of(np.array([score])))   TypeError, numpy 2
            float(activation.of(np.asarray(score)))   fine, shape ()

        Shape in, shape out means a 0-d score comes back 0-d, so no subscript
        is needed and none can be got wrong.

        The two ``float`` calls stay. :class:`NeuronResponse` coerces both of
        its numbers anyway, so leaving them off runs correctly and reads more
        cleanly -- and it hands a declared ``float`` parameter an ``ndarray``,
        which pyright refuses. Correct at runtime and wrong in the types is the
        worse trade, so the coercion is written where the type says it happens.
        """
        score = np.dot(self._weights.values, inputs) + self._bias

        return NeuronResponse(
            score=float(score),
            output=float(self._activation.of(np.asarray(score))),
        )

    def __repr__(self) -> str:
        return (
            f"Neuron(n_inputs={self.n_inputs!r}, bias={self._bias!r}, "
            f"activation={self._activation!r})"
        )
