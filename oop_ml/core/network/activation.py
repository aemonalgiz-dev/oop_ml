"""The bend between one layer and the next, and the only reason depth pays.

Why an activation is not a finishing touch
------------------------------------------
A layer without one is an affine map, and a composition of affine maps is
affine. Measured on a 4x3 followed by a 2x4::

    rows @ first.T @ second.T   agrees with   rows @ (second @ first).T

    np.allclose  200 of 200 random trials
    ==           0 of 200

The two sides associate the same products differently, so they agree to
floating point rather than bit for bit, which is the reassociation this
library's memory-layout note already records. The algebra is exact; the
arithmetic is not, and writing ``==`` there would be the error the design
notes warn about.

Two layers collapse into one 2x3 matrix, and a hundred stacked layers collapse
just as completely. Without a bend between them, depth buys nothing at all: the
network is a single linear model wearing more parameters. The activation is
what makes the second layer worth having, which is why it belongs to the
vocabulary rather than to a configuration dictionary.

Why these are classes and not a closed enum
-------------------------------------------
:mod:`oop_ml.core.distance` uses a closed enum because five of its six metrics
take no parameters, and a bag of keyword arguments whose meaning changes per
member is the magic-string problem wearing a hat. The four here take none
either, so the enum would fit today -- and it would stop fitting at the first
leaky rectifier, whose negative slope is a real parameter, or an exponential
unit, whose saturation point is another. :mod:`oop_ml.core.kernel` already made
this call for the same reason.

The deciding argument is the second method. An activation is not a label a
model looks up, it is a pair of functions that have to travel together: a
backward pass needs the derivative at the very scores the forward pass used,
and a measure that carried only its name would leave the derivative stranded in
a growing ``if``. :class:`~oop_ml.core.tree.impurity.Impurity` is the closest
existing shape, and for the same reason.

Softmax is deliberately absent
------------------------------
It is not one of these, and the difference is a type rather than a taste. The
four below read one number and answer one number, so applying them elementwise
to a layer is the same as applying them to each neuron alone. Softmax reads the
whole layer at once. Measured on three scores::

    [2.0, 1.0, 0.1]   ->  [0.659001, 0.242433, 0.098566]

    move only the third score, 0.1 -> 5.0
    the first output moves         0.659001 -> 0.046613

Nothing changed about the first neuron and its answer fell by a factor of
fourteen, because softmax makes the outputs compete for a fixed total of one.
No elementwise function can do that, and a neuron cannot hold one, since a
neuron has no neighbours to normalise against. It arrives with the output
layer, where the whole row exists, and :func:`oop_ml.core.logistic.stable_softmax`
is already waiting for it.

The derivative each one carries
-------------------------------
Sigmoid's derivative peaks at exactly ``0.25``, at ``z = 0``, so each sigmoid
layer multiplies whatever gradient passes through it by at most a quarter --
``9.537e-07`` through ten layers, in the best case, at the single best point.
That is the vanishing gradient, and it is arithmetic rather than folklore.

The rectifier's derivative is exactly 1 on the positive side, which is why it
took over, and exactly 0 on the other, which is why a rectified neuron whose
score stays negative receives no gradient again and never recovers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy import float64

from oop_ml.core.logistic import stable_logistic
from oop_ml.core.types import FloatArray


class Activation(ABC):
    """One bend, together with the derivative a backward pass will need.

    Both methods are elementwise and shape-preserving: entry ``i`` of the
    answer depends on entry ``i`` of the argument and nothing else. That is the
    property the class is named for, and the property softmax does not have.
    """

    __slots__ = ()

    @abstractmethod
    def of(self, scores: FloatArray) -> FloatArray:
        """Bend every score, elementwise.

        Parameters
        ----------
        scores:
            Any shape. The pre-activation sums, ``weights . inputs + bias``,
            one per neuron per row.

        Returns
        -------
        FloatArray
            The same shape, each entry the bend applied to the score in that
            position.
        """

    @abstractmethod
    def derivative_at(self, scores: FloatArray) -> FloatArray:
        """The slope of :meth:`of` at each score, elementwise.

        Taking the *scores* rather than the outputs is the plainly correct
        signature: it is the derivative of a function, evaluated where the
        function was evaluated, and it can be read straight off the definition.
        Several of these can be computed more cheaply from their own output
        instead, since a sigmoid's slope is ``output * (1 - output)`` and a
        tangent's is ``1 - output ** 2``, which saves recomputing the
        exponential a backward pass has already paid for once. That is an
        optimisation to measure later, against this definition, and not a
        reason to complicate the contract now.

        Parameters
        ----------
        scores:
            Any shape. The same scores :meth:`of` was handed.

        Returns
        -------
        FloatArray
            The same shape, each entry the slope of the bend at that score.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """The formula, for a readout or a saved document to quote."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Activation):
            return NotImplemented
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Identity(Activation):
    """No bend at all, which an output layer predicting a quantity wants.

    Present so that a regression network's last layer is an ordinary layer
    rather than a special case the forward pass has to know about. Stacking two
    of these is the collapse the module docstring measures, so it is only ever
    correct at the end.
    """

    __slots__ = ()

    def of(self, scores: FloatArray) -> FloatArray:
        return scores

    def derivative_at(self, scores: FloatArray) -> FloatArray:
        return np.ones(shape=self.of(scores).shape)

    @property
    def description(self) -> str:
        return "z"


class RectifiedLinear(Activation):
    """``max(0, z)``, the hinge that made deep networks trainable.

    Its derivative is exactly 1 above zero and exactly 0 below, so gradients
    pass through the live half untouched and the dead half stops learning
    permanently. At ``z = 0`` the derivative does not exist; the convention
    here is 0, chosen because it is what the negative branch already answers
    and because a single point of measure zero cannot be reached by an exact
    float often enough to matter.
    """

    __slots__ = ()

    def of(self, scores: FloatArray) -> FloatArray:
        return np.maximum(scores, 0)

    def derivative_at(self, scores: FloatArray) -> FloatArray:
        return (self.of(scores) > 0).astype(float64)

    @property
    def description(self) -> str:
        return "max(0, z)"


class Sigmoid(Activation):
    """``1 / (1 + exp(-z))``, the same squash the logistic models use.

    Reach for :func:`oop_ml.core.logistic.stable_logistic` rather than spelling
    the formula out. The naive form overflows below about ``z = -709`` and
    answers correctly while emitting a warning per row from exactly the neurons
    that are most certain, and that function exists so the trap is written down
    once.

    Its derivative is ``sigmoid(z) * (1 - sigmoid(z))``, which peaks at 0.25 and
    is the arithmetic behind the vanishing gradient.
    """

    __slots__ = ()

    def of(self, scores: FloatArray) -> FloatArray:
        return stable_logistic(scores)

    def derivative_at(self, scores: FloatArray) -> FloatArray:
        sigmoid_values = self.of(scores)
        return sigmoid_values * (1 - sigmoid_values)

    @property
    def description(self) -> str:
        return "1 / (1 + exp(-z))"


class HyperbolicTangent(Activation):
    """``tanh(z)``, the sigmoid recentred on zero.

    Its derivative is ``1 - tanh(z) ** 2``, peaking at 1.0 rather than 0.25, so
    it starves a deep stack four times more slowly than a sigmoid does. It
    saturates at both tails all the same, which is the failure the rectifier
    escapes rather than softens.
    """

    __slots__ = ()

    def of(self, scores: FloatArray) -> FloatArray:
        return np.tanh(scores)

    def derivative_at(self, scores: FloatArray) -> FloatArray:
        return 1 - self.of(scores) ** 2

    @property
    def description(self) -> str:
        return "tanh(z)"
