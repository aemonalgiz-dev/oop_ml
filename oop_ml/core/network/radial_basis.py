"""A unit that answers how *near* a row is, rather than how far along it lies.

What is actually different here
-------------------------------
Every other parameterised layer in this package answers with a weighted sum. A
:class:`~oop_ml.core.network.neuron.Neuron` scores a row by ``weights . row``, a
:class:`~oop_ml.core.network.convolution.Conv2d` filter does the same inside a
window, and in both the score is unbounded above and below and its level surfaces
are parallel planes. A unit here holds a *place* instead of a direction: a centre,
one coordinate per input, and a width, and it answers::

    output = exp(-||row - centre|| ** 2 / (2 * width ** 2))

which is one at the centre, falls smoothly away from it in every direction alike,
and is bounded in ``(0, 1]``. Its level surfaces are spheres. So a dense unit
partitions the input space with a plane and says which side a row fell on, and a
radial unit encloses a region and says how far inside it a row sat.

That difference is the whole reason for the layer. A stack of dense units
approximates a function by folding half-spaces together, and every unit has an
opinion about every row however remote. A radial unit has an opinion about its own
neighbourhood and answers a number indistinguishable from zero everywhere else,
so a layer of them is a *local* description: adding a unit in one region cannot
disturb what the network already answers in another. The classical radial basis
function network is exactly one of these under a linear output layer, and it is
what makes that network fit locally and interpolate rather than extrapolate.

The price is the same locality read the other way. A row far from every centre
gets an answer of zero from all of them and therefore no gradient anywhere, which
is this layer's version of a rectifier's dead half. It is why the centres matter
so much more than a dense layer's starting weights do, and why the usual practice
is to place them on the data rather than to draw them.

Why the score is a squared distance and not a bell
---------------------------------------------------
:class:`~oop_ml.core.network.layer.LayerResponse` carries a score and an output,
and for a dense layer those are the pre-activation sum and the bend applied to it.
The same split is available here and it is the useful one: the score is
``||row - centre|| ** 2`` and the output is the bell taken at it. The backward
pass needs the slope of the bell *at the score that produced it*, which is exactly
the arrangement every other layer here already relies on, so nothing about the
backward walk becomes a special case.

Recording the bell as the score instead would be the mistake worth naming, since
both blocks have the same shape and both hold plausible numbers. The squared
distance is also the only quantity the width gradient can be recovered from, and
it cannot be recovered from the output without a logarithm and a sign no one
would want to write.

Why :class:`~oop_ml.core.network.gradient.LayerGradient` is reused unchanged
-----------------------------------------------------------------------------
A reader meeting ``gradient.weights`` on a layer that has no weights deserves the
sentence, so here it is. ``LayerGradient``'s contract is an *arrangement*, not a
vocabulary: a block of one row per unit against whatever that unit reads, paired
with one number per unit. The centres are ``(n_units, n_features)`` and the widths
are ``(n_units,)``, which is that arrangement exactly, with no reshaping and
nothing folded. Entry ``[unit, feature]`` of the block really is the slope for
that unit's parameter on that input, and entry ``[unit]`` of the vector really is
that unit's single remaining number, which is what the type says it is in both
places.

That is a stronger fit than :class:`~oop_ml.core.network.convolution.Conv2d`
already lives with. A convolution's kernels are four-dimensional and have to be
flattened on the way out and unflattened on the way in, and the module docstring
there argues that this is the interface working rather than a wart, because only
the layer that produced a gradient can say what its axes mean. Nothing is even
flattened here. What is borrowed is the shape and what is not borrowed are the
two names, which belong to the first layer that needed the type and have been the
shared vocabulary of every layer since.

The alternative was measured once and declined, and the reasoning is recorded in
this library's notes on the network review: an abstract gradient base with a
subclass per layer touches fourteen read sites across four modules and five spec
files, to remove a mismatch that is two words long. "A gradient is a block per
unit and a number per unit" is a contract a reader holds in their head. Two of
them is not.

Why the two slopes are worth deriving rather than pattern-matching
--------------------------------------------------------------------
Both are written out in :meth:`RadialBasisLayer.correction_for`. What is worth
knowing before reading them is where each one differs from anything else in this
package.

The centre slope has the *opposite sign* to a weight slope. A dense score rises as
a weight rises; this score is a distance, so it *falls* as the centre approaches
the row. A body that copies the dense form pushes every centre away from the data
it was meant to describe, and the loss still moves on most steps.

The width slope is the only place in this library where a parameter appears cubed.
It is ``output * score / width ** 3``, and the reading a hand naturally writes,
``output * score / width``, has the right sign, the right shape and the right zero
at the centre. It is wrong by a factor of ``width ** 2``, which is *exactly one* at
the default width -- so a spec written on default layers cannot tell the two apart
at all. The gradient check below is run at two widths, neither of them one, for
that single reason.

Why a width may not reach zero
-------------------------------
The width is a denominator and a positive one. At zero the unit is a delta: one at
its centre, zero everywhere else, no slope anywhere, and a division by nothing on
the next forward pass. Negative is worse than degenerate rather than symmetric --
the forward pass only sees ``width ** 2`` and is unchanged, while the width slope
carries ``width ** 3`` and flips sign, so the layer climbs the loss in that one
parameter while every other number in it looks entirely healthy.

So :meth:`RadialBasisLayer.with_parameters` refuses a width at or below zero, and
a step large enough to overshoot is refused by name rather than clamped. Clamping
would hand back a layer nobody configured; refusing says which step did it. The
standard repair is to learn the *logarithm* of the width, which is unconstrained
and makes the guard unreachable, and it is deliberately not done here: it changes
what the parameter is, and therefore what a saved document and a diagram mean, for
a case that a sane learning rate does not reach.

Why the centres are drawn unscaled
-----------------------------------
:class:`~oop_ml.core.network.convolution.Conv2d` scales its draw by
``sqrt(2 / fan_in)``, and that is a statement about a *sum*: a score there adds
``fan_in`` products, so its spread grows like the square root of the count and
saturates the bend before the first step. There is no such sum here. A centre is a
location in the input space, not a multiplier, and the score is a distance whose
scale is set by the data rather than by how many features there are. Scaling the
draw by the fan-in would shrink every centre towards the origin as the input
widens, which is a bias towards one arbitrary point and buys nothing.

Standard normal draws are therefore the honest default *for standardised inputs*,
which is what they assume and what this library's
:class:`~oop_ml.numpy.preprocessing.standardization.standardizer.Standardizer`
produces. All the widths start at one stated value, for the reason a convolution's
biases start at zero: a per-unit starting width would be a claim about how large
each unit's neighbourhood is, and there is no evidence for one until a gradient
supplies it.

Why the pairing block is materialised
--------------------------------------
The forward pass builds ``(n_rows, n_units, n_features)`` of differences, which is
the same three-dimensional pairing this library's neighbour search measured at
1.6 GB and then removed by rewriting the distance as ``||a|| ** 2 - 2 a . b +
||b|| ** 2``. The same rewrite applies to the scores here and would remove the
same array. It is not done yet, and the reason is that the *backward* pass needs
the differences as vectors rather than the distances as numbers, so removing the
block from the forward pass alone would leave it to be rebuilt one method later.
Doing both together is a measured optimisation pass of the kind this library keeps
separate from getting the arithmetic right, and the loops-and-broadcasts version
is what such a pass would be measured against.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from math import isfinite

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.blocks import as_block, check_finite
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray

#: How many extents describe one input here: a row is a single width.
ROW_EXTENTS = 1


def _checked_width(value: float) -> float:
    """Read the starting width, refusing anything a bell cannot be built on.

    Parameters
    ----------
    value:
        The configured width.

    Returns
    -------
    float
        The same number, as a builtin ``float``.

    Raises
    ------
    InvalidValuesError
        If it cannot be read as a number at all, is not finite, or is not above
        zero. ``float("wide")`` raises a bare ``ValueError`` from builtins and
        ``float(None)`` a bare ``TypeError``, and neither is an ``MLLibError``,
        so neither may escape a constructor here.
    """
    try:
        width = float(value)
    except (TypeError, ValueError):
        raise InvalidValuesError(
            "a radial basis unit's width must be a number"
        ) from None
    if not isfinite(width):
        raise InvalidValuesError("a radial basis unit's width must be finite")
    if width <= 0.0:
        raise InvalidValuesError(
            f"a radial basis unit's width must be above zero, got {width}"
        )
    return width


def _checked_seed(value: int) -> int:
    """Read the seed as a whole number of at least zero.

    ``operator.index`` rather than ``isinstance(value, int)``, for the reason
    :mod:`oop_ml.core.network.shape` gives: a count read off a numpy computation
    is a ``numpy.int64`` and is a whole number by every standard except
    ``isinstance``. ``bool`` is excluded ahead of it, since ``True`` indexes as 1
    and would otherwise become a seed of one silently.

    Parameters
    ----------
    value:
        The configured seed.

    Returns
    -------
    int
        The same number, as a builtin ``int``.

    Raises
    ------
    InvalidValuesError
        If it is a bool, is not a whole number, or is negative.
    """
    if isinstance(value, bool):
        raise InvalidValuesError("random_seed must be a whole number, not a bool")
    try:
        whole = operator.index(value)
    except TypeError:
        raise InvalidValuesError("random_seed must be a whole number") from None
    if whole < 0:
        raise InvalidValuesError(f"random_seed must not be negative, got {whole}")
    return whole


class RadialBasisLayer(Layer):
    """Units that answer how close a row sits to a place each of them holds.

    Parameters
    ----------
    reads:
        How wide a row is. A bare integer, since this layer reads a row rather
        than an arrangement; a one-extent sequence such as ``(8,)`` is accepted
        as the same thing, so that a shape read off another layer can be handed
        straight over.
    n_units:
        How many units, which is how many numbers the layer answers with. Each
        one holds its own centre and its own width.
    width:
        What every unit's width starts at. One stated value rather than a draw:
        a per-unit starting width would be a claim about how large each unit's
        neighbourhood is, and there is no evidence for one before a gradient
        supplies it.
    random_seed:
        Which draw the centres start from. Fixed by default, so two layers built
        the same way are the same layer and a spec can rely on it.

    Raises
    ------
    ShapeMismatchError
        If ``reads`` names more than one extent. A picture is not a row, and the
        layer that bridges them is
        :class:`~oop_ml.core.network.flatten.Flatten`.
    InvalidValuesError
        If ``reads`` or ``n_units`` is not a whole number of at least one, if
        ``width`` is not a finite number above zero, or if ``random_seed`` is not
        a whole number of at least zero.

    Notes
    -----
    The shape is settled here in integer comparisons, before any row arrives,
    which is the claim this package opened with. Both counts are handed to
    :class:`~oop_ml.core.network.shape.LayerShape`, which is deliberately the
    only object in this package that decides what a valid extent is; deriving
    that rule again here would be a second implementation of it. That is the
    argument :class:`~oop_ml.core.network.flatten.Flatten` already makes, and it
    is why there is no whole-number guard in this constructor for either count.
    """

    __slots__ = ("_centres", "_n_features", "_n_units", "_shape", "_widths")

    def __init__(
        self,
        reads: int | Sequence[int],
        n_units: int,
        width: float = 1.0,
        random_seed: int = 0,
    ) -> None:
        shape = LayerShape(n_inputs=reads, n_outputs=n_units)
        if len(shape.reads) != ROW_EXTENTS:
            raise ShapeMismatchError(
                f"a radial basis layer reads a row, so one extent, got {shape.reads}"
            )

        n_features = shape.reads[0]
        checked_n_units = shape.answers[0]
        starting_width = _checked_width(width)
        generator = np.random.default_rng(_checked_seed(random_seed))

        self._configure(
            shape=shape,
            centres=generator.normal(size=(checked_n_units, n_features)),
            widths=np.full(checked_n_units, starting_width, dtype=np.float64),
        )

    def _configure(
        self, shape: LayerShape, centres: FloatArray, widths: FloatArray
    ) -> None:
        """Fill every slot from an already validated configuration.

        Separated from ``__init__`` because there are two ways to arrive at a
        layer and only one of them draws random numbers.
        :meth:`with_parameters` builds the next layer of a training run from this
        one's shape and a corrected set of parameters, and rerunning the
        constructor to throw its draw away would be both wasteful and a lie about
        where those numbers came from. It is the argument
        :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked` makes:
        one validating entry point, and a private route for values this class
        produced itself.

        Parameters
        ----------
        shape:
            What the layer reads and answers with, already computed.
        centres:
            ``(n_units, n_features)``, one place per unit.
        widths:
            ``(n_units,)``, every entry above zero.
        """
        self._shape = shape
        self._n_features = shape.reads[0]
        self._n_units = shape.answers[0]

        centre_block = np.array(centres, dtype=np.float64, copy=True)
        width_block = np.array(widths, dtype=np.float64, copy=True)
        centre_block.setflags(write=False)
        width_block.setflags(write=False)
        self._centres = centre_block
        self._widths = width_block

    @property
    def shape(self) -> LayerShape:
        """``(n_features,)`` in, ``(n_units,)`` out.

        Both sides carry their arrangement rather than a bare count, which is
        what lets a stack refuse this layer placed directly above a convolution
        instead of letting the element counts agree by accident.
        """
        return self._shape

    @property
    def centres(self) -> FloatArray:
        """``(n_units, n_features)``, one place per unit, frozen.

        Half the learned parameters, in the arrangement they are used in, which
        is also the arrangement a
        :class:`~oop_ml.core.network.gradient.LayerGradient` carries for them.
        """
        return self._centres

    @property
    def widths(self) -> FloatArray:
        """``(n_units,)``, one width per unit, frozen and every entry above zero.

        How far each unit's answer reaches. At a distance of exactly one width
        the unit answers ``exp(-0.5)``, which is 0.6065, whatever the width is;
        the width is the ruler that distance is measured against.
        """
        return self._widths

    @property
    def n_units(self) -> int:
        """How many units, which is how many numbers the layer answers with."""
        return self._n_units

    def with_parameters(
        self, centres: FloatArray, widths: FloatArray
    ) -> RadialBasisLayer:
        """A new layer of the same shape, carrying these parameters.

        Training does not mutate. A step builds the next layer from this one's
        shape and the corrected numbers, which is what keeps every array frozen
        and every answer reproducible.

        Parameters
        ----------
        centres:
            ``(n_units, n_features)``.
        widths:
            ``(n_units,)``, every entry finite and above zero.

        Returns
        -------
        RadialBasisLayer
            A new layer, same shape, same unit order.

        Raises
        ------
        ShapeMismatchError
            If either block does not match this layer's shape.
        InvalidValuesError
            If either block cannot be read as a float array, holds a non-finite
            entry, or names a width at or below zero. See the module docstring
            for why a non-positive width is a refusal rather than a degenerate
            configuration to be absorbed.
        """
        centre_block = as_block(centres, "centres")
        width_block = as_block(widths, "widths")

        expected = (self._n_units, self._n_features)
        if centre_block.shape != expected:
            raise ShapeMismatchError(
                f"this layer's centres are {expected}, got {centre_block.shape}"
            )
        if width_block.shape != (self._n_units,):
            raise ShapeMismatchError(
                f"this layer's widths are ({self._n_units},), got {width_block.shape}"
            )

        check_finite(centre_block, "centres")
        check_finite(width_block, "widths")
        if not bool(np.all(width_block > 0.0)):
            raise InvalidValuesError(
                "every radial basis width must stay above zero, got a smallest "
                f"of {float(np.min(width_block))}"
            )

        replacement = RadialBasisLayer.__new__(RadialBasisLayer)
        replacement._configure(
            shape=self._shape, centres=centre_block, widths=width_block
        )
        return replacement

    def _gaps_to_every_centre(self, inputs: FloatArray) -> FloatArray:
        """``(n_rows, n_units, n_features)``, each row minus each centre.

        The one block both passes need and neither can avoid: the forward pass
        squares and sums it, and the backward pass needs it unsummed, because a
        centre moves along a direction rather than along a distance. See the
        module docstring for why the matrix-multiply form of a squared distance
        does not remove it.

        The sign is *row minus centre*, which is the direction pointing from the
        centre out towards the row, and it is the sign both slopes below are
        written against.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_features)``.

        Returns
        -------
        FloatArray
            Entry ``[row, unit, feature]`` is that row's coordinate less that
            unit's centre coordinate.
        """
        return inputs[:, None, :] - self._centres[None, :, :]

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """The forward pass, given a block already known to be the right width.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_features)``, already validated by :meth:`respond_to`:
            two-dimensional, at least one row, this layer's width, every value
            finite. Nothing here re-checks any of that.
        purpose:
            Ignored. A distance is a distance whether it is being learned from
            or answered with.

        Returns
        -------
        LayerResponse
            ``scores`` is ``(n_rows, n_units)`` of squared distances, and
            ``outputs`` is the bell taken at them,
            ``exp(-score / (2 * width ** 2))``. See the module docstring for why
            the squared distance is the score rather than the bell.

        Notes
        -----
        The exponent is never positive, since a squared distance is never
        negative and a width never is either, so this cannot overflow. A row far
        from every centre underflows to exactly zero instead, which is the
        correct answer and is what makes the layer local.

        The blocks are wrapped by
        :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked`, which
        is sound because this method allocated both and has shared neither.
        """
        gaps = self._gaps_to_every_centre(inputs)
        scores = np.sum(gaps * gaps, axis=2)
        outputs = np.exp(-scores / (2.0 * self._widths * self._widths))

        return LayerResponse.already_checked(
            inputs=inputs, scores=scores, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """This layer's backward step: two slopes and a block, from one bell.

        The derivation, for one row and one unit. Write ``s`` for the score, the
        squared distance ``sum_k (x_k - c_k) ** 2``, write ``o`` for the output
        ``exp(-s / (2 w ** 2))``, and write ``g`` for the arriving slope of the
        loss at ``o``.

        Start with the bell's slope at its own score::

            d o / d s  =  o * d/ds ( -s / (2 w ** 2) )  =  -o / (2 w ** 2)

        **The centres.** The score falls as the centre approaches the row, which
        is where this layer parts company with every weighted sum in the
        package::

            d s / d c_k  =  -2 (x_k - c_k)

            d L / d c_k  =  g * (-o / (2 w ** 2)) * (-2 (x_k - c_k))
                         =  g * o * (x_k - c_k) / w ** 2

        Two negatives meet and cancel, so the slope points *along* the gap: the
        centre is pulled towards a row the loss wants it nearer to. A body that
        copies a dense layer's sign pushes it away instead, and the loss still
        falls on many steps.

        **The widths.** The width enters only through ``w ** -2`` in the
        exponent, and differentiating that is where the cube comes from::

            d o / d w  =  o * d/dw ( -(s / 2) * w ** -2 )
                       =  o * ( -(s / 2) * (-2) * w ** -3 )
                       =  o * s / w ** 3

            d L / d w  =  g * o * s / w ** 3

        It is zero at the centre, where ``s`` is zero, and positive elsewhere for
        a positive ``g``: widening a unit can only raise what it answers at any
        row it is not already sitting on. The ``w ** 3`` is the term to read
        twice, since ``w`` alone agrees with it exactly at a width of one.

        **The block passed down.** The row enters the same squared gap with the
        opposite sign to the centre, so this is the centre's own contribution
        negated, summed over the units instead of over the rows::

            d s / d x_k  =  +2 (x_k - c_k)

            d L / d x_k  =  - sum_units g * o * (x_k - c_k) / w ** 2

        Writing ``t = g * o / w ** 2`` for the shared factor, all three collapse
        to the three ways of contracting one block::

            d L / d centres      =  sum over rows  of  t * gap
            d L / d widths       =  sum over rows  of  t * score / width
            passed_down          =  - sum over units of  t * gap

        Parameters
        ----------
        response:
            What this layer did on the way up, including the block it read. Its
            ``scores`` are the squared distances and its ``outputs`` the bell,
            both of which the arithmetic above needs; the gaps are recomputed
            from its frozen ``inputs``, since a response has nowhere to carry a
            three-dimensional block and re-deriving it from the same frozen
            numbers cannot disagree with the forward pass.
        arriving:
            ``(n_rows, n_units)``, the slope of the loss at this layer's outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is ``(n_rows, n_features)``, the arriving block for
            the layer beneath. ``gradient`` carries the centre slopes as
            ``weights``, ``(n_units, n_features)``, and the width slopes as
            ``biases``, ``(n_units,)`` -- the arrangement
            :class:`~oop_ml.core.network.gradient.LayerGradient` promises,
            with nothing reshaped and nothing folded. See the module docstring
            for why borrowing that type is honest rather than convenient.

        Raises
        ------
        ShapeMismatchError
            If the response was not produced by a layer of this shape, or if the
            arriving block does not describe this layer's outputs for the rows
            the response holds.
        InvalidValuesError
            If the arriving block cannot be read as a float array.
        """
        checked = self._checked_arriving(response, arriving)
        gaps = self._gaps_to_every_centre(response.inputs)
        factors = checked * response.outputs / self._widths**2
        centre_contributions = factors[:, :, None] * gaps
        return LayerCorrection(
            passed_down=-centre_contributions.sum(axis=1),
            gradient=LayerGradient(
                weights=centre_contributions.sum(axis=0),
                biases=(factors * response.scores / self._widths).sum(axis=0),
            ),
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> RadialBasisLayer:
        """A new layer with every centre and width moved against its slope.

        Downhill is against the slope, so this subtracts. The layer is rebuilt
        rather than mutated, which is what keeps every array frozen.

        Parameters
        ----------
        gradient:
            What this layer's parameters want to change by, with the centre
            slopes in ``weights`` and the width slopes in ``biases``, exactly as
            :meth:`correction_for` sent them.
        learning_rate:
            How far to move.

        Returns
        -------
        RadialBasisLayer
            A new layer, same shape and same unit order, moved parameters.

        Raises
        ------
        ShapeMismatchError
            If no gradient is supplied, since this layer has parameters and has
            nothing to answer with when asked to step by nothing; or if the
            gradient's block does not describe this layer's centres.
        InvalidValuesError
            If the step drives a width to zero or below, by way of
            :meth:`with_parameters`. A step that large is refused by name rather
            than clamped, for the reason the module docstring gives.
        """
        if gradient is None:
            raise ShapeMismatchError(
                "a radial basis layer has parameters and needs a gradient to step by"
            )

        expected = (self._n_units, self._n_features)
        if gradient.weights.shape != expected:
            raise ShapeMismatchError(
                f"this layer's centre gradient is {expected}, got "
                f"{gradient.weights.shape}"
            )

        return self.with_parameters(
            self._centres - learning_rate * gradient.weights,
            self._widths - learning_rate * gradient.biases,
        )

    def __repr__(self) -> str:
        return (
            f"RadialBasisLayer(reads={self._n_features!r}, n_units={self._n_units!r})"
        )
