"""Storing a direction and a length instead of a weight vector.

This is not a normalisation layer, whatever its name says
----------------------------------------------------------
Every other layer with "normalisation" in its name here standardises
*activations*. :class:`~oop_ml.core.network.normalisation.BatchNormalization`
reduces down the rows and rewrites each feature by what the batch looked like;
:class:`~oop_ml.core.network.row_normalisation.WithinRowNormalization` reduces
inside a row and rewrites each row by its own contents. Both of them change what
comes out for a given set of parameters, both of them therefore have a forward
pass of their own, and one of them has to remember what it saw.

This layer standardises nothing at runtime. Its forward pass is a dense layer's,
value for value. What it changes is where the parameters live: instead of
storing a weight vector per neuron it stores a *direction* and a *length*, and
the weight vector is rebuilt from the two on the way past::

    weights[neuron] = magnitude[neuron] * direction[neuron] / norm(direction[neuron])

So the family resemblance is a name and nothing else. Salimans and Kingma, 2016,
and the paper's own argument is that this is what batch normalisation's benefit
looks like once the batch is taken out of it.

What the reparameterisation buys
---------------------------------
A neuron's weight vector has a length and a direction whether anyone stores them
separately or not, and those two do quite different jobs. The direction decides
which way the neuron's boundary faces. The length decides how sharply it
answers, since it multiplies every score before the bend sees it.

Written the ordinary way, neither is a parameter. Both are consequences of the
weights, so gradient descent can only move the length by moving every component
of the vector at once and hoping the total comes out right, and any step that
changes the length also swings the direction. Written this way they separate.
One number per neuron owns the length, the gradient reaches it directly, and a
step of the direction cannot change the length at all, because the length was
divided out.

That is a claim about the *surface*, not about the function. The function is
unchanged, and the first thing the spec asserts is that this layer answers
identically to a :class:`~oop_ml.core.network.layer.DenseLayer` holding the
weights it produces. What differs is that the same learning rate applied to the
same slope arrives somewhere else, which the spec also asserts, because a
reparameterisation that moved the parameters to the same place would be a
renaming.

The backward pass, and the term that is easy to lose
-----------------------------------------------------
Write ``weight_slope`` for the gradient with respect to the effective weights --
which is a dense layer's ordinary ``delta.T @ inputs`` and is computed exactly
that way here -- and ``unit_direction`` for ``direction / norm(direction)``.
Then, per neuron::

    magnitude_slope = weight_slope . unit_direction

    direction_slope = (magnitude / norm(direction))
                      * (weight_slope - magnitude_slope * unit_direction)

The magnitude's slope is the component of the weight slope lying *along* the
direction, which is the whole of what lengthening the vector can achieve. The
direction's slope is the rest of it, rescaled: the same weight slope with that
radial component projected out.

Dropping the projection is the mistake this module is about. It leaves
``(magnitude / norm) * weight_slope``, which has the right shape, plausible
magnitudes, and trains. Measured against a central finite difference on the
spec's wide fixture, five rows of four inputs into three neurons, whose largest
true direction slope is 2.3730, the projected form disagrees by 3.5e-10 and the
unprojected one by 0.6123 -- a quarter of the largest slope it was trying to
report. On the narrow fixture the unprojected error is 0.2202 against a largest
true slope of 0.9015, so the error is a quarter there too, which is the point:
it is not a rounding and it is not a blow-up, it is a wrong direction of travel
that a training curve will not show.

The cheap structural reading is the same kind of thing batch normalisation's
column sums are. Scaling a direction by any positive constant leaves every
answer unchanged, so the loss is flat along the direction itself, so the
gradient can have no component along it -- ``direction_slope . direction`` is
zero for every neuron, exactly. Measured, the projected form gives 5.8e-16 there
and the unprojected one 2.0155.

Why the gradient carries three blocks
---------------------------------------
There are three parameter sets here and
:class:`~oop_ml.core.network.gradient.LayerGradient` carries two, which is the
problem :class:`~oop_ml.core.network.normalisation.BatchStatistics` already
solved once. :class:`ReparameterisedGradient` subclasses it the same way, with
the directions in the ``weights`` block, the biases in the ``biases`` block, and
the magnitudes carried alongside.

Putting the directions in ``weights`` needs saying out loud, because that block
is *not* the slope with respect to the effective weight matrix. It is the slope
with respect to what this layer actually stores and actually steps, and the two
differ by exactly the projection above. A caller reading ``gradient.weights``
off this layer and expecting a dense layer's answer would be reading a different
quantity, which is why the property is also reachable as
:attr:`ReparameterisedGradient.directions` and why both methods here use that
name.

Why it takes an activation
---------------------------
Because otherwise it could never be used. There is no standalone activation
layer in this package -- a bend belongs to a
:class:`~oop_ml.core.network.neuron.Neuron` and therefore to the dense layer
holding it -- so a weight-normalised layer with no bend of its own is an affine
map with nothing able to follow it but another affine map, and
:mod:`oop_ml.core.network.activation` measures what a stack of those collapses
to. It takes one shared bend rather than one per neuron, because the
reparameterisation is the subject here and a layer mixing its activations is a
:class:`~oop_ml.core.network.layer.DenseLayer` concern. It defaults to
:class:`~oop_ml.core.network.activation.Identity`, so an output layer is not a
special case.

Why a direction of zero is refused rather than nursed
-------------------------------------------------------
Every other layer here that divides by something has an ``epsilon``, and this
one deliberately does not.

The difference is what the divisor is. A normalisation layer divides by a spread
measured from *data*, and a constant feature is an ordinary thing for data to
contain -- a rectified unit that is off for the whole batch produces exactly
that -- so the layer has to answer something, and epsilon is what makes the
answer zero rather than ``nan``. A norm here is measured from a *parameter* the
caller chose. A zero vector does not name a direction badly, it names no
direction at all, and there is no number the layer could answer that would be
the right one.

It is also unrecoverable. Both slopes above carry ``magnitude / norm``, so a
neuron whose direction reached zero has an undefined gradient as well as an
undefined answer, and no epsilon large enough to keep the division finite would
be small enough to leave the arithmetic meaning what it says. So the constructor
refuses it, and :meth:`WeightNormalization.stepped_by` inherits that refusal by
building the next layer through the same constructor -- a step that lands
exactly on zero raises rather than producing a network that answers ``nan`` to
everything from then on.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.activation import Activation, Identity
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


def _as_direction_block(values: object) -> FloatArray:
    """Read the direction matrix, refusing in the library's own words.

    Parameters
    ----------
    values:
        The candidate block, ``(n_neurons, n_inputs)``.

    Returns
    -------
    FloatArray
        A private, frozen float64 copy.

    Raises
    ------
    InvalidValuesError
        If it cannot be read as a float array, or carries a non-finite entry.
    ShapeMismatchError
        If it is not two-dimensional.
    """
    try:
        block = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise InvalidValuesError(
            "directions must be readable as a float array"
        ) from error

    if block.ndim != 2:
        raise ShapeMismatchError(
            "directions are one row per neuron, so (n_neurons, n_inputs), got "
            f"{block.ndim} dimensions"
        )
    if not np.isfinite(block).all():
        raise InvalidValuesError("directions must contain only finite values")

    block.setflags(write=False)
    return block


def _as_per_neuron(values: object, n_neurons: int, role: str) -> FloatArray:
    """Read one per-neuron vector, refusing in the library's own words.

    Parameters
    ----------
    values:
        The candidate vector.
    n_neurons:
        How many entries it must have.
    role:
        What it is, for the message.

    Returns
    -------
    FloatArray
        A private, frozen float64 copy.

    Raises
    ------
    InvalidValuesError
        If it cannot be read as a float array, or carries a non-finite entry.
    ShapeMismatchError
        If it is not one-dimensional with exactly ``n_neurons`` entries.
    """
    try:
        block = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise InvalidValuesError(f"{role} must be readable as a float array") from error

    if block.ndim != 1 or block.shape[0] != n_neurons:
        raise ShapeMismatchError(
            f"{role} is one value per neuron, so ({n_neurons},), got {block.shape}"
        )
    if not np.isfinite(block).all():
        raise InvalidValuesError(f"{role} must contain only finite values")

    block.setflags(write=False)
    return block


class ReparameterisedGradient(LayerGradient):
    """A gradient in direction-and-length coordinates rather than weights.

    A :class:`~oop_ml.core.network.gradient.LayerGradient` carries two parameter
    sets and this layer has three, which is the shortfall
    :class:`~oop_ml.core.network.normalisation.BatchStatistics` met first and
    answered the same way. Widening the general type for one layer's benefit
    would be the wrong trade, and a side channel would put the third block
    outside the object a stack already threads, so the layer that needs more
    subclasses and hands back more.

    Parameters
    ----------
    weights:
        ``(n_neurons, n_inputs)``, the slope with respect to the **directions**,
        which is what this layer stores. It is *not* the slope with respect to
        the effective weight matrix; the two differ by the radial projection the
        module docstring derives, and reading this block as a dense layer's
        gradient would silently be reading a different quantity. Also reachable
        as :attr:`directions`, which is the name both of this layer's methods
        use.
    biases:
        ``(n_neurons,)``, one slope per neuron's bias.
    magnitudes:
        ``(n_neurons,)``, one slope per neuron's length. The block that does not
        fit in ``LayerGradient`` and the reason this class exists.

    Raises
    ------
    ShapeMismatchError
        If the weight and bias blocks disagree, by way of ``LayerGradient``, or
        if ``magnitudes`` is not one value per neuron.
    InvalidValuesError
        If ``magnitudes`` carries a non-finite entry.
    """

    __slots__ = ("_magnitudes",)

    def __init__(
        self, weights: FloatArray, biases: FloatArray, magnitudes: FloatArray
    ) -> None:
        super().__init__(weights=weights, biases=biases)
        self._magnitudes = _as_per_neuron(
            magnitudes, self.biases.shape[0], "a magnitude gradient"
        )

    @property
    def directions(self) -> FloatArray:
        """``(n_neurons, n_inputs)``, the slope for the directions.

        The same block as ``weights``, named for what it actually is here. The
        general type calls it ``weights`` because that is what the block means
        for every other layer, and this layer is the one where it does not.
        """
        return self.weights

    @property
    def magnitudes(self) -> FloatArray:
        """``(n_neurons,)``, the slope for the lengths."""
        return self._magnitudes

    @property
    def largest_movement(self) -> float:
        """The biggest single slope, the magnitudes included.

        Overridden because the base reads two blocks and this object has three,
        and a convergence check that could not see the lengths would call a
        network settled while every neuron's sharpness was still moving.
        """
        return max(super().largest_movement, float(np.max(np.abs(self._magnitudes))))

    def __repr__(self) -> str:
        return f"ReparameterisedGradient(shape={self.weights.shape!r})"


class WeightNormalization(Layer):
    """A dense layer whose weights are stored as a direction and a length.

    Parameters
    ----------
    directions:
        ``(n_neurons, n_inputs)``, one row per neuron. Only each row's
        *direction* is read, since its length is divided out, so scaling a row
        by any positive constant changes nothing this layer answers. No row may
        be all zeros.
    magnitudes:
        ``(n_neurons,)``, the length each neuron's weight vector is rebuilt at.
        Defaults to the lengths ``directions`` already has, which makes the
        effective weights exactly the block that was supplied and is what lets a
        caller hand in a weight matrix and get the same layer back.
    biases:
        ``(n_neurons,)``, one per neuron. Defaults to zeros, by the same
        argument the convolution's biases use.
    activation:
        The bend every neuron shares. Defaults to
        :class:`~oop_ml.core.network.activation.Identity`. See the module
        docstring for why this layer has one at all.

    Raises
    ------
    ShapeMismatchError
        If ``directions`` is not two-dimensional, or a supplied vector is not
        one value per neuron.
    InvalidValuesError
        If any supplied block carries a non-finite entry, if the layer would
        have no neurons or no inputs, or if a direction row is all zeros.

    Notes
    -----
    The effective weight matrix is built once here rather than on every forward
    pass, which is sound for the same reason
    :class:`~oop_ml.core.network.layer.DenseLayer`'s cached matrix is: every
    block is frozen and a step builds a new layer, so the cache cannot go stale.
    """

    __slots__ = (
        "_activation",
        "_biases",
        "_directions",
        "_effective_weights",
        "_magnitudes",
        "_norms",
        "_shape",
        "_unit_directions",
    )

    def __init__(
        self,
        directions: FloatArray,
        magnitudes: FloatArray | None = None,
        biases: FloatArray | None = None,
        activation: Activation | None = None,
    ) -> None:
        self._directions = _as_direction_block(directions)
        n_neurons, n_inputs = self._directions.shape

        # LayerShape refuses an extent below one, which is what turns a block of
        # no neurons or no inputs into a typed refusal rather than an empty
        # layer that agrees with everything downstream vacuously.
        self._shape = LayerShape(n_inputs=n_inputs, n_outputs=n_neurons)

        norms = np.linalg.norm(self._directions, axis=1)
        if not np.isfinite(norms).all() or (norms <= 0.0).any():
            raise InvalidValuesError(
                "every direction must have a positive, finite length, since a "
                "zero vector names no direction and both slopes divide by that "
                f"length, got lengths {norms}"
            )
        norms.setflags(write=False)
        self._norms = norms

        unit_directions = self._directions / norms[:, None]
        unit_directions.setflags(write=False)
        self._unit_directions = unit_directions

        self._magnitudes = _as_per_neuron(
            norms if magnitudes is None else magnitudes, n_neurons, "a magnitude"
        )
        self._biases = _as_per_neuron(
            np.zeros(n_neurons) if biases is None else biases, n_neurons, "a bias"
        )
        self._activation = Identity() if activation is None else activation

        effective_weights = self._magnitudes[:, None] * unit_directions
        effective_weights.setflags(write=False)
        self._effective_weights = effective_weights

    @property
    def shape(self) -> LayerShape:
        """The width this layer reads and the width it answers with."""
        return self._shape

    @property
    def n_neurons(self) -> int:
        """How many neurons answer, which is this layer's output width."""
        return self._shape.n_outputs

    @property
    def n_inputs(self) -> int:
        """How many numbers one row holds."""
        return self._shape.n_inputs

    @property
    def directions(self) -> FloatArray:
        """``(n_neurons, n_inputs)``, one stored direction per neuron. Frozen.

        The block as it was supplied, at whatever length it happened to have.
        Only its rows' directions reach the answer.
        """
        return self._directions

    @property
    def magnitudes(self) -> FloatArray:
        """``(n_neurons,)``, the length of each neuron's weight vector. Frozen."""
        return self._magnitudes

    @property
    def biases(self) -> FloatArray:
        """``(n_neurons,)``, one bias per neuron. Frozen."""
        return self._biases

    @property
    def activation(self) -> Activation:
        """The bend every neuron in this layer shares."""
        return self._activation

    @property
    def effective_weights(self) -> FloatArray:
        """``(n_neurons, n_inputs)``, what the reparameterisation produced.

        ``magnitude * direction / norm(direction)`` per row, and the block the
        forward pass actually multiplies by. Exposed because the whole claim of
        this layer is that these weights are an ordinary dense layer's, so a
        caller comparing the two -- or saving one, or drawing one -- needs to be
        able to see them. Frozen, like every other block here.
        """
        return self._effective_weights

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """The forward pass, which is a dense layer's and nothing more.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_inputs)``, already validated by
            :meth:`~oop_ml.core.network.layer.Layer.respond_to`.
        purpose:
            Ignored. Nothing here is measured from the batch and nothing is
            drawn at random, so the answer for a row is the same whether the
            pass is training or predicting. That is the ordinary case; the two
            layers that cannot say it are named in
            :mod:`oop_ml.core.network.purpose`.

        Returns
        -------
        LayerResponse
            ``scores`` are the weighted sums plus biases and ``outputs`` are
            those bent, both ``(n_rows, n_neurons)``, exactly as a dense layer
            would produce them.
        """
        scores = inputs @ self._effective_weights.T + self._biases
        outputs = self._activation.of(scores)

        return LayerResponse.already_checked(
            inputs=inputs, scores=scores, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """A dense layer's backward step, then the change of coordinates.

        Parameters
        ----------
        response:
            What this layer did on the way up, including the block it read.
        arriving:
            ``(n_rows, n_neurons)``, the slope of the loss at this layer's
            outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is ``(n_rows, n_inputs)``, and ``gradient`` is a
            :class:`ReparameterisedGradient` carrying a slope for each of the
            three parameter sets.

        Raises
        ------
        InvalidValuesError
            If ``arriving`` cannot be read as a float array.
        ShapeMismatchError
            If the response did not come from a layer of this shape, or
            ``arriving`` does not describe this layer's outputs for the rows the
            response holds.

        Notes
        -----
        The first half is a dense layer's, unchanged, because the forward pass
        was a dense layer's::

            delta        = arriving * activation slope at the scores
            weight_slope = delta.T @ what this layer read
            bias_slope   = delta.sum(axis=0)
            passed_down  = delta @ effective_weights

        The second half is the chain rule through the reparameterisation, one
        neuron at a time and written as two array expressions::

            magnitude_slope = weight_slope . unit_direction
            direction_slope = (magnitude / norm)
                              * (weight_slope - magnitude_slope * unit_direction)

        The subtracted term is the projection onto the direction, and it is the
        one that is easy to lose. Without it the answer is
        ``(magnitude / norm) * weight_slope``, which conforms, trains, and is
        not the gradient of the loss -- see the module docstring for what a
        finite difference measures it as.

        There is no provenance check to make here beyond the shape one the base
        already runs. Unlike batch normalisation this layer has no second
        forward pass to be confused with, so any response of the right shape
        really did come from this arithmetic.
        """
        arriving = self._checked_arriving(response, arriving)

        delta = arriving * self._activation.derivative_at(response.scores)

        # The dense half. This is the slope with respect to the *effective*
        # weights, which is not what this layer stores and not what it steps.
        weight_slope = delta.T @ response.inputs

        # The change of coordinates. The component of the weight slope lying
        # along the direction is all that lengthening the vector can achieve,
        # so it is the magnitude's whole slope; the direction gets the rest,
        # rescaled, with that radial component projected out.
        magnitude_slope = (weight_slope * self._unit_directions).sum(axis=1)
        direction_slope = (self._magnitudes / self._norms)[:, None] * (
            weight_slope - magnitude_slope[:, None] * self._unit_directions
        )

        return LayerCorrection(
            passed_down=delta @ self._effective_weights,
            gradient=ReparameterisedGradient(
                weights=direction_slope,
                biases=delta.sum(axis=0),
                magnitudes=magnitude_slope,
            ),
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> WeightNormalization:
        """A new layer with all three parameter sets moved against their slopes.

        Parameters
        ----------
        gradient:
            Must be a :class:`ReparameterisedGradient`. A plain
            :class:`~oop_ml.core.network.gradient.LayerGradient` carries no
            magnitudes, and a layer that stepped its directions while leaving
            its lengths where they were would be learning half of what it was
            told.
        learning_rate:
            How far to move.

        Returns
        -------
        WeightNormalization
            A new layer, same width and same bend.

        Raises
        ------
        ShapeMismatchError
            If ``gradient`` is ``None``, is not a
            :class:`ReparameterisedGradient`, or does not describe this layer's
            arrangement.
        InvalidValuesError
            If the step lands a direction exactly on zero. Inherited from the
            constructor rather than checked again here, which is the point of
            rebuilding through it.

        Notes
        -----
        The step is the ordinary subtract-the-slope in all three blocks, and it
        is the *stored* directions that move rather than the unit ones. A step
        therefore changes each direction's length as well as where it points,
        and none of that reaches the answer, because the length is divided out
        again on the way back in. That is the reparameterisation doing its job
        rather than a leak: the neuron's actual length is the magnitude, and
        only the magnitude's own slope can move it.
        """
        if gradient is None:
            raise ShapeMismatchError(
                "a weight normalisation layer has parameters and needs a "
                "gradient to step by"
            )
        if not isinstance(gradient, ReparameterisedGradient):
            raise ShapeMismatchError(
                "a weight normalisation layer steps by a ReparameterisedGradient, "
                "which carries a slope for the lengths as well as for the "
                "directions and the biases; a plain LayerGradient would leave "
                "every neuron's length where it was"
            )
        if gradient.directions.shape != self._directions.shape:
            raise ShapeMismatchError(
                f"this layer's directions are {self._directions.shape}, and the "
                f"gradient describes {gradient.directions.shape}"
            )

        rate = float(learning_rate)
        return WeightNormalization(
            directions=self._directions - rate * gradient.directions,
            magnitudes=self._magnitudes - rate * gradient.magnitudes,
            biases=self._biases - rate * gradient.biases,
            activation=self._activation,
        )

    def __repr__(self) -> str:
        return (
            f"WeightNormalization(n_inputs={self.n_inputs!r}, "
            f"n_neurons={self.n_neurons!r})"
        )
