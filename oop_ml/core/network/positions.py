"""Telling each step where it sits, because attention cannot tell for itself.

The problem it addresses
------------------------
A recurrent layer reads a sequence one step at a time, so the order is built
into the arithmetic: step three is whatever happened after step two. Attention
does the opposite and reads every step at once, which is exactly what makes it
parallel and exactly what makes it blind. Permute the steps and an attention
layer's answer is the same answer permuted; it holds a *set* of measurements
that happen to be written down in a row.

That is fatal rather than inconvenient. "The dog bit the man" and "the man bit
the dog" are the same set. Two identical readings from a sensor, one taken at
the start of an hour and one at the end, are the same set. Anything the order
carried has been discarded before the first weight is applied.

The repair is to make the steps differ before attention sees them. Something
that depends only on *where* a step sits is added to what the step holds, so two
identical measurements at two different positions arrive at the layer above as
two different rows, and the difference between them encodes the distance.

Why a fixed pattern rather than a learned one
----------------------------------------------
A learned table of one vector per position is a real alternative and is an
:class:`~oop_ml.core.network.embedding.Embedding` over positions rather than a
layer of its own. What this fixed pattern buys is that it needs no data: a
position never seen during training still gets a sensible vector, because the
vector is a function rather than a lookup, and every position's vector is
already related to every other's by construction rather than by whatever the fit
happened to arrange.

Why sines and cosines of a geometric series of wavelengths
------------------------------------------------------------
Take the width in pairs. Pair ``k`` of the vector for step ``t`` holds::

    sin( t / wavelength ** (2k / n_dimensions) )
    cos( t / wavelength ** (2k / n_dimensions) )

so pair 0 turns once every ``2 pi`` steps and the last pair turns once every
``2 pi * wavelength`` steps. The near pairs distinguish adjacent steps sharply
and repeat quickly; the far pairs barely move between neighbours and never
repeat over any sequence anyone would write. A vector built from both therefore
says something about fine order and something about coarse order at once, which
a single frequency cannot: too fast and distant positions collide, too slow and
adjacent ones are indistinguishable.

The pairing is what makes relative position expressible at all. A sine and a
cosine of the same angle are a point on a circle, and advancing the step by a
fixed amount rotates that point by a fixed angle whatever the step was -- so
"twelve steps later" is one rotation, applied identically everywhere in the
sequence, and a layer above can in principle learn to read it. A pattern of
sines alone has no such structure.

Every entry lies in ``[-1, 1]``, so the pattern shifts the measurements rather
than swamping them.

Why the width may be odd
-------------------------
The last column of an odd-width vector has no partner. Its sine is kept and the
cosine that would have accompanied it is dropped, which loses the pairing for
that one column and nothing else. Refusing an odd width instead would be a
constraint the arithmetic does not have, and this package refuses only what
cannot work.

Why it learns nothing, and what its backward pass therefore is
----------------------------------------------------------------
The pattern is a constant. Adding a constant to a quantity does not change how
that quantity's slope reaches the loss, so what this layer passes down is what
arrived, unchanged, and its gradient is ``None`` rather than a block of zeros --
the distinction :class:`~oop_ml.core.network.gradient.LayerCorrection` already
draws between having nothing to learn and having something whose slope happens
to be zero.

That is a one-line derivation and is exactly why the spec checks it by finite
difference anyway. The wrong versions all run: scaled by the pattern, or with
the pattern subtracted back off, or multiplied where it should be added. Each of
them descends, and each trains a network towards something other than what was
asked for.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray

SEQUENCE_EXTENTS = 2
"""This layer reads ``(n_steps, n_dimensions)``, which is two extents."""

DEFAULT_WAVELENGTH = 10000.0
"""The longest wavelength in the series, and the usual choice.

Ten thousand steps is longer than any sequence this package is likely to be
handed, so the slowest pair has not completed a turn by the end of one and
therefore never repeats a value within it.
"""


class PositionalEncoding(Layer):
    """Adds a fixed vector to each step, depending only on where the step sits.

    Parameters
    ----------
    reads:
        ``(n_steps, n_dimensions)``, which is also what this layer answers with.
        It never changes the shape, so both of its joins agree by construction
        and it can be inserted into a sound stack without making it unsound.
    wavelength:
        The longest wavelength in the geometric series, so the slowest pair of
        columns turns once every ``2 pi * wavelength`` steps. Any positive
        finite number is accepted. One is permitted and makes every pair the
        same wave, which is degenerate rather than invalid and is useful as a
        control in a search over this setting -- the position
        :class:`~oop_ml.core.network.dropout.Dropout` takes on a drop
        probability of zero.

    Raises
    ------
    ShapeMismatchError
        If ``reads`` is not a sequence of two extents. A bare integer is the
        natural mistake, since it is what a dense layer's width looks like, and
        it earns a message that says so rather than one from builtins.
    InvalidValuesError
        If either extent is not a whole number of at least one, or the
        wavelength is not a positive finite number.

    Notes
    -----
    This layer has no parameters. :meth:`correction_for` answers with
    ``gradient=None`` and :meth:`stepped_by` answers with ``self``.
    """

    __slots__ = ("_pattern", "_shape", "_wavelength")

    def __init__(
        self,
        reads: Sequence[int],
        wavelength: float = DEFAULT_WAVELENGTH,
    ) -> None:
        # Guarded, because ``tuple(8)`` raises a bare TypeError from builtins
        # and every failure in this library is one of its own.
        try:
            extents = tuple(reads)
        except TypeError:
            raise ShapeMismatchError(
                "a positional encoding reads (n_steps, n_dimensions), got "
                f"{reads!r}, which is not a sequence of extents"
            ) from None
        if len(extents) != SEQUENCE_EXTENTS:
            raise ShapeMismatchError(
                "a positional encoding reads (n_steps, n_dimensions), got "
                f"{len(extents)} extents"
            )

        try:
            checked_wavelength = float(wavelength)
        except (TypeError, ValueError):
            raise InvalidValuesError("a wavelength must be a real number") from None
        if not np.isfinite(checked_wavelength) or checked_wavelength <= 0.0:
            raise InvalidValuesError(
                "a wavelength is a positive finite number of steps, got "
                f"{checked_wavelength}"
            )

        # LayerShape is the one object here that reads an extent, so the two
        # counts are validated by building the shape rather than by a second
        # implementation of what an extent is.
        self._shape = LayerShape(n_inputs=extents, n_outputs=extents)
        self._wavelength = checked_wavelength
        self._pattern = self._sinusoids()

    def _sinusoids(self) -> FloatArray:
        """The pattern itself, built once at construction and then frozen.

        Returns
        -------
        FloatArray
            ``(n_steps, n_dimensions)``. Even columns hold the sine of their
            pair's angle and odd columns the cosine of the same angle, so a
            pair is a point on a circle turning at that pair's own rate. An odd
            width leaves the last column a sine whose cosine has nowhere to go.

        Notes
        -----
        The angles are ``step / wavelength ** (2k / n_dimensions)``, one column
        of them per pair, so the exponent runs from zero at the first pair
        towards one at the last and the rates form a geometric series between
        one turn per ``2 pi`` steps and one per ``2 pi * wavelength``.
        """
        n_steps, n_dimensions = self._shape.reads

        steps = np.arange(n_steps, dtype=np.float64).reshape(n_steps, 1)
        pair_columns = np.arange(0, n_dimensions, 2, dtype=np.float64)
        rates = 1.0 / (self._wavelength ** (pair_columns / n_dimensions))
        angles = steps * rates

        pattern = np.zeros((n_steps, n_dimensions), dtype=np.float64)
        pattern[:, 0::2] = np.sin(angles)
        # One column short when the width is odd, since the final sine's cosine
        # has no column to land in. Slicing the angles to the number of odd
        # columns drops it rather than wrapping it round to a pair it does not
        # belong to.
        n_cosines = pattern[:, 1::2].shape[1]
        pattern[:, 1::2] = np.cos(angles[:, :n_cosines])

        pattern.setflags(write=False)
        return pattern

    @property
    def shape(self) -> LayerShape:
        """The same arrangement on both sides. This layer moves no join."""
        return self._shape

    @property
    def pattern(self) -> FloatArray:
        """``(n_steps, n_dimensions)``, frozen, one vector per position.

        Exposed because it is the whole content of the layer, and because a
        spec asserting the interleaving and a diagram showing the bands of
        wavelengths both want to read it directly rather than infer it from an
        answer. Frozen, like every block this library hands outward.
        """
        return self._pattern

    @property
    def wavelength(self) -> float:
        """The longest wavelength in the series."""
        return self._wavelength

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """Add the pattern to every step of every row.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_steps, n_dimensions)``, already known by
            :meth:`~oop_ml.core.network.layer.Layer.respond_to` to be the right
            arrangement, non-empty and finite.
        purpose:
            Ignored. The pattern is a constant either way, which is what makes
            a saved model evaluable without a training-time setting.

        Returns
        -------
        LayerResponse
            ``scores`` and ``outputs`` are the same block, since nothing is
            bent here and so there is no pre-activation value distinct from the
            answer.

        Notes
        -----
        The pattern is ``(n_steps, n_dimensions)`` and the block is
        ``(n_rows, n_steps, n_dimensions)``, so the addition broadcasts over the
        rows: every sequence in the batch is told the same thing about position,
        which is correct, since position means where a step sits inside its own
        sequence rather than which sequence it belongs to.
        """
        outputs = inputs + self._pattern

        return LayerResponse.already_checked(
            inputs=inputs, scores=outputs, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """Hand the arriving slopes straight down, because a constant was added.

        Parameters
        ----------
        response:
            What this layer did on the way up. Only its arrangement and row
            count are read, since adding a constant makes the backward step
            independent of the values that went through.
        arriving:
            ``(n_rows, n_steps, n_dimensions)``, the slope of the loss at this
            layer's outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is the arriving block unchanged, and ``gradient`` is
            ``None`` because this layer holds no parameters for a gradient to
            describe.

        Raises
        ------
        ShapeMismatchError
            If the response did not come from a layer of this shape, or the
            arriving block does not describe this layer's outputs for the rows
            the response holds.
        InvalidValuesError
            If the arriving block cannot be read as a float array.

        Notes
        -----
        There is no arithmetic here, and that is the whole content of the
        method. Output ``[row, step, column]`` is input ``[row, step, column]``
        plus a number that depends on nothing, so the slope of the loss at the
        one is the slope at the other exactly. The spec checks it by finite
        difference regardless, because every wrong version of this line still
        runs and still descends.
        """
        block = self._checked_arriving(response, arriving)

        return LayerCorrection(passed_down=block, gradient=None)

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> PositionalEncoding:
        """This same layer, because there is nothing in it that can move.

        Parameters
        ----------
        gradient:
            ``None`` for this layer, since :meth:`correction_for` never produces
            anything else. A gradient handed here describes parameters this
            layer does not have, and moves nothing.
        learning_rate:
            How far a parameter would move. Unused, for the same reason.

        Returns
        -------
        PositionalEncoding
            ``self``. The pattern is a function of two integers and a
            wavelength, all settled at construction, so an unchanged immutable
            layer has nothing a copy could differ in.
        """
        return self

    def __repr__(self) -> str:
        return (
            f"PositionalEncoding(reads={self._shape.reads!r}, "
            f"wavelength={self._wavelength!r})"
        )
