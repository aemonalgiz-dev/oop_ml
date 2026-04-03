"""Deliberately breaking the network while it learns, so it cannot lean.

The problem it addresses
------------------------
A wide layer trained to convergence tends to arrive at *co-adaptation*: unit 7
learns to be useful only in the company of unit 12, because during training
unit 12 was always there. The pair works, and neither half is a detector of
anything on its own. That is a fragile thing to have built, and it shows up as
the ordinary overfitting story, a training score that keeps improving while a
held-out score stops.

Dropout attacks it directly rather than by penalising weights. On every training
pass each unit is independently silenced with probability ``p``, so no unit can
count on any other being present, and every unit is pushed toward being useful
by itself. The usual framing is that it trains an exponentially large ensemble
of thinned networks sharing one set of weights, and then predicts with the
average of them; the framing that explains the mechanism better is that a
feature which only works alongside a specific partner now gets a bad score
roughly ``p`` of the time.

Why the answer has to be scaled, and where
-------------------------------------------
Silencing a fraction of the units lowers the sum the next layer reads. If a
layer sends 100 numbers averaging 1.0 and a third are zeroed, the next layer's
weighted sum is a third smaller, so a network trained under dropout and then
asked to predict with every unit present sees inputs it has never seen at the
scale it has never seen them at.

There are two places to fix that, and they are not equally good:

* Scale *down* at prediction time by ``1 - p``. This is the original
  formulation and it is why old descriptions of dropout have a step at the end.
  It makes prediction depend on a training-time hyperparameter, so a saved model
  cannot be evaluated without it.
* Scale *up* at training time by ``1 / (1 - p)``. The surviving units are made
  proportionally louder so the expected sum is unchanged, and prediction becomes
  the identity function.

The second is *inverted dropout*, and it is what every current implementation
does and what this one does. The payoff is that
:attr:`~oop_ml.core.network.purpose.PassPurpose.PREDICTING` is a pass-through:
this layer disappears at prediction time, which is the honest description of
what it is for.

The expectation argument, in one line. A unit's contribution during training is
``0`` with probability ``p`` and ``value / (1 - p)`` with probability ``1 - p``,
so its expected contribution is ``(1 - p) * value / (1 - p) = value``, which is
exactly what prediction supplies. The layer is unbiased by construction rather
than approximately.

Why the mask is carried on the response
----------------------------------------
:class:`~oop_ml.core.network.pooling.Pool2d` recomputes which position won each
window rather than remembering it, and argues that carrying a derivable fact is
storage rather than information. The opposite conclusion holds here, and the
reason is the whole difference between the two layers: a pooling winner is a
function of numbers the response already holds, and a dropout mask is a *draw*.
It cannot be recomputed from the inputs, from the outputs, or from anything else
that survives the forward pass.

The tempting shortcut is to read the mask back off the answers -- a zero in the
output must have been dropped. It is wrong, and wrong in the quiet way. An input
that was genuinely zero and genuinely kept also leaves a zero, so it would be
recorded as dropped and its gradient set to zero. Exact zeros are not rare in
the places this layer is used: they are what a rectified unit produces for every
negative score, and they are most of the border of a scanned digit. So
:class:`DropoutResponse` carries the mask, and :meth:`Dropout.correction_for`
refuses a response that does not have one rather than guessing.

Why the generator is the one piece of mutable state here
---------------------------------------------------------
Every other layer in this package is immutable, and this one nearly is: the
probability and the shape are settled at construction and never change. But a
dropout layer that drew the same mask on every pass would silence the same units
forever, which is not dropout at all, it is a smaller network with a strange
initialisation. So the draws have to advance, and something has to hold the
position in the stream.

That is runtime state rather than learned state, and the library already draws
that line: a tree holds a generator too, and the persistence format excludes it
from ``LEARNED_STATE`` for exactly this reason. Two things follow and both are
tested. :meth:`Dropout.stepped_by` answers with ``self``, because the layer has
no parameters and rebuilding it would reset the stream. And a seeded layer is
reproducible over a *sequence* of passes rather than per pass, so the test that
pins it runs several and compares the whole sequence.

What it does not do
-------------------
It has no parameters, so it reports ``gradient=None``. It does not change the
shape, so it can be dropped into a stack anywhere without moving a join. And it
is not a substitute for having enough data; it is a way of spending capacity you
already have more carefully.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


class DropoutResponse(LayerResponse):
    """A layer response that also remembers which units survived the draw.

    See the module docstring for why this exists rather than the mask being
    recovered from the outputs. In short: a zero in the answer is ambiguous
    between "dropped" and "kept, and was already zero", and the second is
    common wherever a rectified unit or a blank pixel is involved.

    ``kept`` is the *scaled* mask rather than a boolean one, holding
    ``1 / (1 - p)`` where the unit survived and ``0`` where it did not. That is
    the number the forward pass multiplied by, so it is also the number the
    backward pass multiplies by, and keeping it in that form means the scale
    appears once in the module rather than once per direction. A scale applied
    forward and forgotten backward is a gradient wrong by a constant factor,
    which trains, converges, and is quietly not the network that was asked for.
    """

    __slots__ = ("_kept",)

    @classmethod
    def recording(
        cls, inputs: FloatArray, outputs: FloatArray, kept: FloatArray
    ) -> DropoutResponse:
        """Wrap a forward pass this layer just built, along with its mask.

        The :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked`
        pattern, which is sound for the same reason: this method's caller
        allocated every block, filled it, and has shared none of them.

        Parameters
        ----------
        inputs:
            The block the layer read.
        outputs:
            The masked, rescaled block. Also used as the scores, since this
            layer applies no bend and so has no pre-activation value distinct
            from its answer.
        kept:
            The scaled mask that produced ``outputs`` from ``inputs``.
        """
        response = cls.__new__(cls)
        for block in (inputs, outputs, kept):
            block.setflags(write=False)
        response._inputs = inputs
        response._scores = outputs
        response._outputs = outputs
        response._kept = kept
        return response

    @property
    def kept(self) -> FloatArray:
        """The scaled mask: ``1 / (1 - p)`` where a unit survived, ``0`` where not.

        Frozen, like every other block this library hands outward. A caller who
        edited it would change what the backward pass believes happened on a
        forward pass that has already been run.
        """
        return self._kept

    def __eq__(self, other: object) -> bool:
        # The mask is the whole reason this class exists, so two responses that
        # thinned differently are not the same response however well their
        # answers happen to line up. The base settles the type check and the
        # blocks; this adds the one field it does not know about.
        if not isinstance(other, DropoutResponse):
            return NotImplemented
        # The base answers NotImplemented when the two types are not identical,
        # which happens for a subclass of this class comparing against this one.
        # ``bool(NotImplemented)`` is True and warns, so calling the base and
        # coercing would report that two responses of different types are equal
        # -- the exact failure this override exists to prevent, reintroduced one
        # level down.
        alike = super().__eq__(other)
        if alike is NotImplemented:
            return NotImplemented
        return bool(alike) and bool(np.array_equal(self._kept, other._kept))

    def __repr__(self) -> str:
        return (
            f"DropoutResponse(n_rows={self.n_rows!r}, "
            f"n_kept={int(np.count_nonzero(self._kept))!r})"
        )


class Dropout(Layer):
    """Silences each unit independently while training, and does nothing while not.

    Parameters
    ----------
    reads:
        The arrangement of one input, which is also what this layer answers
        with. An integer for a row of numbers, a tuple for anything arranged,
        so ``Dropout(512)`` sits above a dense layer and
        ``Dropout((8, 26, 26))`` sits above a convolution. The layer never
        changes the shape, so both of its joins agree by construction and it can
        be inserted into a sound stack without making it unsound.
    drop_probability:
        The chance each unit is silenced on a training pass, in ``[0, 1)``.
        Zero is permitted and makes the layer an expensive identity, which is
        useful as a control in a search over this hyperparameter. One is not:
        it silences the whole layer, leaves nothing for the layers above to
        read, and makes the ``1 / (1 - p)`` scale a division by zero.
    random_seed:
        Fixes the sequence of draws, for a reproducible run. ``None`` draws
        from fresh entropy.

    Raises
    ------
    InvalidValuesError
        If ``drop_probability`` is not a real number in ``[0, 1)``, or if
        ``reads`` holds anything that is not a whole number of at least one.

    Notes
    -----
    This layer has no parameters. :meth:`correction_for` answers with
    ``gradient=None`` and :meth:`stepped_by` answers with ``self``.

    The generator advances as the layer is used, which is the one piece of
    mutable state in this package and is discussed in the module docstring.
    """

    __slots__ = ("_drawing", "_drop_probability", "_random_seed", "_shape")

    def __init__(
        self,
        reads: int | tuple[int, ...],
        drop_probability: float = 0.5,
        random_seed: int | None = None,
    ) -> None:
        try:
            probability = float(drop_probability)
        except (TypeError, ValueError):
            raise InvalidValuesError(
                "a drop probability must be a real number"
            ) from None
        if not np.isfinite(probability):
            raise InvalidValuesError(
                f"a drop probability must be finite, got {probability}"
            )
        if not 0.0 <= probability < 1.0:
            raise InvalidValuesError(
                "a drop probability lies in [0, 1); 1 silences the whole layer "
                f"and leaves nothing to scale, got {probability}"
            )

        # LayerShape does the extent validation, and reads == answers because
        # this layer never changes the arrangement. That is what lets it be
        # inserted anywhere in a sound stack.
        self._shape = LayerShape(n_inputs=reads, n_outputs=reads)
        self._drop_probability = probability
        self._random_seed = random_seed
        self._drawing = np.random.default_rng(random_seed)

    @property
    def shape(self) -> LayerShape:
        """The same arrangement on both sides. This layer moves no join."""
        return self._shape

    @property
    def drop_probability(self) -> float:
        """The chance each unit is silenced on a training pass."""
        return self._drop_probability

    @property
    def keep_probability(self) -> float:
        """The chance each unit survives, which is ``1 - drop_probability``.

        Named because it is the quantity the arithmetic actually uses -- the
        surviving units are scaled by its reciprocal -- and reading
        ``1 / (1 - self._drop_probability)`` at each site invites the sign
        mistake this property removes.
        """
        return 1.0 - self._drop_probability

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """Silence units and rescale the survivors, or pass the block straight through.

        Parameters
        ----------
        inputs:
            ``(n_rows, *shape.reads)``, already known by :meth:`respond_to` to be
            the right arrangement, non-empty and finite.
        purpose:
            The one layer here that reads this, along with
            :class:`~oop_ml.core.network.normalisation.BatchNormalization`.
            Under :attr:`~oop_ml.core.network.purpose.PassPurpose.PREDICTING`
            nothing is dropped and nothing is scaled.

        Returns
        -------
        LayerResponse
            While predicting, an ordinary
            :class:`~oop_ml.core.network.layer.LayerResponse` whose outputs are
            the inputs. While training, a :class:`DropoutResponse` carrying the
            scaled mask, because a backward pass cannot proceed without it.

        Notes
        -----
        The draw is independent per unit *and per row*, so two rows in the same
        block are thinned differently. That is what the mechanism asks for: the
        point is that a unit cannot rely on a partner, and a mask shared across
        the block would let it rely on one for the whole batch.

        The mask is scaled rather than boolean, holding
        ``1 / keep_probability`` where a unit survived. See
        :class:`DropoutResponse` for why it is kept in that form.
        """
        raise NotImplementedError

    def _checked_mask(self, response: LayerResponse) -> FloatArray:
        """The mask this response carries, refusing a response that has none.

        The companion to
        :meth:`~oop_ml.core.network.layer.Layer._checked_arriving`, and the same
        kind of check: not about the data, about whether these two objects
        belong together. That one settles arrangement and row count, which a
        prediction-pass response would satisfy perfectly while carrying no mask
        at all.

        Raises
        ------
        ShapeMismatchError
            If the response is not a :class:`DropoutResponse`. That means it was
            produced while predicting, when nothing was dropped and no backward
            step is defined, or by some other layer of the same shape. Guessing
            a mask from the outputs is the quiet mistake the module docstring
            sets out.
        """
        if not isinstance(response, DropoutResponse):
            raise ShapeMismatchError(
                "a dropout layer's backward step needs the mask its own "
                "forward pass drew, and this response carries none -- it came "
                "from a prediction pass or from another layer"
            )
        return response.kept

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """Send the blame back through the same mask the forward pass drew.

        Parameters
        ----------
        response:
            What this layer did on the way up. Must be a
            :class:`DropoutResponse`, since the mask cannot be recovered from
            anything else.
        arriving:
            ``(n_rows, *shape.answers)``, the slope of the loss at this layer's
            outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is the arriving block seen through the same mask,
            and ``gradient`` is ``None`` because this layer has nothing to
            learn.

        Raises
        ------
        InvalidValuesError
            If ``arriving`` cannot be read as a float array.
        ShapeMismatchError
            If the response was not produced by a layer of this shape, if
            ``arriving`` does not describe this layer's outputs for the rows the
            response holds, or if the response carries no mask -- which means it
            came from a prediction pass, or from another layer entirely, and
            either way the backward step is not defined.

        Notes
        -----
        The whole of it. A unit that was silenced contributed nothing to the
        loss, so it is owed nothing back; a unit that survived was amplified by
        ``1 / keep_probability`` on the way up, so its blame is amplified by the
        same factor on the way down. Both of those are the same multiplication
        by :attr:`DropoutResponse.kept`, which is why the mask is stored
        pre-scaled.

        Forgetting the scale here is the mistake worth naming, because it does
        not announce itself. The gradient comes out uniformly too small by
        ``keep_probability``, the network still trains, the loss still falls,
        and the effective learning rate is quietly not the one that was
        configured. The finite-difference check in the spec is what settles it.
        """
        raise NotImplementedError

    def stepped_by(self, gradient: LayerGradient | None, learning_rate: float) -> Layer:
        """This same layer, because there is nothing in it to move.

        Parameters
        ----------
        gradient:
            Ignored. :meth:`correction_for` always answers ``None`` here.
        learning_rate:
            Ignored, for the same reason.

        Returns
        -------
        Layer
            ``self``, and here that is load-bearing rather than merely
            economical. Rebuilding the layer would build a new generator, so a
            seeded network would draw the identical mask on every epoch and
            dropout would stop being dropout. A test pins it by training for
            several steps and asserting the masks differ.
        """
        return self

    def __repr__(self) -> str:
        return (
            f"Dropout(reads={self._shape.reads!r}, "
            f"drop_probability={self._drop_probability!r}, "
            f"random_seed={self._random_seed!r})"
        )
