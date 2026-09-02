"""Standardising inside one row, which is every normaliser except the batch one.

What the family actually is
----------------------------
Every normalisation layer answers one question, and the answer is the whole
taxonomy: *which axes do you average over?* Batch normalisation reduces down the
rows and gets one statistic per feature. Everything in this module reduces
**inside a single row** and gets one statistic per row.

That sounds like a small difference and it is not. Reaching across the batch is
where all of batch normalisation's difficulty comes from, and none of it is
about normalising:

* It needs running statistics, because a prediction must not depend on which
  other rows happened to travel with it.
* It therefore needs :class:`~oop_ml.core.network.purpose.PassPurpose`, because
  training and predicting genuinely differ.
* It cannot work at a batch of one, where the variance is zero.
* It cannot work on ragged input, where "the batch" is not a rectangle.

Nothing here has any of that. A row is normalised by its own contents, so the
answer for a row is the same whether it arrives alone or among a thousand
others. These layers ignore ``purpose`` entirely, carry no state that is not
learned, need no response subclass, and are the same function at training time
and at prediction time. That is why transformers use layer normalisation rather
than batch normalisation, and it is a fact about the reduction axis rather than
a preference.

Why one base and a boolean
---------------------------
Layer normalisation and RMS normalisation look like two methods and are one,
with a single term switched off. Both compute::

    normalised = (value - centre) / sqrt(spread + epsilon)
    answer     = scale * normalised + shift

and differ only in whether ``centre`` is the row's mean or is zero. The backward
passes differ by exactly the same term. Each input reaches the loss by three
routes when the layer centres, directly and through the mean and through the
spread, and by two when it does not::

    scaled      = arriving * scale
    passed_down = (scaled - centre_route
                   - normalised * mean(scaled * normalised)) / deviation

where ``centre_route`` is ``mean(scaled)`` for a centring layer and zero
otherwise. That was checked against a central finite difference before this base
was written rather than after: layer normalisation agrees to 5.2e-10 and RMS
normalisation to 4.5e-10, on the same fixture.

Writing it once is worth more than the duplication it saves. The naive backward
pass, treating the centre and the deviation as constants and answering
``scaled / deviation``, has the right shape and plausible magnitudes and trains
perfectly well on a gradient that is not the gradient of the loss. There is one
place here for that to be wrong, and one finite-difference check covering every
subclass.

Why the affine is not decoration
---------------------------------
Standardising alone would be a *constraint*, forbidding the layer from ever
answering with anything off-centre, and a network sometimes needs exactly that.
With ``scale`` and ``shift`` restored the layer can represent the identity, so
normalisation costs it no expressiveness. What changes is not what it can
represent but the shape of the surface leading there.

RMS normalisation conventionally drops ``shift`` as well as the centring, and
this one keeps it, defaulting to zeros. Nothing in the mathematics requires
dropping it, a zero shift is exactly the conventional model, and having it means
one class shape serves the whole family.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.blocks import as_per_feature
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


class WithinRowNormalization(Layer):
    """Standardise each row by its own contents, then scale and shift it.

    Parameters
    ----------
    n_features:
        How many values one row holds. The layer answers with the same
        arrangement it reads, so it moves no join in a stack.
    epsilon:
        Added to the spread *inside* the square root, so that a row carrying no
        variation normalises to zeros rather than to ``nan``. Strictly positive.
        Inside rather than outside because only the first is bounded below near
        zero: at a spread of 1e-12 the two readings differ by 288x, and the
        second promotes floating-point noise into a real signal.
    scale:
        ``(n_features,)``, the learned multiplier. Defaults to ones.
    shift:
        ``(n_features,)``, the learned offset. Defaults to zeros, by the same
        argument the convolution's biases use.

    Raises
    ------
    InvalidValuesError
        If ``epsilon`` is not strictly positive, or a supplied vector carries a
        non-finite entry.
    ShapeMismatchError
        If a supplied vector is not one value per feature.

    Notes
    -----
    A subclass supplies :meth:`centres` and nothing else. Everything below --
    the forward pass, the backward pass, the affine, the step -- is shared,
    because the family really does differ by one term.
    """

    __slots__ = ("_epsilon", "_scale", "_shape", "_shift")

    def __init__(
        self,
        n_features: int,
        epsilon: float = 1e-5,
        scale: FloatArray | None = None,
        shift: FloatArray | None = None,
    ) -> None:
        self._shape = LayerShape(n_inputs=n_features, n_outputs=n_features)
        width = self._shape.n_inputs

        try:
            floor = float(epsilon)
        except (TypeError, ValueError):
            raise InvalidValuesError("epsilon must be a real number") from None
        if not np.isfinite(floor) or floor <= 0.0:
            raise InvalidValuesError(
                "epsilon must be strictly positive, since it is what keeps a row "
                f"with no variation from dividing by zero, got {floor}"
            )

        self._epsilon = floor
        self._scale = as_per_feature(
            np.ones(width) if scale is None else scale, width, "a scale"
        )
        self._shift = as_per_feature(
            np.zeros(width) if shift is None else shift, width, "a shift"
        )

    @property
    @abstractmethod
    def centres(self) -> bool:
        """Whether this layer subtracts the row's mean before dividing.

        The single thing the family differs by. ``True`` gives layer
        normalisation and ``False`` gives RMS normalisation, forward and
        backward alike.
        """

    @property
    def shape(self) -> LayerShape:
        """The same width on both sides. This layer moves no join."""
        return self._shape

    @property
    def n_features(self) -> int:
        """How many values one row holds."""
        return self._shape.n_inputs

    @property
    def epsilon(self) -> float:
        """What is added to the spread inside the square root."""
        return self._epsilon

    @property
    def scale(self) -> FloatArray:
        """``(n_features,)``, the learned multiplier. Frozen."""
        return self._scale

    @property
    def shift(self) -> FloatArray:
        """``(n_features,)``, the learned offset. Frozen."""
        return self._shift

    def _standardised(self, inputs: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The normalised block and the deviation it divided by.

        Reduced over ``axis=1`` with ``keepdims``, so both come back shaped to
        broadcast against ``inputs`` without any further work. That is why this
        reduction is easier than batch normalisation's: the statistic keeps the
        same leading axis as the data it describes.

        Returns
        -------
        tuple
            The normalised block ``(n_rows, n_features)`` and the deviation
            ``(n_rows, 1)``. Two halves of one calculation rather than a pairing
            worth naming, which is why this is private and has one caller.
        """
        centre = inputs.mean(axis=1, keepdims=True) if self.centres else 0.0
        spread = ((inputs - centre) ** 2).mean(axis=1, keepdims=True)
        deviation = np.sqrt(spread + self._epsilon)

        return (inputs - centre) / deviation, deviation

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """Standardise the row, then scale and shift it.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_features)``, already validated by :meth:`respond_to`.
        purpose:
            **Ignored, and that is the point of this family.** A row is
            normalised by its own contents, so the answer does not depend on why
            the pass is happening, on which other rows are present, or on
            anything learned from an earlier batch. Batch normalisation cannot
            say that, and everything it needs in order to cope is what this
            module does without.

        Returns
        -------
        LayerResponse
            ``scores`` is the normalised block, before the scale and shift,
            because here the affine plays the part an activation plays
            elsewhere. The backward pass reads it twice.
        """
        normalised, _ = self._standardised(inputs)
        outputs = self._scale * normalised + self._shift

        return LayerResponse.already_checked(
            inputs=inputs, scores=normalised, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """The routes each input has to the loss, collected.

        Parameters
        ----------
        response:
            What this layer did on the way up. Its ``scores`` are the normalised
            block.
        arriving:
            ``(n_rows, n_features)``, the slope of the loss at this layer's
            outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is ``(n_rows, n_features)``, and ``gradient`` is a
            plain :class:`~oop_ml.core.network.gradient.LayerGradient` whose
            weight block is ``(n_features, 1)`` -- one row per feature holding
            that feature's single multiplier.

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
        The deviation is recomputed from ``response.inputs`` rather than carried,
        because unlike a dropout mask it is a function of numbers the response
        already holds and cannot disagree with itself. Same argument
        :class:`~oop_ml.core.network.pooling.Pool2d` makes for recomputing its
        winner, and the opposite of the one
        :class:`~oop_ml.core.network.dropout.Dropout` makes for storing its draw.

        Unlike batch normalisation, no provenance check is needed. There is no
        prediction-time variant of this pass to be confused with, because there
        is only one variant.
        """
        arriving = self._checked_arriving(response, arriving)

        normalised = response.scores
        _, deviation = self._standardised(response.inputs)

        # The easy half. One scale and one shift serve every row, so their
        # slopes are sums down the rows.
        scale_slope = (arriving * normalised).sum(axis=0)
        shift_slope = arriving.sum(axis=0)

        # The half worth reading twice. The centre and the deviation are
        # functions of every value in the row, so each input reaches the loss by
        # three routes when the layer centres and two when it does not. Dropping
        # them leaves ``scaled / deviation``, which has the right shape, trains,
        # and is not the gradient of the loss.
        scaled = arriving * self._scale
        centre_route = scaled.mean(axis=1, keepdims=True) if self.centres else 0.0
        passed_down = (
            scaled
            - centre_route
            - normalised * (scaled * normalised).mean(axis=1, keepdims=True)
        ) / deviation

        return LayerCorrection(
            passed_down=passed_down,
            gradient=LayerGradient(weights=scale_slope[:, None], biases=shift_slope),
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> WithinRowNormalization:
        """A new layer of the same kind with its scale and shift moved.

        Simpler than batch normalisation's step, and the difference is this
        module's whole argument. There are no running statistics to fold in,
        because nothing here is remembered between batches, so this is an
        ordinary subtract-the-slope like every other layer that learns.

        Raises
        ------
        ShapeMismatchError
            If ``gradient`` is ``None`` or does not describe this layer's width.
        """
        if gradient is None:
            raise ShapeMismatchError(
                f"a {type(self).__name__} has parameters and needs a gradient "
                "to step by"
            )
        if gradient.weights.shape != (self.n_features, 1):
            raise ShapeMismatchError(
                f"this layer has {self.n_features} features, so its weight "
                f"gradient is ({self.n_features}, 1), got {gradient.weights.shape}"
            )

        rate = float(learning_rate)
        return type(self)(
            n_features=self.n_features,
            epsilon=self._epsilon,
            scale=self._scale - rate * gradient.weights[:, 0],
            shift=self._shift - rate * gradient.biases,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_features={self.n_features!r}, "
            f"epsilon={self._epsilon!r})"
        )


class LayerNormalization(WithinRowNormalization):
    """Standardise each row to zero mean and unit variance across its features.

    The normaliser transformers use, and the reason is not subtle. A sequence
    model reads rows of wildly differing length and is routinely run on a single
    example, and batch normalisation can do neither. This one treats every row
    on its own, so a batch of one behaves identically to a batch of a thousand.

    Ba, Kiros and Hinton, 2016. It arrived as the answer to batch
    normalisation's trouble with recurrent networks, where the natural batch
    statistic differs at every timestep and there is no sensible running figure
    to keep.

    See :class:`WithinRowNormalization` for the parameters and the arithmetic.
    """

    __slots__ = ()

    @property
    def centres(self) -> bool:
        """True. The row's mean is subtracted before dividing."""
        return True


class RMSNormalization(WithinRowNormalization):
    """Divide each row by its root mean square, without centring it first.

    Layer normalisation with the mean subtraction removed, and the empirical
    finding that motivated it is exactly that: the re-centring was not doing the
    work, the re-scaling was. Zhang and Sennrich, 2019.

    It is cheaper on both passes. The forward pass runs one reduction instead of
    two, and the backward pass has two routes to the loss rather than three,
    because there is no mean for an input to reach the loss through. That is why
    most current large language models use it in preference.

    Conventionally the shift is dropped as well. It is kept here and defaults to
    zeros, which is the conventional model exactly, and which lets one class
    shape serve the whole family.

    See :class:`WithinRowNormalization` for the parameters and the arithmetic.
    """

    __slots__ = ()

    @property
    def centres(self) -> bool:
        """False. The row is divided by its root mean square as it stands."""
        return False
