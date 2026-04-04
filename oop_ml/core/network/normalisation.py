"""Re-centring each feature mid-network, and the state that is not learned.

The problem it addresses
------------------------
A deep stack has a moving-target problem. Layer 5's weights are being tuned
against the distribution layer 4 hands it, and layer 4's weights are changing at
the same time, so the distribution layer 5 is learning about shifts underneath
it on every step. The original paper called that *internal covariate shift*.
Later work argued the diagnosis was at best incomplete and that the real benefit
is a smoother loss surface, which permits a larger learning rate; the
disagreement is about why it works, not whether, and the mechanism below is the
same under either reading.

Either way the fix is the same and it is simple to state. Standardise each
feature across the batch, then give the layer back the freedom to undo that if
it turns out to want to::

    normalised = (value - mean) / sqrt(variance + epsilon)
    answer     = scale * normalised + shift

``scale`` and ``shift`` are learned, one of each per feature. That pairing is
the part worth pausing on. Standardising alone would be a *constraint*: it would
forbid the layer from ever answering with anything off-centre, and a network
sometimes needs exactly that. With the affine restored, the layer can represent
the identity -- ``scale = sqrt(variance + epsilon)``, ``shift = mean`` -- so
normalisation costs it no expressiveness. What changes is not what it can
represent but how the surface leading there is shaped.

Why epsilon is not a fudge
---------------------------
A feature that is constant across the batch has zero variance, and dividing by
its standard deviation is a division by zero. That is not a rare accident: a
rectified unit that is off for every row in the batch produces exactly that, and
so does a genuinely constant input feature. Adding a small epsilon inside the
square root makes the answer zero rather than ``nan``, which is the right answer
-- a feature carrying no information contributes nothing -- and it is why
``epsilon`` sits inside the root rather than being added to the deviation
afterwards. The two differ, and only the first is bounded near zero.

The awkward part, and where this library puts it
--------------------------------------------------
Batch normalisation has state that is neither a hyperparameter nor learned by
gradient: the running mean and running variance it uses when predicting.

They are needed because the training-time definition is unusable for
prediction. Standardising by the batch would make one row's answer depend on
which other rows happened to travel with it, so the same input would score
differently in a batch of 32 and a batch of 64, and a single row on its own has
zero variance and no spread to divide by. So training keeps a running average of
what it saw, and prediction uses that::

    running = momentum * running + (1 - momentum) * this batch's statistic

Every framework updates those inside the forward pass, by mutation. This library
does not mutate, and the awkwardness is real rather than something to be
smoothed over, so it is worth saying exactly where it went.

It went into :meth:`BatchNormalization.stepped_by`, which is already the moment
the layer is rebuilt. :class:`BatchStatistics` is a
:class:`~oop_ml.core.network.gradient.LayerGradient` that also carries the mean
and variance the backward pass just measured, so the step has everything it
needs to build the next layer's running figures alongside its next scale and
shift. Nothing is mutated, nothing rides in a side channel, and the update
happens once per step, which in any ordinary training loop is once per batch --
exactly where a mutating implementation would have put it.

The cost of that choice is stated rather than hidden: a forward pass that is
never followed by a step contributes nothing to the running statistics. For a
training loop that is the correct behaviour, since a batch that was never
learned from should not shape what the model believes about its inputs. For a
caller running training passes deliberately without stepping, it is a
difference from other frameworks and this is where it is written down.

Why the backward pass is not the obvious one
----------------------------------------------
The tempting reading is that ``normalised`` is just ``(value - mean) / deviation``
with the mean and deviation as constants, giving
``d normalised / d value = 1 / deviation``. That is wrong, and it is wrong in a
way that trains.

The mean and the deviation are functions of *every row in the batch*. Change one
input and you change that feature's mean, which changes every other row's
normalised value too. So each input has three routes to the loss: directly
through its own normalised value, through the mean, and through the variance.
Collecting all three gives the standard compact form, with ``n`` rows,
``xhat`` the normalised block and ``d`` the arriving slope after the scale::

    d = arriving * scale
    dx = (n * d - d.sum(axis=0) - xhat * (d * xhat).sum(axis=0)) / (n * deviation)

The two subtracted terms are exactly the mean and variance routes. Dropping them
leaves ``d / deviation``, which has the right shape, plausible magnitudes, and a
network that still converges -- on a gradient that is not the gradient of the
loss. A finite-difference check is what separates the two, and the spec has one
for precisely this reason.

Note that the variance used is the biased one, dividing by ``n`` rather than
``n - 1``. That is not a slip. The derivative above is taken with respect to the
quantity actually computed, so the forward pass and the backward pass have to
agree about which variance that was, and every implementation of this layer uses
the biased form forward. Some use the unbiased form for the *running* variance,
which is a separate figure used only at prediction time and is not what the
gradient flows through.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.blocks import as_per_feature
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


class BatchStatistics(LayerGradient):
    """A gradient that also carries what the batch looked like.

    A :class:`~oop_ml.core.network.gradient.LayerGradient` says what a layer's
    parameters want to change by, and for this layer that is the scale and the
    shift. What it does not say, and what
    :meth:`BatchNormalization.stepped_by` also needs, is the mean and variance
    the batch actually had, since those are what the running figures average.

    Widening ``LayerGradient`` for one layer's benefit would be the wrong trade,
    and passing the statistics through a side channel would put them outside the
    object the stack already threads. So this subclasses it, which is the same
    move :class:`~oop_ml.core.network.dropout.DropoutResponse` makes on the
    response: the general type stays general, and the layer that needs more
    hands back more.

    Parameters
    ----------
    weights:
        ``(n_features, 1)``, the slope for the scale. One row per feature with a
        single weight in it, which is ``LayerGradient``'s ordinary
        ``(n_neurons, n_inputs)`` and not a special case: this layer's answer
        for feature ``j`` is ``scale_j * normalised_j + shift_j``, so each
        feature really is a unit reading one number.

        The transposed reading, ``(1, n_features)``, is the tempting one and it
        is wrong. ``LayerGradient`` checks that the weight block's rows and the
        bias block's entries agree about how many units there are, so a layer
        of more than one feature is refused outright. It raised on the first
        object built, which is the check working.
    biases:
        ``(n_features,)``, the slope for the shift.
    batch_mean:
        ``(n_features,)``, the mean the forward pass standardised by.
    batch_variance:
        ``(n_features,)``, the biased variance it standardised by.

    Raises
    ------
    ShapeMismatchError
        If the blocks disagree, by way of ``LayerGradient``, or if either
        statistic is not one value per feature.
    InvalidValuesError
        If either statistic carries a non-finite entry.
    """

    __slots__ = ("_batch_mean", "_batch_variance")

    def __init__(
        self,
        weights: FloatArray,
        biases: FloatArray,
        batch_mean: FloatArray,
        batch_variance: FloatArray,
    ) -> None:
        super().__init__(weights=weights, biases=biases)
        n_features = self.biases.shape[0]
        self._batch_mean = as_per_feature(batch_mean, n_features, "a batch mean")
        self._batch_variance = as_per_feature(
            batch_variance, n_features, "a batch variance"
        )
        if (self._batch_variance < 0.0).any():
            raise InvalidValuesError("a batch variance cannot be negative")

    @property
    def batch_mean(self) -> FloatArray:
        """``(n_features,)``, the mean the forward pass standardised by."""
        return self._batch_mean

    @property
    def batch_variance(self) -> FloatArray:
        """``(n_features,)``, the biased variance it standardised by."""
        return self._batch_variance

    def __repr__(self) -> str:
        return f"BatchStatistics(n_features={self._batch_mean.shape[0]!r})"


class NormalisationResponse(LayerResponse):
    """A layer response that also remembers the batch it standardised by.

    The backward pass needs the normalised block and the deviation it divided
    by. The normalised block is the response's ``scores``, since for this layer
    the affine plays the part an activation plays elsewhere, so what is left to
    carry is the deviation and the statistics the step will average.

    Recomputing them from ``inputs`` would work while training and would be
    silently wrong after a prediction pass, which standardises by the running
    figures rather than the batch. Carrying them removes the question: a
    response that has them came from a training pass, and one that does not is
    refused.
    """

    __slots__ = ("_batch_mean", "_batch_variance", "_deviation")

    @classmethod
    def recording(
        cls,
        inputs: FloatArray,
        normalised: FloatArray,
        outputs: FloatArray,
        batch_mean: FloatArray,
        batch_variance: FloatArray,
        deviation: FloatArray,
    ) -> NormalisationResponse:
        """Wrap a training forward pass this layer just built, with its statistics.

        The :meth:`~oop_ml.core.network.layer.LayerResponse.already_checked`
        pattern; see there for why skipping the copy is sound.

        Parameters
        ----------
        inputs:
            The block the layer read.
        normalised:
            The standardised block, before the scale and shift. Kept as the
            response's ``scores``.
        outputs:
            ``scale * normalised + shift``.
        batch_mean:
            ``(n_features,)``, what it centred by.
        batch_variance:
            ``(n_features,)``, the biased variance it used.
        deviation:
            ``(n_features,)``, ``sqrt(batch_variance + epsilon)``. Carried
            rather than recomputed so that the forward and backward passes
            cannot disagree about epsilon.
        """
        response = cls.__new__(cls)
        for block in (
            inputs,
            normalised,
            outputs,
            batch_mean,
            batch_variance,
            deviation,
        ):
            block.setflags(write=False)
        response._inputs = inputs
        response._scores = normalised
        response._outputs = outputs
        response._batch_mean = batch_mean
        response._batch_variance = batch_variance
        response._deviation = deviation
        return response

    @property
    def normalised(self) -> FloatArray:
        """``(n_rows, n_features)``, standardised, before the scale and shift.

        The same block as ``scores``, named for what it is here. The backward
        pass reads it twice and the reading is clearer for saying so.
        """
        return self._scores

    @property
    def batch_mean(self) -> FloatArray:
        """``(n_features,)``, what this pass centred by."""
        return self._batch_mean

    @property
    def batch_variance(self) -> FloatArray:
        """``(n_features,)``, the biased variance this pass used."""
        return self._batch_variance

    @property
    def deviation(self) -> FloatArray:
        """``(n_features,)``, ``sqrt(batch_variance + epsilon)``."""
        return self._deviation

    def __eq__(self, other: object) -> bool:
        # The statistics are the whole reason this class exists, so two
        # responses standardised by different batches are not the same
        # response. The base settles the type check and the blocks; this adds
        # the fields it does not know about. The deviation is a function of the
        # variance and epsilon, so comparing the first two settles it.
        if not isinstance(other, NormalisationResponse):
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
        return (
            bool(alike)
            and bool(np.array_equal(self._batch_mean, other._batch_mean))
            and bool(np.array_equal(self._batch_variance, other._batch_variance))
        )

    def __repr__(self) -> str:
        return (
            f"NormalisationResponse(n_rows={self.n_rows!r}, "
            f"n_features={self._batch_mean.shape[0]!r})"
        )


class BatchNormalization(Layer):
    """Standardises each feature across the batch, then scales and shifts it.

    Parameters
    ----------
    n_features:
        How many features one row holds. This layer answers with the same
        arrangement it reads, so it moves no join in a stack.
    momentum:
        How much of the running statistics to keep on each step, in ``[0, 1)``.
        The default of 0.9 keeps 90% of the running figure and takes 10% from
        the batch, which is a slow average over roughly the last ten batches.
        Zero replaces the running figures with the latest batch outright, which
        is legal and makes prediction depend on whichever batch came last.
    epsilon:
        Added to the variance inside the square root, to keep a constant
        feature from dividing by zero. Strictly positive.
    scale:
        ``(n_features,)``, the learned multiplier, one per feature. Defaults to
        all ones, which starts the layer standardising and nothing more.
    shift:
        ``(n_features,)``, the learned offset. Defaults to all zeros, for the
        same reason and by the same argument the convolution's biases use: an
        offset the layer has no evidence for yet is one the gradient will
        supply the moment there is any.
    running_mean:
        ``(n_features,)``, what prediction centres by. Defaults to zeros, which
        is what an untrained layer believes and is why predicting through one
        that has never been trained is close to an identity rather than an
        error.
    running_variance:
        ``(n_features,)``, what prediction scales by. Defaults to ones, for the
        same reason. Must not be negative.

    Raises
    ------
    InvalidValuesError
        If ``n_features`` is below one, if ``momentum`` is not in ``[0, 1)``,
        if ``epsilon`` is not strictly positive, if any supplied vector carries
        a non-finite entry, or if ``running_variance`` carries a negative one.
    ShapeMismatchError
        If any supplied vector is not one value per feature.

    Notes
    -----
    The gradient's ``weights`` block is ``(n_features, 1)``, one row per feature
    holding that feature's single scale. That is
    :class:`~oop_ml.core.network.gradient.LayerGradient`'s ordinary
    ``(n_neurons, n_inputs)`` rather than a shape being bent to fit, because
    this layer's answer for feature ``j`` is ``scale_j * normalised_j +
    shift_j``, which really is a unit reading one number.

    See the module docstring for why the running statistics update in
    :meth:`stepped_by` rather than in the forward pass, and for what that costs.
    """

    __slots__ = (
        "_epsilon",
        "_momentum",
        "_running_mean",
        "_running_variance",
        "_scale",
        "_shape",
        "_shift",
    )

    def __init__(
        self,
        n_features: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        scale: FloatArray | None = None,
        shift: FloatArray | None = None,
        running_mean: FloatArray | None = None,
        running_variance: FloatArray | None = None,
    ) -> None:
        # LayerShape validates the extent and refuses zero, and reads == answers
        # because this layer never changes the arrangement.
        self._shape = LayerShape(n_inputs=n_features, n_outputs=n_features)
        width = self._shape.n_inputs

        try:
            keep = float(momentum)
            floor = float(epsilon)
        except (TypeError, ValueError):
            raise InvalidValuesError(
                "momentum and epsilon must be real numbers"
            ) from None
        if not np.isfinite(keep) or not 0.0 <= keep < 1.0:
            raise InvalidValuesError(
                f"momentum lies in [0, 1); 1 would never learn, got {keep}"
            )
        if not np.isfinite(floor) or floor <= 0.0:
            raise InvalidValuesError(
                "epsilon must be strictly positive, since it is what keeps a "
                f"constant feature from dividing by zero, got {floor}"
            )

        self._momentum = keep
        self._epsilon = floor
        self._scale = as_per_feature(
            np.ones(width) if scale is None else scale, width, "a scale"
        )
        self._shift = as_per_feature(
            np.zeros(width) if shift is None else shift, width, "a shift"
        )
        self._running_mean = as_per_feature(
            np.zeros(width) if running_mean is None else running_mean,
            width,
            "a running mean",
        )
        self._running_variance = as_per_feature(
            np.ones(width) if running_variance is None else running_variance,
            width,
            "a running variance",
        )
        if (self._running_variance < 0.0).any():
            raise InvalidValuesError("a running variance cannot be negative")

    @property
    def shape(self) -> LayerShape:
        """The same width on both sides. This layer moves no join."""
        return self._shape

    @property
    def n_features(self) -> int:
        """How many features one row holds."""
        return self._shape.n_inputs

    @property
    def momentum(self) -> float:
        """How much of the running statistics each step keeps."""
        return self._momentum

    @property
    def epsilon(self) -> float:
        """What is added to the variance inside the square root."""
        return self._epsilon

    @property
    def scale(self) -> FloatArray:
        """``(n_features,)``, the learned multiplier. Frozen."""
        return self._scale

    @property
    def shift(self) -> FloatArray:
        """``(n_features,)``, the learned offset. Frozen."""
        return self._shift

    @property
    def running_mean(self) -> FloatArray:
        """``(n_features,)``, what prediction centres by. Frozen."""
        return self._running_mean

    @property
    def running_variance(self) -> FloatArray:
        """``(n_features,)``, what prediction scales by. Frozen."""
        return self._running_variance

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """Standardise, then scale and shift.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_features)``, already known by :meth:`respond_to` to be
            the right arrangement, non-empty and finite.
        purpose:
            Which statistics to standardise by. Under
            :attr:`~oop_ml.core.network.purpose.PassPurpose.TRAINING` the batch's
            own mean and biased variance; under
            :attr:`~oop_ml.core.network.purpose.PassPurpose.PREDICTING` the
            running figures, so that one row's answer never depends on which
            other rows it travelled with.

        Returns
        -------
        LayerResponse
            While predicting, an ordinary
            :class:`~oop_ml.core.network.layer.LayerResponse`. While training, a
            :class:`NormalisationResponse` carrying the statistics, because
            neither the backward pass nor the step can proceed without them.

        Notes
        -----
        The variance is the biased one, ``((values - mean) ** 2).mean(axis=0)``,
        which is what :func:`numpy.var` gives by default. See the module
        docstring for why the backward pass depends on that choice.

        The deviation is ``sqrt(variance + epsilon)``, with epsilon inside the
        root rather than added afterwards. A constant feature then normalises to
        zero rather than to ``nan``.
        """
        if purpose == PassPurpose.PREDICTING:
            deviation = np.sqrt(self.running_variance + self.epsilon)
            normalised = (inputs - self.running_mean) / deviation
            outputs = self.scale * normalised + self.shift

            # A plain response carrying no statistics, which is exactly what
            # ``_checked_statistics`` needs in order to refuse to differentiate a
            # pass that standardised by the running figures.
            return LayerResponse.already_checked(
                inputs=inputs,
                scores=normalised,
                outputs=outputs,
            )

        batch_mean = inputs.mean(axis=0)
        batch_variance = inputs.var(axis=0)
        deviation = np.sqrt(batch_variance + self.epsilon)
        normalised = (inputs - batch_mean) / deviation
        outputs = self.scale * normalised + self.shift

        return NormalisationResponse.recording(
            inputs=inputs,
            normalised=normalised,
            outputs=outputs,
            batch_mean=batch_mean,
            batch_variance=batch_variance,
            deviation=deviation,
        )

    def _checked_statistics(self, response: LayerResponse) -> NormalisationResponse:
        """The response, refusing one that carries no batch statistics.

        The companion to
        :meth:`~oop_ml.core.network.layer.Layer._checked_arriving`, and the same
        kind of check: not about the data, about whether these two objects
        belong together. A prediction-pass response has the right arrangement
        and the right row count and standardised by the running figures, so the
        gradient taken from it would be the gradient of a different function.

        Raises
        ------
        ShapeMismatchError
            If the response is not a :class:`NormalisationResponse`.
        """
        if not isinstance(response, NormalisationResponse):
            raise ShapeMismatchError(
                "a batch normalisation layer's backward step needs the "
                "statistics its own training pass measured, and this response "
                "carries none -- it came from a prediction pass or from "
                "another layer"
            )
        return response

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """The three routes each input has to the loss, collected.

        Parameters
        ----------
        response:
            What this layer did on the way up. Must be a
            :class:`NormalisationResponse`, since a prediction pass standardised
            by different numbers.
        arriving:
            ``(n_rows, n_features)``, the slope of the loss at this layer's
            outputs.

        Returns
        -------
        LayerCorrection
            ``passed_down`` is ``(n_rows, n_features)``, and ``gradient`` is a
            :class:`BatchStatistics` carrying the slopes for the scale and the
            shift along with the batch's own mean and variance, which
            :meth:`stepped_by` needs for the running figures.

        Raises
        ------
        InvalidValuesError
            If ``arriving`` cannot be read as a float array.
        ShapeMismatchError
            If the response was not produced by a layer of this shape, if
            ``arriving`` does not describe this layer's outputs for the rows the
            response holds, or if the response carries no statistics.

        Notes
        -----
        The two parameter gradients are the easy half, and they are sums down
        the rows because one scale serves every row::

            d scale = (arriving * normalised).sum(axis=0)
            d shift = arriving.sum(axis=0)

        The block passed down is the half worth reading twice, because the mean
        and the variance are themselves functions of every row. With ``n`` rows
        and ``d = arriving * scale``::

            passed_down = (n * d - d.sum(axis=0)
                           - normalised * (d * normalised).sum(axis=0))
                          / (n * deviation)

        Dropping the two subtracted terms leaves ``d / deviation``, which is the
        answer if the mean and deviation were constants. They are not, it has
        the right shape, and it trains. The finite-difference check in the spec
        is what tells the two apart.

        The gradient block returned is ``(n_features, 1)``, one row per feature
        holding that feature's single weight, which is what
        :class:`~oop_ml.core.network.gradient.LayerGradient` means by
        ``(n_neurons, n_inputs)``. ``d scale`` above is ``(n_features,)`` and
        needs a trailing axis added to sit in it.
        """
        recorded = self._checked_statistics(response)
        arriving = self._checked_arriving(response, arriving)

        normalised = recorded.normalised
        n_rows = float(recorded.n_rows)

        # The easy half. One scale and one shift serve every row, so their
        # slopes are sums down the rows.
        scale_slope = (arriving * normalised).sum(axis=0)
        shift_slope = arriving.sum(axis=0)

        # The half worth reading twice. The mean and the deviation are functions
        # of every row, so each input reaches the loss by three routes rather
        # than one. The two subtracted terms are the mean route and the variance
        # route, and dropping them leaves ``scaled / deviation``, which trains
        # and is not the gradient of the loss.
        scaled = arriving * self.scale
        passed_down = (
            n_rows * scaled
            - scaled.sum(axis=0)
            - normalised * (scaled * normalised).sum(axis=0)
        ) / (n_rows * recorded.deviation)

        return LayerCorrection(
            passed_down=passed_down,
            gradient=BatchStatistics(
                weights=scale_slope[:, None],
                biases=shift_slope,
                batch_mean=recorded.batch_mean,
                batch_variance=recorded.batch_variance,
            ),
        )

    def stepped_by(self, gradient: LayerGradient | None, learning_rate: float) -> Layer:
        """A new layer with moved parameters and updated running statistics.

        The one place in this package where a step does something besides
        subtract a slope, and the module docstring says why it is here.

        Parameters
        ----------
        gradient:
            Must be a :class:`BatchStatistics`. A plain
            :class:`~oop_ml.core.network.gradient.LayerGradient` carries the
            parameter slopes but not the batch's mean and variance, so the
            running figures could not be updated and prediction would keep
            using whatever it was last told.
        learning_rate:
            How far to move the scale and the shift.

        Returns
        -------
        Layer
            A new layer, same width, same momentum and epsilon.

        Raises
        ------
        ShapeMismatchError
            If ``gradient`` is ``None``, is not a :class:`BatchStatistics`, or
            does not describe this layer's width.

        Notes
        -----
        Downhill is against the slope, so the parameters subtract. The running
        figures are an exponential moving average and move *toward* the batch::

            running = momentum * running + (1 - momentum) * batch

        Those are two different kinds of update sharing one method, which is
        exactly the awkwardness this layer has and the reason it is written down
        here rather than smoothed over.
        """
        if gradient is None:
            raise ShapeMismatchError(
                "a batch normalisation layer has parameters and needs a "
                "gradient to step by"
            )
        if not isinstance(gradient, BatchStatistics):
            raise ShapeMismatchError(
                "a batch normalisation layer steps by a BatchStatistics, which "
                "carries the batch's mean and variance as well as the parameter "
                "slopes; a plain LayerGradient cannot update the running figures"
            )
        if gradient.batch_mean.shape != (self.n_features,):
            raise ShapeMismatchError(
                f"this layer has {self.n_features} features and the gradient "
                f"describes {gradient.batch_mean.shape[0]}"
            )

        rate = float(learning_rate)
        keep = self._momentum
        return BatchNormalization(
            n_features=self.n_features,
            momentum=keep,
            epsilon=self._epsilon,
            scale=self._scale - rate * gradient.weights[:, 0],
            shift=self._shift - rate * gradient.biases,
            running_mean=keep * self._running_mean + (1.0 - keep) * gradient.batch_mean,
            running_variance=(
                keep * self._running_variance + (1.0 - keep) * gradient.batch_variance
            ),
        )

    def __repr__(self) -> str:
        return (
            f"BatchNormalization(n_features={self.n_features!r}, "
            f"momentum={self._momentum!r}, epsilon={self._epsilon!r})"
        )
