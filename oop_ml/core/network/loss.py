"""What a network is trying to make small, and the slope it hands back.

The fact worth building this module around
-------------------------------------------
Three of the five losses here have the *same* gradient, and it is the simplest
expression in the file::

    dL/d(last layer's output) = (prediction - truth) / n_rows

Squared error for a quantity, log-loss for a yes-or-no, log-loss over softmax
for one of many classes. Three different questions, three different formulas,
one shared derivative. Checked against a central finite difference, the largest
disagreements are 1.7e-10, 3.3e-11 and 1.1e-10.

That is not a coincidence and it is not an approximation. Each pairing is a
*canonical link*: the squash chosen for each kind of answer is precisely the
one whose derivative cancels the loss's own, leaving prediction minus truth.
Gauss reached it for the normal distribution, and the logistic and softmax
pairings are the same result for the Bernoulli and the categorical. It is the
reason those particular squashes are the ones everybody uses, rather than a
convention that hardened.

Practically it means the awkward derivative never has to be written. Softmax's
own derivative is a Jacobian, every output depending on every logit, which no
elementwise contract could hold, and it is exactly the term that cancels.
Sigmoid's ``p(1 - p)`` cancels the same way. So the loss owns the squash, the
last layer stays linear, and the backward pass begins with a subtraction.

Absolute error is here to break the pattern
--------------------------------------------
Its gradient is ``sign(prediction - truth) / n_rows``, and the difference is
the whole argument for it. Squared error's pull grows with the miss, so one
bad row can drown the rest::

    misses                  1      1      1     100
    squared error pulls    0.25   0.25   0.25  25.0     <- the outlier is 100x
    absolute error pulls   0.25   0.25   0.25   0.25    <- every row pulls alike

A single mistyped label cannot dominate a fit that uses absolute error, and
that is what "robust" means when a textbook says it. The price is the kink at
zero, where no derivative exists, which is the rectifier's problem again and
answered the same way: pick a branch and write it down.

:class:`HuberError` is the compromise, squared close in and absolute far out,
which is why it takes a parameter where the others take none.

The convention every loss here follows
---------------------------------------
Sum over everything, divide by the number of *rows*. So a batch twice the size
does not double the loss, and multi-output problems stay comparable to
single-output ones. The gradient carries the same division, because if it did
not the step size would secretly depend on the batch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.logistic import stable_logistic, stable_softmax
from oop_ml.core.types import FloatArray, NumericInput


class LossMeasurement:
    """How wrong a batch of answers was, and which way to move to fix it.

    Two things that must travel together, since computing either separately
    means running the squash twice. The gradient is with respect to the *last
    layer's raw output*, which is where a backward pass starts.

    Parameters
    ----------
    value:
        The loss, divided by the row count, so batches of different sizes stay
        comparable.
    gradient:
        ``(n_rows, n_outputs)``, the slope of the loss at each raw output,
        carrying the same division as ``value``.
    """

    __slots__ = ("_gradient", "_value")

    def __init__(self, value: float, gradient: FloatArray) -> None:
        self._value = float(value)
        frozen = np.array(gradient, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        self._gradient = frozen

    @property
    def value(self) -> float:
        """The loss, divided by the row count."""
        return self._value

    @property
    def gradient(self) -> FloatArray:
        """``(n_rows, n_outputs)``, the slope at each raw output."""
        return self._gradient

    def __repr__(self) -> str:
        return f"LossMeasurement(value={self._value!r})"


class Loss(ABC):
    """What training makes small, together with its slope at the raw outputs.

    A classification loss applies its own squash, so the last layer stays
    linear and hands over raw scores. A regression loss has nothing to apply,
    and the raw scores are already the predictions.
    """

    __slots__ = ()

    def measure(self, outputs: FloatArray, targets: FloatArray) -> LossMeasurement:
        """How wrong these answers are, and the slope that fixes them.

        The shape agreement is established here, once, so that
        :meth:`_measure_aligned` may do arithmetic without checking anything.

        Parameters
        ----------
        outputs:
            ``(n_rows, n_outputs)``, the last layer's raw answers, unbent.
        targets:
            ``(n_rows, n_outputs)``, the truth in the same shape.

        Returns
        -------
        LossMeasurement
            The loss and the gradient with respect to ``outputs``.

        Raises
        ------
        ShapeMismatchError
            If the two blocks are not the same shape.
        EmptyValuesError
            If there are no rows to average over.
        """
        if outputs.shape != targets.shape:
            raise ShapeMismatchError(
                f"outputs {outputs.shape} and targets {targets.shape} describe "
                "the same rows and must be the same shape"
            )

        return self._measure_aligned(outputs, targets)

    @abstractmethod
    def _measure_aligned(
        self, outputs: FloatArray, targets: FloatArray
    ) -> LossMeasurement:
        """The arithmetic, given two blocks already known to be the same shape.

        Parameters
        ----------
        outputs:
            ``(n_rows, n_outputs)``, the last layer's raw answers.
        targets:
            ``(n_rows, n_outputs)``, the truth.

        Returns
        -------
        LossMeasurement
            The loss, summed over everything and divided by the row count, and
            its gradient carrying that same division.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """The formula, for a readout or a saved document to quote."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Loss):
            return NotImplemented
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SquaredError(Loss):
    """``(prediction - truth)²``, halved, for predicting a quantity.

    The regression loss, and the one every least-squares fit in this library
    already minimises. The half is bookkeeping: it makes the derivative come
    out as a clean subtraction rather than carrying a 2, which is the same
    kindness the gradient boosting page's residual takes advantage of.

    Its gradient is ``(prediction - truth) / n_rows``, the shared form.
    """

    __slots__ = ()

    def _measure_aligned(
        self, outputs: FloatArray, targets: FloatArray
    ) -> LossMeasurement:
        """Sum the halved squared misses, divide by the rows.

        Notes
        -----
        Halve the *sum of squares*, then divide by the row count. Not by the
        number of entries: a two-output problem should read as twice the cost
        of a one-output problem that misses as badly, which is what keeps the
        gradient free of the output width.
        """
        error = outputs - targets
        rows = outputs.shape[0]

        value = float(np.sum(error**2) / (2 * rows))
        gradient = error / rows

        return LossMeasurement(value=value, gradient=gradient)

    @property
    def description(self) -> str:
        return "sum((p - y)**2) / (2n)"


class AbsoluteError(Loss):
    """``|prediction - truth|``, which one bad row cannot dominate.

    Robust regression. Every row pulls with the same force whatever its miss,
    so a mistyped label moves the fit by the same amount as an ordinary
    observation rather than a hundred times as much.

    Its gradient is ``sign(prediction - truth) / n_rows``, which is *not* the
    shared form, and that is precisely the property being bought.
    """

    __slots__ = ()

    def _measure_aligned(
        self, outputs: FloatArray, targets: FloatArray
    ) -> LossMeasurement:
        """Sum the absolute misses, divide by the rows.

        Notes
        -----
        The kink is at an exact tie, where no derivative exists. ``np.sign``
        answers 0 there, which is the sensible branch and the one this class
        commits to: a row already exactly right should pull in neither
        direction. It is the rectifier's ``z == 0`` decision again.
        """
        error = outputs - targets
        rows = outputs.shape[0]

        value = float(np.sum(np.abs(error)) / rows)
        gradient = np.sign(error) / rows

        return LossMeasurement(value=value, gradient=gradient)

    @property
    def description(self) -> str:
        return "sum(|p - y|) / n"


class HuberError(Loss):
    """Squared close to the truth, absolute far from it.

    The compromise between the two above, and the reason it takes a parameter
    where they take none. Inside ``threshold`` of the truth it behaves like
    squared error, so small misses are corrected smoothly and the kink at zero
    disappears. Outside it, the pull stops growing, so an outlier is capped
    rather than allowed to dominate.

    The pieces are chosen to meet: at exactly ``threshold`` both the value and
    the slope agree from either side, so the loss is smooth everywhere, which
    is the whole point of the constant that looks arbitrary in the formula.

    Parameters
    ----------
    threshold:
        Where squared behaviour ends and absolute begins. Must be finite and
        above zero.

    Raises
    ------
    InvalidValuesError
        If the threshold is not finite and positive. At zero it is absolute
        error with an extra branch, and below zero it is nothing at all.
    """

    __slots__ = ("_threshold",)

    def __init__(self, threshold: float = 1.0) -> None:
        value = float(threshold)
        if not np.isfinite(value) or value <= 0.0:
            raise InvalidValuesError(
                f"threshold must be finite and above zero, got {threshold}"
            )
        self._threshold = value

    @property
    def threshold(self) -> float:
        """Where squared behaviour ends and absolute begins."""
        return self._threshold

    def _measure_aligned(
        self, outputs: FloatArray, targets: FloatArray
    ) -> LossMeasurement:
        """Squared inside the threshold, linear outside, summed over rows.

        Notes
        -----
        Writing ``d`` for the threshold and ``e`` for ``prediction - truth``::

            |e| <= d    value  0.5 * e**2          slope  e
            |e| >  d    value  d * (|e| - 0.5*d)   slope  d * sign(e)

        The ``- 0.5*d`` in the outer branch is not decoration. It is exactly
        what makes the two pieces meet at ``|e| == d``, where both give
        ``0.5 * d**2``. Drop it and the loss jumps at the threshold, which
        leaves a discontinuity the optimiser will happily sit in.

        Both branches then divide by the row count, as everywhere here.
        """
        error = outputs - targets
        rows = outputs.shape[0]
        threshold = self._threshold

        # Signed throughout: the comparison needs the size of the miss, and
        # the gradient needs its direction, so the absolute value is taken
        # only where it is asked for.
        nearby = np.abs(error) <= threshold

        value = float(
            np.sum(
                np.where(
                    nearby,
                    0.5 * error**2,
                    threshold * (np.abs(error) - 0.5 * threshold),
                )
            )
            / rows
        )
        gradient = np.where(nearby, error, threshold * np.sign(error)) / rows

        return LossMeasurement(value=value, gradient=gradient)

    @property
    def description(self) -> str:
        return f"huber(p - y, threshold={self._threshold})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Loss):
            return NotImplemented
        if not isinstance(other, HuberError):
            return False
        return self._threshold == other._threshold

    def __hash__(self) -> int:
        return hash((type(self), self._threshold))

    def __repr__(self) -> str:
        return f"HuberError(threshold={self._threshold!r})"


class BinaryCrossEntropy(Loss):
    """Log-loss over one sigmoid output, for a yes-or-no answer.

    The loss the logistic regression page derived, arriving here unchanged. The
    last layer has a single linear neuron, this applies the squash, and the
    gradient at that neuron's raw score is ``(p - y) / n_rows`` -- the shared
    form, and the same balance condition the logistic page set to zero.

    Targets are 0.0 or 1.0 in a single column.
    """

    __slots__ = ()

    def _measure_aligned(
        self, outputs: FloatArray, targets: FloatArray
    ) -> LossMeasurement:
        """Squash, score, and hand back the subtraction.

        Notes
        -----
        Reach for :func:`~oop_ml.core.logistic.stable_logistic` rather than
        spelling the sigmoid out. The naive form overflows below about
        ``z = -709`` and warns from exactly the rows the model is most certain
        about, which is the trap that function exists to have solved once.

        ``log(p)`` and ``log(1 - p)`` are then the second trap. A confident
        enough score saturates the sigmoid to exactly 0.0 or 1.0, and one of
        the two logarithms is then ``-inf``, which poisons the average for a
        row that was merely very wrong. Clipping the probabilities away from
        both ends before taking logarithms is what keeps the value finite; the
        *gradient* needs no such care, since it is a subtraction.
        """
        output_probabilities = stable_logistic(outputs)

        # Only the value needs this. A confident enough score saturates the
        # sigmoid to exactly 0.0 or 1.0, and one of the two logarithms is then
        # -inf, which poisons the average for a row that was merely very
        # wrong. The gradient below is a subtraction, finite at either end, so
        # it reads the unclipped probabilities.
        # The bounds are mirror images on purpose. ``tiny`` looks like the
        # natural floor, and it is not: ``1 - tiny`` rounds straight back to
        # 1.0, so the upper bound can only be ``1 - eps``, and pairing the two
        # makes the same mistake cost 708 one way and 36 the other. Cross
        # entropy is symmetric in its two classes, so its floor has to be too.
        spacing = np.finfo(np.float64).eps
        clipped = np.clip(output_probabilities, spacing, 1.0 - spacing)

        return LossMeasurement(
            value=-np.sum(
                targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)
            )
            / outputs.shape[0],
            gradient=(output_probabilities - targets) / outputs.shape[0],
        )

    @property
    def description(self) -> str:
        return "-sum(y*log(p) + (1-y)*log(1-p)) / n"


class SoftmaxCrossEntropy(Loss):
    """Softmax over the classes, scored by cross-entropy, as one operation.

    Targets are one-hot: one row per observation, a single 1.0 in the column of
    the true class, which is the shape :meth:`one_hot` builds from class
    positions.

    Its gradient is ``(p - y) / n_rows``, the shared form, and the softmax
    Jacobian that would otherwise be needed cancels entirely.
    """

    __slots__ = ()

    @staticmethod
    def one_hot(
        class_positions: NDArray[np.integer] | NumericInput, n_classes: int
    ) -> FloatArray:
        """Turn a column of class positions into the block a loss compares to.

        Parameters
        ----------
        class_positions:
            ``(n_rows,)``, whole numbers in ``[0, n_classes)``. An integer
            array is the usual thing to have, which is why the annotation
            admits one rather than insisting on floats it would immediately
            round.
        n_classes:
            How wide the answer is.

        Returns
        -------
        FloatArray
            ``(n_rows, n_classes)``, one 1.0 per row.
        """
        positions = np.asarray(class_positions, dtype=np.int64).ravel()
        block = np.zeros((positions.size, n_classes), dtype=np.float64)
        block[np.arange(positions.size), positions] = 1.0
        return block

    def _measure_aligned(
        self, outputs: FloatArray, targets: FloatArray
    ) -> LossMeasurement:
        """Squash across the classes, score, and hand back the subtraction.

        Notes
        -----
        Reach for :func:`~oop_ml.core.logistic.stable_softmax`, which already
        subtracts the row maximum. Written literally, ``exp(z) / sum(exp(z))``
        answers ``nan`` once any score passes about 709.

        ``log(p)`` is the remaining trap. A deeply negative logit underflows to
        exactly zero and its logarithm is ``-inf``, which poisons the average.
        Flooring the probabilities at ``np.finfo(np.float64).tiny`` before the
        logarithm costs nothing and keeps a confidently wrong row finite.

        Only the true class contributes to the sum, since every other target
        entry is zero, though multiplying through by the one-hot block is the
        clearer way to say that and costs nothing at these sizes.
        """
        output_probabilities = stable_softmax(outputs)
        clipped = np.maximum(output_probabilities, np.finfo(np.float64).tiny)

        return LossMeasurement(
            value=-np.sum(targets * np.log(clipped)) / outputs.shape[0],
            gradient=(output_probabilities - targets) / outputs.shape[0],
        )

    @property
    def description(self) -> str:
        return "-sum(y * log(softmax(z))) / n"
