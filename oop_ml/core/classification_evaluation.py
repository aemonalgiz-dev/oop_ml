"""Predicted labels paired with the truth, and the metrics that follow.

R^2 has nothing to say about a classifier. Asking how much variance a set of
zeroes and ones explains is not a question with a useful answer, so the
regression evaluation object cannot simply be pointed at a different target and
reused.

What replaces it is the confusion matrix. Every binary prediction lands in one
of four boxes, according to what was predicted and what was true, and every
metric worth having is some ratio of those four counts. Accuracy, precision,
recall and F1 are not four separate ideas so much as four different questions
asked of the same table.

The reason there are four of them rather than one is that the two mistakes are
not interchangeable. Consider a screening test for a condition that one person
in a thousand has. Predict "no" for everybody and you are 99.9% accurate while
being completely useless, because accuracy quietly rewards you for the class
that dominates. Precision and recall refuse to average the two errors together,
which is what makes them worth the extra reading.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.column import Column, ColumnSource
from oop_ml.core.exceptions import InvalidValuesError, UndefinedMetricError
from oop_ml.core.types import FloatArray
from oop_ml.core.validation import ValueRole


class ConfusionMatrix:
    """The four counts every binary classification metric is built from.

    Parameters
    ----------
    true_positives:
        Predicted 1, actually 1.
    true_negatives:
        Predicted 0, actually 0.
    false_positives:
        Predicted 1, actually 0. A false alarm.
    false_negatives:
        Predicted 0, actually 1. A miss.

    Raises
    ------
    InvalidValuesError
        If any count is negative.
    """

    __slots__ = (
        "_false_negatives",
        "_false_positives",
        "_true_negatives",
        "_true_positives",
    )

    def __init__(
        self,
        true_positives: int,
        true_negatives: int,
        false_positives: int,
        false_negatives: int,
    ) -> None:
        counts = (true_positives, true_negatives, false_positives, false_negatives)
        if any(count < 0 for count in counts):
            raise InvalidValuesError("confusion matrix counts cannot be negative")

        self._true_positives = int(true_positives)
        self._true_negatives = int(true_negatives)
        self._false_positives = int(false_positives)
        self._false_negatives = int(false_negatives)

    @classmethod
    def of(cls, actual: Column, predicted: Column) -> ConfusionMatrix:
        """Count the four outcomes from two aligned 0/1 columns.

        Both columns are assumed already validated and aligned, which
        :class:`ClassificationEvaluation` guarantees before calling this.
        """
        actual_positive = actual.values == 1.0
        predicted_positive = predicted.values == 1.0

        return cls(
            true_positives=int(np.sum(actual_positive & predicted_positive)),
            true_negatives=int(np.sum(~actual_positive & ~predicted_positive)),
            false_positives=int(np.sum(~actual_positive & predicted_positive)),
            false_negatives=int(np.sum(actual_positive & ~predicted_positive)),
        )

    @property
    def true_positives(self) -> int:
        """Predicted 1, actually 1."""
        return self._true_positives

    @property
    def true_negatives(self) -> int:
        """Predicted 0, actually 0."""
        return self._true_negatives

    @property
    def false_positives(self) -> int:
        """Predicted 1, actually 0, which is a false alarm."""
        return self._false_positives

    @property
    def false_negatives(self) -> int:
        """Predicted 0, actually 1, which is a miss."""
        return self._false_negatives

    @property
    def n_samples(self) -> int:
        """How many observations the four counts were taken over."""
        return (
            self._true_positives
            + self._true_negatives
            + self._false_positives
            + self._false_negatives
        )

    @property
    def predicted_positive(self) -> int:
        """How many rows the model called positive, right or wrong."""
        return self._true_positives + self._false_positives

    @property
    def actual_positive(self) -> int:
        """How many rows genuinely were positive."""
        return self._true_positives + self._false_negatives

    def __repr__(self) -> str:
        return (
            f"ConfusionMatrix(tp={self._true_positives}, tn={self._true_negatives}, "
            f"fp={self._false_positives}, fn={self._false_negatives})"
        )


class ClassificationEvaluation:
    """What a classifier predicted, against what actually happened.

    The counterpart to :class:`~oop_ml.core.evaluation.RegressionEvaluation`,
    and deliberately a sibling of it rather than a subclass. The two share the
    idea of pairing predictions with truth and nothing else; there is no metric
    they have in common, and no caller who wants one would accept the other.

    Parameters
    ----------
    actual_values:
        The true labels, which must be 0 or 1.
    predicted_values:
        The predicted labels, which must also be 0 or 1. These are labels rather
        than probabilities, so a threshold has already been applied by the time
        anything reaches here.

    Raises
    ------
    EmptyValuesError
        If either input is empty.
    NonEqualArrayLengthError
        If the two inputs are different lengths.
    NonBinaryLabelsError
        If either input contains a value that is not 0 or 1.
    """

    __slots__ = ("_actual_column", "_predicted_column")

    def __init__(
        self, actual_values: ColumnSource, predicted_values: ColumnSource
    ) -> None:
        self._actual_column = Column.of(actual_values, ValueRole.ACTUAL_VALUES)
        self._predicted_column = Column.of(predicted_values, ValueRole.PREDICTED_VALUES)

        self._actual_column.check_equal_length(self._predicted_column)
        self._actual_column.check_is_binary()
        self._predicted_column.check_is_binary()

    @property
    def actual_values(self) -> FloatArray:
        """The true labels, as validated."""
        return self._actual_column.values

    @property
    def predicted_values(self) -> FloatArray:
        """The predicted labels, as validated."""
        return self._predicted_column.values

    @property
    def n_samples(self) -> int:
        """How many observations were scored."""
        return self._actual_column.n_samples

    @property
    def confusion_matrix(self) -> ConfusionMatrix:
        """The four counts, from which every metric below is derived."""
        return ConfusionMatrix.of(self._actual_column, self._predicted_column)

    @staticmethod
    def _ratio(numerator: int, denominator: int, undefined_because: str) -> float:
        """One count over another, with a zero denominator named rather than nan.

        Every metric below is one of these. Left to numpy a zero denominator
        gives back nan, which then propagates silently through whatever the
        caller does next, and an undefined metric is a fact about the data worth
        stopping on. A model that never predicted positive has not earned a
        precision of zero; it has no precision at all, and those are different
        claims.

        Raises
        ------
        UndefinedMetricError
            If ``denominator`` is zero.
        """
        if denominator == 0:
            raise UndefinedMetricError(undefined_because)

        return numerator / denominator

    @property
    def accuracy(self) -> float:
        """Share of predictions that were correct.

        ``(true_positives + true_negatives) / n_samples``.

        Read this one with suspicion whenever the classes are unbalanced. On a
        target that is 99% negative, predicting negative for everything scores
        0.99 while finding nothing at all, so a high accuracy is only evidence
        when the two classes are of comparable size.

        Returns
        -------
        float
            Between 0.0 and 1.0.
        """
        raise NotImplementedError

    @property
    def precision(self) -> float:
        """Of the rows called positive, the share that really were.

        ``true_positives / (true_positives + false_positives)``.

        This is the metric to care about when a false alarm is expensive, since
        it asks how much you can trust a positive prediction when you see one.

        Raises
        ------
        UndefinedMetricError
            If nothing was predicted positive, leaving a zero denominator.
            Note that this is genuinely undefined rather than zero: a model that
            never fires has not made a wrong positive call, and it has not made
            a right one either.
        """
        raise NotImplementedError

    @property
    def recall(self) -> float:
        """Of the rows that really were positive, the share that were found.

        ``true_positives / (true_positives + false_negatives)``.

        The metric to care about when a miss is expensive. Precision and recall
        trade against each other through the decision threshold: lower it and
        you catch more of the real positives while raising more false alarms.

        Raises
        ------
        UndefinedMetricError
            If there were no positive rows at all, leaving a zero denominator.
        """
        raise NotImplementedError

    @property
    def f1_score(self) -> float:
        """The harmonic mean of precision and recall.

        ``2 * precision * recall / (precision + recall)``.

        Harmonic rather than arithmetic, and the choice matters. An arithmetic
        mean of precision 1.0 and recall 0.0 is a respectable-looking 0.5,
        whereas the harmonic mean is 0.0, which is the honest answer for a model
        that found none of the positives. The harmonic mean is dragged down by
        whichever of the two is worse, so it cannot be gamed by sacrificing one
        of them entirely.

        Raises
        ------
        UndefinedMetricError
            If either precision or recall is undefined, or if both are zero.
        """
        raise NotImplementedError

    @property
    def specificity(self) -> float:
        """Of the rows that really were negative, the share correctly rejected.

        ``true_negatives / (true_negatives + false_positives)``.

        Recall's mirror image, and the pair of them is what a ROC curve plots.

        Raises
        ------
        UndefinedMetricError
            If there were no negative rows at all.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        matrix = self.confusion_matrix

        return (
            f"ClassificationEvaluation(n_samples={self.n_samples}, "
            f"tp={matrix.true_positives}, tn={matrix.true_negatives}, "
            f"fp={matrix.false_positives}, fn={matrix.false_negatives})"
        )
