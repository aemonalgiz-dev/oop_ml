"""Predicted classes against the truth, when there are more than two of them.

The binary confusion matrix has four cells because there are two ways to be
right and two ways to be wrong. With ``K`` classes there are ``K * K`` cells,
the diagonal is right and everything else is a particular confusion -- not
merely an error, but *this* class mistaken for *that* one, which is usually the
question worth asking.

Precision and recall survive the move, but they stop being single numbers.
Each is defined per class: precision for class ``k`` is the share of rows
called ``k`` that really were, recall for class ``k`` is the share of real
``k`` rows that were found. A model can be excellent at the common class and
useless at the rare one, and that is exactly the fact a single number would
hide.

Macro against micro
-------------------
Collapsing the per-class scores to one number is a choice with consequences,
and there is no default that is right for everybody.

*Macro* averages the per-class scores, so every class counts equally however
rare it is. A model that ignores a class present in 5% of rows is punished as
hard as one that ignores a class present in 45%.

*Micro* pools the counts first and then divides, so every *row* counts equally
and the common classes dominate. In the single-label case where every row gets
exactly one prediction, micro-precision, micro-recall and accuracy are all the
same number, which is worth knowing before reporting them as though they were
three pieces of evidence.

On a target split 77 / 41 / 182 those two tell noticeably different stories
about the same model. Pick the one that matches what a mistake costs, and say
which you picked.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.data.column import Column, ColumnSource
from oop_ml.core.exceptions import InvalidValuesError, UndefinedMetricError
from oop_ml.core.types import FloatArray, IndexArray
from oop_ml.core.validation import ValueRole


class MultiClassConfusionMatrix:
    """A ``K x K`` table of actual class against predicted class.

    Row ``i``, column ``j`` counts the rows whose true class was ``i`` and whose
    predicted class was ``j``. The diagonal is therefore the correct
    predictions, and an off-diagonal cell names a specific confusion rather
    than a generic error.

    Parameters
    ----------
    counts:
        A square, non-negative integer array.

    Raises
    ------
    InvalidValuesError
        If ``counts`` is not square, or holds a negative entry.
    """

    __slots__ = ("_counts",)

    def __init__(self, counts: IndexArray) -> None:
        as_array = np.asarray(counts)

        if as_array.ndim != 2 or as_array.shape[0] != as_array.shape[1]:
            raise InvalidValuesError(
                f"a confusion matrix must be square; got shape {as_array.shape}"
            )
        if np.any(as_array < 0):
            raise InvalidValuesError("confusion matrix counts cannot be negative")

        self._counts = as_array.astype(np.int64)

    @classmethod
    def of(
        cls, actual: Column, predicted: Column, n_classes: int
    ) -> MultiClassConfusionMatrix:
        """Count every (actual, predicted) pair from two aligned label columns.

        Both columns are assumed already validated and aligned, which
        :class:`MultiClassEvaluation` guarantees before calling this.
        """
        counts = np.zeros((n_classes, n_classes), dtype=np.int64)
        np.add.at(
            counts,
            (actual.values.astype(np.int64), predicted.values.astype(np.int64)),
            1,
        )

        return cls(counts)

    @property
    def counts(self) -> IndexArray:
        """The table itself, as a copy so a caller cannot corrupt it."""
        return self._counts.copy()

    @property
    def n_classes(self) -> int:
        """How many classes the table spans."""
        return int(self._counts.shape[0])

    @property
    def n_samples(self) -> int:
        """How many observations the table was built from."""
        return int(self._counts.sum())

    @property
    def correct(self) -> int:
        """Rows on the diagonal: predicted class equal to actual class."""
        return int(np.trace(self._counts))

    def true_positives_for(self, class_index: int) -> int:
        """Rows of class ``class_index`` that were predicted as it."""
        self._check_class(class_index)
        return int(self._counts[class_index, class_index])

    def predicted_as(self, class_index: int) -> int:
        """Rows the model called ``class_index``, right or wrong.

        The column total, which is precision's denominator.
        """
        self._check_class(class_index)
        return int(self._counts[:, class_index].sum())

    def actually_are(self, class_index: int) -> int:
        """Rows that genuinely were ``class_index``.

        The row total, which is recall's denominator.
        """
        self._check_class(class_index)
        return int(self._counts[class_index, :].sum())

    def _check_class(self, class_index: int) -> None:
        """Raise :class:`InvalidValuesError` if the class is not in the table."""
        if not 0 <= class_index < self.n_classes:
            raise InvalidValuesError(
                f"class {class_index} is outside a table spanning "
                f"0 to {self.n_classes - 1}"
            )

    def __repr__(self) -> str:
        return (
            f"MultiClassConfusionMatrix(n_classes={self.n_classes}, "
            f"n_samples={self.n_samples}, correct={self.correct})"
        )


class MultiClassEvaluation:
    """What a multi-class classifier predicted, against what actually happened.

    A sibling of
    :class:`~oop_ml.core.evaluation.classification.ClassificationEvaluation`
    rather than a generalisation of it, for the same reason that one is a
    sibling of the regression evaluation: the binary object hands back a single
    precision, and no caller who wants one number would accept a vector.

    Parameters
    ----------
    actual_values:
        The true classes, whole positions running ``0 .. K - 1``.
    predicted_values:
        The predicted classes, on the same scale.
    n_classes:
        How many classes the table should span. Defaults to whatever the actual
        column contains. Pass it explicitly when scoring a fold that may not
        contain every class, or the table will silently be the wrong size.

    Raises
    ------
    EmptyValuesError
        If either input is empty.
    NonEqualArrayLengthError
        If the two inputs are different lengths.
    NonBinaryLabelsError
        If either input holds a negative or fractional value.
    SingleClassError
        If the actual column holds fewer than two classes, or leaves a gap.
    InvalidValuesError
        If a predicted class falls outside ``0 .. n_classes - 1``.
    """

    __slots__ = (
        "_actual_column",
        "_confusion_matrix",
        "_n_classes",
        "_predicted_column",
    )

    def __init__(
        self,
        actual_values: ColumnSource,
        predicted_values: ColumnSource,
        n_classes: int | None = None,
    ) -> None:
        self._actual_column = Column.of(actual_values, ValueRole.ACTUAL_VALUES)
        self._predicted_column = Column.of(predicted_values, ValueRole.PREDICTED_VALUES)

        self._actual_column.check_equal_length(self._predicted_column)
        self._actual_column.check_is_label_encoded()

        self._n_classes = (
            self._actual_column.n_classes if n_classes is None else int(n_classes)
        )
        self._check_predictions_are_known_classes()

        # Counted once. Every per-class metric reads this table, so macro_f1
        # used to rebuild it 2K times -- sixteen passes over the data at eight
        # classes to answer a question settled on the first.
        self._confusion_matrix = MultiClassConfusionMatrix.of(
            self._actual_column, self._predicted_column, self._n_classes
        )

    def _check_predictions_are_known_classes(self) -> None:
        """Raise if the model named a class the table has no column for."""
        predicted = self._predicted_column.values

        if np.any(predicted < 0.0) or np.any(predicted != np.floor(predicted)):
            raise InvalidValuesError(
                "predicted values must be whole class positions starting at 0"
            )
        if predicted.size and predicted.max() >= self._n_classes:
            raise InvalidValuesError(
                f"predicted class {int(predicted.max())} is outside a problem "
                f"with {self._n_classes} classes"
            )

    @property
    def actual_values(self) -> FloatArray:
        """The true classes, as validated."""
        return self._actual_column.values

    @property
    def predicted_values(self) -> FloatArray:
        """The predicted classes, as validated."""
        return self._predicted_column.values

    @property
    def n_samples(self) -> int:
        """How many observations were scored."""
        return self._actual_column.n_samples

    @property
    def n_classes(self) -> int:
        """How many classes the evaluation spans."""
        return self._n_classes

    @property
    def confusion_matrix(self) -> MultiClassConfusionMatrix:
        """The ``K x K`` table every metric below is derived from."""
        return self._confusion_matrix

    @staticmethod
    def _ratio(numerator: int, denominator: int, undefined_because: str) -> float:
        """One count over another, with an empty denominator named, not nan.

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
        """Share of rows whose predicted class was the true one.

        The diagonal over the whole table. Read it with the same suspicion the
        binary case deserves, and more: with many classes a majority-class
        predictor can score well while getting every rare class wrong.
        """
        matrix = self.confusion_matrix

        return self._ratio(
            matrix.correct,
            matrix.n_samples,
            "accuracy is undefined with no observations",
        )

    def precision_for(self, class_index: int) -> float:
        """Of the rows called ``class_index``, the share that really were.

        Raises
        ------
        InvalidValuesError
            If ``class_index`` is not a class of this problem.
        UndefinedMetricError
            If the model never predicted this class, leaving an empty column.
        """
        matrix = self.confusion_matrix

        return self._ratio(
            matrix.true_positives_for(class_index),
            matrix.predicted_as(class_index),
            f"precision for class {class_index} is undefined because the model "
            f"never predicted it",
        )

    def recall_for(self, class_index: int) -> float:
        """Of the rows that really were ``class_index``, the share found.

        Raises
        ------
        InvalidValuesError
            If ``class_index`` is not a class of this problem.
        UndefinedMetricError
            If no row genuinely belongs to this class, leaving an empty row.
        """
        matrix = self.confusion_matrix

        return self._ratio(
            matrix.true_positives_for(class_index),
            matrix.actually_are(class_index),
            f"recall for class {class_index} is undefined because no row belongs to it",
        )

    def f1_for(self, class_index: int) -> float:
        """The harmonic mean of this class's precision and recall.

        Raises
        ------
        InvalidValuesError
            If ``class_index`` is not a class of this problem.
        UndefinedMetricError
            If either component is undefined, or both are zero.
        """
        precision = self.precision_for(class_index)
        recall = self.recall_for(class_index)

        if precision + recall == 0.0:
            raise UndefinedMetricError(
                f"F1 for class {class_index} is undefined when its precision "
                f"and recall are both zero"
            )

        return 2.0 * precision * recall / (precision + recall)

    @property
    def macro_precision(self) -> float:
        """Mean of the per-class precisions, every class weighted equally.

        The average is unweighted on purpose: a class present in 5% of rows
        moves this number exactly as much as one present in 45%, which is the
        whole point of asking for it.

        Raises
        ------
        UndefinedMetricError
            If any class's precision is undefined. Averaging over the classes
            that happen to be defined would quietly change which question is
            being answered.
        """
        # per_class_precision already loops the classes and already propagates
        # an undefined one, so this is genuinely just the mean of it. float() at
        # the boundary because public methods hand back Python floats, not
        # numpy scalars.
        return float(np.mean(self.per_class_precision))

    @property
    def macro_recall(self) -> float:
        """Mean of the per-class recalls, every class weighted equally.

        Raises
        ------
        UndefinedMetricError
            If any class's recall is undefined.
        """
        return float(np.mean(self.per_class_recall))

    @property
    def macro_f1_score(self) -> float:
        """Mean of the per-class F1 scores.

        Note this is the average of the per-class harmonic means, not the
        harmonic mean of :attr:`macro_precision` and :attr:`macro_recall`.
        Those are different numbers, and the second is not a standard metric
        however reasonable it looks.

        Raises
        ------
        UndefinedMetricError
            If any class's F1 is undefined.
        """
        return float(np.mean(self.per_class_f1_score))

    @property
    def micro_precision(self) -> float:
        """Precision with the per-class counts pooled before dividing.

        Every *row* counts equally here rather than every class, so the common
        classes dominate. Where each row receives exactly one prediction --
        which is the only case this object models -- the pooled numerator is
        the diagonal and the pooled denominator is every row, so this equals
        :attr:`micro_recall` and :attr:`accuracy` exactly. Worth knowing before
        reporting all three as though they were separate evidence.

        Raises
        ------
        UndefinedMetricError
            If there are no observations.
        """
        matrix = self.confusion_matrix

        found = sum(
            matrix.true_positives_for(index) for index in range(self._n_classes)
        )
        belong = sum(matrix.predicted_as(index) for index in range(self._n_classes))

        return self._ratio(
            found, belong, "micro-recall is undefined with no observations"
        )

    @property
    def micro_recall(self) -> float:
        """Recall with the per-class counts pooled before dividing.

        Equal to :attr:`micro_precision` and :attr:`accuracy`, for the reason
        given there.

        Raises
        ------
        UndefinedMetricError
            If there are no observations.
        """
        matrix = self.confusion_matrix

        found = sum(
            matrix.true_positives_for(index) for index in range(self._n_classes)
        )
        belong = sum(matrix.actually_are(index) for index in range(self._n_classes))

        return self._ratio(
            found, belong, "micro-recall is undefined with no observations"
        )

    @property
    def per_class_precision(self) -> FloatArray:
        """Every class's precision, in class order.

        Raises
        ------
        UndefinedMetricError
            If any class's precision is undefined.
        """
        return np.array([self.precision_for(index) for index in range(self._n_classes)])

    @property
    def per_class_f1_score(self) -> FloatArray:
        return np.array([self.f1_for(index) for index in range(self._n_classes)])

    @property
    def per_class_recall(self) -> FloatArray:
        """Every class's recall, in class order.

        Raises
        ------
        UndefinedMetricError
            If any class's recall is undefined.
        """
        return np.array([self.recall_for(index) for index in range(self._n_classes)])

    def __repr__(self) -> str:
        return (
            f"MultiClassEvaluation(n_samples={self.n_samples}, "
            f"n_classes={self.n_classes}, "
            f"correct={self.confusion_matrix.correct})"
        )
