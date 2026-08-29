"""Score a model on data it was not fitted to, and average over folds.

Theory
------
Fitting and scoring on the same rows answers the wrong question. It measures how
well the model memorised what it saw, and every model here is built to maximise
exactly that. Cross-validation answers the question you actually have: how well
does this model do on rows it has never seen?

The procedure is one sentence. Divide the rows into ``k`` folds; for each fold,
fit on the other ``k - 1`` and score on the one held out; average the scores.

Two properties make it worth the ``k`` fits:

* Every row is scored exactly once, so nothing goes to waste, unlike a single
  hold-out where a quarter of the data only ever serves as a ruler.
* **Every fit sees most of the data**, so the models being scored resemble the
  model you would finally train on everything.

The spread across folds is worth as much as the average. A mean R^2 of 0.8 from
folds at 0.79, 0.81, 0.80 is a different claim from the same mean out of 0.2,
0.95, 1.0. The second is telling you the estimate itself is unreliable, and no
single number can tell you that on its own.

Choosing a hyperparameter
-------------------------
This is what makes ``penalty``, ``degree`` and ``learning_rate`` choosable at
all. Cross-validate the model at each candidate value, keep the value with the
best mean held-out score. Nothing in the derivations gives you those numbers,
because they depend on the noise level and the collinearity of *your* data --
quantities you do not know and cannot compute directly.

Averaging a ratio, and why classification does it differently
------------------------------------------------------------
For a regressor the summary is the mean of the folds' R^2 scores, and the
spread across them is half the value. A classifier's metrics are ratios over
rows -- precision is ``true positives / predicted positives`` -- and there are
two ways to combine ``k`` of those, which disagree.

Averaging the folds' precisions counts each *fold* equally. Adding the
confusion matrices together first and dividing once counts each *row* equally.
Measured on five folds, four of 200 rows scoring 0.900 and one of 10 scoring
0.200::

    average the folds' precisions   0.7600
    pool the matrices, then divide  0.8829

A fold holding 1.2% of the data moved the first number by 0.12.

This library pools, and the reason is the undefined case rather than taste.
Recall on a fold with no positives is ``0/0``. Averaged, that fold has to
contribute a ``nan`` or a convention, and neither is a measurement; pooled, it
adds zero to both the numerator and the denominator and distorts nothing.
Accuracy is always defined, so it is the one the classification result also
reports a spread for.

A warning this design does not yet enforce
------------------------------------------
Anything *learned* from the data must be learned inside the fold. Standardizing before
splitting lets the training rows see the test rows' mean, which is leakage, and it
flatters every score computed afterwards. Until a pipeline object exists to make that
structural, the caller has to fit the
:class:`~oop_ml.preprocessing.standardization.standardizer.Standardizer` on
``split.training`` and only transform ``split.testing``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, Self, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from oop_ml.core.base.estimator import MultiClassClassifier, Regressor
from oop_ml.core.data.column import Column
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.evaluation.multiclass import (
    MultiClassConfusionMatrix,
    MultiClassEvaluation,
)
from oop_ml.core.evaluation.regression import RegressionEvaluation
from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.validation import ValueRole
from oop_ml.model_selection.splitting import KFold

EvaluationT = TypeVar("EvaluationT", covariant=True)


class Foldable(Protocol[EvaluationT]):
    """Anything that can be fitted to features and then score itself on them.

    A structural protocol rather than a base class, because the fold loop cares
    about exactly two methods and both ``Regressor`` and ``MultiClassClassifier``
    already have them. Naming a common base instead would mean naming
    ``Estimator``, which declares no ``evaluate`` at all -- the evaluations are
    one per task, so no single signature covers both.

    The type parameter is what each entry point recovers on the way out: hand
    the loop a regressor and the list is of regression evaluations.
    """

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self: ...

    def evaluate(
        self, input_values: Sequence[Feature], actual_values: Feature
    ) -> EvaluationT: ...


class RegressionCrossValidationResult:
    """What each fold scored, and the summary across them.

    Parameters
    ----------
    fold_evaluations:
        One :class:`~oop_ml.core.evaluation.regression.RegressionEvaluation`
        per fold, each computed on that fold's held-out rows.

    Raises
    ------
    EmptyValuesError
        If no evaluations are supplied.
    """

    __slots__ = ("_fold_evaluations",)

    def __init__(self, fold_evaluations: Sequence[RegressionEvaluation]) -> None:
        if not fold_evaluations:
            raise EmptyValuesError("cross-validation needs at least one fold")

        self._fold_evaluations = tuple(fold_evaluations)

    @property
    def n_folds(self) -> int:
        """How many folds were scored."""
        return len(self._fold_evaluations)

    @property
    def mean_r2_score(self) -> float:
        """Average held-out R^2 across the folds.

        The headline number, and the one to compare candidate hyperparameters
        on. Read it beside :attr:`r2_score_spread`.
        """
        return (
            sum(evaluation.r2_score for evaluation in self._fold_evaluations)
            / self.n_folds
        )

    @property
    def mean_squared_error(self) -> float:
        """Average held-out mean squared error across the folds."""
        return (
            sum(evaluation.mean_squared_error for evaluation in self._fold_evaluations)
            / self.n_folds
        )

    @property
    def r2_score_spread(self) -> float:
        """Difference between the best and worst fold's R^2.

        How much the estimate depends on which rows happened to be held out. A
        wide spread means the mean is not to be trusted on its own.
        """
        scores = [evaluation.r2_score for evaluation in self._fold_evaluations]

        return max(scores) - min(scores)

    def __iter__(self) -> Iterator[RegressionEvaluation]:
        return iter(self._fold_evaluations)

    def __len__(self) -> int:
        return self.n_folds

    def __repr__(self) -> str:
        return (
            f"RegressionCrossValidationResult(n_folds={self.n_folds}, "
            f"mean_r2_score={self.mean_r2_score:.4f})"
        )


class ClassificationCrossValidationResult:
    """What each fold scored on a classification task, pooled across them.

    The sibling of :class:`RegressionCrossValidationResult`, and it summarises
    differently for the reason the module docstring gives: a classifier's
    metrics are ratios over rows, so the folds' confusion matrices are added
    together and the metric computed once, rather than each fold's metric being
    computed and then averaged.

    That is not a preference. Recall on a fold with no positives is ``0/0``.
    Averaged, such a fold contributes a ``nan`` or a convention; pooled, it adds
    zero to both sides of the ratio and distorts nothing. Stratified folding
    makes empty folds rare and cannot always prevent them, so the summary has
    to survive one.

    Parameters
    ----------
    fold_evaluations:
        One :class:`~oop_ml.core.evaluation.multiclass.MultiClassEvaluation`
        per fold, in fold order.

    Raises
    ------
    EmptyValuesError
        If no folds are supplied.
    InvalidValuesError
        If the folds disagree about how many classes there are, which means
        they were not produced by the same fitted configuration.
    """

    __slots__ = ("_fold_evaluations",)

    def __init__(self, fold_evaluations: Sequence[MultiClassEvaluation]) -> None:
        if not fold_evaluations:
            raise EmptyValuesError("at least one fold evaluation is required")

        widths = {evaluation.n_classes for evaluation in fold_evaluations}
        if len(widths) > 1:
            raise InvalidValuesError(
                f"folds disagree about the class count: {sorted(widths)}"
            )

        self._fold_evaluations = tuple(fold_evaluations)

    @property
    def n_folds(self) -> int:
        """How many folds were scored."""
        return len(self._fold_evaluations)

    @property
    def n_classes(self) -> int:
        """How many classes the folds spanned."""
        return self._fold_evaluations[0].n_classes

    @property
    def pooled_confusion_matrix(self) -> MultiClassConfusionMatrix:
        """Every fold's counts added into one matrix.

        The whole of the summarising, and the only method here that is not a
        one-line read off the result. Each fold's matrix has the same shape --
        the constructor refuses folds that disagree -- so the pooled counts are
        the element-wise sum, and every metric below is then the ordinary
        single-matrix formula applied once.

        Because each row was held out exactly once, this pooled matrix holds
        one entry per row of the whole dataset. That is what makes it the
        honest denominator: a metric read off it is a metric over the data,
        not an average of metrics over pieces of it.
        """
        output_confusion_matrix = np.zeros(
            (self.n_classes, self.n_classes), dtype=np.int64
        )

        for fold_evaluation in self._fold_evaluations:
            output_confusion_matrix += fold_evaluation.confusion_matrix.counts

        return MultiClassConfusionMatrix(output_confusion_matrix)

    @property
    def pooled_accuracy(self) -> float:
        """Share of all held-out rows put in the right class."""
        return self._pooled_evaluation.accuracy

    @property
    def pooled_macro_precision(self) -> float:
        """Macro precision read off the pooled matrix.

        Macro because the per-class scores are averaged with equal weight,
        pooled because each class's counts came from every fold before that
        average was taken. The two words describe different steps and both
        belong in the name.
        """
        return self._pooled_evaluation.macro_precision

    @property
    def pooled_macro_recall(self) -> float:
        """Macro recall read off the pooled matrix.

        The metric that most needs pooling. A fold holding no rows of some
        class has an undefined recall for it, and pooling is what stops that
        fold having to contribute a convention instead of a count.
        """
        return self._pooled_evaluation.macro_recall

    @property
    def pooled_macro_f1_score(self) -> float:
        """Macro F1 read off the pooled matrix."""
        return self._pooled_evaluation.macro_f1_score

    @property
    def accuracy_spread(self) -> float:
        """Difference between the best and worst fold's accuracy.

        The one metric it is safe to compare fold by fold, because accuracy is
        defined on any fold with rows in it. Read it beside the pooled figure
        for the same reason ``r2_score_spread`` is read beside the mean: 0.85
        from folds at 0.84, 0.86, 0.85 is a different claim from the same 0.85
        out of 0.6, 0.95, 1.0.
        """
        scores = [evaluation.accuracy for evaluation in self._fold_evaluations]

        return max(scores) - min(scores)

    @property
    def _pooled_evaluation(self) -> MultiClassEvaluation:
        """The pooled matrix, wearing the evaluation that can read metrics off it.

        Rebuilding an evaluation rather than reimplementing precision and
        recall here, so a classification metric has exactly one definition in
        this library however it was arrived at.

        Build it from ``_held_out_rows`` and ``_held_out_predictions``, which
        are the folds laid end to end. Every row was held out exactly once, so
        that concatenation is one prediction per row of the whole dataset, and
        no metric on it depends on the order rows arrive in.

        This is the second route to the same table.
        ``pooled_confusion_matrix`` adds the folds' counts; this counts the
        folds' rows. ``test_the_two_routes_to_the_pooled_table_agree`` pins
        that they match, the way every other pair of routes in this library
        carries an agreement test.
        """
        return MultiClassEvaluation(
            actual_values=self._held_out_rows,
            predicted_values=self._held_out_predictions,
            n_classes=self.n_classes,
        )

    @property
    def _held_out_rows(self) -> Column:
        """Every fold's true classes, end to end in fold order."""
        return Column.selecting(
            np.concatenate(
                [evaluation.actual_values for evaluation in self._fold_evaluations]
            ),
            ValueRole.ACTUAL_VALUES,
        )

    @property
    def _held_out_predictions(self) -> Column:
        """Every fold's predicted classes, aligned with ``_held_out_rows``."""
        return Column.selecting(
            np.concatenate(
                [evaluation.predicted_values for evaluation in self._fold_evaluations]
            ),
            ValueRole.PREDICTED_VALUES,
        )

    def __iter__(self) -> Iterator[MultiClassEvaluation]:
        return iter(self._fold_evaluations)

    def __len__(self) -> int:
        return self.n_folds

    def __repr__(self) -> str:
        return (
            f"ClassificationCrossValidationResult(n_folds={self.n_folds}, "
            f"pooled_accuracy={self.pooled_accuracy:.4f})"
        )


class CrossValidation(BaseModel):
    """Fit a model on each training fold and score it on the held-out one.

    Parameters
    ----------
    folds:
        The splitter that divides the rows. Defaults to five-fold.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    folds: KFold = KFold()

    def evaluate(
        self, model: Regressor[Sequence[Feature], Feature], dataset: Dataset
    ) -> RegressionCrossValidationResult:
        """Cross-validate a regressor over ``dataset``.

        For each split the folds produce: fit the model on ``split.training``,
        then evaluate it on ``split.testing``, which holds rows the fit never
        saw. Collect one
        :class:`~oop_ml.core.evaluation.regression.RegressionEvaluation` per
        fold and return them as a :class:`RegressionCrossValidationResult`.

        The model is refitted from scratch on every fold. That is the point:
        ``k`` independent estimates of how this *configuration* performs, not
        one model incrementally trained.

        Raises
        ------
        TooFewValuesError
            If the dataset has fewer rows than folds.
        """
        return RegressionCrossValidationResult(self._fold_evaluations(model, dataset))

    def evaluate_classifier(
        self,
        model: MultiClassClassifier[Sequence[Feature], Feature],
        dataset: Dataset,
    ) -> ClassificationCrossValidationResult:
        """Cross-validate a classifier over ``dataset``.

        The same loop, differing only in what the folds hand back, because
        ``MultiClassClassifier.evaluate`` returns a
        :class:`~oop_ml.core.evaluation.multiclass.MultiClassEvaluation` where
        ``Regressor.evaluate`` returns a regression one. The seam is the one
        the evaluations already have and this follows it rather than inventing
        a second.

        A separate method rather than one that inspects its argument: which
        result a caller gets is decided by which model they hand over, and a
        return type that changes based on a runtime check is one a reader has
        to trace to predict.

        **Fold with a stratified splitter.** Nothing here enforces it, and on
        an imbalanced target plain folding leaves folds holding no positive at
        all -- measured, three of ten at a 5% positive rate. Pooling survives
        that where averaging would not, but a fold with no positives still
        contributes nothing to the recall it was supposed to help measure.

        Raises
        ------
        TooFewValuesError
            If the dataset has fewer rows than folds.
        """
        return ClassificationCrossValidationResult(
            self._fold_evaluations(model, dataset)
        )

    def _fold_evaluations(
        self,
        model: Foldable[EvaluationT],
        dataset: Dataset,
    ) -> list[EvaluationT]:
        """Fit and score once per fold, whatever the task.

        The loop both entry points share. It asks the model to evaluate itself
        and never looks at what comes back, which is what lets one body serve a
        regressor and a classifier without knowing which it has -- and the type
        parameter is what carries that back out, so the caller still knows.
        """
        evaluations = []

        for split in self.folds.split(dataset):
            training, testing = split.training, split.testing

            model.fit(training.input_features, training.target_feature)
            evaluations.append(
                model.evaluate(testing.input_features, testing.target_feature)
            )

        return evaluations
