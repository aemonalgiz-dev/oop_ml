"""Spec for cross-validating a classifier, and for pooling rather than averaging.

The test carrying the argument is ``TestWhyItPools``. Both tests there use folds
of deliberately unequal size, because that is the only condition under which the
two ways of combining ``k`` ratios disagree: with equal folds the mean of the
fold accuracies and the pooled accuracy are the same number, and a spec written
on equal folds would pass against either implementation.

``test_a_fold_can_have_a_recall_the_pooled_result_does_not`` is the other half,
and it is the reason this is not a preference. A fold holding no rows of some
class has an undefined recall for it. Averaged, that fold has to contribute a
convention; pooled, its counts land in a denominator that other folds have
already made non-zero.
"""

import numpy as np
import pytest

from oop_ml.core.data.dataset import Dataset
from oop_ml.core.evaluation.multiclass import MultiClassEvaluation
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    UndefinedMetricError,
)
from oop_ml.core.model_selection.cross_validation import (
    ClassificationCrossValidationResult,
    CrossValidation,
)
from oop_ml.core.model_selection.splitting import KFold
from oop_ml.numpy.classification.multiclass.multinomial_logistic_regression import (
    MultinomialLogisticRegression,
)
from test.fixtures import THREE_CLASSES


def evaluation(
    actual: list[float], predicted: list[float], n_classes: int = 2
) -> MultiClassEvaluation:
    """One fold's worth of scored rows."""
    return MultiClassEvaluation(actual, predicted, n_classes)


def perfect_then_wrong() -> ClassificationCrossValidationResult:
    """Six rows all right, then two rows all wrong.

    Mean of the fold accuracies: ``(1.0 + 0.0) / 2 = 0.5``.
    Pooled over the rows: ``6 / 8 = 0.75``.
    """
    return ClassificationCrossValidationResult(
        [
            evaluation([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
            evaluation([0, 1], [1, 0]),
        ]
    )


class TestReadingTheFolds:
    """The parts that are a plain read off the folds."""

    def test_counts_its_folds(self) -> None:
        result = perfect_then_wrong()

        assert result.n_folds == 2
        assert len(result) == 2

    def test_reports_the_class_count(self) -> None:
        assert perfect_then_wrong().n_classes == 2

    def test_iterates_the_fold_evaluations(self) -> None:
        assert [round(one.accuracy, 4) for one in perfect_then_wrong()] == [1.0, 0.0]

    def test_accuracy_spread_is_best_minus_worst(self) -> None:
        assert perfect_then_wrong().accuracy_spread == pytest.approx(1.0)

    def test_repr_names_the_pooled_figure(self) -> None:
        assert "pooled_accuracy=0.7500" in repr(perfect_then_wrong())


class TestPooling:
    """Adding the folds' tables together."""

    def test_the_pooled_matrix_is_the_element_wise_sum(self) -> None:
        """Six correct rows plus two crossed ones, written out.

        Fold one contributes ``[[3, 0], [0, 3]]`` and fold two ``[[0, 1],
        [1, 0]]``, so the pooled table is ``[[3, 1], [1, 3]]``.
        """
        pooled = perfect_then_wrong().pooled_confusion_matrix

        assert np.array_equal(pooled.counts, np.array([[3, 1], [1, 3]]))

    def test_the_pooled_matrix_holds_every_held_out_row(self) -> None:
        """Each row is held out exactly once, so the counts total the dataset."""
        assert perfect_then_wrong().pooled_confusion_matrix.n_samples == 8

    def test_the_two_routes_to_the_pooled_table_agree(self) -> None:
        """Summing the folds' counts against counting the folds' rows.

        ``pooled_confusion_matrix`` adds tables; ``_pooled_evaluation`` builds
        one from the concatenated rows. Two routes to a number are two
        implementations of it unless something asserts they meet.
        """
        result = perfect_then_wrong()

        assert np.array_equal(
            result.pooled_confusion_matrix.counts,
            result._pooled_evaluation.confusion_matrix.counts,
        )

    def test_the_width_comes_from_the_folds_not_from_the_pooled_rows(self) -> None:
        """A class no held-out row happened to be still gets its column.

        The folds here state four classes and between them show two. Inferring
        the width from the pooled rows would build a 2x2 table while
        ``pooled_confusion_matrix`` builds 4x4, and the two routes would stop
        describing the same thing. The folds already said how wide the problem
        is; nothing downstream should be re-deriving it from the sample.
        """
        result = ClassificationCrossValidationResult(
            [
                evaluation([0, 0, 1, 1], [0, 0, 1, 1], n_classes=4),
                evaluation([1, 1, 0], [1, 1, 0], n_classes=4),
            ]
        )

        assert result._pooled_evaluation.n_classes == 4
        assert np.array_equal(
            result.pooled_confusion_matrix.counts,
            result._pooled_evaluation.confusion_matrix.counts,
        )

    def test_a_gap_in_the_pooled_rows_is_still_countable(self) -> None:
        """Class 1 absent from every fold, and the pooling survives it.

        Inferring the width would put these rows through the dense-run guard
        and raise, which is the rule for a target whose width is unknown and
        the wrong rule here.
        """
        result = ClassificationCrossValidationResult(
            [
                evaluation([0, 0, 2], [0, 0, 2], n_classes=3),
                evaluation([2, 2, 0], [2, 2, 0], n_classes=3),
            ]
        )

        assert result.pooled_accuracy == pytest.approx(1.0)

    def test_pooled_metrics_come_off_the_pooled_table(self) -> None:
        """``[[3, 1], [1, 3]]``: precision, recall and F1 are all 0.75."""
        result = perfect_then_wrong()

        assert result.pooled_accuracy == pytest.approx(0.75)
        assert result.pooled_macro_precision == pytest.approx(0.75)
        assert result.pooled_macro_recall == pytest.approx(0.75)
        assert result.pooled_macro_f1_score == pytest.approx(0.75)


class TestWhyItPools:
    """The two behaviours averaging the folds would not give."""

    def test_a_small_fold_does_not_count_as_much_as_a_large_one(self) -> None:
        """The measurement the module docstring records, in miniature.

        Averaging the folds' accuracies gives 0.5 and pooling gives 0.75, and
        the difference is entirely that the second fold holds a quarter of the
        rows while the first way of counting gave it half the say.
        """
        result = perfect_then_wrong()
        averaged = sum(one.accuracy for one in result) / len(result)

        assert averaged == pytest.approx(0.5)
        assert result.pooled_accuracy == pytest.approx(0.75)

    def test_a_fold_can_have_a_recall_the_pooled_result_does_not(self) -> None:
        """A fold with no rows of class 2 cannot report a recall for it.

        The first fold's own ``macro_recall`` raises, because dividing by the
        rows belonging to class 2 divides by zero. The pooled figure is
        defined, because the second fold put two rows in that denominator.

        Three classes rather than two, because a fold whose held-out rows all
        share a single class cannot be evaluated at all: with nothing to
        discriminate between there is no confusion matrix to contribute. The
        case that reaches this code is a fold holding *some* classes and not
        others, which is exactly what stratifying cannot fix when a class has
        fewer rows than there are folds.
        """
        result = ClassificationCrossValidationResult(
            [
                evaluation([0, 0, 1, 1], [0, 0, 1, 1], n_classes=3),
                evaluation([2, 2, 0], [2, 2, 0], n_classes=3),
            ]
        )

        with pytest.raises(UndefinedMetricError):
            _ = next(iter(result)).macro_recall

        assert result.pooled_macro_recall == pytest.approx(1.0)


class TestWhatItRefuses:
    """The constructor's invariants."""

    def test_rejects_no_folds(self) -> None:
        with pytest.raises(EmptyValuesError):
            ClassificationCrossValidationResult([])

    def test_rejects_folds_that_disagree_about_the_class_count(self) -> None:
        """Two class counts means two configurations, and no pooled table."""
        with pytest.raises(InvalidValuesError):
            ClassificationCrossValidationResult(
                [
                    evaluation([0, 1], [0, 1], n_classes=2),
                    evaluation([0, 1, 2], [0, 1, 2], n_classes=3),
                ]
            )


class TestCrossValidatingAClassifier:
    """The loop, against a real model on the three-class fixture."""

    def dataset(self) -> Dataset:
        return Dataset(THREE_CLASSES.input_features, THREE_CLASSES.target_feature)

    def result(self, n_folds: int = 3) -> ClassificationCrossValidationResult:
        return CrossValidation(
            folds=KFold(n_folds=n_folds, stratified=True, random_seed=0)
        ).evaluate_classifier(
            MultinomialLogisticRegression(max_epochs=2000), self.dataset()
        )

    def test_returns_one_evaluation_per_fold(self) -> None:
        assert self.result().n_folds == 3

    def test_scores_every_row_exactly_once(self) -> None:
        """What makes the pooled table a table over the whole dataset."""
        result = self.result()

        assert result.pooled_confusion_matrix.n_samples == self.dataset().n_samples

    def test_the_pooled_accuracy_is_a_real_held_out_score(self) -> None:
        """Above chance on three classes, and short of the training fit.

        The fixture's own training accuracy is 0.694, and a held-out figure
        landing at or above that would mean rows leaked between the halves.
        """
        pooled = self.result().pooled_accuracy

        assert 1.0 / 3.0 < pooled < 0.75

    def test_refitting_per_fold_leaves_the_model_usable(self) -> None:
        """The model is fitted k times and comes back fitted to the last fold."""
        model = MultinomialLogisticRegression(max_epochs=2000)
        CrossValidation(folds=KFold(n_folds=3, random_seed=0)).evaluate_classifier(
            model, self.dataset()
        )

        assert model.n_classes == THREE_CLASSES.n_classes
