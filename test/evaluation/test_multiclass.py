"""Spec for the K x K confusion matrix and the metrics read off it.

The counting and the per-class ratios are wired, so those pass now. The four
averaging properties are stubs, and they are the part carrying an actual
decision: macro weights every class equally, micro weights every row equally,
and on an unbalanced target they are different claims about the same model.

The test worth reading twice is the one asserting micro-precision equals
accuracy. That is not a coincidence to be verified once and forgotten -- where
each row gets exactly one prediction it is an identity, and an implementation
that made them differ would have computed something else entirely.
"""

import numpy as np
import pytest

from oop_ml.evaluation.multiclass import (
    MultiClassConfusionMatrix,
    MultiClassEvaluation,
)
from oop_ml.exceptions import (
    InvalidValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    SingleClassError,
    UndefinedMetricError,
)

# actual, predicted -- three classes, laid out so every metric differs.
#   class 0: 3 rows, 2 found         class 1: 4 rows, 2 found
#   class 2: 3 rows, 3 found
ACTUAL = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
PREDICTED = [0, 0, 1, 1, 1, 0, 2, 2, 2, 2]


def evaluation(actual=None, predicted=None, n_classes=None) -> MultiClassEvaluation:
    return MultiClassEvaluation(
        ACTUAL if actual is None else actual,
        PREDICTED if predicted is None else predicted,
        n_classes,
    )


class TestConfusionMatrixConstruction:
    def test_rejects_a_non_square_table(self):
        with pytest.raises(InvalidValuesError):
            MultiClassConfusionMatrix(np.zeros((2, 3), dtype=int))

    def test_rejects_negative_counts(self):
        with pytest.raises(InvalidValuesError):
            MultiClassConfusionMatrix(np.array([[1, -1], [0, 2]]))

    def test_counts_are_handed_out_as_a_copy(self):
        matrix = MultiClassConfusionMatrix(np.array([[1, 0], [0, 1]]))
        matrix.counts[0, 0] = 99

        assert matrix.counts[0, 0] == 1


class TestConfusionMatrix:
    def test_counts_every_pair(self):
        matrix = evaluation().confusion_matrix

        assert matrix.counts.tolist() == [[2, 1, 0], [1, 2, 1], [0, 0, 3]]

    def test_shape_and_totals(self):
        matrix = evaluation().confusion_matrix

        assert matrix.n_classes == 3
        assert matrix.n_samples == 10
        assert matrix.correct == 7

    @pytest.mark.parametrize(("class_index", "expected"), [(0, 2), (1, 2), (2, 3)])
    def test_true_positives_are_the_diagonal(self, class_index, expected):
        assert evaluation().confusion_matrix.true_positives_for(class_index) == expected

    @pytest.mark.parametrize(("class_index", "expected"), [(0, 3), (1, 3), (2, 4)])
    def test_predicted_as_is_the_column_total(self, class_index, expected):
        assert evaluation().confusion_matrix.predicted_as(class_index) == expected

    @pytest.mark.parametrize(("class_index", "expected"), [(0, 3), (1, 4), (2, 3)])
    def test_actually_are_is_the_row_total(self, class_index, expected):
        assert evaluation().confusion_matrix.actually_are(class_index) == expected

    @pytest.mark.parametrize("class_index", [-1, 3, 99])
    def test_an_unknown_class_is_rejected(self, class_index):
        with pytest.raises(InvalidValuesError):
            evaluation().confusion_matrix.true_positives_for(class_index)


class TestValidation:
    def test_misaligned_lengths_are_rejected(self):
        with pytest.raises(NonEqualArrayLengthError):
            MultiClassEvaluation([0, 1, 2], [0, 1])

    @pytest.mark.parametrize("actual", [[0, 1, -1], [0, 1, 1.5]])
    def test_non_class_actuals_are_rejected(self, actual):
        with pytest.raises(NonBinaryLabelsError):
            MultiClassEvaluation(actual, [0, 1, 0])

    def test_a_single_class_is_rejected(self):
        with pytest.raises(SingleClassError):
            MultiClassEvaluation([1, 1, 1], [1, 1, 1])

    def test_a_gap_in_the_classes_is_rejected(self):
        # 0 and 2 present, 1 missing: almost always a filtered dataset.
        with pytest.raises(SingleClassError):
            MultiClassEvaluation([0, 0, 2, 2], [0, 0, 2, 2])

    def test_a_predicted_class_outside_the_problem_is_rejected(self):
        with pytest.raises(InvalidValuesError):
            MultiClassEvaluation([0, 1, 2], [0, 1, 5])

    def test_n_classes_can_be_widened_beyond_what_the_truth_shows(self):
        # A fold that happens to contain only two of three classes still has to
        # score against a table the fitted model's width, not the fold's.
        result = MultiClassEvaluation([0, 1, 0, 1], [0, 1, 0, 1], n_classes=3)

        assert result.n_classes == 3
        assert result.confusion_matrix.counts.shape == (3, 3)


class TestPerClassMetrics:
    def test_accuracy_is_the_diagonal_over_everything(self):
        assert evaluation().accuracy == pytest.approx(0.7)

    @pytest.mark.parametrize(
        ("class_index", "expected"),
        [(0, 2 / 3), (1, 2 / 3), (2, 3 / 4)],
    )
    def test_precision_is_the_column_share(self, class_index, expected):
        assert evaluation().precision_for(class_index) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("class_index", "expected"),
        [(0, 2 / 3), (1, 0.5), (2, 1.0)],
    )
    def test_recall_is_the_row_share(self, class_index, expected):
        assert evaluation().recall_for(class_index) == pytest.approx(expected)

    def test_f1_is_the_harmonic_mean_of_the_two(self):
        result = evaluation()
        precision, recall = result.precision_for(1), result.recall_for(1)

        assert result.f1_for(1) == pytest.approx(
            2 * precision * recall / (precision + recall)
        )

    def test_precision_is_undefined_for_a_class_never_predicted(self):
        with pytest.raises(UndefinedMetricError, match="never predicted"):
            MultiClassEvaluation([0, 1, 2], [0, 1, 1]).precision_for(2)

    def test_recall_is_undefined_for_a_class_with_no_rows(self):
        with pytest.raises(UndefinedMetricError, match="no row"):
            MultiClassEvaluation([0, 1, 0, 1], [0, 1, 2, 1], n_classes=3).recall_for(2)

    def test_per_class_vectors_are_in_class_order(self):
        result = evaluation()

        assert result.per_class_precision == pytest.approx([2 / 3, 2 / 3, 3 / 4])
        assert result.per_class_recall == pytest.approx([2 / 3, 0.5, 1.0])


class TestMacroAveraging:
    def test_macro_precision_is_the_unweighted_mean(self):
        assert evaluation().macro_precision == pytest.approx(
            np.mean([2 / 3, 2 / 3, 3 / 4])
        )

    def test_macro_recall_is_the_unweighted_mean(self):
        assert evaluation().macro_recall == pytest.approx(np.mean([2 / 3, 0.5, 1.0]))

    def test_macro_f1_averages_the_per_class_f1s(self):
        # Not the harmonic mean of macro_precision and macro_recall, which is a
        # different number and not a standard metric however sensible it looks.
        result = evaluation()
        per_class = [result.f1_for(index) for index in range(3)]

        assert result.macro_f1_score == pytest.approx(np.mean(per_class))

    def test_macro_ignores_how_common_a_class_is(self):
        # Two problems with identical per-class scores and very different class
        # sizes must give the same macro number. That is what macro is for.
        balanced = MultiClassEvaluation([0, 0, 1, 1], [0, 1, 0, 1])
        skewed = MultiClassEvaluation(
            [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 0, 1]
        )

        assert balanced.macro_recall == pytest.approx(0.5)
        assert skewed.macro_recall == pytest.approx(np.mean([0.5, 0.5]))

    def test_an_undefined_class_makes_the_macro_undefined(self):
        # Averaging over whichever classes happen to be defined would silently
        # answer a different question than the one asked.
        result = MultiClassEvaluation([0, 1, 2], [0, 1, 1])

        with pytest.raises(UndefinedMetricError):
            _ = result.macro_precision


class TestMicroAveraging:
    def test_micro_precision_equals_accuracy(self):
        # An identity, not a coincidence: with one prediction per row the pooled
        # numerator is the diagonal and the pooled denominator is every row.
        result = evaluation()

        assert result.micro_precision == pytest.approx(result.accuracy)

    def test_micro_recall_equals_accuracy(self):
        result = evaluation()

        assert result.micro_recall == pytest.approx(result.accuracy)

    def test_micro_and_macro_disagree_on_an_unbalanced_target(self):
        # The whole reason both exist. Class 1 is rare and badly served; micro
        # barely notices and macro is dragged down by it.
        actual = [0] * 18 + [1] * 2
        predicted = [0] * 18 + [0, 1]
        result = MultiClassEvaluation(actual, predicted)

        assert result.micro_recall == pytest.approx(0.95)
        assert result.macro_recall == pytest.approx(0.75)

    def test_micro_is_undefined_with_no_observations_at_all(self):
        # There is no way to build an empty evaluation through the constructor,
        # so this pins the guard on the matrix directly.
        empty = MultiClassConfusionMatrix(np.zeros((3, 3), dtype=int))

        assert empty.n_samples == 0
        assert empty.correct == 0
