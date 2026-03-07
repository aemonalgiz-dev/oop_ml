"""Spec for ConfusionMatrix and ClassificationEvaluation -- red until the
metrics land.

The confusion counts are plumbing and are already implemented, so those tests
pass now and guard the rest. Everything derived from them is a stub, and each
metric is specified over several inputs rather than one, because the interesting
cases are the degenerate ones: a model that never fires, a target with no
positives, and the unbalanced case where accuracy flatters a useless classifier.
"""

import pytest

from oop_ml.evaluation.classification import (
    ClassificationEvaluation,
    ConfusionMatrix,
)
from oop_ml.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonBinaryLabelsError,
    NonEqualArrayLengthError,
    UndefinedMetricError,
)

# actual, predicted, (tp, tn, fp, fn)
COUNTED_CASES = [
    ([1, 1, 0, 0], [1, 0, 1, 0], (1, 1, 1, 1)),
    ([1, 1, 1, 1], [1, 1, 1, 1], (4, 0, 0, 0)),
    ([0, 0, 0, 0], [0, 0, 0, 0], (0, 4, 0, 0)),
    ([1, 0, 1, 0], [0, 1, 0, 1], (0, 0, 2, 2)),
]


class TestConfusionMatrixCounts:
    @pytest.mark.parametrize(("actual", "predicted", "expected"), COUNTED_CASES)
    def test_counts_the_four_outcomes(self, actual, predicted, expected):
        matrix = ClassificationEvaluation(actual, predicted).confusion_matrix

        counted = (
            matrix.true_positives,
            matrix.true_negatives,
            matrix.false_positives,
            matrix.false_negatives,
        )
        assert counted == expected

    @pytest.mark.parametrize(("actual", "predicted", "expected"), COUNTED_CASES)
    def test_the_counts_partition_the_rows(self, actual, predicted, expected):
        matrix = ClassificationEvaluation(actual, predicted).confusion_matrix

        assert matrix.n_samples == len(actual)

    def test_reports_the_marginals(self):
        matrix = ClassificationEvaluation([1, 1, 0, 0], [1, 0, 1, 0]).confusion_matrix

        assert matrix.predicted_positive == 2
        assert matrix.actual_positive == 2

    def test_negative_counts_are_rejected(self):
        with pytest.raises(InvalidValuesError):
            ConfusionMatrix(
                true_positives=-1,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
            )


class TestValidation:
    def test_misaligned_lengths_raise(self):
        with pytest.raises(NonEqualArrayLengthError):
            ClassificationEvaluation([1, 0, 1], [1, 0])

    def test_empty_input_raises(self):
        with pytest.raises(EmptyValuesError):
            ClassificationEvaluation([], [])

    @pytest.mark.parametrize(
        ("actual", "predicted"),
        [([0, 1, 2], [0, 1, 1]), ([0, 1, 1], [0, 0.5, 1]), ([0, 1], [-1, 1])],
        ids=["three-class actual", "probability predicted", "negative label"],
    )
    def test_non_binary_input_raises(self, actual, predicted):
        with pytest.raises(NonBinaryLabelsError):
            ClassificationEvaluation(actual, predicted)


class TestAccuracy:
    @pytest.mark.parametrize(
        ("actual", "predicted", "expected"),
        [
            ([1, 1, 0, 0], [1, 0, 1, 0], 0.5),
            ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),
            ([1, 0, 1, 0], [0, 1, 0, 1], 0.0),
            ([1, 0, 0, 0], [0, 0, 0, 0], 0.75),
        ],
    )
    def test_share_of_predictions_that_were_correct(self, actual, predicted, expected):
        evaluation = ClassificationEvaluation(actual, predicted)

        assert evaluation.accuracy == pytest.approx(expected)

    def test_accuracy_flatters_a_useless_model_on_unbalanced_data(self):
        # Nineteen negatives and one positive. Predicting all-negative finds
        # nothing at all and still scores 0.95, which is the whole reason
        # precision and recall exist.
        actual = [1] + [0] * 19
        predicted = [0] * 20

        assert ClassificationEvaluation(actual, predicted).accuracy == pytest.approx(
            0.95
        )


class TestPrecisionAndRecall:
    @pytest.mark.parametrize(
        ("actual", "predicted", "expected"),
        [
            ([1, 1, 0, 0], [1, 0, 1, 0], 0.5),
            ([1, 1, 0, 0], [1, 1, 0, 0], 1.0),
            ([1, 1, 0, 0], [0, 0, 1, 1], 0.0),
        ],
    )
    def test_precision_is_share_of_positive_calls_that_were_right(
        self, actual, predicted, expected
    ):
        assert ClassificationEvaluation(actual, predicted).precision == pytest.approx(
            expected
        )

    @pytest.mark.parametrize(
        ("actual", "predicted", "expected"),
        [
            ([1, 1, 0, 0], [1, 0, 1, 0], 0.5),
            ([1, 1, 0, 0], [1, 1, 0, 0], 1.0),
            ([1, 1, 1, 0], [0, 0, 0, 0], 0.0),
        ],
    )
    def test_recall_is_share_of_real_positives_that_were_found(
        self, actual, predicted, expected
    ):
        assert ClassificationEvaluation(actual, predicted).recall == pytest.approx(
            expected
        )

    def test_precision_is_undefined_when_nothing_was_called_positive(self):
        # Not zero. A model that never fires has made no wrong positive call,
        # and it has made no right one either.
        with pytest.raises(UndefinedMetricError):
            _ = ClassificationEvaluation([1, 0, 1, 0], [0, 0, 0, 0]).precision

    def test_recall_is_undefined_when_there_were_no_positives(self):
        with pytest.raises(UndefinedMetricError):
            _ = ClassificationEvaluation([0, 0, 0, 0], [0, 1, 0, 1]).recall

    def test_lowering_the_bar_trades_precision_for_recall(self):
        # Same truth, two predictors: the cautious one calls a single positive,
        # the eager one calls everything positive.
        actual = [1, 1, 1, 0, 0, 0]
        cautious = ClassificationEvaluation(actual, [1, 0, 0, 0, 0, 0])
        eager = ClassificationEvaluation(actual, [1, 1, 1, 1, 1, 1])

        assert cautious.precision > eager.precision
        assert cautious.recall < eager.recall


class TestF1Score:
    @pytest.mark.parametrize(
        ("actual", "predicted", "expected"),
        [
            ([1, 1, 0, 0], [1, 0, 1, 0], 0.5),
            ([1, 1, 0, 0], [1, 1, 0, 0], 1.0),
            # precision 1/3, recall 1.0 -> harmonic mean 0.5
            ([1, 0, 0], [1, 1, 1], 0.5),
        ],
    )
    def test_harmonic_mean_of_precision_and_recall(self, actual, predicted, expected):
        assert ClassificationEvaluation(actual, predicted).f1_score == pytest.approx(
            expected
        )

    def test_harmonic_mean_refuses_to_reward_sacrificing_one_side(self):
        # Precision 1.0, recall 0.25. The arithmetic mean would be a
        # respectable 0.625; the harmonic mean is 0.4 and is the honest number.
        evaluation = ClassificationEvaluation([1, 1, 1, 1], [1, 0, 0, 0])

        assert evaluation.precision == pytest.approx(1.0)
        assert evaluation.recall == pytest.approx(0.25)
        assert evaluation.f1_score == pytest.approx(0.4)


class TestSpecificity:
    @pytest.mark.parametrize(
        ("actual", "predicted", "expected"),
        [
            ([1, 1, 0, 0], [1, 0, 1, 0], 0.5),
            ([0, 0, 0, 0], [0, 0, 0, 0], 1.0),
            ([0, 0, 1, 1], [1, 1, 1, 1], 0.0),
        ],
    )
    def test_share_of_real_negatives_correctly_rejected(
        self, actual, predicted, expected
    ):
        assert ClassificationEvaluation(actual, predicted).specificity == pytest.approx(
            expected
        )

    def test_specificity_is_undefined_when_there_were_no_negatives(self):
        with pytest.raises(UndefinedMetricError):
            _ = ClassificationEvaluation([1, 1, 1], [1, 0, 1]).specificity
