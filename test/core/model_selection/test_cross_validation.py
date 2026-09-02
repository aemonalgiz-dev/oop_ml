"""Spec for CrossValidation -- red until ``evaluate`` lands.

The dataset below is an exact plane over 20 rows, so every fold can be fitted
perfectly and every held-out R^2 must be 1.0. That makes the arithmetic
checkable: any fold scoring below 1.0 means rows leaked between the halves or a
fold was fitted on the wrong rows.
"""

import pytest

from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.evaluation.regression import RegressionEvaluation
from oop_ml.core.exceptions import EmptyValuesError, TooFewValuesError
from oop_ml.core.model_selection.cross_validation import (
    CrossValidation,
    RegressionCrossValidationResult,
)
from oop_ml.core.model_selection.splitting import KFold
from oop_ml.numpy.regression.least_squares.multiple_feature_regression import (
    MultipleLinearRegression,
)
from oop_ml.numpy.regression.penalised.ridge_regression import RidgeRegression


def exact_dataset(n_samples: int = 20) -> Dataset:
    """y = 1 + 2*x1 + 3*x2 exactly, so a correct fit scores 1.0 on any subset."""
    first = [value for value in range(n_samples)]
    second = [(value * 7) % 11 for value in range(n_samples)]

    return Dataset(
        [Feature("x1", first), Feature("x2", second)],
        Feature(
            "y",
            [1 + 2 * one + 3 * two for one, two in zip(first, second, strict=True)],
        ),
    )


class TestRegressionCrossValidationResult:
    def make_result(self) -> RegressionCrossValidationResult:
        return RegressionCrossValidationResult(
            [
                RegressionEvaluation([1, 2, 3], [1, 2, 3]),
                RegressionEvaluation([1, 2, 3], [2, 2, 2]),
            ]
        )

    def test_counts_its_folds(self):
        result = self.make_result()

        assert result.n_folds == 2
        assert len(result) == 2

    def test_iterates_the_fold_evaluations(self):
        assert all(
            isinstance(evaluation, RegressionEvaluation)
            for evaluation in self.make_result()
        )

    def test_mean_r2_averages_the_folds(self):
        # 1.0 and 0.0
        assert self.make_result().mean_r2_score == pytest.approx(0.5)

    def test_mean_squared_error_averages_the_folds(self):
        # 0.0 and 2/3
        assert self.make_result().mean_squared_error == pytest.approx(1 / 3)

    def test_spread_reports_best_minus_worst(self):
        assert self.make_result().r2_score_spread == pytest.approx(1.0)

    def test_no_folds_raises(self):
        with pytest.raises(EmptyValuesError):
            RegressionCrossValidationResult([])


class TestCrossValidation:
    def test_defaults_to_five_folds(self):
        assert CrossValidation().folds.n_folds == 5

    @pytest.mark.parametrize("n_folds", [2, 4, 5])
    def test_produces_one_evaluation_per_fold(self, n_folds):
        result = CrossValidation(folds=KFold(n_folds=n_folds, random_seed=0)).evaluate(
            MultipleLinearRegression(), exact_dataset()
        )

        assert result.n_folds == n_folds

    def test_an_exactly_fittable_plane_scores_one_on_every_fold(self):
        # Each fold is fitted without its held-out rows and still predicts them
        # perfectly, because the relationship is exact.
        result = CrossValidation(folds=KFold(random_seed=0)).evaluate(
            MultipleLinearRegression(), exact_dataset()
        )

        for evaluation in result:
            assert evaluation.r2_score == pytest.approx(1.0)

        assert result.mean_r2_score == pytest.approx(1.0)
        assert result.r2_score_spread == pytest.approx(0.0)

    def test_every_row_is_scored_exactly_once_across_the_folds(self):
        dataset = exact_dataset()

        result = CrossValidation(folds=KFold(random_seed=0)).evaluate(
            MultipleLinearRegression(), dataset
        )

        scored = sum(evaluation.n_samples for evaluation in result)

        assert scored == dataset.n_samples

    def test_the_model_is_refitted_per_fold_not_left_fitted_from_before(self):
        # A fresh model goes in unfitted; cross-validation must not depend on
        # the caller having fitted it first.
        model = MultipleLinearRegression()

        CrossValidation(folds=KFold(random_seed=0)).evaluate(model, exact_dataset())

        assert model.is_fitted is True

    def test_a_penalised_model_scores_worse_on_an_exact_plane(self):
        # Ridge trades fit for stability. On data with no noise there is nothing
        # to stabilise, so the penalty can only cost accuracy -- which is what
        # held-out scoring should reveal.
        dataset = exact_dataset()
        folds = KFold(random_seed=0)

        ordinary = CrossValidation(folds=folds).evaluate(
            MultipleLinearRegression(), dataset
        )
        penalised = CrossValidation(folds=folds).evaluate(
            RidgeRegression(penalty=10.0), dataset
        )

        assert penalised.mean_r2_score < ordinary.mean_r2_score

    def test_choosing_a_penalty_by_held_out_score(self):
        # The whole point: nothing in the derivations gives you the penalty, so
        # you measure it. On noiseless data the answer must be the smallest.
        dataset = exact_dataset()
        folds = KFold(random_seed=0)

        scored = {
            penalty: CrossValidation(folds=folds)
            .evaluate(RidgeRegression(penalty=penalty), dataset)
            .mean_r2_score
            for penalty in (0.0, 1.0, 10.0, 100.0)
        }

        assert max(scored, key=lambda penalty: scored[penalty]) == 0.0

    def test_too_few_rows_for_the_folds_raises(self):
        with pytest.raises(TooFewValuesError):
            CrossValidation(folds=KFold(n_folds=10)).evaluate(
                MultipleLinearRegression(), exact_dataset(4)
            )
