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

A warning this design does not yet enforce
------------------------------------------
Anything *learned* from the data must be learned inside the fold. Standardizing
before splitting lets the training rows see the test rows' mean, which is
leakage, and it flatters every score computed afterwards. Until a pipeline
object exists to make that structural, the caller has to fit the
:class:`~oop_ml.preprocessing.standardizer.Standardizer` on ``split.training`` and only
transform ``split.testing``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from pydantic import BaseModel, ConfigDict

from oop_ml.core.base.estimator import Regressor
from oop_ml.core.data.feature import Feature
from oop_ml.core.evaluation.regression import RegressionEvaluation
from oop_ml.core.exceptions import EmptyValuesError
from oop_ml.model_selection.dataset import Dataset
from oop_ml.model_selection.splitting import KFold


class CrossValidationResult:
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
            f"CrossValidationResult(n_folds={self.n_folds}, "
            f"mean_r2_score={self.mean_r2_score:.4f})"
        )


class CrossValidation(BaseModel):
    """Fit a model on each training fold and score it on the held-out one.

    Parameters
    ----------
    folds:
        The splitter that divides the rows. Defaults to five-fold.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    folds: KFold = KFold()

    def evaluate(
        self, model: Regressor[Sequence[Feature], Feature], dataset: Dataset
    ) -> CrossValidationResult:
        """Cross-validate ``model`` over ``dataset``.

        For each split the folds produce: fit the model on ``split.training``, then
        evaluate it on ``split.testing``, which holds rows the fit never saw. Collect
        one :class:`~oop_ml.core.evaluation.regression.RegressionEvaluation`
        per fold and return them as a :class:`CrossValidationResult`.

        ``Regressor.evaluate(input_values, actual_values)`` produces the
        evaluation in one call, so each fold is a fit followed by an evaluate.

        The model is refitted from scratch on every fold. That is the point:
        ``k`` independent estimates of how this *configuration* performs, not
        one model incrementally trained.

        No numpy is involved anywhere here. It is a loop over
        ``self.folds.split(dataset)``, and for each ``split``:

        1. ``model.fit(training.input_features, training.target_feature)``
        2. ``model.evaluate(testing.input_features, testing.target_feature)``

        naming ``training = split.training`` and ``testing = split.testing``
        first, since each is used twice.

        Collect the evaluations and return
        ``CrossValidationResult(evaluations)``.

        Returns
        -------
        CrossValidationResult
            One evaluation per fold, plus the summary across them.

        Raises
        ------
        TooFewValuesError
            If the dataset has fewer rows than folds.
        """

        regression_evaluations = []

        for split in self.folds.split(dataset):
            model.fit(split.training.input_features, split.training.target_feature)
            regression_evaluations.append(
                model.evaluate(
                    split.testing.input_features, split.testing.target_feature
                )
            )

        return CrossValidationResult(regression_evaluations)
