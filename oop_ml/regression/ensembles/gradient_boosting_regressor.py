"""Adding shallow trees one at a time, each fitted to what is still wrong.

Bagging and boosting look alike from a distance -- both hold many trees -- and
agree about almost nothing. Bagging fits its members in parallel on resamples
and averages them, which fights variance. Boosting fits its members in sequence
on the previous ones' mistakes and adds them up, which fights bias.

That reversal decides the member. Averaging wants the member with the most
variance and the least bias, so it takes an unpruned tree. Boosting wants the
opposite -- a stump, depth two or three -- because a deep member would explain
the residuals in one round and leave every later round fitting noise, with
nothing averaging that noise away.

Why "gradient"
--------------
Fitting a member to ``target - prediction`` is the recipe everyone learns, and
it is a special case. Squared error's derivative with respect to the prediction
is ``-(target - prediction)``, so the residual *is* the negative gradient. State
it that way and the same machinery covers losses whose residual is not a
subtraction -- absolute error, which yields the sign, or log loss, which yields
``target - probability``. Each round is one step of gradient descent, taken in
the space of functions rather than the space of parameters, with the member
serving as the step direction and the learning rate as the step size.

The learning rate
-----------------
Adding only a fraction of each member buys the same shrinkage a ridge penalty
buys, and for the same reason: committing less per step leaves less room to
commit to noise. It trades computation directly for accuracy -- 0.01 with a
thousand rounds beats 0.5 with twenty -- and rate and round count have to move
together, since halving one and not doubling the other changes how far the
ensemble travels in total.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import Field

from oop_ml.core.base.ensemble import BoostingEnsemble, BoostingMember
from oop_ml.core.base.estimator import Regressor
from oop_ml.core.data.feature import Feature
from oop_ml.core.types import FloatArray
from oop_ml.regression.trees.decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingRegressor(
    BoostingEnsemble, Regressor[Sequence[Feature], Feature]
):
    """Predict a quantity as a constant plus a sum of shrunken corrections.

    Parameters
    ----------
    max_depth:
        How deep each round's tree may go. Small on purpose -- 3 lets a member
        express a three-way interaction and no more, which is about the depth
        where this family stops improving on tabular data.
    min_samples_split, min_samples_leaf:
        Passed to every member.
    n_rounds, learning_rate:
        Inherited from
        :class:`~oop_ml.core.base.ensemble.BoostingEnsemble`.
    """

    max_depth: int | None = Field(default=3, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)

    def _prototype(self, round_number: int) -> BoostingMember:
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

    def _residuals(
        self, target_values: FloatArray, predictions: FloatArray
    ) -> FloatArray:
        """What squared error leaves unexplained: the plain difference.

        The negative gradient of ``(target - prediction)^2 / 2`` with respect
        to the prediction, which for this loss and no other simplifies to a
        subtraction.

        Parameters
        ----------
        target_values, predictions:
            ``(n_rows,)`` each, aligned.

        Returns
        -------
        FloatArray
            ``(n_rows,)``, what the next member should be fitted on.
        """
        return target_values - predictions

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit the rounds in sequence.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        return self._fit_rounds(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """The starting constant plus every round's shrunken contribution.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        ordered = list(self._matched_rows(input_values))

        running = np.full(len(ordered[0]), self.initial_prediction, dtype=np.float64)
        for member in self.members:
            running += self.learning_rate * member.predict(ordered)

        return running
