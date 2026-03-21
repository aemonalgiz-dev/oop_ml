"""Averaging many regressors fitted on resamples of the same data.

The plainest ensemble there is, and it wraps anything. Fit ``n_members`` copies
of a prototype on bootstrap resamples, average their predictions, done.

Whether it helps depends entirely on what it wraps. Averaging cuts variance and
leaves bias alone, so a member that is already stable gains nothing -- bagging a
linear regression is almost exactly a linear regression, because every resample
produces nearly the same line. Bagging a deep tree is transformative, because
every resample produces a visibly different tree.

That is the rule for this whole family: the member has to be unstable for the
average to have anything to work with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from oop_ml.core.base.ensemble import AveragingEnsemble, AveragingMember
from oop_ml.core.base.estimator import Regressor
from oop_ml.core.data.feature import Feature
from oop_ml.core.types import FloatArray
from oop_ml.regression.trees.decision_tree_regressor import DecisionTreeRegressor


class BaggingRegressor(AveragingEnsemble, Regressor[Sequence[Feature], Feature]):
    """Predict a quantity as the mean of many members' predictions.

    Parameters
    ----------
    base_model:
        The prototype every member is a deep copy of. Defaults to an unpruned
        tree, which is the member this family is built for: averaging fights
        variance, so the ideal member is the one with the most of it and the
        least bias.
    n_members, random_seed:
        Inherited from
        :class:`~oop_ml.core.base.ensemble.AveragingEnsemble`.
    """

    base_model: Regressor = DecisionTreeRegressor()

    def _prototype(self, position: int) -> AveragingMember:
        return self.base_model

    def _member_answer(
        self, member: AveragingMember, input_values: Sequence[Feature]
    ) -> FloatArray:
        assert isinstance(member, Regressor)
        return member.predict(input_values)

    def _combine(self, member_predictions: FloatArray) -> FloatArray:
        """The mean of what the members said.

        Every member counts equally. Weighting them by training performance is
        tempting and wrong: they were each fitted on a different resample, so a
        member that scores better may only have drawn an easier one.

        Parameters
        ----------
        member_predictions:
            ``(n_members, n_queries)``.

        Returns
        -------
        FloatArray
            ``(n_queries,)``.
        """
        return member_predictions.mean(axis=0)

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit every member on its own resample.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        return self._fit_members(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """The members' mean prediction, one value per row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return self._combine(self._member_predictions(input_values))
