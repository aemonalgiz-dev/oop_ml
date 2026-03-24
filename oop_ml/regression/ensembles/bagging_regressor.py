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
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.ensemble.member_predictions import MemberPredictions
from oop_ml.core.evaluation.regression import RegressionEvaluation
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
        """The configured member, its own seed offset by position.

        A member carrying a random seed -- a tree restricting features, say --
        would otherwise be deep-copied with the identical seed into every
        member, and all of them would replay one random stream: the same
        degeneracy the forest's per-member offset exists to prevent, arrived
        at through configuration instead of code.
        """
        member_fields = type(self.base_model).model_fields
        member_seed = getattr(self.base_model, "random_seed", None)

        if "random_seed" in member_fields and member_seed is not None:
            configured = {
                name: getattr(self.base_model, name) for name in member_fields
            }

            return type(self.base_model)(
                **{**configured, "random_seed": member_seed + position}
            )

        return self.base_model

    def _member_answer(
        self, member: AveragingMember, input_values: Sequence[Feature]
    ) -> FloatArray:
        assert isinstance(member, Regressor)
        return member.predict(input_values).values

    def _combine(self, member_predictions: MemberPredictions) -> FloatArray:
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
        return member_predictions.values.mean(axis=0)

    def out_of_bag_evaluate(self) -> RegressionEvaluation:
        """Score the fit against rows each member never drew.

        The bagged answer to a train/test split. Every member missed about
        36.8% of the training set, so the ensemble already contains a held-out
        estimate; this reads it out rather than setting data aside up front.

        Read the caveats in
        :mod:`~oop_ml.core.ensemble.out_of_bag` before trusting the number.
        The short version is that it is conservative: each row is judged by
        roughly ``0.368 * n_members`` members, so it measures a smaller
        ensemble than the one you fitted.

        Rows no member missed are excluded, so this can be computed over fewer
        rows than the training set holds.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        estimate = self.out_of_bag_estimate()
        assert self._training is not None
        actual = self._training.target_feature.values[estimate.covered]

        return RegressionEvaluation(actual, estimate.covered_predictions)

    def out_of_bag_score(self) -> float:
        """R^2 against the rows each member never drew.

        A thin convenience over ``out_of_bag_evaluate``, mirroring what
        ``score`` is to ``evaluate``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return self.out_of_bag_evaluate().r2_score

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

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The members' mean prediction, one value per row.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return Predictions.already_checked(
            self._combine(self._member_predictions(input_values))
        )
