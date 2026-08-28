"""Averaging many classifiers fitted on resamples of the same data.

The same machinery as the regression side, with one decision that is not
obvious: the members are averaged on their *probabilities*, not on their votes.

Counting votes throws away how sure each member was. Suppose ten members and a
row where six of them put class 1 at 0.51 and four put class 0 at 0.99. The vote
says class 1, six to four. The averaged probability says class 0, 0.70 to 0.30,
because the four were nearly certain and the six were barely committed. The
second answer uses information the first one discarded, and it is the reason
this family reads ``predict_probabilities`` from its members.

It also makes the ensemble's own probabilities better than any member's. A
single tree's leaf reports 1.0 whenever it is pure, which an unstopped tree
makes true everywhere; averaging a hundred such claims across members that
disagree produces a graded number where each member had only a certainty.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import PrivateAttr

from oop_ml.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.core.base.ensemble import AveragingEnsemble, AveragingMember
from oop_ml.core.base.estimator import MultiClassClassifier
from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ProbabilityMatrix
from oop_ml.core.ensemble.member_predictions import MemberPredictions
from oop_ml.core.evaluation.multiclass import MultiClassEvaluation
from oop_ml.core.types import FloatArray


class BaggingClassifier(
    AveragingEnsemble, MultiClassClassifier[Sequence[Feature], Feature]
):
    """Predict a class by averaging many members' probability matrices.

    Multi-class rather than binary, for the reason
    :class:`~oop_ml.classification.trees.decision_tree_classifier.DecisionTreeClassifier`
    gives: averaging probability columns is the same operation for two classes
    and twenty, so there is nothing for a binary special case to simplify.

    Parameters
    ----------
    base_model:
        The prototype every member is a deep copy of. Defaults to an unpruned
        tree.
    n_members, random_seed:
        Inherited from
        :class:`~oop_ml.core.base.ensemble.AveragingEnsemble`.
    """

    base_model: MultiClassClassifier = DecisionTreeClassifier()

    _n_classes: int | None = PrivateAttr(default=None)

    @property
    def n_classes(self) -> int:
        """How many classes the fit saw.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    def _prototype(self, position: int) -> AveragingMember:
        return self.base_model

    def _validated_target(self, target_values: Feature) -> Column:
        """The target, insisted upon as whole class positions ``0 .. K-1``.

        Raises
        ------
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If fewer than two classes are present, or they leave a gap.
        """
        target_column = super()._validated_target(target_values)
        target_column.check_is_label_encoded()

        return target_column

    def _member_answer(
        self, member: AveragingMember, input_values: Sequence[Feature]
    ) -> FloatArray:
        assert isinstance(member, MultiClassClassifier)
        return member.predict_probabilities(input_values).values

    def _combine(self, member_predictions: MemberPredictions) -> FloatArray:
        """The class with the highest averaged probability, per query.

        Ties go to the lowest class index, matching the rule a single tree's
        leaf follows.

        Parameters
        ----------
        member_predictions:
            ``(n_members, n_queries, n_classes)``.

        Returns
        -------
        FloatArray
            ``(n_queries,)`` of class positions, as floats on the ``0 .. K-1``
            scale the target uses.
        """
        consensus = member_predictions.values.mean(axis=0)
        return consensus.argmax(-1).astype(np.float64)

    def out_of_bag_evaluate(self) -> MultiClassEvaluation:
        """Score the fit against rows each member never drew.

        The bagged answer to a train/test split, and the class count comes
        from the fitted model rather than from the covered rows, so a rare
        class missing from them still produces a table of the right size.

        Read the caveats in
        :mod:`~oop_ml.core.ensemble.out_of_bag` before trusting the number.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        estimate = self.out_of_bag_estimate()
        assert self._training is not None
        actual = self._training.target_feature.values[estimate.covered]

        return MultiClassEvaluation(
            actual, estimate.covered_predictions, self.n_classes
        )

    def out_of_bag_score(self) -> float:
        """Accuracy against the rows each member never drew.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return self.out_of_bag_evaluate().accuracy

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit every member on its own resample, recording the class count.

        The count is read before the members are fitted, for the same reason
        ``DecisionTreeClassifier.fit`` reads it first: a member fitted on a
        resample can easily miss a rare class, and its probability matrix would
        then be narrower than the others and refuse to stack.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap.
        """
        self._n_classes = self._validated_target(target_values).n_classes

        return self._fit_members(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The most probable class per row, averaged across members.

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

    def predict_probabilities(
        self, input_values: Sequence[Feature]
    ) -> ProbabilityMatrix:
        """The members' mean probability matrix, ``(n_queries, n_classes)``.

        Rows sum to 1, because each member's does and a mean of things that sum
        to 1 sums to 1.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return ProbabilityMatrix(
            self._member_predictions(input_values).values.mean(axis=0)
        )
