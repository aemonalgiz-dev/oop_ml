"""Regression by carving the feature space into boxes and averaging each one.

The prediction surface is piecewise constant -- the same shape a neighbour
regressor produces, arrived at from the opposite direction. k-NN gets there by
averaging whichever rows happen to be near a query; a tree gets there by
deciding in advance where the boundaries go and averaging within them. One
defers every decision to prediction time, the other makes them all at fit time.

Why the leaf predicts a mean
----------------------------
Not by convention. The criterion is variance about the node's own mean, and the
mean is precisely the constant minimising that, so a leaf reporting it is a leaf
making the smallest squared error a single number can make for its rows. Change
the criterion to absolute error and the right constant becomes the median --
which is why that is a member of
:class:`~oop_ml.core.tree.criterion.RegressionCriterion` rather than a flag.

The identity behind a split
---------------------------
::

    parent variance = weighted mean of child variances
                      + variance of the child means

A split minimises the first term, so its gain *is* the second. Choosing a
threshold is choosing where to maximise the spread between two group means,
which is a one-way analysis of variance run at every candidate cut.

It cannot extrapolate either
----------------------------
Query beyond the edge of the training data and the answer is whatever the
outermost box holds, forever -- the same flattening a neighbour model does, and
for a related reason: there is no slope anywhere in the model to continue. Least
squares would extend its line, confidently and just as unfoundedly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np

from oop_ml.core.base.estimator import Regressor
from oop_ml.core.base.tree_model import TreeModel
from oop_ml.core.data.feature import Feature
from oop_ml.core.tree.criterion import RegressionCriterion
from oop_ml.core.tree.impurity import Impurity
from oop_ml.core.tree.node import LeafNode
from oop_ml.core.types import FloatArray


class DecisionTreeRegressor(TreeModel, Regressor[Sequence[Feature], Feature]):
    """Predict a quantity as the mean of its box.

    Parameters
    ----------
    criterion:
        How splits are scored. See
        :class:`~oop_ml.core.tree.criterion.RegressionCriterion`.
    max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease:
        The stopping rules, inherited from
        :class:`~oop_ml.core.base.tree_model.TreeModel`. Left at their defaults
        the tree grows until every leaf is pure, which on continuous targets
        means one leaf per distinct row.
    """

    criterion: RegressionCriterion = RegressionCriterion.SQUARED_ERROR

    @property
    def _impurity(self) -> Impurity:
        return self.criterion.impurity

    def _leaf(self, target_values: FloatArray) -> LeafNode:
        """A leaf predicting the mean of these targets.

        Parameters
        ----------
        target_values:
            ``(n_rows,)``, the targets of every training row that reached this
            node. Never empty.

        Returns
        -------
        LeafNode
            Carrying the mean as its prediction, the row count, and the
            impurity of these targets under the configured criterion.
        """
        return LeafNode(
            prediction=np.mean(target_values),
            n_samples=len(target_values),
            impurity=self._impurity.of(target_values),
        )

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Grow the tree. Unlike a neighbour model, all the work is here.

        Parameters
        ----------
        input_values:
            One or more predictor columns, all the same length as the target.
        target_values:
            The response being regressed on.

        Returns
        -------
        Self
            This model, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        return self._fit_tree(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """The mean of the box each row falls in, one value per row.

        A handful of comparisons per row and no reference to the training data,
        which is the trade a tree makes against a neighbour model.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return np.array(
            [leaf.prediction for leaf in self._leaves_for(input_values)],
            dtype=np.float64,
        )
