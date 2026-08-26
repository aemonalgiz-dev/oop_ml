"""Regression by averaging the nearest rows.

Theory
------
Every regressor so far assumed a shape. Least squares assumes a hyperplane;
polynomial features assume a curve of a chosen degree; a penalty assumes
smaller coefficients are more likely to be right. This assumes nothing about
the shape at all. It says only that rows near each other in feature space tend
to have similar targets, and answers each query with the mean of its ``k``
nearest neighbours' targets.

That is a weaker assumption, and weaker assumptions cut both ways. Where the
truth is a plane, least squares will beat this with a handful of numbers while
this carries the whole training set around. Where the truth is some shape
nobody guessed, this finds it and least squares cannot.

What the prediction looks like
------------------------------
A step function. Move a query point slightly and its neighbour set usually does
not change, so the prediction does not move either; cross a boundary where one
neighbour swaps for another and it jumps. The fitted surface is piecewise
constant, with one flat region per distinct neighbour set, and no amount of
data makes it smooth -- only larger ``k`` does, by averaging over more of them.

It also cannot extrapolate. Ask about a row beyond the edge of the training
data and its neighbours are all on the same side, so the answer is their mean:
the surface flattens out rather than continuing whatever trend was there. Least
squares would happily extend the line, correctly or otherwise. Which of those
behaviours you want is a question about the problem, not about the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from oop_ml.core.base.estimator import Regressor
from oop_ml.core.base.neighbour_model import NeighbourModel
from oop_ml.core.data.feature import Feature
from oop_ml.core.types import FloatArray


class KNearestNeighboursRegressor(
    NeighbourModel, Regressor[Sequence[Feature], Feature]
):
    """Predict a quantity as the mean of its nearest neighbours' targets.

    Parameters
    ----------
    n_neighbours:
        How many neighbours are averaged. Inherited from
        :class:`~oop_ml.core.base.neighbour_model.NeighbourModel`.
    metric:
        What "near" means. Standardise the features first, or whichever column
        happens to be measured in larger units will decide every answer.
    """

    def _combine(self, neighbour_targets: FloatArray) -> FloatArray:
        """The mean of each query's neighbours' targets.

        Every neighbour counts the same regardless of how near it is. The
        obvious refinement is to weight by inverse distance so a close
        neighbour speaks louder, which is a real improvement and also a
        different model -- it has no ``k`` at which it stops caring, and it
        needs a rule for the case where a query sits exactly on a training row
        and the weight is infinite. Unweighted first.

        Parameters
        ----------
        neighbour_targets:
            ``(n_queries, n_neighbours)``, nearest first.

        Returns
        -------
        FloatArray
            One predicted value per query.
        """
        # axis=1 walks across a row -- one query's k neighbours -- and leaves
        # one number per query. axis=0 would average each neighbour position
        # across queries, which is arithmetic nobody asked for.
        return neighbour_targets.mean(axis=1)

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Remember the rows. There is nothing else to do.

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
        TooFewValuesError
            If there are fewer rows than ``n_neighbours``.
        """
        return self._remember(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """The mean target of the nearest rows, one value per query.

        All of the work is here rather than in ``fit``, which is the trade a
        non-parametric model makes.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return self._combine(self._neighbour_targets(input_values))
