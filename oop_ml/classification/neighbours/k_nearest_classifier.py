"""Classification by asking the nearest rows to vote.

Theory
------
Logistic regression draws one boundary and commits to it being a hyperplane.
This draws no boundary at all. The decision surface is whatever falls out of
where the training rows sit -- piecewise linear, arbitrarily shaped, and
capable of enclosing a region entirely inside another class if that is what the
data shows. Nothing about the model constrains it, which is its strength and
its whole failure mode at once.

Multi-class comes free
----------------------
Note what is *not* here. There is no reference class, no flat ridge in the
likelihood, no one-vs-rest wrapper, and no softmax. A vote does not care how
many candidates are on the ballot: count the classes among the ``k`` nearest
and take the largest count. Two classes and twenty are the same code.

That is the one place this family is simpler than logistic regression rather
than merely different, and it is worth noticing why. Logistic regression needed
all that machinery because it was estimating parameters, and parameters for K
classes are not identifiable without pinning one down. With no parameters there
is nothing to pin.

Probabilities, honestly
-----------------------
``predict_probabilities`` returns the share of the ``k`` neighbours belonging
to each class. That is a probability in the sense that it is non-negative and
sums to one, and it is not a probability in the sense that a fitted model
offers: with ``k = 5`` the only values obtainable are 0, 0.2, 0.4, 0.6, 0.8 and
1.0, so a row cannot be reported as 0.53 confident no matter how the data
looks. The resolution is ``1/k``, and reading these as calibrated confidence is
a mistake the coarseness should make obvious.

Ties
----
Two classes can hold the same count -- guaranteed whenever ``k`` is even and
there are two classes, and possible otherwise. The tie has to break somehow and
the rule should be stated rather than inherited from whatever ``argmax``
happens to do. Lowest class index wins here, which is arbitrary but
deterministic, and deterministic is the property that matters: the alternative
is a model that answers differently on identical input.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import PrivateAttr

from oop_ml.core.base.estimator import MultiClassClassifier
from oop_ml.core.base.neighbour_model import NeighbourModel
from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.types import FloatArray


class KNearestNeighboursClassifier(
    NeighbourModel, MultiClassClassifier[Sequence[Feature], Feature]
):
    """Predict a class by majority vote among the nearest neighbours.

    Built on :class:`~oop_ml.core.base.estimator.MultiClassClassifier` rather
    than ``Classifier``, because two classes are not a special case here -- the
    vote is the same operation either way, and the binary base would force a
    single ``predict_probability`` where this naturally produces one column per
    class.

    Parameters
    ----------
    n_neighbours:
        How many neighbours vote. An even value guarantees ties are reachable
        with two classes; the tie rule is stated on :meth:`predict`.
    metric:
        What "near" means. Standardise the features first.
    """

    _n_classes: int | None = PrivateAttr(default=None)

    @property
    def n_classes(self) -> int:
        """How many classes the remembered rows span.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._n_classes is not None
        return self._n_classes

    def _validated_target(self, target_values: Feature) -> Column:
        """The target, insisted upon as whole class positions ``0 .. K-1``.

        Where classification tightens the contract, mirroring what
        ``LinearClassifier`` does for the linear frame.

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

    def _combine(self, neighbour_targets: FloatArray) -> FloatArray:
        """The most common class among each query's neighbours.

        Ties go to the lowest class index. Return floats on the same ``0..K-1``
        scale the target uses, so the answer goes straight into a
        :class:`~oop_ml.core.evaluation.multiclass.MultiClassEvaluation` like
        any other prediction.

        Parameters
        ----------
        neighbour_targets:
            ``(n_queries, n_neighbours)`` of class positions, nearest first.

        Returns
        -------
        FloatArray
            One predicted class per query, as ``0.0 .. K-1``.
        """
        # argmax returns the FIRST maximum it meets, so a tie already resolves
        # to the lowest class index. That is the documented rule arriving for
        # free rather than a happy accident, which is why nothing here breaks
        # ties by hand.
        return np.argmax(self._class_shares(neighbour_targets), axis=1).astype(
            np.float64
        )

    def _class_shares(self, neighbour_targets: FloatArray) -> FloatArray:
        """What share of each query's neighbours belongs to each class.

        Parameters
        ----------
        neighbour_targets:
            ``(n_queries, n_neighbours)`` of class positions.

        Returns
        -------
        FloatArray
            ``(n_queries, n_classes)``, every row summing to 1.

        Notes
        -----
        Every class needs a column even when no neighbour voted for it, or the
        matrix silently changes width depending on which rows were queried.
        Counting into an array of the fitted width is what keeps column ``k``
        meaning class ``k``, and it is why ``minlength`` is not optional below.

        Tallying votes is a scatter-add -- the same cell comes up repeatedly,
        because several neighbours vote the same way, so plain
        ``counts[rows, columns] += 1`` would keep only the last write.
        ``np.add.at`` is the accumulating form that gets this right, and it is
        also slow: it gives up buffering to guarantee the ordering, and cost
        5.7x what the route below does over 20000 queries.

        The trick is that a two-dimensional tally is a one-dimensional tally
        of ``row * n_classes + class``, because that expression visits every
        cell exactly once -- it is the address arithmetic numpy itself does
        under a C-contiguous index. Flattened that way the whole thing is one
        ``bincount``, which is a single pass in C, and the shape comes back
        with a reshape that copies nothing.
        """
        n_queries, n_voters = neighbour_targets.shape

        offsets = np.repeat(np.arange(n_queries) * self.n_classes, n_voters)
        cells = neighbour_targets.astype(np.int64).ravel() + offsets

        counts = np.bincount(cells, minlength=n_queries * self.n_classes)

        return counts.reshape(n_queries, self.n_classes) / n_voters

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Remember the rows and how many classes they span.

        Parameters
        ----------
        input_values:
            One or more predictor columns, all the same length as the target.
        target_values:
            The classes, as whole positions running ``0 .. K - 1``.

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
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap.
        """
        self._n_classes = self._validated_target(target_values).n_classes

        return self._remember(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """The class most of the nearest rows belong to, as ``0.0 .. K-1``.

        Ties break to the lowest class index.

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

    def predict_probabilities(self, input_values: Sequence[Feature]) -> FloatArray:
        """Each class's share of the nearest neighbours, ``(n_queries, K)``.

        Rows sum to 1, but read the module docstring before treating these as
        calibrated: with ``k`` neighbours the only reachable values are
        multiples of ``1/k``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return self._class_shares(self._neighbour_targets(input_values))
