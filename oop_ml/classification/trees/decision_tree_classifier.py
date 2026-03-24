"""Classification by asking questions until the rows in a box mostly agree.

Multi-class comes free
----------------------
The same thing that was true of a neighbour vote is true here, and for the same
reason. There is no reference class, no flat ridge in a likelihood, no
one-vs-rest wrapper and no softmax, because all of that machinery existed to
make *parameters* identifiable and there are no parameters. Counting rows in a
leaf works identically for two classes and twenty.

What the boundary looks like
----------------------------
A union of axis-aligned boxes, which is a strictly different vocabulary from
the hyperplane a logistic model draws. Where a class sits in a ring around
another one, logistic regression fits the best straight line available and the
best straight line available is worth nothing; a tree encloses the region with
four cuts. Where the true boundary is a rotated plane, the positions reverse and
the tree needs a staircase of splits to say what one line says exactly.

Interactions are native
-----------------------
Because a question is asked inside a region an earlier question already carved
out, "studying only helps if you slept" is expressible without anyone naming an
interaction. A linear model needs that product handed to it as a column. This is
the tree's real advantage over the linear classifiers here, and it is also
precisely the mechanism that lets an unstopped tree carve out a box containing
one row.

Probabilities, honestly
-----------------------
``predict_probabilities`` returns each class's share of the leaf's training
rows. A leaf holding four rows can only report multiples of a quarter, and a
pure leaf reports 1.0 no matter how few rows it holds -- which is the more
misleading of the two, since an unstopped tree makes every leaf pure by
construction and would then claim total certainty everywhere.

Ties
----
Two classes can hold the same count in a leaf. The tie has to break somehow and
the rule should be stated rather than inherited from whatever ``argmax``
happens to do: lowest class index wins, which is arbitrary and deterministic,
and deterministic is the property that matters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import MultiClassClassifier
from oop_ml.core.base.tree_model import TreeModel
from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ProbabilityMatrix
from oop_ml.core.tree.criterion import ClassificationCriterion
from oop_ml.core.tree.impurity import Impurity
from oop_ml.core.tree.node import ClassificationLeaf, LeafNode


class DecisionTreeClassifier(
    TreeModel, MultiClassClassifier[Sequence[Feature], Feature]
):
    """Predict a class by majority vote among the training rows of its box.

    Built on :class:`~oop_ml.core.base.estimator.MultiClassClassifier` rather
    than ``Classifier``, because two classes are not a special case here -- the
    count is the same operation either way, and the binary base would force a
    single ``predict_probability`` where this naturally produces one column per
    class.

    Parameters
    ----------
    criterion:
        How splits are scored. See
        :class:`~oop_ml.core.tree.criterion.ClassificationCriterion`.
    max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease:
        The stopping rules, inherited from
        :class:`~oop_ml.core.base.tree_model.TreeModel`. Left at their defaults
        the tree grows until every leaf is pure, which is memorisation.
    n_known_classes:
        How many classes the problem has, stated by the caller, **and stating
        it changes what the target is allowed to hold** -- the same rule
        :class:`~oop_ml.core.evaluation.multiclass.MultiClassEvaluation` has
        for the same reason. Left as ``None`` the width is inferred, so the
        classes must run densely from zero. Stated, the target need only hold
        whole positions inside the width: a class may be missing, and even a
        single-class target fits, as a lone leaf. That is the case a bagged
        ensemble's member is in -- a bootstrap resample of imbalanced data
        misses a rare class routinely, and the ensemble states the width so
        every member answers with probability rows of the same shape.
    """

    criterion: ClassificationCriterion = ClassificationCriterion.GINI
    n_known_classes: int | None = Field(default=None, ge=2)

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

    @property
    def _impurity(self) -> Impurity:
        return self.criterion.impurity

    def _validated_target(self, target_values: Feature) -> Column:
        """The target, insisted upon as whole class positions ``0 .. K-1``.

        Where classification tightens the contract, mirroring what
        ``LinearClassifier`` does for the linear frame and what
        ``KNearestNeighboursClassifier`` does for the neighbour one.

        Raises
        ------
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the width is being inferred and fewer than two classes are
            present or they leave a gap. With ``n_known_classes`` stated the
            dense run is not required -- see the field's docstring.
        InvalidValuesError
            If ``n_known_classes`` is stated and the target names a class at
            or beyond it.
        """
        target_column = super()._validated_target(target_values)

        if self.n_known_classes is None:
            target_column.check_is_label_encoded()
        else:
            target_column.check_are_class_positions(self.n_known_classes)

        return target_column

    def _leaf(self, target_values: Column) -> LeafNode:
        """A leaf predicting the most common class among these targets.

        Ties go to the lowest class index. The prediction is returned on the
        same ``0 .. K-1`` scale the target uses, so it goes straight into a
        :class:`~oop_ml.core.evaluation.multiclass.MultiClassEvaluation` like
        any other prediction.

        Every fitted class needs an entry in the shares even when no row in
        this leaf belongs to it, or the probability matrix would change width
        depending on which leaf a query happened to reach. The width is the
        number of classes the *fit* saw, not the number present in this leaf --
        a pure leaf still reports a full-width row.

        Read that width from the private ``self._n_classes``, **not** from the
        public ``self.n_classes``. This method runs during growth, and growth
        happens before ``_mark_fitted``, so the public property is still
        raising ``NotFittedError`` at the moment a leaf is built. It is the
        same ordering trap the neighbour models hit three times from the other
        direction, and it is invisible until something actually calls ``fit``.

        Parameters
        ----------
        target_values:
            ``(n_rows,)`` of class positions, for every training row that
            reached this node. Never empty.

        Returns
        -------
        ClassificationLeaf
            Carrying the majority class, the shares over all ``n_classes``, the
            row count, and the impurity of these targets under the configured
            criterion.
        """
        assert self._n_classes is not None

        # Width is the classes the FIT saw, not the ones present here, so a
        # pure leaf still reports a full row and predict_probabilities can
        # stack them. _n_classes rather than n_classes: this runs during
        # growth, and the public property is still refusing until fit ends.
        counts = np.bincount(
            target_values.values.astype(np.int64), minlength=self._n_classes
        )

        return ClassificationLeaf(
            # argmax takes the first maximum, so the documented tie-break --
            # lowest class index -- arrives without being written.
            prediction=float(np.argmax(counts)),
            class_shares=counts / target_values.n_samples,
            n_samples=target_values.n_samples,
            impurity=self._impurity.of(target_values),
        )

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Grow the tree, and record how many classes it spans.

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
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the width is inferred and the target holds fewer than two
            classes or leaves a gap.
        InvalidValuesError
            If ``n_known_classes`` is stated and the target names a class at
            or beyond it.
        """
        # Before _fit_tree, because _leaf reads n_classes on the way down and
        # a leaf built during growth would otherwise size its shares off a
        # model that does not yet know how wide they should be. The stated
        # width wins over the inferred one: a resample missing the top class
        # would otherwise infer too narrow and stack against nothing.
        target_column = self._validated_target(target_values)
        self._n_classes = (
            target_column.n_classes
            if self.n_known_classes is None
            else self.n_known_classes
        )

        return self._fit_tree(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The majority class of the box each row falls in, as ``0.0 .. K-1``.

        Ties within a leaf break to the lowest class index.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        return Predictions.already_checked(
            np.array(
                [leaf.prediction for leaf in self._leaves_for(input_values)],
                dtype=np.float64,
            )
        )

    def predict_probabilities(
        self, input_values: Sequence[Feature]
    ) -> ProbabilityMatrix:
        """Each class's share of the leaf's rows, ``(n_queries, K)``.

        Rows sum to 1. Read the module docstring before treating these as
        calibrated: a pure leaf reports 1.0 regardless of how few rows built
        it, and an unstopped tree makes every leaf pure.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        leaves = self._leaves_for(input_values)

        # Every leaf a fitted classifier grows is a ClassificationLeaf, because
        # _leaf is the only thing that makes one. The assert states that rather
        # than trusting it silently.
        shares = []
        for leaf in leaves:
            assert isinstance(leaf, ClassificationLeaf)
            shares.append(leaf.class_shares)

        return ProbabilityMatrix(
            np.array(shares, dtype=np.float64).reshape(len(leaves), self.n_classes)
        )
