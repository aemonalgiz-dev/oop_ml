"""The frame for models that answer by asking a sequence of yes/no questions.

The second non-parametric family here, and the interesting thing is how little
it shares with the first. A neighbour model does nothing at ``fit`` and carries
the whole training set into every prediction. A tree does an expensive search at
``fit``, throws the data away, and then answers with a handful of comparisons --
about twelve at depth twelve, whether it was grown on a thousand rows or a
million. Same family, opposite costs.

It also drops both of k-NN's liabilities. A split compares one column against a
threshold, so nothing is summed across features and no column can drown another
by being measured in larger numbers; standardising a tree's inputs changes
precisely nothing. And there is no distance, so the curse of dimensionality
arrives in a different and gentler form.

What it pays for that
---------------------
**The boundary is axis-aligned.** Every split is parallel to an axis, so a
diagonal boundary is only reachable as a staircase and needs many splits to
approximate what one line would say. Where the real structure happens to be
axis-aligned this costs nothing; where it is a rotated plane it costs a lot.

**The search is greedy, and cannot be otherwise.** Finding the optimal tree is
NP-hard, so each node takes the best split available now and never reconsiders.
That is usually fine and occasionally fatal: on a parity target every single
first split has a gain of exactly zero, while a depth-two tree would be perfect.
A greedy search stalls on the first move of a structure it could finish easily.

**Interactions come free, which cuts both ways.** Because a question is asked
inside the region another question already carved out, a tree represents
"studying only helps if you slept" natively, where a linear model needs the
product term handed to it. The same mechanism is what lets a deep tree carve out
a region containing one row, which is memorisation wearing a respectable hat.

Stopping is the whole of the regularisation
-------------------------------------------
Grown without limit, a tree puts one leaf per distinct row: training error zero,
test error dreadful, and exactly the failure ``k=1`` produced for a neighbour
model. Four hyperparameters stand between the recursion and that outcome, and
they are the only defence this model has.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.tree.impurity import Impurity
from oop_ml.core.tree.node import LeafNode, TreeNode
from oop_ml.core.tree.split import Split
from oop_ml.core.types import FloatArray


class TreeModel(Fittable):
    """A model that recursively splits its rows and answers from a leaf.

    Parameters
    ----------
    max_depth:
        How many questions may be asked of any one row. ``None`` for no limit,
        which is the setting that memorises.
    min_samples_split:
        A node holding fewer rows than this becomes a leaf without trying.
    min_samples_leaf:
        A split leaving either child with fewer rows than this is not
        considered at all. Unlike the others this constrains the *search*
        rather than only the stop, because a split has to be rejected before it
        is chosen, not after.
    min_impurity_decrease:
        A split buying less than this is not worth making. Zero admits any
        split with a strictly positive gain.
    """

    max_depth: int | None = Field(default=None, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    min_impurity_decrease: float = Field(default=0.0, ge=0.0)

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _root: TreeNode | None = PrivateAttr(default=None)

    @property
    def root(self) -> TreeNode:
        """The top of the fitted tree.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._root is not None
        return self._root

    @property
    def depth(self) -> int:
        """How many questions the longest path asks. A single leaf is zero.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        return self.root.depth

    @property
    def n_leaves(self) -> int:
        """How many regions the fitted tree carves the feature space into.

        Worth reading beside ``n_samples``: as the two approach each other the
        model is closer to having memorised than to having learned.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        return self.root.n_leaves

    def describe(self) -> str:
        """The fitted tree as indented text.

        Most of the reason to choose a tree is that you can read it, so this is
        part of the model rather than something a caller writes against the
        node internals.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        """
        return "\n".join(self.root.description_lines())

    @property
    @abstractmethod
    def _impurity(self) -> Impurity:
        """The measure this task scores splits with."""

    @abstractmethod
    def _leaf(self, target_values: FloatArray) -> LeafNode:
        """The leaf that answers for a node holding these targets.

        The one line separating a tree regressor from a tree classifier, and
        the only thing besides the criterion that a concrete model supplies.

        Parameters
        ----------
        target_values:
            The targets of every training row that reached this node. Never
            empty -- ``min_samples_leaf`` is at least 1 and the search rejects
            any split that would empty a child.

        Returns
        -------
        LeafNode
            Carrying the prediction, the row count and the impurity.
        """

    def _candidate_thresholds(self, column: FloatArray) -> FloatArray:
        """Every threshold worth trying on one column.

        Sorting the distinct values, the only cuts that separate rows
        differently from each other are the midpoints between consecutive
        ones -- so there are at most ``n - 1`` of them however many rows there
        are, and a constant column offers none at all.

        Midpoints rather than the observed values themselves so that the
        boundary sits between the two training rows it separates instead of on
        top of one of them, which is a slightly better guess about where an
        unseen row belongs.

        Parameters
        ----------
        column:
            One feature's values for the rows at this node.

        Returns
        -------
        FloatArray
            ``(n_distinct - 1,)``, ascending. Empty when the column is
            constant, which is the signal that this feature cannot split
            these rows at all.
        """
        distinct = np.unique(column)
        if distinct.size < 2:
            return np.empty(0, dtype=np.float64)

        return (distinct[:-1] + distinct[1:]) / 2.0

    def _best_split(
        self, feature_matrix: FloatArray, target_values: FloatArray
    ) -> Split | None:
        """The highest-gain split of these rows, or ``None`` if there is none.

        An exhaustive search: every feature, every candidate threshold from
        :meth:`_candidate_thresholds`, scored by
        :meth:`~oop_ml.core.tree.impurity.Impurity.gain`. There is nothing
        clever available here -- the gain of one candidate says nothing about
        the gain of the next, so they all have to be tried.

        A candidate is only admissible when **both** children hold at least
        ``min_samples_leaf`` rows. That check belongs here rather than in
        :meth:`_grow`, because a split has to be excluded before it can win,
        not rejected after.

        Ties go to the first candidate met, scanning features in fitted order
        and thresholds ascending. Arbitrary, but deterministic, and
        deterministic is the property that matters -- the alternative is a
        model that grows a different tree on identical input.

        Parameters
        ----------
        feature_matrix:
            ``(n_rows, n_features)`` for the rows at this node, in fitted
            column order.
        target_values:
            ``(n_rows,)``, aligned with those rows.

        Returns
        -------
        Split | None
            The best admissible split, carrying the gain it achieved, or
            ``None`` when no candidate is admissible -- every column constant,
            or every split leaving a child too small.

        Notes
        -----
        ``None`` rather than a zero-gain ``Split``, because "there is no
        question to ask" and "the best question buys nothing" are different
        facts about the node and the caller acts on them the same way only by
        coincidence.
        """
        raise NotImplementedError

    def _grow(
        self, feature_matrix: FloatArray, target_values: FloatArray, depth: int
    ) -> TreeNode:
        """Build the subtree for these rows, recursively.

        This node becomes a leaf, via :meth:`_leaf`, when **any** of:

        - ``depth`` has reached ``max_depth``
        - it holds fewer than ``min_samples_split`` rows
        - its impurity is zero, so there is nothing left to separate
        - :meth:`_best_split` finds no admissible candidate
        - the best candidate's gain is not strictly positive
        - that gain is below ``min_impurity_decrease``

        Otherwise it becomes a
        :class:`~oop_ml.core.tree.node.DecisionNode` holding that split, and
        both children are grown at ``depth + 1``.

        Parameters
        ----------
        feature_matrix:
            ``(n_rows, n_features)`` for the rows reaching this node.
        target_values:
            ``(n_rows,)``, aligned with them.
        depth:
            How many questions have already been asked to get here. The root is
            zero.

        Returns
        -------
        TreeNode
            A ``LeafNode`` or a ``DecisionNode``, whichever the rules above
            call for.

        Notes
        -----
        Requiring the gain to be *strictly* positive is what makes a parity
        target unreachable: there, every first split scores exactly zero and
        the recursion stops at the root, even though two levels would separate
        the classes perfectly. That is not a bug to be worked around by
        admitting zero-gain splits -- doing so would grow a great deal of
        useless structure everywhere else for the sake of one pathological
        shape. It is the price of being greedy, and it is why ensembles that
        randomise the split choice exist.
        """
        raise NotImplementedError

    def _fit_tree(
        self, input_values: Sequence[Feature], target_values: Feature
    ) -> Self:
        """Validate the inputs and grow the tree. The whole of fitting.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        self._feature_names = tuple(feature.name for feature in feature_set)

        # Row-major here, unlike every linear model in the library. A tree
        # never forms X.T @ v; it reads one column at a time while searching
        # and then slices rows to hand each child its share, and slicing rows
        # is what C order makes contiguous.
        matrix = np.ascontiguousarray(feature_set.feature_matrix)
        self._root = self._grow(matrix, self._validated_target(target_values).values, 0)

        self._mark_fitted()
        return self

    def _matched_rows(self, input_values: Sequence[Feature]) -> FloatArray:
        """The query rows, in the column order the fit saw.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        assert self._feature_names is not None

        return FeatureSet.matching(self._feature_names, input_values).feature_matrix

    def _leaves_for(self, input_values: Sequence[Feature]) -> list[LeafNode]:
        """The leaf each query row reaches, in the order supplied.

        Everything ``predict`` needs on either task, so a concrete model is
        only ever a read off the leaf away from an answer.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        # _matched_rows carries the fitted-state guard, so it has to run before
        # any assert about what the fit stored -- otherwise an unfitted call
        # dies on an AssertionError rather than saying NotFittedError.
        rows = self._matched_rows(input_values)

        assert self._root is not None
        root = self._root

        return [root.leaf_for(row) for row in rows]

    def _validated_target(self, target_values: Feature) -> Column:
        """The target as a column, checked against whatever the task requires.

        The seam between the two tree models, mirroring the one
        :class:`~oop_ml.core.base.neighbour_model.NeighbourModel` uses. The
        default is the regression answer: a column is already numeric and
        finite, and averaging asks nothing further of it.
        """
        return target_values.column
