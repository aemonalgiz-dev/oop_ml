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
from oop_ml.core.tree.criterion import ClassificationCriterion
from oop_ml.core.tree.impurity import Impurity
from oop_ml.core.tree.node import DecisionNode, LeafNode, TreeNode
from oop_ml.core.tree.search import (
    SplitCandidate,
    SplitRejection,
    SplitSearch,
)
from oop_ml.core.tree.split import GAIN_TIE_TOLERANCE, Split
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
    classification_criterion: ClassificationCriterion = Field(
        default=ClassificationCriterion.GINI
    )

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
    def _impurity(self) -> Impurity:
        """The measure this task scores splits with."""
        return self.classification_criterion.impurity

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

    @staticmethod
    def _candidate_thresholds(column: FloatArray) -> FloatArray:
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

    def split_search(
        self, feature_matrix: FloatArray, target_values: FloatArray
    ) -> SplitSearch:
        """Every candidate this node considered, kept or not, and why.

        The observed route. :meth:`_best_split` answers the same question and
        discards its working; this keeps all of it, in the order it was
        scanned, so a step-by-step explanation has something to walk.

        Deliberately slower. It scores candidates that :meth:`_best_split`
        excludes without scoring -- a split leaving a child too small still
        gets a gain here, because "excluded despite scoring well" is the
        case worth seeing and hiding the number would conceal it. Sized for
        looking at, not for the twenty-thousand-row fit in an application.

        Parameters
        ----------
        feature_matrix:
            ``(n_rows, n_features)`` for the rows at this node.
        target_values:
            ``(n_rows,)``, aligned with them.

        Returns
        -------
        SplitSearch
            Iterable over every candidate. ``search.best`` is the same answer
            :meth:`_best_split` returns, and a test asserts that.
        """
        assert self._feature_names is not None

        candidates: list[SplitCandidate] = []

        for index, name in enumerate(self._feature_names):
            column = feature_matrix[:, index]

            for threshold in self._candidate_thresholds(column):
                goes_left = column < threshold
                rows_left = int(goes_left.sum())
                rows_right = int(column.size) - rows_left

                gain = self._impurity.gain(
                    target_values,
                    target_values[goes_left],
                    target_values[~goes_left],
                )

                if (
                    rows_left < self.min_samples_leaf
                    or rows_right < self.min_samples_leaf
                ):
                    rejection = SplitRejection.TOO_FEW_ROWS
                elif gain == 0.0:
                    rejection = SplitRejection.NO_GAIN
                elif gain < self.min_impurity_decrease:
                    rejection = SplitRejection.BELOW_MINIMUM_DECREASE
                else:
                    rejection = SplitRejection.ADMITTED

                candidates.append(
                    SplitCandidate(
                        Split(index, name, float(threshold), gain),
                        rejection,
                        rows_left,
                        rows_right,
                    )
                )

        return SplitSearch(candidates)

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

        assert self._feature_names is not None

        n_rows = int(target_values.size)
        if n_rows < 2:
            return None

        best: Split | None = None

        for index, name in enumerate(self._feature_names):
            column = feature_matrix[:, index]

            # Sort once, then every cut on this feature is read off one swept
            # array. Scoring candidates one at a time recomputes both
            # children's impurity from scratch, which makes a feature cost
            # O(n^2); this makes it O(n log n), and measured on one node it is
            # the difference between a second and a millisecond.
            order = np.argsort(column, kind="stable")
            sorted_column = column[order]
            gains = self._impurity.gains_at_every_prefix(target_values[order])

            # Entry i of gains is the cut leaving i + 1 rows on the left. Such
            # a cut is only a real question where the sorted value actually
            # changes -- between two equal values there is no threshold that
            # separates them -- and only legal where both children clear
            # min_samples_leaf.
            rows_left = np.arange(1, n_rows)
            eligible = (
                (sorted_column[:-1] < sorted_column[1:])
                & (rows_left >= self.min_samples_leaf)
                & (n_rows - rows_left >= self.min_samples_leaf)
            )
            if not eligible.any():
                continue

            # argmax takes the first maximum, so a tie within a feature keeps
            # the lower threshold, and the strict comparison below keeps the
            # earlier feature. Same tie rule as scoring them one at a time.
            scored = np.where(eligible, gains, -np.inf)
            highest = float(scored.max())
            tolerance = GAIN_TIE_TOLERANCE * max(1.0, abs(highest))

            # The first candidate within a tolerance of the best, not the
            # numerically largest: on a tie the lower threshold wins, and the
            # recorded search has to reach the same answer.
            at = int(np.argmax(scored >= highest - tolerance))
            gain = float(gains[at])

            if gain == 0.0 or gain < self.min_impurity_decrease:
                continue

            threshold = (sorted_column[at] + sorted_column[at + 1]) / 2.0
            candidate = Split(index, name, float(threshold), gain)
            if candidate.beats(best):
                best = candidate

        return best

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
        if depth == self.max_depth:
            return self._leaf(target_values)

        best_split = self._best_split(feature_matrix, target_values)

        if best_split is None:
            return self._leaf(target_values)

        if best_split.gain < self.min_impurity_decrease:
            return self._leaf(target_values)

        send_left = best_split.sends_left(feature_matrix)

        if len(send_left) < self.min_samples_split:
            return self._leaf(target_values)

        return DecisionNode(
            split=best_split,
            left=self._grow(
                feature_matrix[send_left], target_values[send_left], depth + 1
            ),
            right=self._grow(
                feature_matrix[~send_left], target_values[~send_left], depth + 1
            ),
            n_samples=len(target_values),
            impurity=self._impurity.of(target_values),
        )

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
        matrix = np.ascontiguousarray(feature_set.feature_matrix, dtype=np.float64)
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
