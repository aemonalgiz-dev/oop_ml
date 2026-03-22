"""How mixed a node is, which is the only thing a split is scored on.

A tree makes one decision over and over -- which ``(feature, threshold)`` pair
to split on -- and every part of that decision reduces to a single number per
node. These are the classes that produce it.

Why the measure has to be strictly concave
------------------------------------------
Because the child sizes are exactly the counts that went each way, the weighted
mean of the children's class proportions equals the parent's, identically. So
the weighted child impurity is not merely some number: it is the *chord*
between the two children, read off at the parent's own position. Gain is
therefore the vertical gap between the curve and its chord.

That turns the choice of measure into a question about geometry. A strictly
concave function lies above its chords, so any split with unequal children
yields a strictly positive gain, and better-separated children pull the chord
further down. A straight line lies *on* its chords, and the gain collapses.

Misclassification rate -- ``1 - max(p)`` -- is two straight lines meeting at
one half, which is exactly why it is not offered here. Where both children fall
on the same side of the kink the gain is not merely small, it is zero. On a
parent that is 30% class B splitting into children at 10% and 40%, weighted one
third to two thirds:

===================  ========  ========  ======
measure              parent    children  gain
===================  ========  ========  ======
misclassification      0.3000    0.3000  0.0000
gini                   0.4200    0.3800  0.0400
===================  ========  ========  ======

A split separating a 10% node from a 40% node is plainly worth making, and
accuracy scores it at nothing. The reason nobody grows trees on accuracy is
concavity rather than convention.

Gini against entropy
--------------------
Both are strictly concave, both are zero when pure and maximal when uniform,
and on two classes Gini is very nearly the quadratic approximation of entropy.
They disagree about which split is best only rarely, and where they disagree
the difference is usually noise. Gini avoids a logarithm per candidate; entropy
leans very slightly harder toward balanced children. Neither is where a tree's
accuracy comes from -- the stopping rules are.

Regression is the same machinery
--------------------------------
Swap impurity for variance and the leaf predicts a mean instead of a majority.
Nothing else changes, and there is an identity underneath::

    parent variance = weighted mean of child variances
                      + variance of the child means

The first term is what a split minimises, so the gain *is* the second term. A
regression split maximises the spread between the two group means, which is a
one-way analysis of variance run at every candidate threshold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from oop_ml.core.data.column import Column
from oop_ml.core.types import FloatArray
from oop_ml.core.validation import ValueRole


class Impurity(ABC):
    """How mixed one node's targets are, and what a split of them buys."""

    __slots__ = ()

    def of(self, target_values: Column) -> float:
        """How mixed this node is. Zero when every row agrees.

        Parameters
        ----------
        target_values:
            The targets of the rows at one node. Class positions for a
            classification measure, quantities for a regression one.

        Returns
        -------
        float
            Impurity, at or above zero, and zero exactly when the node is pure.

        Notes
        -----
        This used to take a bare array and define the empty node as zero, with
        a long note explaining why: Gini returns 1.0 on nothing, because the
        sum of no squared proportions is zero; variance returns ``nan``,
        because the mean of nothing is undefined, and a ``nan`` then makes
        every comparison against it false, so a split search would silently
        never choose that candidate.

        Taking a :class:`~oop_ml.core.data.column.Column` retires the whole
        discussion. A column cannot be empty, so the case is not defined away,
        it is unrepresentable, and no measure can be written that gets it
        wrong. That is the difference between a rule stated in a contract and
        an invariant carried by a type.

        The coercion to ``float`` stays. A measure whose formula ends in a
        numpy reduction hands back a ``float64``, which is duck-compatible
        enough that neither the tests nor the type checker notice, and which
        then travels outward into every gain and every leaf.
        """
        return float(self._of_non_empty(target_values.values))

    @abstractmethod
    def _of_non_empty(self, target_values: FloatArray) -> float:
        """How mixed this node is, given at least one row.

        The formula, and nothing else. Emptiness has already been handled by
        :meth:`of`, so this can divide by the row count without checking it.

        Parameters
        ----------
        target_values:
            ``(n_rows,)``, never empty.

        Returns
        -------
        float
            Impurity, at or above zero, and zero exactly when the node is pure.
        """

    @staticmethod
    def target_probabilities(target_values: FloatArray) -> FloatArray:
        """What share of these rows belongs to each class present.

        The step that turns a column of class labels into the ``p_k`` the Gini
        and entropy formulas are written in terms of: ``[0, 0, 1, 1]`` holds
        two rows of each class, so it becomes ``[0.5, 0.5]``.

        Returns
        -------
        FloatArray
            One entry per class *present in this node*, summing to 1 -- not one
            per row, and not one per class the fit saw. A class with no rows
            here is absent rather than zero, which is what keeps ``log2`` away
            from zero in the entropy measure.
        """
        _, counts = np.unique(target_values, return_counts=True)

        return counts / counts.sum()

    def gain(
        self,
        parent_values: Column,
        left_values: Column,
        right_values: Column,
    ) -> float:
        """How much impurity a split removes.

        The children are weighted by how many rows went each way::

            gain = impurity(parent)
                   - (n_left / n) * impurity(left)
                   - (n_right / n) * impurity(right)

        The weighting is not decoration. Without it, peeling a single row off
        into its own pure child scores perfectly every time, and the tree
        degenerates into one leaf per observation.

        Parameters
        ----------
        parent_values:
            The targets before the split.
        left_values, right_values:
            The targets after it. Together these hold every parent row exactly
            once, which is what makes the weighted mean of the children's
            proportions equal the parent's.

        Returns
        -------
        float
            The reduction in impurity. At or above zero for any strictly
            concave measure, and zero when both children have the same
            composition as the parent.
        """
        parent_impurity = self.of(parent_values)
        left_impurity = self.of(left_values)
        right_impurity = self.of(right_values)

        n = len(left_values) + len(right_values)

        return (
            parent_impurity
            - len(left_values) * left_impurity / n
            - len(right_values) * right_impurity / n
        )

    def gains_at_every_prefix(self, sorted_targets: FloatArray) -> FloatArray:
        """The gain from cutting after 1 row, after 2 rows, and so on.

        The whole reason a split search can be fast. Scoring candidates one at
        a time recomputes the impurity of both children from scratch every
        time, so one feature costs O(n^2); sorting the column once and sweeping
        the boundary upward lets each cut reuse the running counts from the one
        before it, and the feature costs O(n log n).

        This is on ``Impurity`` rather than in the search because the reuse is
        different for every measure -- class counts for Gini and entropy,
        running sums for variance -- and a search that knew which was which
        would be a search with the measures hard-coded into it.

        Parameters
        ----------
        sorted_targets:
            ``(n_rows,)``, ordered by whichever feature is being cut. The
            order is the caller's business; this only needs to know that
            cutting after position ``i`` puts the first ``i`` on the left.

        Returns
        -------
        FloatArray
            ``(n_rows - 1,)``. Entry ``i`` is the gain from a cut leaving
            ``i + 1`` rows on the left, identical to
            ``gain(targets, targets[:i + 1], targets[i + 1:])``.

        Notes
        -----
        The default here is that identical loop, so a measure the library has
        never heard of still works -- correctly, and at the O(n^2) cost the
        subclasses below exist to avoid. Every override is covered by a test
        asserting it agrees with this.
        """
        role = ValueRole.TARGET_VALUES
        parent = Column.selecting(sorted_targets, role)

        return np.array(
            [
                self.gain(
                    parent,
                    Column.selecting(sorted_targets[:cut], role),
                    Column.selecting(sorted_targets[cut:], role),
                )
                for cut in range(1, sorted_targets.size)
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _cumulative_class_counts(sorted_targets: FloatArray) -> FloatArray:
        """Class counts on the left of every cut, as ``(n_rows - 1, n_classes)``.

        One pass. The counts after ``i + 1`` rows are the counts after ``i``
        plus one, which is what makes the sweep O(n) instead of O(n^2), and
        ``cumsum`` over a one-hot encoding is how numpy says that in one call.
        """
        classes = sorted_targets.astype(np.int64)
        n_rows = classes.size
        n_classes = int(classes.max()) + 1

        one_hot = np.zeros((n_rows, n_classes), dtype=np.float64)
        one_hot[np.arange(n_rows), classes] = 1.0

        return np.cumsum(one_hot, axis=0)[:-1]


class GiniImpurity(Impurity):
    """The chance two rows drawn from this node disagree.

    ::

        1 - sum(p_k ** 2)

    Read it as a probability and the formula explains itself: draw two rows at
    random with replacement, and ``sum(p_k ** 2)`` is the chance they match, so
    one minus it is the chance they do not. Zero when the node is pure, and
    maximal at ``1 - 1/K`` when every class is equally represented.
    """

    __slots__ = ()

    def _of_non_empty(self, target_values: FloatArray) -> float:
        return float(1 - np.sum(self.target_probabilities(target_values) ** 2))

    def gains_at_every_prefix(self, sorted_targets: FloatArray) -> FloatArray:
        """One sweep: the counts after i+1 rows are the counts after i plus
        one. See :meth:`Impurity.gains_at_every_prefix`."""
        left_counts = self._cumulative_class_counts(sorted_targets)
        total_counts = np.bincount(
            sorted_targets.astype(np.int64), minlength=left_counts.shape[1]
        ).astype(np.float64)

        n_rows = sorted_targets.size
        rows_left = np.arange(1, n_rows, dtype=np.float64)[:, None]
        rows_right = n_rows - rows_left
        right_counts = total_counts[None, :] - left_counts

        left_impurity = 1.0 - ((left_counts / rows_left) ** 2).sum(axis=1)
        right_impurity = 1.0 - ((right_counts / rows_right) ** 2).sum(axis=1)

        weighted = (
            rows_left[:, 0] * left_impurity + rows_right[:, 0] * right_impurity
        ) / n_rows

        return float(self._of_non_empty(sorted_targets)) - weighted


class EntropyImpurity(Impurity):
    """Bits needed to encode which class a row of this node belongs to.

    ::

        -sum(p_k * log2(p_k))

    Zero when pure -- a constant needs no bits -- and ``log2(K)`` when every
    class is equally likely, which for two classes is exactly 1.

    A class with no rows contributes nothing, and must contribute nothing
    rather than a ``nan``: ``0 * log2(0)`` is the limit ``0``, and the
    implementation has to say so, because numpy will say ``nan``.
    """

    __slots__ = ()

    def _of_non_empty(self, target_values: FloatArray) -> float:
        target_probabilities = self.target_probabilities(target_values)
        log_p_k = np.log2(target_probabilities)
        per_class_entropy = target_probabilities * log_p_k
        return float(-np.sum(per_class_entropy))

    def gains_at_every_prefix(self, sorted_targets: FloatArray) -> FloatArray:
        """One sweep over cumulative class counts. See
        :meth:`Impurity.gains_at_every_prefix`."""
        left_counts = self._cumulative_class_counts(sorted_targets)
        total_counts = np.bincount(
            sorted_targets.astype(np.int64), minlength=left_counts.shape[1]
        ).astype(np.float64)

        n_rows = sorted_targets.size
        rows_left = np.arange(1, n_rows, dtype=np.float64)[:, None]
        rows_right = n_rows - rows_left
        right_counts = total_counts[None, :] - left_counts

        weighted = (
            rows_left[:, 0] * self._bits(left_counts, rows_left)
            + rows_right[:, 0] * self._bits(right_counts, rows_right)
        ) / n_rows

        return float(self._of_non_empty(sorted_targets)) - weighted

    @staticmethod
    def _bits(counts: FloatArray, row_totals: FloatArray) -> FloatArray:
        """Entropy of each row of counts, absent classes costing nothing.

        ``where=`` leaves the zeros untouched rather than evaluating log2(0),
        which is the vectorised form of the rule the scalar measure follows.
        """
        shares = counts / row_totals
        logarithms = np.zeros_like(shares)
        np.log2(shares, out=logarithms, where=shares > 0.0)

        return -(shares * logarithms).sum(axis=1)


class VarianceImpurity(Impurity):
    """The mean squared distance from this node's own mean.

    The regression counterpart, and the reason a regression leaf predicts the
    mean: the mean is precisely the constant that minimises this, so a leaf
    that reports it is a leaf making the smallest squared error available to a
    single number.
    """

    __slots__ = ()

    def _of_non_empty(self, target_values: FloatArray) -> float:
        y_mean = np.mean(target_values)
        mean_squared_error = np.power(target_values - y_mean, 2)

        return float(np.sum(mean_squared_error) / len(target_values))

    def gains_at_every_prefix(self, sorted_targets: FloatArray) -> FloatArray:
        """One sweep over running sums, with no sum of squares anywhere.

        The identity in the module docstring says the gain *is* the variance
        of the child means, so only the means are needed -- and a mean needs
        only a running sum::

            gain = (S_left^2 / n_left + S_right^2 / n_right - S^2 / n) / n

        Centring first is what makes that safe: uncentred, it subtracts two
        large nearly-equal numbers to recover a small one, the same
        cancellation that made the Euclidean expansion unsafe until it was
        centred. Gain is unchanged by shifting every target equally, so the
        centring costs nothing but conditioning.

        The ``- S^2 / n`` term is kept even though centring is *supposed* to
        make S zero. It does not, quite: the mean carries its own rounding, so
        the centred total is a small non-zero number, and dropping the term
        lets that error through the cumulative sum multiplied by the row
        count. Measured against exact rational arithmetic on targets near 1e9,
        assuming a zero total cost 6.1e-07 relative error where keeping the
        term costs 1.7e-13.
        """
        centred = sorted_targets - sorted_targets.mean()
        total = centred.sum()

        rows_left = np.arange(1, centred.size, dtype=np.float64)
        rows_right = centred.size - rows_left
        sum_left = np.cumsum(centred)[:-1]
        sum_right = total - sum_left

        return (
            sum_left**2 / rows_left
            + sum_right**2 / rows_right
            - total**2 / centred.size
        ) / centred.size
