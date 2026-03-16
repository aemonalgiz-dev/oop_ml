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

from oop_ml.core.types import FloatArray


class Impurity(ABC):
    """How mixed one node's targets are, and what a split of them buys."""

    __slots__ = ()

    @abstractmethod
    def of(self, target_values: FloatArray) -> float:
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
            An empty node is defined as zero: there is nothing in it to
            disagree.
        """

    def gain(
        self,
        parent_values: FloatArray,
        left_values: FloatArray,
        right_values: FloatArray,
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
        raise NotImplementedError


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

    def of(self, target_values: FloatArray) -> float:
        raise NotImplementedError


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

    def of(self, target_values: FloatArray) -> float:
        raise NotImplementedError


class VarianceImpurity(Impurity):
    """The mean squared distance from this node's own mean.

    The regression counterpart, and the reason a regression leaf predicts the
    mean: the mean is precisely the constant that minimises this, so a leaf
    that reports it is a leaf making the smallest squared error available to a
    single number.
    """

    __slots__ = ()

    def of(self, target_values: FloatArray) -> float:
        raise NotImplementedError
