"""How far apart two *groups* are, given how far apart their members are.

A distance metric answers the question for two rows. Agglomerative clustering
needs it answered for two sets of rows, and there is no single right way to do
that: a group is not a point, so "the distance between these two groups" is a
choice rather than a measurement. The choice is the whole character of the
method, and the four here behave differently enough that a caller should pick
deliberately.

- **Single**, the smallest distance between any member of one and any member of
  the other. Two groups are close if they touch anywhere, so a chain of rows
  each near the next comes out as one group however long it is. That is the
  same reachability
  :class:`~oop_ml.numpy.clustering.dbscan.DBSCAN` uses, and it finds long thin
  shapes for the same reason -- and suffers the same chaining, where a thin
  bridge of rows fuses two groups that are otherwise plainly separate.
- **Complete**, the largest such distance. Two groups are close only if *every*
  pair is close, which makes compact roughly round groups and is the opposite
  failure: a group is broken up rather than a bridge being crossed.
- **Average**, the mean over every pair. Between the two, and the usual choice
  when neither failure is wanted.
- **Ward**, which is not a summary of the pairwise distances at all. It merges
  whichever pair adds least to the total spread within groups, which makes it
  the hierarchical relative of k-means, and it inherits k-means' one
  restriction: spread is measured about a mean, and a mean minimises squared
  *Euclidean* distance specifically, so ward is defined for that metric and no
  other.

The first three read the pairwise block and nothing else, which is why any
metric serves them. Ward reads the group means, which is why it does not.
"""

from __future__ import annotations

from enum import StrEnum


class Linkage(StrEnum):
    """What the distance between two groups is taken to be.

    A closed set rather than a callable, for the reason
    :class:`~oop_ml.core.distance.metric.DistanceMetric` is one: a wrong value
    should be a type error rather than a surprise at runtime, and every member
    here is a rule with a name rather than a rule with parameters.
    """

    SINGLE = "single"
    """The nearest pair. Follows chains, and can be fooled by a bridge."""

    COMPLETE = "complete"
    """The furthest pair. Makes compact groups, and can split a long one."""

    AVERAGE = "average"
    """The mean over every pair. Between the other two."""

    WARD = "ward"
    """The least increase in within-group spread. Euclidean only, and why."""

    @property
    def reads_the_pairwise_block(self) -> bool:
        """Whether this rule is a summary of the member-to-member distances.

        True for three of the four, which is what lets them work under any
        metric. Ward reads the group means instead and is the exception the
        model has to branch on.
        """
        return self is not Linkage.WARD
