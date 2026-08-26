"""Which impurity measure to score splits with, one closed enum per task.

Two enums rather than one, because the valid choices genuinely differ. Variance
is meaningless on class positions -- it would read class 2 as twice as far from
class 0 as class 1 is -- and Gini is meaningless on a quantity. A single enum
covering all three would need a runtime check in every model saying "not that
one", which is the bad message at runtime the rest of the library avoids by
making the wrong value unrepresentable.

The pattern is the one :mod:`~oop_ml.core.distance.metric` uses: the enum is the
closed vocabulary a user picks from, and each member carries the object that
does the work, so a member cannot be added without also being given a way to
compute it.
"""

from __future__ import annotations

from enum import StrEnum

from oop_ml.core.tree.impurity import (
    EntropyImpurity,
    GiniImpurity,
    Impurity,
    VarianceImpurity,
)


class ClassificationCriterion(StrEnum):
    """How a classification tree scores a candidate split.

    Both members are strictly concave, which is the property that makes a split
    worth anything at all -- see :mod:`~oop_ml.core.tree.impurity` for why
    misclassification rate is absent rather than merely unfashionable.

    Attributes
    ----------
    GINI:
        The chance two rows drawn from a node disagree. The default, and
        cheaper by a logarithm per candidate split.
    ENTROPY:
        Bits needed to encode a row's class. Leans very slightly harder toward
        balanced children, and picks a different split rarely enough that it is
        not where accuracy comes from.
    """

    _impurity: Impurity

    GINI = ("gini", GiniImpurity())
    ENTROPY = ("entropy", EntropyImpurity())

    def __new__(cls, value: str, impurity: Impurity) -> ClassificationCriterion:
        """Bind each member to the object that computes it."""
        member = str.__new__(cls, value)
        member._value_ = value
        member._impurity = impurity

        return member

    @property
    def impurity(self) -> Impurity:
        """The measure this member names."""
        return self._impurity


class RegressionCriterion(StrEnum):
    """How a regression tree scores a candidate split.

    One member today. It is an enum rather than nothing at all so that adding
    absolute error later -- which is a different leaf prediction, the median
    rather than the mean, and not merely a different formula -- is a new member
    rather than a new parameter with a magic string in it.

    Attributes
    ----------
    SQUARED_ERROR:
        Variance about the node's own mean, which is the squared error a leaf
        makes when it predicts that mean.
    """

    _impurity: Impurity

    SQUARED_ERROR = ("squared_error", VarianceImpurity())

    def __new__(cls, value: str, impurity: Impurity) -> RegressionCriterion:
        """Bind each member to the object that computes it."""
        member = str.__new__(cls, value)
        member._value_ = value
        member._impurity = impurity

        return member

    @property
    def impurity(self) -> Impurity:
        """The measure this member names."""
        return self._impurity
