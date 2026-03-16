"""The vocabulary a tree is built from, separate from the models that grow one.

:mod:`~oop_ml.core.tree.criterion` is what a user chooses between, one closed
enum per task. :mod:`~oop_ml.core.tree.impurity` is how each choice is actually
computed. :mod:`~oop_ml.core.tree.split` is one question bound to the name of
the feature it asks about, and :mod:`~oop_ml.core.tree.node` is what a fitted
tree is made of.

The enum-plus-strategy split is the same shape
:mod:`~oop_ml.core.distance` uses, for the same reason: the vocabulary should
be closed so a misspelling cannot reach runtime, while the calculation stays an
object so a new measure is a new class rather than another branch in a growing
``if``.

This sits in ``core`` rather than beside the models because none of it is a
model. A ``Split`` is a fact about a column, and impurity is a fact about a set
of targets; gradient boosting and random forests would want both, and neither
of those is a decision tree.
"""
