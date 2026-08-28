"""How much each feature mattered, measured two ways that disagree.

The two live apart because they answer different questions and only one of them
is cheap.

Mean decrease in impurity is a property of a *fitted tree*: add up the impurity
each feature removed at every node it won, weighted by how many rows passed
through. It falls out of what the tree already stores, so it costs nothing, and
it is biased. A feature offering hundreds of candidate thresholds wins splits
more often than one offering a single threshold, on chance alone, so continuous
and high-cardinality columns score higher than they deserve.

Permutation importance is a property of *any fitted model*: shuffle one column,
score again, and see how much worse it got. It costs a full scoring pass per
feature and it measures what the model actually leans on rather than what the
split search happened to like, which is the honest question and usually the one
worth asking.
"""

from oop_ml.core.importance.importances import (
    FeatureImportance,
    FeatureImportances,
)
from oop_ml.core.importance.permutation import PermutationImportance

__all__ = [
    "FeatureImportance",
    "FeatureImportances",
    "PermutationImportance",
]
