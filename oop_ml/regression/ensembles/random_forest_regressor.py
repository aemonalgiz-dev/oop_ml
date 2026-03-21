"""Bagged trees that are also stopped from all asking the same question first.

Bagging alone leaves a floor. Resamples of the same data share most of their
rows, so every member finds the same strongest split and puts it at the root --
measured on six features, thirty bagged trees did that thirty times out of
thirty. Members that similar are correlated, and correlation is a floor that no
number of members can get below.

A forest lowers the floor rather than chasing it. At every node the tree may
only consider a random subset of the features, so sometimes the strongest one is
not on the menu and the tree is forced to find structure elsewhere. Allowed
three features per node on that same data, five different features appeared at
the root.

It is a trade, not a free improvement. Restriction raises each individual tree's
variance even as it lowers their correlation, so there is an optimum rather than
a direction -- and a forest of one tree is strictly worse than a lone tree.
"""

from __future__ import annotations

from pydantic import Field

from oop_ml.core.base.ensemble import AveragingMember
from oop_ml.regression.ensembles.bagging_regressor import BaggingRegressor
from oop_ml.regression.trees.decision_tree_regressor import DecisionTreeRegressor


class RandomForestRegressor(BaggingRegressor):
    """Bagged trees with a per-node feature restriction.

    Parameters
    ----------
    max_features:
        How many features each node may consider. ``None`` restricts nothing,
        which makes this exactly bagged trees.
    max_depth, min_samples_split, min_samples_leaf:
        Passed to every member. The defaults leave the trees unpruned, which is
        deliberate: the averaging is the regularisation here, and stopping
        rules that are essential in a lone tree mostly get in the way.
    n_members, random_seed:
        Inherited.
    """

    max_features: int | None = Field(default=None, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)

    def _prototype(self, position: int) -> AveragingMember:
        """A tree configured to restrict its features.

        Built here rather than accepted as a prototype, because a forest is
        specifically bagged *trees* -- the restriction is a tree feature and
        nothing else in the library has one.

        The member's seed is offset by its position. Handing every tree the
        ensemble's own seed would make all of them draw the same features at
        every node, which leaves the forest exactly as correlated as plain
        bagging while looking like it did something.
        """
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_seed=(
                None if self.random_seed is None else self.random_seed + position
            ),
        )
