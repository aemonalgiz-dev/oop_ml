"""Bagged classification trees, restricted in what each node may consider.

The argument is the one the regression forest makes, and this is where it is
usually met first. Resamples share most of their rows, so bagged trees find the
same strongest split and put it at the root every time; members that similar are
correlated, and correlation is a floor that no number of members can get below.

Restricting each node to a random subset of the features forces some trees to
build around a feature they would never have chosen. Each individual tree gets
worse and the ensemble gets better, which is the trade worth remembering: this
is not a free improvement, it is variance spent on decorrelation.

It also fixes the tree's worst failure. On a parity target -- the class is 1 when
exactly one of two features is 1 -- every single-feature split scores a gain of
exactly zero, so the greedy search has no reason to prefer any of them and the
first move is effectively arbitrary. A lone tree that starts wrong stays wrong.
A forest starts differently in every member, and some of them start right.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from oop_ml.core.base.ensemble import AveragingMember
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.numpy.classification.ensembles.bagging_classifier import BaggingClassifier
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)


class RandomForestClassifier(BaggingClassifier):
    """Bagged classification trees with a per-node feature restriction.

    Parameters
    ----------
    max_features:
        How many features each node may consider. ``None`` restricts nothing,
        which makes this exactly bagged trees.
    max_depth, min_samples_split, min_samples_leaf:
        Passed to every member. Unpruned by default -- the averaging is the
        regularisation here, and the stopping rules that are a lone tree's only
        defence mostly get in the way.
    n_members, random_seed:
        Inherited.
    """

    max_features: int | None = Field(default=None, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _refuse_a_configured_base_model(self) -> Self:
        """Raise if a caller configured the one field a forest ignores.

        See ``RandomForestRegressor`` for the argument; the two must refuse
        identically.
        """
        if self.base_model != type(self).model_fields["base_model"].default:
            raise InvalidValuesError(
                "a forest builds its own trees and ignores base_model; "
                "configure max_depth, min_samples_split, min_samples_leaf and "
                "max_features on the forest itself"
            )

        return self

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
        return DecisionTreeClassifier(
            n_known_classes=self._n_classes,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_seed=(
                None if self.random_seed is None else self.random_seed + position
            ),
        )
