"""Predictor columns paired with the target they are fit against.

Every model here takes ``fit(input_values, target_values)``, which is two
arguments the caller is trusted to keep aligned. That works while the data sits
still. The moment it is subset -- into a training half, a fold, or a bootstrap
resample -- the pairing has to survive being carried around, and passing two
sequences and hoping they stay in step is exactly the shape of mistake this
library avoids elsewhere.

:class:`Dataset` is that pairing, and ``select_rows`` is the one operation every
caller of it wants: take these rows from every column at once. A split takes a
disjoint pair of index sets, a fold takes one, and a bootstrap sample takes an
index set with repeats -- the same method serves all three, because none of
them is doing anything to the data except choosing rows.

It lives in ``core.data`` rather than beside the splitters because two packages
now speak it. ``model_selection`` was the first, and ``core.base.ensemble`` is
the second: a bagged member is fitted on ``dataset.select_rows(sample.drawn)``
and nothing else about resampling has to be written twice.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.types import IndexArray


class Dataset:
    """Predictor columns paired with the target they are fit against.

    Parameters
    ----------
    input_features:
        The predictor columns. Names must be unique and lengths must match.
    target_feature:
        The response column, aligned row-for-row with the predictors.

    Raises
    ------
    EmptyValuesError
        If no features are supplied.
    NonUniqueFeaturesError
        If two features share a name.
    NonEqualArrayLengthError
        If the columns are not all the same length, or the target does not match.
    """

    __slots__ = ("_input_features", "_target_feature")

    def __init__(
        self, input_features: Sequence[Feature], target_feature: Feature
    ) -> None:
        feature_set = FeatureSet(input_features)
        feature_set.check_aligned_with(target_feature)

        self._input_features = tuple(input_features)
        self._target_feature = target_feature

    @property
    def input_features(self) -> list[Feature]:
        """The predictor columns, in the order supplied.

        A list rather than the internal tuple, because every model's ``fit``
        takes a ``Sequence[Feature]`` and this is the value handed straight to
        it.
        """
        return list(self._input_features)

    @property
    def target_feature(self) -> Feature:
        """The response column."""
        return self._target_feature

    @property
    def n_samples(self) -> int:
        """Number of rows, shared by every column."""
        return self._target_feature.n_samples

    @property
    def n_features(self) -> int:
        """Number of predictor columns."""
        return len(self._input_features)

    def select_rows(self, row_indices: IndexArray | Sequence[int]) -> Dataset:
        """A new dataset holding only the rows at ``row_indices``, in that order.

        Every column is subset identically, which is what keeps the predictors
        and the target aligned through a split. Names are preserved, so a model
        fitted on one subset can predict on another.
        """
        indices = np.asarray(row_indices, dtype=np.intp)

        return Dataset(
            [
                Feature(feature.name, feature.values[indices])
                for feature in self._input_features
            ],
            Feature(self._target_feature.name, self._target_feature.values[indices]),
        )

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        described = ", ".join(feature.name for feature in self._input_features)
        return (
            f"Dataset({described} -> {self._target_feature.name}, "
            f"n_samples={self.n_samples})"
        )
