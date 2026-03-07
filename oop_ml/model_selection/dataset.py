"""Features and their target as one object, and a partition of one.

Every model so far has taken ``fit(input_values, target_values)``, which is two
arguments the caller is trusted to keep aligned. That works well enough while the
data sits still, although the moment it gets split, the pairing has to survive
being carried around as a training half, a testing half, and one of each per
fold. Passing four sequences and hoping they stay in step is exactly the shape of
mistake this library goes out of its way to avoid elsewhere.

:class:`Dataset` is that pairing. :class:`DataSplit` is two of them, being the
halves of one partition, named rather than positional so that nobody has to
remember whether training or testing came first.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oop_ml.core.feature import Feature
from oop_ml.core.feature_set import FeatureSet
from oop_ml.types import IndexArray


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


class DataSplit:
    """One partition of a dataset into a part to learn from and a part to score on.

    Parameters
    ----------
    training:
        The rows the model is allowed to see.
    testing:
        The rows held back, used only to measure how well it generalises.

    Raises
    ------
    EmptyValuesError
        If either half has no rows.
    """

    __slots__ = ("_testing", "_training")

    def __init__(self, training: Dataset, testing: Dataset) -> None:
        self._training = training
        self._testing = testing

    @property
    def training(self) -> Dataset:
        """The rows the model learns from."""
        return self._training

    @property
    def testing(self) -> Dataset:
        """The rows held back for scoring."""
        return self._testing

    @property
    def n_samples(self) -> int:
        """Total rows across both halves."""
        return self._training.n_samples + self._testing.n_samples

    def __repr__(self) -> str:
        return (
            f"DataSplit(training={self._training.n_samples}, "
            f"testing={self._testing.n_samples})"
        )
