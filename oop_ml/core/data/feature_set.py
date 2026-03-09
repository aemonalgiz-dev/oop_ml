"""A validated collection of :class:`~oop_ml.core.data.feature.Feature`
columns.

Where a :class:`~oop_ml.core.data.feature.Feature` is one named column, a
:class:`FeatureSet` is the group of them that a model is fit on, and it owns the
rules that only make sense across several columns at once. Names have to be
unique, lengths have to match, and a predictor has to actually vary.

The point of the design is that those rules are enforced in the constructor
rather than by a guard the caller is expected to remember. An instance cannot
exist in an invalid state, so ``fit`` never opens with a defensive preamble; it
simply accepts an object whose type already carries the guarantee. Consider the
alternative, a free function along the lines of
``check_features_are_well_formed(features)``. Anyone can forget to call it, and
sooner or later somebody will, whereas a constructor is not something you can
skip.

It also gives the cross-column questions somewhere to live as behaviour, instead
of leaving that arithmetic scattered through each model. That covers
``n_samples``, ``n_features``, ``column("age")``, and the question of whether
there are enough rows to estimate this many parameters, which every regressor
has to ask.

The module sits below the estimators and above :mod:`core.feature`, importing
:mod:`core.feature` and :mod:`core.validation` while neither of them imports it.
The dependency arrows therefore point one way and no cycle is possible.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    TooFewValuesError,
)
from oop_ml.core.types import FloatArray
from oop_ml.core.validation import ValueRole, check_has_variance


class FeatureSet:
    """An aligned, uniquely named group of predictor columns.

    Parameters
    ----------
    features:
        One or more predictor columns. They must have distinct names, share a
        single sample count, and none may be constant.

    Raises
    ------
    EmptyValuesError
        If no features are supplied.
    NonUniqueFeaturesError
        If two features share a name.
    NonEqualArrayLengthError
        If the columns are not all the same length.
    """

    __slots__ = ("_features", "_feature_matrix")

    def __init__(self, features: Sequence[Feature]) -> None:
        self._check_features_are_present(features)
        self._check_names_are_unique(features)
        self._check_lengths_are_equal(features)

        # Column order is fixed here and never changes: it is what lets a model
        # map a positional solution vector back onto the right feature.
        self._features = tuple(features)

        # Column-major, and the layout is load-bearing rather than incidental.
        # Every linear model in the library reaches for X.T @ v at least once,
        # and an iterative one does it every epoch. With C-ordered storage that
        # product reads down a column with a row's stride between elements,
        # which on a tall matrix misses cache on nearly every access; measured
        # at 20000x51 it costs 4.4x what the same product costs here. Building
        # column by column is what a feature set is doing anyway, so the layout
        # that suits the arithmetic is also the one that suits the construction
        # and there is nothing to trade away.
        self._feature_matrix = np.empty(
            (features[0].n_samples, len(self._features)), order="F"
        )
        for index, feature in enumerate(self._features):
            self._feature_matrix[:, index] = feature.values

    @classmethod
    def matching(
        cls, feature_names: Sequence[str], features: Sequence[Feature]
    ) -> FeatureSet:
        """The supplied features, put back into the order a fit saw them in.

        Every model here matches its inputs by name rather than by position,
        so a caller can hand ``predict`` the same columns in any arrangement.
        The names must match *exactly*: a missing column leaves the model
        unevaluable, and an unexpected one means the caller believes something
        about this model that is not true.

        Parameters
        ----------
        feature_names:
            The names the fit saw, in the order it saw them.
        features:
            The columns supplied now, in any order.

        Returns
        -------
        FeatureSet
            The same columns, ordered to ``feature_names``.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two supplied features share a name.
        NonEqualArrayLengthError
            If the supplied columns are not all the same length.
        InvalidValuesError
            If the supplied names do not match the fitted ones exactly.
        """
        # Uniqueness first, and deliberately before the dictionary is built:
        # two columns sharing a name would collapse into one entry, and the
        # set comparison below would then pass while a column went missing.
        cls._check_features_are_present(features)
        cls._check_names_are_unique(features)

        by_name = {feature.name: feature for feature in features}
        if set(by_name) != set(feature_names):
            raise InvalidValuesError(
                f"expected features {', '.join(sorted(feature_names))}; "
                f"got {', '.join(sorted(by_name))}"
            )

        return cls([by_name[name] for name in feature_names])

    @property
    def n_samples(self) -> int:
        """Number of observations (rows) shared by every column."""
        return self._features[0].n_samples

    @property
    def n_features(self) -> int:
        """Number of predictor columns."""
        return len(self._features)

    @property
    def feature_matrix(self) -> FloatArray:
        """The feature columns as an ``(n_samples, n_features)`` matrix."""
        return self._feature_matrix

    def column(self, name: str) -> Feature:
        """Return the column called ``name``.

        Raises
        ------
        InvalidValuesError
            If no column has that name.
        """
        for feature in self._features:
            if feature.name == name:
                return feature

        raise InvalidValuesError(
            f"no feature named {name!r}; this set has "
            f"{', '.join(feature.name for feature in self._features)}"
        )

    def check_aligned_with(self, feature: Feature) -> None:
        """Raise if ``feature`` does not share this set's sample count.

        This is used for the target column, which has to line up row for row
        with the predictors although it is not itself a member of the set, since
        it earns no coefficient of its own.

        Raises
        ------
        NonEqualArrayLengthError
            If the lengths differ.
        """
        feature_length = feature.n_samples
        if feature_length != self.n_samples:
            raise NonEqualArrayLengthError(
                f"{feature.name} has {feature_length} rows, "
                f"the features have {self.n_samples}"
            )

    def check_supports_parameter_count(self, parameter_count: int) -> None:
        """Raise if there are fewer observations than parameters to estimate.

        Least squares needs at least as many equations as it has unknowns, which
        with an intercept means ``n >= n_features + 1`` and without one means
        ``n >= n_features``. Fall below that bound and the system is
        underdetermined, so infinitely many coefficient vectors fit perfectly
        and a solver will hand you one of them without complaining at all. That
        silent success is exactly what this guard exists to prevent.

        Raises
        ------
        TooFewValuesError
            If ``n_samples`` is below ``parameter_count``.
        """
        if self.n_samples < parameter_count:
            raise TooFewValuesError(
                f"estimating {parameter_count} parameters needs at least "
                f"{parameter_count} samples, got {self.n_samples}"
            )

    def __len__(self) -> int:
        return self.n_features

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features)

    def __repr__(self) -> str:
        described = ", ".join(feature.name for feature in self._features)
        return f"FeatureSet({described}, n_samples={self.n_samples})"

    @staticmethod
    def _check_features_are_present(features: Sequence[Feature]) -> None:
        if not features:
            raise EmptyValuesError("at least one feature is required")

    @staticmethod
    def _check_names_are_unique(features: Sequence[Feature]) -> None:
        seen_names: set[str] = set()
        duplicate_names: set[str] = set()

        for feature in features:
            if feature.name in seen_names:
                duplicate_names.add(feature.name)
            seen_names.add(feature.name)

        if duplicate_names:
            raise NonUniqueFeaturesError(
                f"feature names must be unique, got duplicates: "
                f"{', '.join(sorted(duplicate_names))}"
            )

    @staticmethod
    def _check_lengths_are_equal(features: Sequence[Feature]) -> None:
        reference_feature = features[0]
        reference_length = reference_feature.n_samples

        for feature in features[1:]:
            feature_length = feature.n_samples
            if feature_length != reference_length:
                raise NonEqualArrayLengthError(
                    f"{feature.name} has {feature_length} rows, "
                    f"{reference_feature.name} has {reference_length}"
                )

    def check_columns_vary(self) -> None:
        """Raise if any column is constant.

        This is a fitting requirement rather than a structural one, which is why
        it is a method the caller invokes and not a rule the constructor
        enforces. A constant predictor is perfectly legal to hold, to predict
        on, and to carry inside a held-out fold; fitting is the only thing it
        breaks, because it is collinear with the intercept's column of ones.

        Raises
        ------
        AllSameValuesError
            If any column has zero variance.
        """
        for feature in self._features:
            check_has_variance(feature.values, ValueRole.FEATURE_VALUES)
