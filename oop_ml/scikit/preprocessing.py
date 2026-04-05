"""The preprocessing family, with scikit-learn doing the arithmetic.

Every class here has a namesake in ``oop_ml.numpy`` with the same name, the
same pydantic fields, the same base class and the same learned properties, so
that a caller can swap one backend for the other at the import line and change
nothing else. A standardizer still answers with a
:class:`~oop_ml.numpy.preprocessing.standardization.scaling.FeatureScalings`,
an affine scaler with an
:class:`~oop_ml.numpy.preprocessing.rescaling.affine.AffineScalings`, and a
polynomial expansion with a
:class:`~oop_ml.numpy.preprocessing.polynomial.terms.PolynomialTerms`. What
differs is who reads the statistics off the training rows.

By-name matching, which the engines do not have
------------------------------------------------
Every scikit-learn transformer is positional. It learns ``n_features_in_``
columns and refuses anything but exactly that many, in the order it saw them.
This library's transformers match by name, and the scalers accept a *subset*
of the fitted features while refusing an unknown one, because a held-out fold
with one column is a legitimate thing to rescale and a column the fit never
saw has no centre and no spread.

The scalers keep that rule by not asking the engine to transform at all. What
a scaler learns is two numbers per column, and the engine reports them as
fitted attributes; the value object that holds them,
:class:`~oop_ml.numpy.preprocessing.rescaling.affine.AffineScaling` or its
standardizing twin, already owns ``(value - centre) / spread`` and its inverse.
So the engine is asked to *learn* and the learned object is asked to *apply*,
which is the line the Boltzmann wrapper draws for the same reason. Keeping an
engine that nothing would read again is the dead field the serving audit
removed elsewhere.

Three of the four engines subtract and divide as the value object does, and
measured over 600 random blocks apiece the two routes come back bit-identical
for the standardizing, max-abs and robust engines. The min-max engine is the
exception, and it is worth naming rather than rounding away. It folds the pair
into a multiply and an add, ``value * scale_ + min_``, so it is a different
arithmetic reading the same two numbers, and none of those 600 blocks came
back bit-identical, 25454 of the 55839 entries differed, and the largest
disagreement was 2.2204e-16, one unit in the last place of 1.0 on a column
mapped onto the unit interval. Either route is correct, and the value object's
is the one this library's own ``inverse_transform`` undoes.

The polynomial expansion is the other way round. There the engine's work is
the products, so the engine is kept, the supplied features are put back into
the fitted order by name before it is called, and the columns it answers with
are named from the exponent table it reports.

Where the engines patch what this library refuses
--------------------------------------------------
A column with no spread cannot be scaled, and this library refuses it with
:class:`~oop_ml.core.exceptions.AllSameValuesError`. Every engine here
instead substitutes a spread of one and carries on, which turns a column
carrying no information into a column of zeros and says nothing.

The substitution is not confined to an exact zero, which is the half that
catches a reader out. The min-max, max-abs and robust engines replace any
spread below ten machine epsilons, 2.2204e-15, so a column of readings whose
range is 4e-16 is patched exactly as readily as a constant one; the
standardizer applies a relative bound instead, described on that class. A
patched spread of one is indistinguishable afterwards from a column whose
spread genuinely is one, so nothing downstream can recover the difference.

The wrappers therefore read the *unpatched* attribute wherever the engine
keeps one, ``var_`` rather than ``scale_`` for the standardizer,
``data_range_`` for min-max and ``max_abs_`` for max-abs, so the value objects
see the true spread, scale by it when it is real and refuse it when it is
zero. The robust engine keeps no unpatched attribute at all, so that wrapper
reads its spread off the quartiles directly and :class:`RobustScaler` records
why a guard in front of the engine was not enough on its own.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr
from sklearn.preprocessing import MaxAbsScaler as EngineMaxAbsScaler
from sklearn.preprocessing import MinMaxScaler as EngineMinMaxScaler
from sklearn.preprocessing import PolynomialFeatures as EnginePolynomialFeatures
from sklearn.preprocessing import RobustScaler as EngineRobustScaler
from sklearn.preprocessing import StandardScaler

from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.exceptions import AllSameValuesError, InvalidValuesError
from oop_ml.core.types import FloatArray
from oop_ml.numpy.preprocessing.polynomial.terms import PolynomialTerm, PolynomialTerms
from oop_ml.numpy.preprocessing.rescaling.affine import AffineScaling, AffineScalings
from oop_ml.numpy.preprocessing.standardization.scaling import (
    FeatureScaling,
    FeatureScalings,
)
from oop_ml.scikit.plumbing import matrix_of

QUARTILES: tuple[float, float] = (0.25, 0.75)
"""The two quantiles whose distance is the interquartile range."""

ENGINE_QUARTILE_PERCENTAGES: tuple[float, float] = (25.0, 75.0)
"""The same two quantiles as the engine states them, in percent."""

UNIT_INTERVAL: tuple[float, float] = (0.0, 1.0)
"""Where the min-max engine is told to put a column, which is where this
library's ``MinMaxScaler`` puts one."""


def check_names_were_learned(
    learned_names: Sequence[str], supplied: Sequence[Feature], scaler_name: str
) -> None:
    """Raise unless every supplied feature is one the fit learned a scaling for.

    A subset is fine, since transforming one column of a held-out set is
    legitimate. An unknown name is the error, because a column the fit never
    saw has no centre and no spread, and inventing one would answer a question
    nobody asked.

    Raises
    ------
    InvalidValuesError
        If a supplied feature was not among the fitted ones.
    """
    known = set(learned_names)
    unknown = [feature.name for feature in supplied if feature.name not in known]

    if unknown:
        raise InvalidValuesError(
            f"this {scaler_name} learned {sorted(known)} and was handed "
            f"{unknown}, which it has no scaling for"
        )


def affine_scalings_of(
    feature_names: Sequence[str], centres: FloatArray, spreads: FloatArray
) -> AffineScalings:
    """One :class:`AffineScaling` per column, from the engine's two arrays.

    Raises
    ------
    AllSameValuesError
        If any spread is zero or negative, which is
        :class:`~oop_ml.numpy.preprocessing.rescaling.affine.AffineScaling`
        refusing a column it cannot divide by, named.
    """
    return AffineScalings(
        [
            AffineScaling(name=name, centre=float(centre), spread=float(spread))
            for name, centre, spread in zip(
                feature_names, centres, spreads, strict=True
            )
        ]
    )


class Standardizer(Transformer[Sequence[Feature]]):
    """Centre each feature at zero and rescale it to unit standard deviation,
    by ``StandardScaler``.

    Takes no hyperparameters, exactly as the numpy backend's does.

    Translation
    -----------
    The engine's ``with_mean`` and ``with_std`` are left on, which is the
    whole transformation. The engine divides by the *population* standard
    deviation, over ``n`` rather than ``n - 1``, which is what the numpy
    backend's ``Column.standard_deviation`` computes, so the two backends'
    scalings agree to rounding and the standardized columns agree with them.

    The spread is read off the engine's ``var_`` as its square root rather
    than off ``scale_``, because ``scale_`` is patched to one on a column the
    engine judges constant and ``var_`` is not. The judgement the engine makes
    there is relative, a variance no larger than
    ``n * eps * var + (n * mean * eps) ** 2`` counts as none, where this
    library's is exact, every value identical.

    The gap between those two rules is reachable rather than theoretical. Nine
    values a few last bits apart above 1.0 have a standard deviation of about
    1.3009e-15 and are judged constant, so a wrapper reading ``scale_`` would
    report a spread of exactly 1.0 and hand back the column barely moved.
    Reading the true variance keeps this library's rule, and a genuinely
    constant column is refused by ``check_columns_vary`` before the engine
    sees it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _scalings: FeatureScalings | None = PrivateAttr(default=None)

    @property
    def scalings(self) -> FeatureScalings:
        """The statistics learned per feature, addressable by name.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._scalings is not None
        return self._scalings

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Learn each feature's mean and standard deviation.

        Returns
        -------
        Self
            This standardizer, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If the features are not all the same length.
        AllSameValuesError
            If any feature is constant, since a zero spread has nothing to
            rescale by.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()

        feature_names = [feature.name for feature in feature_set]
        engine: Any = StandardScaler(with_mean=True, with_std=True)
        engine.fit(matrix_of(feature_set))

        means = np.asarray(engine.mean_, dtype=np.float64)
        spreads = np.sqrt(np.asarray(engine.var_, dtype=np.float64))

        learned = FeatureScalings(
            [
                FeatureScaling(name, float(mean), float(spread))
                for name, mean, spread in zip(
                    feature_names, means, spreads, strict=True
                )
            ]
        )

        self._scalings = learned
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Rescale each feature using the statistics learned during ``fit``.

        Matches by name and never by position, so the caller may pass a
        subset of the fitted features in any order.

        Returns
        -------
        list[Feature]
            One standardized feature per input, in the order supplied.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If a supplied feature was not seen during ``fit``.
        """
        scalings = self.scalings
        check_names_were_learned(
            [scaling.name for scaling in scalings], input_values, "standardizer"
        )

        return [
            Feature(feature.name, scalings[feature.name].standardize(feature.values))
            for feature in input_values
        ]

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "Standardizer(unfitted)"

        return f"Standardizer(n_features={self.scalings.n_features})"


class EngineScaler(Transformer[Sequence[Feature]]):
    """Learn a centre and a spread per column from an engine, then rescale by them.

    The frame the three affine wrappers share. A subclass supplies
    :meth:`_learned_scalings`, which runs its engine over the training matrix
    and reads the two numbers per column back into an
    :class:`~oop_ml.numpy.preprocessing.rescaling.affine.AffineScalings`.
    Everything around that, the fit, the by-name matching, the round trip and
    every refusal, is written once, as it is on the numpy backend's
    ``FeatureScaler``.

    Not exported, and not a reproduction of the numpy frame. That one reads
    its two numbers with ``centre_of`` and ``spread_of``; this one reads them
    off a fitted engine, and the two frames have nothing but the property
    name in common.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _scalings: AffineScalings | None = PrivateAttr(default=None)

    @property
    def scalings(self) -> AffineScalings:
        """What the fit learned, one scaling per column.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._scalings is not None
        return self._scalings

    @abstractmethod
    def _learned_scalings(self, feature_set: FeatureSet) -> AffineScalings:
        """Run the engine over ``feature_set`` and read its centres and spreads.

        Raises
        ------
        AllSameValuesError
            If any column has no spread under this scaler's reading of spread.
        """

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Learn one centre and one spread per feature.

        Nothing is committed until every scaling has been built, so a fit
        that raises partway leaves the previous one intact.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If the features are not all the same length.
        AllSameValuesError
            If any feature has no spread under this scaler's reading of spread.
        """
        feature_set = FeatureSet(input_values)

        learned = self._learned_scalings(feature_set)

        self._scalings = learned
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Rescale each supplied feature by what the fit learned for it.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonUniqueFeaturesError
            If two features share a name.
        InvalidValuesError
            If a feature was not among the fitted ones.
        """
        scalings = self.scalings
        supplied = FeatureSet(input_values)
        check_names_were_learned(scalings.names, list(supplied), type(self).__name__)

        return [
            Feature(feature.name, scalings[feature.name].scale(feature.values))
            for feature in supplied
        ]

    def inverse_transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Undo :meth:`transform`, so an answer can be read in original units.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonUniqueFeaturesError
            If two features share a name.
        InvalidValuesError
            If a feature was not among the fitted ones.
        """
        scalings = self.scalings
        supplied = FeatureSet(input_values)
        check_names_were_learned(scalings.names, list(supplied), type(self).__name__)

        return [
            Feature(feature.name, scalings[feature.name].restore(feature.values))
            for feature in supplied
        ]

    def __repr__(self) -> str:
        fitted = self._scalings.n_features if self._scalings is not None else None
        return f"{type(self).__name__}(n_features={fitted!r})"


class MinMaxScaler(EngineScaler):
    """Squash each column into ``[0, 1]``, by ``MinMaxScaler``.

    Takes no hyperparameters, exactly as the numpy backend's does.

    Translation
    -----------
    The engine's ``feature_range`` is pinned to the unit interval, which is
    the only range the numpy backend produces. The centre is the engine's
    ``data_min_`` and the spread its ``data_range_``, both unpatched; its
    ``scale_`` and ``min_`` are the same two numbers folded into a multiply
    and an add, with the spread patched to one on any column whose range falls
    below ten machine epsilons, and are not read.

    Not mirrored from the numpy backend
    -----------------------------------
    ``centre_of``, ``spread_of``
        The numpy frame's two readings of a column, the smallest value and the
        range. Here the engine reads them, and a public method that reproduced
        its arithmetic would be a second implementation of the thing wrapped.
    """

    def _learned_scalings(self, feature_set: FeatureSet) -> AffineScalings:
        engine: Any = EngineMinMaxScaler(feature_range=UNIT_INTERVAL)
        engine.fit(matrix_of(feature_set))

        return affine_scalings_of(
            [feature.name for feature in feature_set],
            np.asarray(engine.data_min_, dtype=np.float64),
            np.asarray(engine.data_range_, dtype=np.float64),
        )


class MaxAbsScaler(EngineScaler):
    """Divide each column by its largest magnitude, by ``MaxAbsScaler``.

    Takes no hyperparameters, exactly as the numpy backend's does.

    Translation
    -----------
    The engine takes no parameters worth naming. The centre is zero, since
    neither backend moves anything, and the spread is the engine's
    ``max_abs_``, unpatched; its ``scale_`` is the same number with anything
    below ten machine epsilons replaced by one, and is not read.

    A constant column of sevens is accepted and answers all ones, on both
    backends, because seven really is that column's magnitude. The column
    refused is the all-zero one, which has no magnitude either.

    Not mirrored from the numpy backend
    -----------------------------------
    ``centre_of``, ``spread_of``
        The numpy frame's two readings of a column. Here the engine reads
        them; see :class:`MinMaxScaler`.
    """

    def _learned_scalings(self, feature_set: FeatureSet) -> AffineScalings:
        engine: Any = EngineMaxAbsScaler()
        engine.fit(matrix_of(feature_set))

        spreads = np.asarray(engine.max_abs_, dtype=np.float64)

        return affine_scalings_of(
            [feature.name for feature in feature_set],
            np.zeros_like(spreads),
            spreads,
        )


class RobustScaler(EngineScaler):
    """Centre on the median and divide by the interquartile range, by ``RobustScaler``.

    Takes no hyperparameters, exactly as the numpy backend's does.

    Translation
    -----------
    The engine's ``with_centering`` and ``with_scaling`` are left on, its
    ``quantile_range`` pinned to the quartiles, and ``unit_variance`` left
    off, since the numpy backend divides by the interquartile range itself
    and not by the range a normal distribution would have. The centre is the
    engine's ``center_``. ``with_scaling`` stays on so the engine is asked
    for the whole robust fit rather than a configuration describing some
    other model, and its ``scale_`` is nonetheless not what the spread is
    read from, for the reason below.

    Why the spread is not read off ``scale_``
    -----------------------------------------
    This engine keeps no unpatched spread. It computes the interquartile range
    and then replaces it with one wherever the range falls below ten machine
    epsilons, so a patched column is indistinguishable afterwards from a
    column whose quartiles genuinely sit one apart.

    A guard in front of the engine refusing a range of zero is not enough,
    because the engine substitutes over a band and not at a point. Measured,
    nine readings spaced 1e-16 apart have a real interquartile range of 4e-16
    and are patched, so reading ``scale_`` reported a spread of 1.0, handed
    the column back moved by 4e-16 where the numpy backend divides by 4e-16
    and answers the ramp from -1 to 1. Nothing raised, and the two backends
    disagreed by a factor of 2.5e15 on data neither of them should have
    struggled with.

    So the spread is read off the two quartiles directly, which is the engine's
    own number by a route the engine agrees with. Both sides place a quantile
    by linear interpolation between the two nearest sorted values, the engine
    through ``nanpercentile`` and this through ``quantile``; measured over
    2320 columns of 2 to 59 rows spanning magnitudes from 1e-8 to 1e8, the two
    ranges are bit-identical every time, and over 1140 further columns the
    engine's ``scale_`` equalled this range on every one it did not patch. A
    range of zero is still refused, which is the case the numpy backend's
    docstring measures, seven identical values out of eight, and is a real
    data shape rather than a degenerate one.

    Not mirrored from the numpy backend
    -----------------------------------
    ``centre_of``, ``spread_of``
        The numpy frame's two readings of a column. Here the engine reads the
        centre; see :class:`MinMaxScaler`.
    """

    def _learned_scalings(self, feature_set: FeatureSet) -> AffineScalings:
        spreads = self._interquartile_ranges(feature_set)

        engine: Any = EngineRobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=ENGINE_QUARTILE_PERCENTAGES,
            unit_variance=False,
        )
        engine.fit(matrix_of(feature_set))

        return affine_scalings_of(
            [feature.name for feature in feature_set],
            np.asarray(engine.center_, dtype=np.float64),
            spreads,
        )

    @staticmethod
    def _interquartile_ranges(feature_set: FeatureSet) -> FloatArray:
        """The distance between each column's quartiles, one entry per column.

        Raises
        ------
        AllSameValuesError
            If a column's interquartile range is zero. The column may still
            vary elsewhere; there is no robust spread in it to divide by.
        """
        ranges: list[float] = []

        for feature in feature_set:
            first, third = np.quantile(feature.values, QUARTILES)
            interquartile_range = float(third - first)

            if interquartile_range <= 0.0:
                raise AllSameValuesError(
                    f"{feature.name!r} has an interquartile range of 0.0, so it "
                    "cannot be rescaled; both quartiles sit inside one repeated "
                    "value, and the engine would silently divide by one instead"
                )

            ranges.append(interquartile_range)

        return np.asarray(ranges, dtype=np.float64)


class PolynomialFeatures(Transformer[Sequence[Feature]]):
    """Expand features into every power and product up to ``degree``, by
    ``PolynomialFeatures``.

    Parameters
    ----------
    degree, include_interactions:
        As on the numpy backend.

    Translation
    -----------
    ``degree`` is the engine's ``degree`` unchanged. The engine's
    ``include_bias`` is off, because the constant column it would add is the
    model's intercept and not a feature, which is why the numpy backend's
    terms never carry a zeroth power.

    ``include_interactions`` has no engine parameter. The engine's
    ``interaction_only`` is the *opposite* switch, keeping only the products of
    distinct features and dropping every power, where this library's
    ``include_interactions=False`` keeps only the powers and drops every
    product. So the engine is always asked for the full expansion, and the
    columns whose exponent row names more than one feature are left out on
    the way through when interactions are not wanted. The engine's
    ``powers_`` is that exponent table, one row per output column, and it is
    what each :class:`~oop_ml.numpy.preprocessing.polynomial.terms.PolynomialTerm`
    is built from, so the names are the numpy backend's names by the numpy
    backend's rule.

    The engine emits its columns in ascending total degree and, within a
    degree, in the order ``combinations_with_replacement`` walks the
    features, which is the loop the numpy backend's ``_build_terms`` is
    written as. Measured over every feature count and every degree from one to
    five, up to the 251 columns five features at degree five produce, the
    engine's exponent table and that loop agree row for row.

    ``transform`` demands every fitted feature, as on the numpy backend,
    because ``x1*x2`` cannot be computed with a column missing. The supplied
    features are put back into the fitted order by name before the engine
    sees them, since the engine reads positions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    degree: int = Field(default=2, ge=1)
    include_interactions: bool = True

    _engine: EnginePolynomialFeatures | None = PrivateAttr(default=None)
    _terms: PolynomialTerms | None = PrivateAttr(default=None)
    _feature_names: tuple[str, ...] = PrivateAttr(default=())
    _kept_columns: tuple[int, ...] = PrivateAttr(default=())

    @property
    def terms(self) -> PolynomialTerms:
        """The expansion fixed during ``fit``, in column order.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._terms is not None
        return self._terms

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Work out which terms the expansion will produce.

        Returns
        -------
        Self
            This transformer, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If the features are not all the same length.
        AllSameValuesError
            If any feature is constant, since its powers would be constant
            columns collinear with the intercept.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()

        feature_names = tuple(feature.name for feature in feature_set)
        engine: Any = EnginePolynomialFeatures(
            degree=self.degree, interaction_only=False, include_bias=False
        )
        engine.fit(matrix_of(feature_set))

        exponent_table = np.asarray(engine.powers_, dtype=np.int64)
        kept_columns = tuple(
            position
            for position, exponents in enumerate(exponent_table)
            if self.include_interactions or np.count_nonzero(exponents) == 1
        )
        terms = PolynomialTerms(
            [
                self._term_from(feature_names, exponent_table[position])
                for position in kept_columns
            ]
        )

        self._engine = engine
        self._terms = terms
        self._feature_names = feature_names
        self._kept_columns = kept_columns
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Compute the expanded columns for the supplied features.

        Matches by name, never by position, and returns the columns in the
        order fixed at ``fit``.

        Returns
        -------
        list[Feature]
            One feature per term, named for the term that produced it.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If a feature the expansion needs was not supplied.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        NonUniqueFeaturesError
            If two supplied features share a name.
        """
        self._check_fitted()
        assert self._engine is not None

        supplied = FeatureSet(list(input_values))
        values_by_name = {feature.name: feature.values for feature in supplied}
        missing = [name for name in self._feature_names if name not in values_by_name]

        if missing:
            raise InvalidValuesError(
                f"the expansion needs {', '.join(sorted(missing))}; "
                f"got {', '.join(sorted(values_by_name))}"
            )

        matrix = np.column_stack([values_by_name[name] for name in self._feature_names])
        expanded = np.asarray(self._engine.transform(matrix), dtype=np.float64)

        return [
            Feature(term.name, expanded[:, column])
            for term, column in zip(self.terms, self._kept_columns, strict=True)
        ]

    @staticmethod
    def _term_from(
        feature_names: Sequence[str], exponents: Sequence[int]
    ) -> PolynomialTerm:
        """One term from one row of the engine's exponent table.

        A zero exponent means the feature is absent from the term, and the
        term's own constructor refuses a zero, so only the positive ones are
        kept. Walking the names in fitted order is what makes the term print
        ``x1^2*x2`` rather than ``x2*x1^2``.
        """
        return PolynomialTerm(
            {
                name: int(exponent)
                for name, exponent in zip(feature_names, exponents, strict=True)
                if exponent > 0
            }
        )

    def __repr__(self) -> str:
        if not self.is_fitted:
            return (
                f"PolynomialFeatures(degree={self.degree}, "
                f"include_interactions={self.include_interactions}, unfitted)"
            )

        return (
            f"PolynomialFeatures(degree={self.degree}, "
            f"include_interactions={self.include_interactions}, "
            f"n_terms={self.terms.n_terms})"
        )
